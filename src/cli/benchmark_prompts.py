"""
src/cli/benchmark_prompts.py
Banc d'essai scientifique (Benchmark) des stratégies de prompting pour Llama 3.2.
Compare scientifiquement les architectures de prompts inspirées du cours M2 IA (TP1 prompt_service.py)
sur des offres réelles PostgreSQL pour déterminer le prompt le plus rapide, fiable et précis.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

# Résolution des imports et de l'encodage Windows
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pydantic import ValidationError
from sqlalchemy import func
from src.config import settings
from src.database.db_session import SessionLocal
from src.database.sql_models import Offre
from src.parsers.job_enricher import JobEnrichmentOutput, JobEnricher
from src.utils.ollama_client import OllamaClient


# ==============================================================================
# 1. STRUCTURES DE DONNÉES DU BENCHMARK
# ==============================================================================
@dataclass
class JobSample:
    id: str
    company: str
    poste: str
    snippet: str
    full_length: int


@dataclass
class StrategyMetrics:
    name: str
    description: str
    total_runs: int = 0
    pydantic_valid_count: int = 0
    total_latency_seconds: float = 0.0
    total_skills_extracted: int = 0
    diploma_detected_count: int = 0
    loop_detected_count: int = 0  # Détection de répétitions anormales
    results_detail: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return (self.pydantic_valid_count / self.total_runs * 100.0) if self.total_runs > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        return (self.total_latency_seconds / self.total_runs) if self.total_runs > 0 else 0.0

    @property
    def avg_skills(self) -> float:
        return (self.total_skills_extracted / self.pydantic_valid_count) if self.pydantic_valid_count > 0 else 0.0

    @property
    def overall_score(self) -> float:
        """Score consolidé sur 100 valorisant la validité, la vitesse et la pertinence."""
        # 50% Taux de validité Pydantic
        score_valid = (self.success_rate / 100.0) * 50.0
        # 30% Rapidité (pénalité si > 30s)
        latency_score = max(0.0, 30.0 - (self.avg_latency * 0.75))
        # 20% Richesse (objectif: 3 à 8 compétences pertinentes)
        skills_score = min(20.0, self.avg_skills * 3.5)
        # Pénalité si boucles infinies détectées
        loop_penalty = self.loop_detected_count * 25.0
        return max(0.0, round(score_valid + latency_score + skills_score - loop_penalty, 1))


# ==============================================================================
# 2. LES 4 STRATÉGIES DE PROMPTS TESTÉES (Inspirées du TP1 M2 IA)
# ==============================================================================

def prompt_zeroshot_simple(job: JobSample) -> List[Dict[str, str]]:
    """Stratégie 1 : Zero-Shot Simple avec demande de JSON (Notebook 1)."""
    text = (
        f"Tu es un recruteur IT. Extrais sous format JSON avec les clés 'competences_necessaires', "
        f"'niveau_etude_min' (ex: 5 pour Master/Ingénieur, 0 sinon), et 'min_experience_years'.\n\n"
        f"Poste : {job.poste}\n"
        f"Texte :\n{job.snippet}"
    )
    return [{"role": "user", "content": text}]


def prompt_zeroshot_structured(job: JobSample) -> List[Dict[str, str]]:
    """Stratégie 2 : Zero-Shot Structuré avec Délimiteurs Hermétiques [SYSTEM] > [INSTRUCTIONS] > [DATA] (Notebook 3)."""
    system_prompt = (
        "Tu es un extracteur d'entités de recrutement technique en format JSON strict.\n"
        "RÈGLES IMPÉRATIVES :\n"
        "1. Priorité absolue : SYSTEM > INSTRUCTIONS > DATA.\n"
        "2. RÈGLE ANTI-HALLUCINATION : N'extrais STRICTEMENT que les compétences et diplômes explicitement présents dans [DATA].\n"
        "3. INTERDICTION FORMELLE DE RÉPÉTER LES MÊMES TERMES EN BOUCLE.\n"
        "4. Réponds UNIQUEMENT par un objet JSON valide commençant par { et finissant par }."
    )
    user_prompt = (
        f"[INSTRUCTIONS]\n"
        f"Extrais les critères pour le poste '{job.poste}' sous ce format JSON strict :\n"
        f"{{\"competences_necessaires\": [\"...\"], \"niveau_etude_min\": 5, \"min_experience_years\": 0.0}}\n\n"
        f"[DATA]\n"
        f"{job.snippet}\n\n"
        f"JSON :"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def prompt_fewshot_standard(job: JobSample) -> List[Dict[str, str]]:
    """Stratégie 3 : Few-Shot Learning avec 2 exemples Input -> Assistant JSON (Notebook 6)."""
    system_prompt = (
        "Tu es un assistant RH technique. Réponds STRICTEMENT par un objet JSON valide. "
        "Ne génère aucun commentaire. N'invente aucune compétence absente de l'extrait."
    )
    shot1_user = (
        "Poste : Développeur Python Backend\n"
        "Extrait : Profil : De formation Bac+3 ou Bac+5 informatique, 2 ans d'expérience. "
        "Compétences requises : Python, Django, PostgreSQL, Docker."
    )
    shot1_assistant = json.dumps({
        "competences_necessaires": ["Python", "Django", "PostgreSQL", "Docker"],
        "niveau_etude_min": 3,
        "min_experience_years": 2.0
    }, ensure_ascii=False)

    shot2_user = (
        "Poste : Data Scientist R&D\n"
        "Extrait : Diplôme d'ingénieur ou Master 2 requis. Connaissances approfondies en PyTorch, "
        "SQL et Git. Stage de fin d'études bienvenu."
    )
    shot2_assistant = json.dumps({
        "competences_necessaires": ["PyTorch", "SQL", "Git"],
        "niveau_etude_min": 5,
        "min_experience_years": 0.0
    }, ensure_ascii=False)

    target_user = f"Poste : {job.poste}\nExtrait : {job.snippet}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": shot1_user},
        {"role": "assistant", "content": shot1_assistant},
        {"role": "user", "content": shot2_user},
        {"role": "assistant", "content": shot2_assistant},
        {"role": "user", "content": target_user}
    ]


def prompt_fewshot_compact(job: JobSample) -> List[Dict[str, str]]:
    """Stratégie 4 : Few-Shot Compact (1 seul exemple ultra-concis calibré pour mini contexte CPU)."""
    system_prompt = (
        "Tu es un extracteur d'entités JSON. Extrais uniquement les outils informatiques, "
        "le niveau d'études post-bac (entier, ex: 5 pour Master/Ingénieur) et l'expérience."
    )
    shot_user = "Poste : Analyste Data\nExtrait : Bac+5 ingénieur, maîtrise SQL et Tableau. Débutant."
    shot_assistant = '{"competences_necessaires": ["SQL", "Tableau"], "niveau_etude_min": 5, "min_experience_years": 0.0}'

    target_user = f"Poste : {job.poste}\nExtrait : {job.snippet}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": shot_user},
        {"role": "assistant", "content": shot_assistant},
        {"role": "user", "content": target_user}
    ]


PROMPT_REGISTRY = {
    "zeroshot_simple": {
        "fn": prompt_zeroshot_simple,
        "desc": "Zero-Shot Simple (JSON demandé sans règles strictes)",
    },
    "zeroshot_structured": {
        "fn": prompt_zeroshot_structured,
        "desc": "Zero-Shot Structuré avec délimiteurs hermétiques [SYSTEM] > [DATA]",
    },
    "fewshot_standard": {
        "fn": prompt_fewshot_standard,
        "desc": "Few-Shot 2 Exemples (Pattern Matching en contexte)",
    },
    "fewshot_compact": {
        "fn": prompt_fewshot_compact,
        "desc": "Few-Shot 1 Exemple Compact (Optimisé légèreté CPU)",
    },
}


# ==============================================================================
# 3. CHARGEMENT DU JEU DE TEST RÉEL (PostgreSQL)
# ==============================================================================
def load_benchmark_jobs(limit: int = 3) -> List[JobSample]:
    """Charge des offres réelles représentatives issues de PostgreSQL."""
    db = SessionLocal()
    samples: List[JobSample] = []
    try:
        # Cible les offres riches de JobSpy (LinkedIn/Indeed)
        offres = db.query(Offre).filter(func.length(Offre.description) >= 1000).limit(limit).all()
        for o in offres:
            snippet = JobEnricher.extract_requirements_snippet(o.description, max_chars=1100)
            samples.append(JobSample(
                id=o.id,
                company=o.company,
                poste=o.poste,
                snippet=snippet,
                full_length=len(o.description)
            ))
    finally:
        db.close()

    if not samples:
        # Fallback de secours si la base est vide
        samples = [
            JobSample(
                id="test_talan",
                company="Talan",
                poste="Stage Consultant Data Analytics",
                snippet="Data préparation Dataiku, Dbt. Profil : Cursus ingénieur (Bac+5). Solides compétences SQL et Python.",
                full_length=1500
            ),
            JobSample(
                id="test_celio",
                company="Celio",
                poste="Lead Data Scientist",
                snippet="Formation Bac+5 / Master IA. Maîtrise de Python, PyTorch, Scikit-Learn, Docker, Kubernetes. 5 ans d'expérience.",
                full_length=2000
            )
        ]
    return samples


# ==============================================================================
# 4. EXÉCUTION DU BENCHMARK
# ==============================================================================
def execute_benchmark(
    jobs: List[JobSample],
    model_name: str,
    target_strategies: List[str]
) -> List[StrategyMetrics]:
    client = OllamaClient(default_model=model_name, timeout=240)
    metrics_list: List[StrategyMetrics] = []

    print("\n" + "=" * 80)
    print(f"🏁 LANCEMENT DU BENCHMARK COMPARATIF DE PROMPTS")
    print(f"• Modèle testé       : {model_name}")
    print(f"• Nombre d'offres   : {len(jobs)} offres réelles")
    print(f"• Stratégies testées : {len(target_strategies)}")
    print("=" * 80)

    for strat_key in target_strategies:
        strat_info = PROMPT_REGISTRY[strat_key]
        strat_fn = strat_info["fn"]
        metrics = StrategyMetrics(name=strat_key, description=strat_info["desc"])

        print(f"\n▶ Évaluation de la stratégie : [{strat_key}]")
        print(f"  {strat_info['desc']}")
        print("-" * 80)

        for idx, job in enumerate(jobs, 1):
            metrics.total_runs += 1
            messages = strat_fn(job)

            t0 = time.time()
            try:
                raw = client.chat(
                    messages=messages,
                    model=model_name,
                    format="json",
                    temperature=0.0,
                    num_ctx=1024,
                    num_predict=200
                )
                lat = round(time.time() - t0, 2)
                metrics.total_latency_seconds += lat

                # Détection de boucle infinie (répétitions anormales)
                if raw.count("Kubernetes") > 3 or raw.count("Docker") > 3 or len(raw) > 900:
                    metrics.loop_detected_count += 1
                    print(f"  [{idx}/{len(jobs)}] {job.company[:12]} | ⏱️ {lat}s | ⚠️ BOUCLE INFINIE DÉTECTÉE !")
                    continue

                # Validation Pydantic
                parsed = client._clean_and_parse(raw, JobEnrichmentOutput)
                metrics.pydantic_valid_count += 1
                metrics.total_skills_extracted += len(parsed.competences_necessaires)
                if parsed.niveau_etude_min > 0:
                    metrics.diploma_detected_count += 1

                skills_str = ", ".join(parsed.competences_necessaires[:4])
                if len(parsed.competences_necessaires) > 4:
                    skills_str += f" (+{len(parsed.competences_necessaires) - 4})"

                print(
                    f"  [{idx}/{len(jobs)}] {job.company[:12]} | ⏱️ {lat}s | "
                    f"✅ Validé | Bac+{parsed.niveau_etude_min} | Exp: {parsed.min_experience_years}a | "
                    f"Compétences ({len(parsed.competences_necessaires)}): [{skills_str}]"
                )

            except ValidationError as ve:
                lat = round(time.time() - t0, 2)
                metrics.total_latency_seconds += lat
                print(f"  [{idx}/{len(jobs)}] {job.company[:12]} | ⏱️ {lat}s | ❌ Erreur Pydantic: {ve}")
            except Exception as e:
                lat = round(time.time() - t0, 2)
                metrics.total_latency_seconds += lat
                print(f"  [{idx}/{len(jobs)}] {job.company[:12]} | ⏱️ {lat}s | ❌ Erreur Inférence: {e}")

        metrics_list.append(metrics)

    return metrics_list


# ==============================================================================
# 5. RESTITUTION DU CLASSEMENT ET DU PROMPT GAGNANT
# ==============================================================================
def display_leaderboard(metrics_list: List[StrategyMetrics], model_name: str, num_jobs: int):
    # Tri par score global décroissant
    sorted_metrics = sorted(metrics_list, key=lambda m: m.overall_score, reverse=True)
    winner = sorted_metrics[0]

    print("\n" + "=" * 90)
    print("🏆 CLASSEMENT OFFICIEL DU BENCHMARK (LEADERBOARD MASTER 2 IA)")
    print("=" * 90)
    print(f"{'Rang':<5} | {'Stratégie':<22} | {'Succès Pydantic':<16} | {'Latence Moy.':<13} | {'Compétences':<12} | {'Score /100'}")
    print("-" * 90)

    for rank, m in enumerate(sorted_metrics, 1):
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"#{rank}"))
        print(
            f"{medal:<5} | {m.name:<22} | {m.success_rate:>13.1f} % | {m.avg_latency:>10.1f} s | "
            f"{m.avg_skills:>10.1f}  | {m.overall_score:>8.1f} pts"
        )
    print("=" * 90)

    print(f"\n🎯 VERDICT : LE PROMPT LE PLUS ADAPTÉ EST >>> [{winner.name.upper()}] <<<")
    print(f"   • Description : {winner.description}")
    print(f"   • Taux de succès Pydantic : {winner.success_rate:.1f} %")
    print(f"   • Temps moyen par offre   : {winner.avg_latency:.1f} secondes")
    print(f"   • Compétences moyennes    : {winner.avg_skills:.1f} par offre")
    print(f"   • Score global            : {winner.overall_score} / 100")
    print("\n💡 Pour appliquer ce prompt gagnant en production, nous l'activerons par défaut dans 'job_enricher.py'.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark comparatif des prompts pour Llama 3.2 (M2 IA)")
    parser.add_argument("--model", type=str, default="llama3.2:1b", help="Modèle Ollama (défaut: llama3.2:1b)")
    parser.add_argument("--num-jobs", type=int, default=3, help="Nombre d'offres réelles à tester (défaut: 3)")
    parser.add_argument("--strategy", type=str, choices=list(PROMPT_REGISTRY.keys()) + ["all"], default="all")

    args = parser.parse_args()

    client = OllamaClient(default_model=args.model)
    if not client.is_available():
        print(f"❌ Ollama n'est pas joignable sur {settings.OLLAMA_BASE_URL}. Vérifiez qu'Ollama est lancé.")
        sys.exit(1)

    jobs = load_benchmark_jobs(limit=args.num_jobs)
    print(f"📦 {len(jobs)} offres réelles chargées pour l'évaluation :")
    for j in jobs:
        print(f"   • [{j.company}] {j.poste} ({j.full_length} chars -> snippet {len(j.snippet)} chars)")

    target_strategies = list(PROMPT_REGISTRY.keys()) if args.strategy == "all" else [args.strategy]
    metrics = execute_benchmark(jobs, args.model, target_strategies)
    display_leaderboard(metrics, args.model, len(jobs))


if __name__ == "__main__":
    main()
