"""
tests/test_prompt_strategies.py
Banc de test comparatif des stratégies de prompting (Zero-Shot, Délimiteurs, Few-Shot, CoT)
adaptées à Llama 3.2 pour l'extraction structurée d'offres d'emploi.

Inspiré directement de la méthodologie du cours M2 IA (Tp1 prompt_service.py).
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Résolution des imports et de l'encodage Windows
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pydantic import ValidationError
from src.config import settings
from src.parsers.job_enricher import JobEnrichmentOutput, JobEnricher
from src.utils.ollama_client import OllamaClient
from src.database.db_session import SessionLocal
from src.database.sql_models import Offre


# ==============================================================================
# 1. ÉCHANTILLON DE TEST RÉEL (Offre Talan issue de PostgreSQL ou Fallback)
# ==============================================================================
def get_sample_job() -> Dict[str, str]:
    """Récupère l'offre Talan réelle en base ou un exemple type."""
    db = SessionLocal()
    try:
        offre = db.query(Offre).filter(Offre.company == "Talan").first()
        if offre:
            snippet = JobEnricher.extract_requirements_snippet(offre.description)
            return {
                "poste": offre.poste,
                "company": offre.company,
                "snippet": snippet
            }
    finally:
        db.close()

    # Fallback si PostgreSQL non dispo
    return {
        "poste": "Stage Consultant Data Analytics & Visualisation",
        "company": "Talan",
        "snippet": (
            "Data préparation Dataiku, Dbt, Alteryx.\n"
            "Qualifications requises :\n"
            "- Cursus ingénieur ou double diplôme (Bac+5)\n"
            "- Solides compétences en SQL et Python\n"
            "- Stage de fin d'études (débutant accepté)\n"
            "- Anglais courant"
        )
    }


# ==============================================================================
# 2. CONSTRUCTEURS DE PROMPTS (Adaptés du cours Tp1/prompt_service.py)
# ==============================================================================

def build_zeroshot_simple(poste: str, text: str) -> List[Dict[str, str]]:
    """
    Stratégie 1 : Zero-Shot Simple (Notebooks 1 & 5).
    Demande de JSON sans délimiteurs stricts.
    """
    prompt = (
        f"Tu es un assistant RH. Extrais du texte ci-dessous les compétences techniques, "
        f"le niveau d'études en années post-bac (ex: 5 pour Master/Ingénieur, 0 si non précisé) "
        f"et les années d'expérience requises sous format JSON avec les clés : "
        f"'competences_necessaires', 'niveau_etude_min', 'min_experience_years'.\n\n"
        f"Poste : {poste}\n"
        f"Texte : {text}"
    )
    return [{"role": "user", "content": prompt}]


def build_zeroshot_structured(poste: str, text: str) -> List[Dict[str, str]]:
    """
    Stratégie 2 : Zero-Shot Structuré avec Délimiteurs Hermétiques (Notebooks 3 & 4).
    Hiérarchie [SYSTEM] > [INSTRUCTIONS] > [DATA] et règles anti-hallucination.
    """
    system_prompt = (
        "Tu es un extracteur d'entités de recrutement IT en JSON strict.\n"
        "RÈGLES IMPÉRATIVES :\n"
        "1. Priorité absolue : SYSTEM > INSTRUCTIONS > DATA.\n"
        "2. RÈGLE ANTI-HALLUCINATION : N'extrais STRICTEMENT que les outils et diplômes explicitement présents dans [DATA].\n"
        "3. INTERDICTION FORMELLE DE FAIRE DES LISTES OU DE RÉPÉTER DES MOTS EN BOUCLE.\n"
        "4. Réponds UNIQUEMENT par un objet JSON valide débutant par { et finissant par }.\n"
        "Format requis :\n"
        "{\"competences_necessaires\": [\"...\"], \"niveau_etude_min\": 5, \"min_experience_years\": 0.0}"
    )
    user_prompt = (
        f"[INSTRUCTIONS]\n"
        f"Extrais les critères pour le poste '{poste}' sous le format JSON obligatoire :\n\n"
        f"[DATA]\n"
        f"{text}\n\n"
        f"JSON :"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def build_fewshot(poste: str, text: str) -> List[Dict[str, str]]:
    """
    Stratégie 3 : Few-Shot Learning (Notebook 6).
    Apprentissage en contexte avec 2 exemples réels (Input -> JSON Output)
    extrêmement efficace pour stabiliser les petits modèles comme Llama 3.2.
    """
    system_prompt = (
        "Tu es un extracteur de critères de recrutement IT en JSON.\n"
        "Réponds STRICTEMENT par un objet JSON respectant les clés exactes. "
        "Ne génère AUCUN texte superflu. N'invente aucun outil non présent dans le texte."
    )

    shot1_user = (
        "Poste : Développeur Python Backend\n"
        "Extrait : Profil : De formation Bac+3 ou Bac+5 informatique, vous avez 2 ans d'expérience. "
        "Compétences exigées : Python, Django, PostgreSQL, Docker."
    )
    shot1_assistant = json.dumps({
        "competences_necessaires": ["Python", "Django", "PostgreSQL", "Docker"],
        "niveau_etude_min": 3,
        "min_experience_years": 2.0
    }, ensure_ascii=False)

    shot2_user = (
        "Poste : Data Scientist R&D NLP\n"
        "Extrait : Diplôme d'ingénieur ou Master 2 requis. Connaissances approfondies en PyTorch, "
        "HuggingFace et SQL. Débutant / Stage de fin d'études bienvenu."
    )
    shot2_assistant = json.dumps({
        "competences_necessaires": ["PyTorch", "HuggingFace", "SQL"],
        "niveau_etude_min": 5,
        "min_experience_years": 0.0
    }, ensure_ascii=False)

    target_user = f"Poste : {poste}\nExtrait : {text}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": shot1_user},
        {"role": "assistant", "content": shot1_assistant},
        {"role": "user", "content": shot2_user},
        {"role": "assistant", "content": shot2_assistant},
        {"role": "user", "content": target_user}
    ]


def build_cot(poste: str, text: str) -> List[Dict[str, str]]:
    """
    Stratégie 4 : Chain-of-Thought / Raisonnement Décomposé (Notebook TP1).
    Décompose l'analyse (repérage diplôme, outils, expérience) avant le JSON final.
    """
    system_prompt = (
        "Tu es un analyste de recrutement méthodique.\n"
        "Décompose ton analyse en 2 étapes rapides :\n"
        "1. Énumère brièvement le diplôme identifié, l'expérience et les outils techniques.\n"
        "2. Conclus OBLIGATOIREMENT par un bloc JSON final sous balises ```json ... ```\n"
        "au format exact :\n"
        "{\"competences_necessaires\": [...], \"niveau_etude_min\": 5, \"min_experience_years\": 0.0}"
    )
    user_prompt = f"Analyse et extrais les critères de cette offre :\nPoste : {poste}\n\nTexte :\n{text}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


STRATEGIES = {
    "zeroshot_simple": build_zeroshot_simple,
    "zeroshot_structured": build_zeroshot_structured,
    "fewshot": build_fewshot,
    "cot": build_cot,
}


# ==============================================================================
# 3. MOTEUR D'ÉVALUATION ET DE BENCHMARK
# ==============================================================================

def run_strategy_benchmark(strategy_name: str, sample: Dict[str, str], model_name: str, client: OllamaClient):
    print("\n" + "=" * 75)
    print(f"🧪 STRATÉGIE : {strategy_name.upper()} | MODÈLE : {model_name}")
    print("=" * 75)

    builder = STRATEGIES[strategy_name]
    messages = builder(sample["poste"], sample["snippet"])

    # format="json" activé pour forcer la sortie JSON sauf pour CoT qui fait un raisonnement préalable
    use_format_json = "json" if strategy_name != "cot" else None

    start_time = time.time()
    try:
        raw_output = client.chat(
            messages=messages,
            model=model_name,
            format=use_format_json,
            temperature=0.0,
            num_ctx=1024,
            num_predict=256
        )
        duration = round(time.time() - start_time, 2)
        print(f"⏱️ Temps de réponse : {duration}s")
        print("\n📝 Sortie brute reçue :")
        print("-" * 50)
        print(raw_output.strip())
        print("-" * 50)

        # Validation Pydantic
        parsed = client._clean_and_parse(raw_output, JobEnrichmentOutput)
        print("✅ Validation Pydantic : SUCCÈS")
        print(f"   • Diplôme min (Bac+) : {parsed.niveau_etude_min}")
        print(f"   • Expérience min      : {parsed.min_experience_years} an(s)")
        print(f"   • Compétences ({len(parsed.competences_necessaires)}) : {parsed.competences_necessaires}")

    except ValidationError as ve:
        print(f"❌ Validation Pydantic : ÉCHEC (Schéma invalide : {ve})")
    except Exception as e:
        print(f"❌ Erreur d'exécution : {e}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark des stratégies de prompting pour Llama 3.2 (M2 IA)")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:1b",
        help="Modèle Ollama à tester (ex: llama3.2:1b ou llama3.2:3b)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=list(STRATEGIES.keys()) + ["all"],
        default="all",
        help="Stratégie de prompt à tester (par défaut: all)"
    )

    args = parser.parse_args()

    client = OllamaClient(default_model=args.model)
    if not client.is_available():
        print(f"❌ Impossible de joindre Ollama sur {settings.OLLAMA_BASE_URL}. Vérifiez qu'Ollama est lancé.")
        sys.exit(1)

    sample = get_sample_job()
    print("🎯 Échantillon de test chargé :")
    print(f"   • Poste      : {sample['poste']} ({sample['company']})")
    print(f"   • Extrait    : {len(sample['snippet'])} caractères")

    if args.strategy == "all":
        for strat in STRATEGIES.keys():
            run_strategy_benchmark(strat, sample, args.model, client)
    else:
        run_strategy_benchmark(args.strategy, sample, args.model, client)


if __name__ == "__main__":
    main()
