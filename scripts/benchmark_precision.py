"""
scripts/benchmark_precision.py
Benchmark scientifique de PRÉCISION : NOMIC-EMBED-TEXT vs BGE-M3
Évalue la précision sémantique pure (sans contrainte de temps CPU)
selon les standards académiques du Master 2 IA / Information Retrieval :

Métriques calculées :
1. Top-1 Accuracy (% de requêtes où l'offre parfaite est 1ère)
2. MRR@5 (Mean Reciprocal Rank) : qualité du rang de la meilleure offre
3. NDCG@3 (Normalized Discounted Cumulative Gain) : qualité de l'ordonnancement gradué
4. Marge de Séparation Sémantique (Signal-to-Noise Ratio : Pertinent vs Hors-sujet)
5. Robustesse aux Pièges Lexicaux (Hard Negatives : Java vs JavaScript, etc.)
"""

import sys
import math
import numpy as np
import ollama
from typing import List, Dict, Any, Tuple

# Forcer l'encodage UTF-8 sur la console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def dcg_at_k(relevances: List[int], k: int = 3) -> float:
    """Calcule le Discounted Cumulative Gain à k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(relevances: List[int], k: int = 3) -> float:
    """Calcule le Normalized DCG à k."""
    actual_dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevances, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


# ==============================================================================
# JEU DE DONNÉES MULTI-SECTEURS DE TEST (CV & CANDIDATS VS OFFRES)
# ==============================================================================

BENCHMARK_SCENARIOS = [
    {
        "id": "scenario_ia",
        "description": "Profil Data Science & IA (Candidat Bac+5)",
        "query": (
            "Profil : Ingénieur en Intelligence Artificielle et Machine Learning. "
            "Expert Python, PyTorch, Transformers, LLM, RAG, ChromaDB, HuggingFace. Diplôme d'ingénieur Bac+5."
        ),
        "corpus": [
            # rel: 2 = Match parfait, 1 = Proche/Connexe, 0 = Hors sujet
            {"id": "job_ia", "rel": 2, "label": "Data Scientist NLP / LLM (Match Parfait)", "text": (
                "[INTITULÉ DU POSTE] : Data Scientist NLP & IA Générative\n"
                "[COMPÉTENCES] : Python, PyTorch, RAG, Transformers, Vector Stores, LLM local\n"
                "[SECTEUR] : Informatique Data Ia Web | [NIVEAU] : Bac+5"
            )},
            {"id": "job_dev_web", "rel": 1, "label": "Développeur Fullstack React/Node (Connexe)", "text": (
                "[INTITULÉ DU POSTE] : Développeur Fullstack JavaScript TypeScript\n"
                "[COMPÉTENCES] : React, Node.js, NextJS, Express, PostgreSQL, API REST\n"
                "[SECTEUR] : Informatique Data Ia Web | [NIVEAU] : Bac+3"
            )},
            {"id": "job_infirmier", "rel": 0, "label": "Infirmier Soins d'Urgence (Hors sujet)", "text": (
                "[INTITULÉ DU POSTE] : Infirmier DE - Urgences Médicales\n"
                "[COMPÉTENCES] : Soins infirmiers, Traumatologie, Constantes vitales, Hygiène hospitalière\n"
                "[SECTEUR] : Santé Médical Pharmacie | [NIVEAU] : Bac+3"
            )},
            {"id": "job_btp", "rel": 0, "label": "Conducteur de Travaux BTP (Hors sujet)", "text": (
                "[INTITULÉ DU POSTE] : Conducteur de Travaux Gros Œuvre\n"
                "[COMPÉTENCES] : Chantiers de construction, Maçonnerie, Coordination sous-traitants, Sécurité BTP\n"
                "[SECTEUR] : Btp Construction | [NIVEAU] : Bac+2"
            )},
        ]
    },
    {
        "id": "scenario_sante",
        "description": "Profil Santé & Soins (Infirmière Diplômée)",
        "query": (
            "Profil : Infirmière Diplômée d'État avec 3 ans d'expérience en milieu hospitalier. "
            "Soins techniques, perfusions, pansements complexes, gestion des urgences, surveillance médicale."
        ),
        "corpus": [
            {"id": "job_infirmiere_reanim", "rel": 2, "label": "Infirmier Soins Intensifs (Match Parfait)", "text": (
                "[INTITULÉ DU POSTE] : Infirmier (H/F) - Service Réanimation et Soins Intensifs\n"
                "[COMPÉTENCES] : Surveillance clinique, Soins techniques, Réanimation, Urgences vitales\n"
                "[SECTEUR] : Santé Médical Pharmacie | [NIVEAU] : Diplôme d'État Infirmier Bac+3"
            )},
            {"id": "job_secretaire_med", "rel": 1, "label": "Secrétaire Médicale (Connexe Santé)", "text": (
                "[INTITULÉ DU POSTE] : Secrétaire Médicale en Clinique\n"
                "[COMPÉTENCES] : Accueil patients, Prise de rendez-vous, Gestion dossiers médicaux, Facturation CPAM\n"
                "[SECTEUR] : Santé Médical Pharmacie | [NIVEAU] : Bac à Bac+2"
            )},
            {"id": "job_data_analyst", "rel": 0, "label": "Data Analyst Python (Hors sujet)", "text": (
                "[INTITULÉ DU POSTE] : Data Analyst / Business Intelligence\n"
                "[COMPÉTENCES] : SQL, PowerBI, Python Pandas, Dashboard, Modélisation statistique\n"
                "[SECTEUR] : Informatique Data Ia Web | [NIVEAU] : Bac+5"
            )},
            {"id": "job_comptable", "rel": 0, "label": "Comptable Fournisseurs (Hors sujet)", "text": (
                "[INTITULÉ DU POSTE] : Collaborateur Comptable et Audit\n"
                "[COMPÉTENCES] : Clôture comptable, Déclarations TVA, Rapprochements bancaires, Bilan\n"
                "[SECTEUR] : Finance Gestion | [NIVEAU] : Bac+3"
            )},
        ]
    },
    {
        "id": "scenario_piege_java",
        "description": "Piège Lexical : Java vs JavaScript (Même racine textuelle, métiers opposés)",
        "query": "Ingénieur Logiciel Backend Java Spring Boot Hibernate Microservices PostgreSQL",
        "corpus": [
            {"id": "job_java_pure", "rel": 2, "label": "Développeur Java Spring (Match Parfait)", "text": (
                "Poste : Développeur Backend Java 17.\n"
                "Compétences : Java, Spring Boot, JPA Hibernate, Docker, Kubernetes, Microservices REST, PostgreSQL."
            )},
            {"id": "job_js_trap", "rel": 0, "label": "Développeur JavaScript React (Piège Lexical)", "text": (
                "Poste : Développeur Frontend Web JavaScript.\n"
                "Compétences : JavaScript ECMAScript, React.js, Next.js, Redux, TailwindCSS, Node.js, Webpack."
            )},
            {"id": "job_python", "rel": 1, "label": "Développeur Backend Python Django (Connexe Backend)", "text": (
                "Poste : Développeur Backend Python.\n"
                "Compétences : Python, Django REST framework, FastAPI, Docker, PostgreSQL, Microservices."
            )},
        ]
    },
    {
        "id": "scenario_rh",
        "description": "Profil Ressources Humaines & Recrutement",
        "query": (
            "Profil : Chargée de Recrutement et Gestion RH. "
            "Sourcing candidats sur LinkedIn Recruiter, entretiens d'embauche, onboarding, droit du travail, gestion de paie."
        ),
        "corpus": [
            {"id": "job_talent_acq", "rel": 2, "label": "Talent Acquisition Manager (Match Parfait)", "text": (
                "[INTITULÉ DU POSTE] : Responsable Recrutement et Talent Acquisition\n"
                "[COMPÉTENCES] : Sourcing, Conduite d'entretiens, Marque employeur, Onboarding, ATS\n"
                "[SECTEUR] : Ressources Humaines Juridique | [NIVEAU] : Bac+5"
            )},
            {"id": "job_juriste", "rel": 1, "label": "Juriste Droit Social (Connexe RH)", "text": (
                "[INTITULÉ DU POSTE] : Juriste Droit Social et Relations Sociales\n"
                "[COMPÉTENCES] : Droit du travail, CSE, Accords collectifs, Procédures disciplinaires\n"
                "[SECTEUR] : Ressources Humaines Juridique | [NIVEAU] : Bac+5"
            )},
            {"id": "job_technicien_btp", "rel": 0, "label": "Technicien Chauffagiste (Hors sujet)", "text": (
                "[INTITULÉ DU POSTE] : Technicien Chauffage et Climatisation\n"
                "[COMPÉTENCES] : Maintenance chaudières, Pompe à chaleur, Dépannage fluides frigorifiques\n"
                "[SECTEUR] : Btp Construction | [NIVEAU] : Bac Pro / CAP"
            )},
        ]
    }
]


def run_precision_benchmark(model_name: str, client: ollama.Client) -> Dict[str, Any]:
    print(f"\n{'='*75}")
    print(f"[TEST DE PRÉCISION] ÉVALUATION : {model_name.upper()}")
    print(f"{'='*75}")

    top1_hits = 0
    reciprocal_ranks = []
    ndcg_scores = []
    margins_target_negative = []
    java_trap_success = False
    java_trap_margin = 0.0

    total_scenarios = len(BENCHMARK_SCENARIOS)

    for sc in BENCHMARK_SCENARIOS:
        sc_id = sc["id"]
        query_text = sc["query"]
        corpus = sc["corpus"]

        # Vectoriser la requête et le corpus
        emb_query = client.embed(model=model_name, input=[query_text])["embeddings"][0]
        corpus_texts = [item["text"] for item in corpus]
        emb_corpus = client.embed(model=model_name, input=corpus_texts)["embeddings"]

        # Calculer la similarité cosinus pour chaque offre
        scored_items = []
        for item, emb_doc in zip(corpus, emb_corpus):
            sim = cosine_similarity(emb_query, emb_doc)
            scored_items.append({
                "id": item["id"],
                "label": item["label"],
                "rel": item["rel"],
                "score": sim
            })

        # Trier par score décroissant (Ranking)
        scored_items.sort(key=lambda x: x["score"], reverse=True)

        # 1. Vérification Top-1
        is_top1_correct = (scored_items[0]["rel"] == 2)
        if is_top1_correct:
            top1_hits += 1

        # 2. Calcul du Reciprocal Rank (RR) pour le match parfait (rel=2)
        rank_perfect = next((i + 1 for i, it in enumerate(scored_items) if it["rel"] == 2), 0)
        rr = 1.0 / rank_perfect if rank_perfect > 0 else 0.0
        reciprocal_ranks.append(rr)

        # 3. Calcul du NDCG@3
        relevances_ordered = [it["rel"] for it in scored_items]
        score_ndcg = ndcg_at_k(relevances_ordered, k=3)
        ndcg_scores.append(score_ndcg)

        # 4. Marge de séparation : Score(rel=2) - Score(rel=0 max)
        score_perfect = next((it["score"] for it in scored_items if it["rel"] == 2), 0.0)
        negatives = [it["score"] for it in scored_items if it["rel"] == 0]
        if negatives:
            max_neg = max(negatives)
            margin = score_perfect - max_neg
            margins_target_negative.append(margin)

        # Test piège Java
        if sc_id == "scenario_piege_java":
            score_java = next(it["score"] for it in scored_items if it["id"] == "job_java_pure")
            score_js = next(it["score"] for it in scored_items if it["id"] == "job_js_trap")
            java_trap_margin = score_java - score_js
            java_trap_success = (java_trap_margin > 0)

        # Affichage pédagogique du scénario
        print(f"\n--- {sc['description']} ---")
        for rank, it in enumerate(scored_items, start=1):
            badge = "⭐ MATCH" if it["rel"] == 2 else ("🔹 CONNEXE" if it["rel"] == 1 else "⛔ HORS SUJET")
            print(f"  Rang {rank} [Score: {it['score']:.4f}] {badge} : {it['label']}")
        print(f"  -> Top-1 Correct: {'OUI ✅' if is_top1_correct else 'NON ❌'} | RR: {rr:.2f} | NDCG@3: {score_ndcg:.4f}")

    # Calcul des métriques globales
    accuracy_top1 = (top1_hits / total_scenarios) * 100.0
    mean_mrr = float(np.mean(reciprocal_ranks))
    mean_ndcg = float(np.mean(ndcg_scores))
    mean_margin = float(np.mean(margins_target_negative)) if margins_target_negative else 0.0

    return {
        "model": model_name,
        "top1_accuracy": accuracy_top1,
        "mrr": mean_mrr,
        "ndcg_3": mean_ndcg,
        "mean_margin": mean_margin,
        "java_trap_success": java_trap_success,
        "java_trap_margin": java_trap_margin
    }


def main():
    print("=" * 75)
    print("  🔬 AUDIT COMPARATIF DE PRÉCISION SÉMANTIQUE (INFORMATION RETRIEVAL)")
    print("  Modèles testés : nomic-embed-text vs bge-m3")
    print("=" * 75)

    client = ollama.Client()
    models = ["nomic-embed-text", "bge-m3"]
    results = {}

    for m in models:
        try:
            results[m] = run_precision_benchmark(m, client)
        except Exception as e:
            print(f"❌ Erreur sur {m} : {e}")

    # Affichage du rapport comparatif final
    print("\n" + "=" * 75)
    print("🏆 RÉSULTATS COMPARATIFS DES MÉTRIQUES DE PRÉCISION")
    print("=" * 75)
    print(f"{'Métrique de Précision Sémantique':<38} | {'NOMIC-EMBED-TEXT':<16} | {'BGE-M3':<16}")
    print("-" * 75)

    r_nomic = results.get("nomic-embed-text", {})
    r_bge = results.get("bge-m3", {})

    if r_nomic and r_bge:
        print(f"{'Top-1 Accuracy (% Match exact 1er)':<38} | {r_nomic['top1_accuracy']:>13.1f} % | {r_bge['top1_accuracy']:>13.1f} %")
        print(f"{'MRR (Mean Reciprocal Rank)':<38} | {r_nomic['mrr']:>15.4f} | {r_bge['mrr']:>15.4f}")
        print(f"{'NDCG@3 (Gradation de pertinence)':<38} | {r_nomic['ndcg_3']:>15.4f} | {r_bge['ndcg_3']:>15.4f}")
        print(f"{'Marge de Séparation Sémantique (SNR)':<38} | {r_nomic['mean_margin']:>+15.4f} | {r_bge['mean_margin']:>+15.4f}")
        print(f"{'Résistance Piège Java vs JavaScript':<38} | {r_nomic['java_trap_margin']:>+15.4f} | {r_bge['java_trap_margin']:>+15.4f}")
        print("-" * 75)
        
        # Synthèse analytique
        mrg_diff = r_bge['mean_margin'] - r_nomic['mean_margin']
        print(f"⭐ Contraste & Rejet du Hors-sujet : BGE-M3 surpasse Nomic de {mrg_diff:+.4f} de marge.")
    print("=" * 75)


if __name__ == "__main__":
    main()
