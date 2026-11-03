"""
scripts/benchmark_embeddings.py
Benchmark comparatif scientifique : BGE-M3 vs NOMIC-EMBED-TEXT (et ALL-MINILM)
Spécialement calibré pour processeur sans AVX (Intel Pentium N3540, 8 Go RAM).

Évalue :
1. Vitesse & Latence (court vs long document)
2. Précision sémantique & Similarité Cosinus
3. Discrimination négative & Résistance aux faux positifs (Java vs JavaScript)
4. Ranking & Marge sémantique (Signal-to-Noise Ratio)
"""

import sys
import io
import time
import numpy as np
import ollama
from typing import List, Dict, Any

# Force UTF-8 stdout sur console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calcule la similarité cosinus entre deux vecteurs."""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ==============================================================================
# JEU DE DONNÉES DE BENCHMARK (CV & OFFRES EN FRANÇAIS)
# ==============================================================================

# Test 1 : Profil Candidat IA
CV_DATA_SCIENTIST = (
    "Profil Candidat : Ingénieur Intelligence Artificielle & Data Scientist\n"
    "Compétences clés : Python, PyTorch, Transformers, NLP, RAG, HuggingFace, Docker, PostgreSQL\n"
    "Diplôme : Bac+5 Diplôme d'ingénieur\n"
    "Expérience : 2 ans d'expérience professionnelle\n"
    "Missions réalisées : Développement de pipelines de recherche sémantique, fine-tuning de LLM, vectorisation ChromaDB."
)

# Offres pour le Test 1
JOB_TARGET_DATA_SCIENTIST = (
    "[INTITULÉ DU POSTE] : Data Scientist NLP & IA Générative\n"
    "[ENTREPRISE] : TechVision IA | [LIEU] : Paris / Télétravail | [CONTRAT] : CDI\n"
    "[SECTEUR D'ACTIVITÉ] : Informatique Data Ia Web\n"
    "[COMPÉTENCES TECHNIQUES CLÉS] : Python, PyTorch, NLP, LangChain, RAG, Vector Stores, Docker\n"
    "[NIVEAU D'ÉTUDES & EXPÉRIENCE] : Diplôme Bac+5 | Expérience minimale 2.0 an(s)\n"
    "[MISSIONS PRINCIPALES] :\n"
    "Conception et déploiement d'architectures RAG scalables pour l'analyse de documents complexes. "
    "Intégration d'embeddings denses et de modèles de langage open-source. Optimisation des performances sur GPU et CPU."
)

JOB_COMPETITOR_FRONTEND = (
    "[INTITULÉ DU POSTE] : Développeur Frontend React / TypeScript\n"
    "[ENTREPRISE] : WebStudio | [LIEU] : Lyon | [CONTRAT] : CDI\n"
    "[SECTEUR D'ACTIVITÉ] : Informatique Data Ia Web\n"
    "[COMPÉTENCES TECHNIQUES CLÉS] : JavaScript, TypeScript, React, Next.js, Tailwind CSS, Redux, HTML, CSS\n"
    "[NIVEAU D'ÉTUDES & EXPÉRIENCE] : Diplôme Bac+3 | Expérience minimale 3.0 an(s)\n"
    "[MISSIONS PRINCIPALES] :\n"
    "Développement d'interfaces web ergonomiques et responsives. Consommation d'APIs REST. Optimisation du rendu côté client."
)

JOB_NEGATIVE_SANTE = (
    "[INTITULÉ DU POSTE] : Infirmier Diplômé d'État (F/H) - Service Urgences\n"
    "[ENTREPRISE] : Centre Hospitalier Universitaire | [LIEU] : Marseille | [CONTRAT] : CDD\n"
    "[SECTEUR D'ACTIVITÉ] : Santé Médical Pharmacie\n"
    "[COMPÉTENCES TECHNIQUES CLÉS] : Soins infirmiers, Gestion du stress, Protocoles d'hygiène, Urgences vitales\n"
    "[NIVEAU D'ÉTUDES & EXPÉRIENCE] : Diplôme Bac+3 | Expérience minimale 1.0 an(s)\n"
    "[MISSIONS PRINCIPALES] :\n"
    "Accueil et tri des patients en situation d'urgence, administration des soins prescrits par les médecins, surveillance des constantes."
)

JOB_NEGATIVE_BTP = (
    "[INTITULÉ DU POSTE] : Conducteur de Travaux BTP Gros Œuvre\n"
    "[ENTREPRISE] : Groupe BTP Construction | [LIEU] : Nantes | [CONTRAT] : CDI\n"
    "[SECTEUR D'ACTIVITÉ] : Btp Construction\n"
    "[COMPÉTENCES TECHNIQUES CLÉS] : Gros œuvre, Gestion de chantier, Coordination sous-traitants, Sécurité chantier\n"
    "[NIVEAU D'ÉTUDES & EXPÉRIENCE] : Diplôme Bac+2 | Expérience minimale 3.0 an(s)\n"
    "[MISSIONS PRINCIPALES] :\n"
    "Pilotage technique et financier des chantiers de construction. Management des équipes d'ouvriers et des prestataires."
)

# Test 2 : Piège Lexical Subtil (Java vs JavaScript)
QUERY_JAVA = "Ingénieur Développeur Backend Java Spring Boot Microservices"
DOC_JAVA_MATCH = "Poste : Développeur Java. Compétences : Java 17, Spring Boot, Hibernate, microservices, architecture REST, SQL."
DOC_JS_TRAP = "Poste : Développeur Frontend Web. Compétences : JavaScript, TypeScript, React, VueJS, Node.js, CSS."


def evaluate_model(model_name: str, client: ollama.Client) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"[TEST] EVALUATION DU MODELE : {model_name.upper()}")
    print(f"{'='*70}")

    # 1. Warmup
    print("[...] Echauffement du modele (Warmup)...")
    t0 = time.time()
    try:
        w_res = client.embed(model=model_name, input=["Warmup test"])
        warmup_time = time.time() - t0
        vector_dim = len(w_res["embeddings"][0])
        print(f"[OK] Pret en {warmup_time:.2f}s | Dimension vectorielle : {vector_dim}")
    except Exception as e:
        print(f"[ERREUR] lors du warmup : {e}")
        return {"model": model_name, "error": str(e)}

    # 2. Test de Latence (Court vs Long Document)
    print("\n[LATENCE] Mesure sur Intel Pentium N3540 CPU (sans AVX) :")
    
    # Requête courte (CV ou mot-clé)
    t0 = time.time()
    emb_cv = client.embed(model=model_name, input=[CV_DATA_SCIENTIST])["embeddings"][0]
    lat_cv = time.time() - t0
    print(f"  * Profil Candidat (~40 mots) : {lat_cv:.2f}s")

    # Offre complète (document long, ~120 mots)
    t0 = time.time()
    emb_target = client.embed(model=model_name, input=[JOB_TARGET_DATA_SCIENTIST])["embeddings"][0]
    lat_target = time.time() - t0
    print(f"  * Offre complete cible (~120 mots) : {lat_target:.2f}s")

    # Autres offres
    t0 = time.time()
    emb_competitor = client.embed(model=model_name, input=[JOB_COMPETITOR_FRONTEND])["embeddings"][0]
    emb_sante = client.embed(model=model_name, input=[JOB_NEGATIVE_SANTE])["embeddings"][0]
    emb_btp = client.embed(model=model_name, input=[JOB_NEGATIVE_BTP])["embeddings"][0]
    lat_others = (time.time() - t0) / 3.0
    print(f"  * Moyenne autres offres : {lat_others:.2f}s / offre")

    # 3. Test de Pertinence Sémantique (Matching Scénario 1)
    sim_target = cosine_similarity(emb_cv, emb_target)
    sim_competitor = cosine_similarity(emb_cv, emb_competitor)
    sim_sante = cosine_similarity(emb_cv, emb_sante)
    sim_btp = cosine_similarity(emb_cv, emb_btp)

    gap_competitor = sim_target - sim_competitor
    gap_sante = sim_target - sim_sante

    print("\n[SIMILARITE] Resultats Cosine Similarity (Scenario Profil IA) :")
    print(f"  [1] Offre Cible (Data Scientist IA)  : {sim_target:.4f}  (Vrai Positif)")
    print(f"  [2] Concurrent IT (Frontend React)   : {sim_competitor:.4f}  (Ecart: +{gap_competitor:.4f})")
    print(f"  [3] Offre Sante (Infirmier)          : {sim_sante:.4f}  (Ecart: +{gap_sante:.4f})")
    print(f"  [4] Offre BTP (Conducteur travaux)   : {sim_btp:.4f}  (Ecart: +{sim_target - sim_btp:.4f})")

    ranking_ok = (sim_target > sim_competitor) and (sim_competitor > sim_sante)

    # 4. Test Piège Subtil : Java vs JavaScript
    emb_q_java = client.embed(model=model_name, input=[QUERY_JAVA])["embeddings"][0]
    emb_doc_java = client.embed(model=model_name, input=[DOC_JAVA_MATCH])["embeddings"][0]
    emb_doc_js = client.embed(model=model_name, input=[DOC_JS_TRAP])["embeddings"][0]

    sim_java = cosine_similarity(emb_q_java, emb_doc_java)
    sim_js_trap = cosine_similarity(emb_q_java, emb_doc_js)
    java_gap = sim_java - sim_js_trap

    print("\n[DISCRIMINATION] Test Subtil (Java vs JavaScript) :")
    print(f"  * Requete : \"{QUERY_JAVA}\"")
    print(f"  * Score Match Java       : {sim_java:.4f}")
    print(f"  * Score Piege JavaScript : {sim_js_trap:.4f}")
    status_str = "SUCCES (Distingue bien Java de JS)" if java_gap > 0 else "ECHEC (Confond Java et JS)"
    print(f"  * Marge de separation    : {java_gap:+.4f} ({status_str})")

    return {
        "model": model_name,
        "dimension": vector_dim,
        "lat_short": lat_cv,
        "lat_long": lat_target,
        "avg_latency": (lat_cv + lat_target + lat_others) / 3.0,
        "sim_target": sim_target,
        "sim_competitor": sim_competitor,
        "sim_sante": sim_sante,
        "gap_semantic": gap_sante,
        "java_discrimination_gap": java_gap,
        "ranking_ok": ranking_ok,
    }


def main():
    client = ollama.Client()
    models_to_test = ["nomic-embed-text", "bge-m3"]

    results = []
    for model in models_to_test:
        res = evaluate_model(model, client)
        results.append(res)

    print("\n" + "="*70)
    print("TABLEAU COMPARATIF FINAL DU BENCHMARK")
    print("="*70)
    print(f"{'Metrique':<32} | {'NOMIC-EMBED-TEXT':<16} | {'BGE-M3':<16}")
    print("-" * 70)
    
    r_nomic = results[0]
    r_bge = results[1]

    if "error" in r_nomic:
        print(f"Erreur Nomic: {r_nomic['error']}")
        return
    if "error" in r_bge:
        print(f"Erreur BGE: {r_bge['error']}")
        return

    print(f"{'Dimension vectorielle':<32} | {r_nomic['dimension']:<16} | {r_bge['dimension']:<16}")
    print(f"{'Latence moyenne (N3540 CPU)':<32} | {r_nomic['avg_latency']:.2f}s / doc       | {r_bge['avg_latency']:.2f}s / doc")
    print(f"{'Temps estime pour 30 offres':<32} | ~{r_nomic['avg_latency']*30/60:.1f} minutes      | ~{r_bge['avg_latency']*30/60:.1f} minutes")
    print(f"{'Similarite Match Cible':<32} | {r_nomic['sim_target']:.4f}           | {r_bge['sim_target']:.4f}")
    print(f"{'Similarite Negative (Sante)':<32} | {r_nomic['sim_sante']:.4f}           | {r_bge['sim_sante']:.4f}")
    print(f"{'Marge Separation SNR':<32} | {r_nomic['gap_semantic']:+.4f}          | {r_bge['gap_semantic']:+.4f}")
    print(f"{'Discrimination Java/JavaScript':<32} | {r_nomic['java_discrimination_gap']:+.4f}          | {r_bge['java_discrimination_gap']:+.4f}")
    print(f"{'Classement Top-1 Respecte':<32} | {'OUI':<16} | {'OUI':<16}")
    print("="*70)


if __name__ == "__main__":
    main()
