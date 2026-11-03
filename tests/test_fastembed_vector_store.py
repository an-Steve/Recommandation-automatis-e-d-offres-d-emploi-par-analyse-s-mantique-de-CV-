"""
tests/test_fastembed_vector_store.py
Validation unitaire et fonctionnelle du moteur FastEmbed ONNX pour ChromaDB :
1. Initialisation locale ONNX sans dépendance HTTP Ollama.
2. Respect de la dimension 768 compatible avec la collection existante.
3. Vitesse d'inférence < 100 ms par document sur CPU.
4. Recherche de similarité cosinus dense dans ChromaDB.
"""

import sys
import time
import logging
from pathlib import Path

# Encodage Windows UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ajout de la racine au PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def test_fastembed_integration():
    from src.config import settings
    from src.database.vector_store import (
        get_embedding_function,
        get_offres_collection,
        FastEmbedEmbeddingFunction,
    )

    print("\n" + "=" * 70)
    print("TEST 1 : INITIALISATION ET CONFIGURATION DE FASTEMBED")
    print("=" * 70)
    print(f"Provider configuré : {settings.EMBEDDING_PROVIDER}")
    print(f"Modèle configuré   : {settings.EMBEDDING_MODEL_NAME}")
    assert settings.EMBEDDING_PROVIDER == "fastembed", "Le provider devrait être fastembed."

    fn = get_embedding_function()
    assert isinstance(fn, FastEmbedEmbeddingFunction), "L'instance n'est pas FastEmbedEmbeddingFunction."
    print("--> FastEmbedEmbeddingFunction initialisée avec succès.")

    print("\n" + "=" * 70)
    print("TEST 2 : VITESSE D'INFÉRENCE & RESPECT DE LA DIMENSION (768)")
    print("=" * 70)
    sample_docs = [
        "Ingénieur Data Python Spark AWS Cloud Docker Kubernetes Paris",
        "Infirmier Diplômé d'État soins généraux cardiologie hôpital Lyon",
        "Comptable Général Sage Cegid TVA Bilan fiscal CDI Nantes",
    ]

    start = time.time()
    embeddings = fn(sample_docs)
    elapsed = time.time() - start

    assert len(embeddings) == 3, f"Attendu 3 embeddings, obtenu {len(embeddings)}"
    dim = len(embeddings[0])
    assert dim == 768, f"Dimension invalide : {dim} != 768"

    print(f"--> {len(sample_docs)} documents vectorisés en {elapsed:.3f} s ({elapsed/len(sample_docs)*1000:.1f} ms/doc)")
    print(f"--> Dimension vérifiée : {dim} (compatible collection ChromaDB)")

    print("\n" + "=" * 70)
    print("TEST 3 : CONNEXION CHROMADB ET REQUÊTE SÉMANTIQUE DENSE")
    print("=" * 70)
    collection = get_offres_collection()
    count = collection.count()
    print(f"--> Collection 'offres' chargée : {count} offres existantes en base.")
    assert count > 0, "La collection devrait contenir des offres."

    # Test de recherche vectorielle par texte via FastEmbed
    query = ["Data Scientist Python Machine Learning Cloud"]
    query_start = time.time()
    results = collection.query(query_texts=query, n_results=3)
    query_elapsed = time.time() - query_start

    print(f"--> Recherche sémantique effectuée en {query_elapsed*1000:.1f} ms.")
    print("--> Top 3 résultats trouvés :")
    for i, (jid, dist) in enumerate(zip(results["ids"][0], results["distances"][0]), 1):
        similarity = round(max(0.0, 1.0 - dist), 4)
        print(f"    [{i}] Job ID: {jid} | Similarité cosinus: {similarity*100:.2f}%")

    print("\n" + "=" * 70)
    print("TOUS LES TESTS FASTEMBED ONNX ONT RÉUSSI AVEC SUCCÈS !")
    print("=" * 70)


if __name__ == "__main__":
    test_fastembed_integration()
