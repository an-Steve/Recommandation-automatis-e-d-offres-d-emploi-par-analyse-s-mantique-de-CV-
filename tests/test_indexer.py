"""
tests/test_indexer.py
Test unitaire et d'intégration validant le flux ELT (PostgreSQL -> Ollama bge-m3 -> ChromaDB).
"""

import sys
from pathlib import Path

# Résolution des imports et de l'encodage Windows
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.pipeline.job_indexer import JobIndexer
from src.database.db_session import SessionLocal
from src.database.sql_models import Offre
from src.database.vector_store import get_offres_collection


def main():
    print("=" * 65)
    print("🚀 TEST DU PIPELINE ELT : DÉDUPLICATION, SQL & OLLAMA BGE-M3")
    print("=" * 65)

    sample_jobs = [
        {
            "poste": "Ingénieur RAG & NLP Senior",
            "company": "Lab IA Paris",
            "location": "Paris",
            "type_de_contrat": "CDI",
            "description": (
                "Poste de Data Scientist / Ingénieur NLP spécialisé en architectures RAG.\n"
                "Technologies principales : Python, PyTorch, LangChain, ChromaDB et Docker.\n"
                "Formation requise : Master 2 IA ou diplôme d'ingénieur (Bac+5)."
            ),
            "source_url": "https://example.com/job/rag-123"
        }
    ]

    indexer = JobIndexer(batch_size=4)

    # 1. Ingestion brute dans PostgreSQL (is_indexed = False)
    print("\n[1/4] Enregistrement brut dans PostgreSQL...")
    stats = indexer.save_raw_jobs_to_postgres(sample_jobs)
    assert stats["saved"] == 1, "L'offre test aurait dû être insérée dans PostgreSQL !"
    print("      ✅ Offre enregistrée avec succès dans PostgreSQL (is_indexed=False)")

    # 2. Vérification de l'état dans PostgreSQL
    db = SessionLocal()
    test_id = None
    try:
        offre = db.query(Offre).filter(Offre.poste == "Ingénieur RAG & NLP Senior").first()
        assert offre is not None, "Offre introuvable dans PostgreSQL !"
        assert offre.is_indexed is False, "L'offre doit avoir is_indexed=False avant embedding !"
        test_id = offre.id
        print(f"      ✅ Offre confirmée en base (ID: {test_id}, is_indexed={offre.is_indexed})")
    finally:
        db.close()

    # 3. Exécution du worker d'embedding par lots Ollama
    print("\n[2/4] Exécution du worker d'indexation vectorielle via Ollama...")
    indexed_count = indexer.index_pending_jobs()
    assert indexed_count == 1, "Le worker aurait dû vectoriser 1 offre !"

    # 4. Vérification que PostgreSQL est mis à jour (is_indexed = True)
    db = SessionLocal()
    try:
        offre_maj = db.query(Offre).filter(Offre.id == test_id).first()
        assert offre_maj.is_indexed is True, "is_indexed doit être True après le passage du worker !"
        print(f"      ✅ Statut PostgreSQL mis à jour : is_indexed={offre_maj.is_indexed} à {offre_maj.indexed_at}")
    finally:
        db.close()

    # 5. Recherche vectorielle dans ChromaDB via Ollama
    print("\n[3/4] Recherche vectorielle cosinus dans ChromaDB via Ollama...")
    collection = get_offres_collection()
    res = collection.query(
        query_texts=["Recherche un profil IA NLP maîtrise de Python et PyTorch"],
        n_results=1
    )
    matched_id = res["ids"][0][0]
    print(f"      ✅ Vecteur retrouvé dans ChromaDB ! ID correspondant : {matched_id}")
    assert matched_id == test_id, "L'ID retourné par ChromaDB doit correspondre à PostgreSQL !"

    # 6. Nettoyage
    print("\n[4/4] Nettoyage des données de test...")
    db = SessionLocal()
    try:
        db.query(Offre).filter(Offre.id == test_id).delete()
        db.commit()
        collection.delete(ids=[test_id])
        print("      🧹 Offre de test supprimée proprement des deux bases.")
    finally:
        db.close()

    print("\n" + "=" * 65)
    print("🎉 SUCCÈS TOTAL : Votre pipeline ELT avec Ollama est 100 % opérationnel !")
    print("=" * 65)


if __name__ == "__main__":
    main()
