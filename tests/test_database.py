"""
Script de test d'intégration pour le socle de base de données.
Valide la lecture/écriture synchronisée dans PostgreSQL et ChromaDB.
"""
from pathlib import Path
import sys

# Support des caractères UTF-8 / émojis sous Windows PowerShell / CMD
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ajoute automatiquement la racine du projet (C:/MASTER2_IA/Projet_CV) au sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import settings
from src.database.db_session import check_postgres_connection, get_db
from src.database.sql_models import Offre
from src.database.vector_store import (
    upsert_job_vector,
    query_similar_jobs,
    delete_job_vector,
    count_indexed_jobs
)


def run_integration_test():
    print("=" * 60)
    print("🚀 TEST D'INTÉGRATION : PostgreSQL + ChromaDB")
    print(f"🔧 Device détecté pour l'IA : {settings.device.upper()}")
    print("=" * 60)
    
    # 1. Vérification de la connexion PostgreSQL
    print("\n[1/4] Test de connexion PostgreSQL...")
    if not check_postgres_connection():
        print("❌ ÉCHEC : Impossible de joindre PostgreSQL. Vérifiez DATABASE_URL dans votre .env.")
        sys.exit(1)
    print("✅ Connexion PostgreSQL réussie !")

    # 2. Test d'insertion dans PostgreSQL
    print("\n[2/4] Test d'insertion d'une offre dans PostgreSQL...")
    db_gen = get_db()
    db = next(db_gen)
    
    test_job_id = "test_job_999"
    try:
        # Nettoyage préventif si le test avait déjà tourné
        existing = db.query(Offre).filter(Offre.id == test_job_id).first()
        if existing:
            db.delete(existing)
            db.commit()

        # Insertion de test
        offre_test = Offre(
            id=test_job_id,
            poste="Data Scientist NLP Test",
            company="Lab IA Test",
            type_de_contrat="CDI",
            niveau_etude_min=5,
            competences_necessaires=["Python", "PyTorch", "NLP", "ChromaDB"],
            description="Conception de systèmes RAG pour projet Master 2.",
            mission="Développement et évaluation des modèles.",
            is_active=True
        )
        db.add(offre_test)
        db.commit()
        print(f"✅ Offre insérée dans PostgreSQL : {offre_test.id} - {offre_test.poste}")
    except Exception as e:
        db.rollback()
        print(f"❌ Erreur lors de l'insertion SQL : {e}")
        sys.exit(1)

    # 3. Test d'insertion et de recherche dans ChromaDB
    print("\n[3/4] Test vectoriel ChromaDB...")
    try:
        # Vecteur factice de 1024 dimensions (comme BGE-M3)
        fake_vector = [0.05] * 1024
        fake_document = "Poste: Data Scientist NLP | Compétences: Python, PyTorch | Contrat: CDI"
        fake_metadata = {"type_de_contrat": "CDI", "niveau_etude_min": 5, "is_active": True}
        
        upsert_job_vector(test_job_id, fake_document, fake_vector, fake_metadata)
        print(f"✅ Vecteur inséré dans ChromaDB. Total indexé : {count_indexed_jobs()} offre(s)")
        
        # Test de recherche cosinus avec pré-filtrage
        results = query_similar_jobs(fake_vector, n_results=1, where_filter={"type_de_contrat": "CDI"})
        assert len(results) > 0, "Aucun résultat trouvé dans ChromaDB"
        print(f"✅ Recherche vectorielle réussie ! ID trouvé : {results[0]['job_id']} (Score cosinus: {results[0]['cosine_similarity']})")
    except Exception as e:
        print(f"❌ Erreur ChromaDB : {e}")
        sys.exit(1)

    # 4. Nettoyage de l'offre de test
    print("\n[4/4] Nettoyage des données de test...")
    db.delete(db.query(Offre).filter(Offre.id == test_job_id).first())
    db.commit()
    delete_job_vector(test_job_id)
    db.close()
    print("✅ Données de test nettoyées proprement.")

    print("\n" + "=" * 60)
    print("🎉 SUCCÈS TOTAL : Votre socle PostgreSQL + ChromaDB est 100% opérationnel !")
    print("=" * 60)


if __name__ == "__main__":
    run_integration_test()
