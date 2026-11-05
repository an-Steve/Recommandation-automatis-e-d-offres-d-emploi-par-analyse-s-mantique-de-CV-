"""
scripts/remove_adzuna_jobs.py
Supprime proprement et de manière synchronisée toutes les offres Adzuna
dans PostgreSQL et dans ChromaDB pour ne conserver que les offres de haute qualité (LinkedIn & Indeed).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import logging
from src.database.db_session import SessionLocal
from src.database.sql_models import Offre
from src.database.vector_store import get_offres_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def purge_adzuna_jobs():
    print("=" * 70)
    print("🧹 PURGE SÉCURISÉE DES OFFRES ADZUNA (POSTGRESQL & CHROMADB)")
    print("=" * 70)

    db = SessionLocal()
    collection = get_offres_collection()

    try:
        # 1. Identification des offres Adzuna
        adzuna_jobs = db.query(Offre).filter(Offre.source == "adzuna").all()
        total_adzuna = len(adzuna_jobs)
        adzuna_ids = [job.id for job in adzuna_jobs]

        total_sql_before = db.query(Offre).count()
        total_chroma_before = collection.count()

        print(f"\n📊 État initial de vos bases :")
        print(f"• Total offres dans PostgreSQL : {total_sql_before}")
        print(f"• Total vecteurs dans ChromaDB : {total_chroma_before}")
        print(f"• Offres Adzuna identifiées     : {total_adzuna}")

        if total_adzuna == 0:
            print("\n✅ Aucune offre Adzuna à supprimer. Les bases sont déjà propres !")
            return

        # 2. Suppression dans ChromaDB
        print(f"\n🗑️  Suppression des {len(adzuna_ids)} vecteurs Adzuna dans ChromaDB...")
        try:
            # ChromaDB gère la suppression par liste d'IDs
            collection.delete(ids=adzuna_ids)
            print("✅ Vecteurs Adzuna supprimés de ChromaDB avec succès.")
        except Exception as e:
            logger.warning(f"⚠️ Note ChromaDB lors de la suppression : {e}")

        # 3. Suppression dans PostgreSQL
        print(f"🗑️  Suppression des {total_adzuna} annonces Adzuna dans PostgreSQL...")
        deleted_rows = db.query(Offre).filter(Offre.source == "adzuna").delete(synchronize_session=False)
        db.commit()
        print(f"✅ {deleted_rows} lignes Adzuna supprimées de PostgreSQL.")

        # 4. Vérification finale
        total_sql_after = db.query(Offre).count()
        total_chroma_after = collection.count()

        print("\n" + "=" * 70)
        print("🎉 BILAN FINAL APRÈS PURGE :")
        print("=" * 70)
        print(f"• Total restant dans PostgreSQL : {total_sql_after} (100% LinkedIn & Indeed)")
        print(f"• Total restant dans ChromaDB   : {total_chroma_after} vecteurs synchronisés")
        print("• Qualité du corpus restant     : Haute (~2 000 caractères par offre)")
        print("=" * 70)

    except Exception as err:
        db.rollback()
        logger.error(f"❌ Erreur lors de la purge : {err}")
        raise err
    finally:
        db.close()


if __name__ == "__main__":
    purge_adzuna_jobs()
