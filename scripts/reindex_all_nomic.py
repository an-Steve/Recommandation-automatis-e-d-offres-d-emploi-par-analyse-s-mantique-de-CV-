"""
scripts/reindex_all_nomic.py
Réinitialise la collection ChromaDB et réindexe l'ensemble des offres
stockées dans PostgreSQL avec le modèle 'nomic-embed-text' (768 dimensions).
Supporte la reprise automatique : si des offres sont déjà indexées en 768 dims,
continue là où il s'était arrêté (sauf si --reset est passé).
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.database.vector_store import chroma_client, get_offres_collection
from src.database.db_session import SessionLocal
from src.database.sql_models import Offre
from src.pipeline.job_indexer import JobIndexer
from src.config import settings


def reindex(force_reset: bool = False):
    print("=" * 70)
    print(f"🔄 MIGRATION VECTORIELLE CHROMADB -> {settings.EMBEDDING_MODEL_NAME.upper()}")
    print("=" * 70)

    db = SessionLocal()
    try:
        pending_count = db.query(Offre).filter(Offre.is_indexed == False).count()
        total_count = db.query(Offre).count()
    finally:
        db.close()

    # Si force_reset ou si aucune offre n'est en attente de migration, on réinitialise tout
    if force_reset or pending_count == 0:
        try:
            chroma_client.delete_collection(name=settings.CHROMA_COLLECTION_OFFRES)
            print("🗑️ Ancienne collection ChromaDB supprimée avec succès.")
        except Exception as e:
            print(f"ℹ️ Aucune collection existante à supprimer ({e}).")

        collection = get_offres_collection()
        print(f"✨ Nouvelle collection '{settings.CHROMA_COLLECTION_OFFRES}' initialisée.")

        db = SessionLocal()
        try:
            updated = db.query(Offre).update({Offre.is_indexed: False})
            db.commit()
            print(f"🔄 {updated} offre(s) dans PostgreSQL réinitialisées à 'is_indexed = False'.")
        finally:
            db.close()
    else:
        collection = get_offres_collection()
        already_done = total_count - pending_count
        print(f"⏩ Reprise intelligente : {already_done}/{total_count} offre(s) déjà indexées ({pending_count} restantes).")

    # Lancement de la vectorisation offre par offre (batch_size=1 pour CPU)
    print(f"\n⚡ Lancement de l'indexation avec {settings.EMBEDDING_MODEL_NAME} (batch_size=1)...")
    indexer = JobIndexer(batch_size=1)
    indexed_count = indexer.index_pending_jobs()

    print("\n" + "=" * 70)
    print(f"🎉 SUCCÈS : {indexed_count} offres vectorisées dans cette session ({settings.EMBEDDING_MODEL_NAME}).")
    print(f"📊 Total vecteurs dans ChromaDB : {collection.count()}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migration et indexation ChromaDB nomic-embed-text")
    parser.add_argument("--reset", action="store_true", help="Forcer la suppression et réindexation totale depuis 0")
    args = parser.parse_args()
    reindex(force_reset=args.reset)
