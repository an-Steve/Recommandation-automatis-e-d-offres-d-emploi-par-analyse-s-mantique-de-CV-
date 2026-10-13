"""
src/database/export_chroma.py
Script d'exportation des données de ChromaDB (Documents canoniques + Métadonnées)
vers un fichier JSON pour audit qualité et vérification de la structure.
"""
import os
import sys
import json
import logging
from pathlib import Path

# Ajout de la racine du projet dans sys.path pour supporter l'exécution directe
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.database.vector_store import get_offres_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def export_chroma_data(output_file: str = "data/chroma_dump.json") -> str:
    """
    Extrait l'ensemble des documents textuels canoniques et leurs métadonnées
    stockés dans la collection ChromaDB 'offres'.
    """
    collection = get_offres_collection()
    count = collection.count()
    logger.info(f"📊 Nombre total d'offres dans ChromaDB : {count}")

    if count == 0:
        logger.warning("⚠️ La collection ChromaDB est vide. Lancez la ré-indexation d'abord.")
        return ""

    # Extraction des IDs, documents et métadonnées (sans les vecteurs bruts pour garder le fichier léger)
    data = collection.get(include=["documents", "metadatas"])

    items = []
    for job_id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        items.append({
            "id": job_id,
            "metadata": meta,
            "canonical_document": doc
        })

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_count": count,
            "exported_count": len(items),
            "items": items
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Export réussi ! {len(items)} offres enregistrées dans {output_path.resolve()}")
    return str(output_path.resolve())


if __name__ == "__main__":
    export_chroma_data()
