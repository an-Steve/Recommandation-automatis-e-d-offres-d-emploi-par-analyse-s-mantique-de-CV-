"""
src/pipeline/batch_enrich_jobs.py
Worker par Lots pour l'Enrichissement LLM Complet (CPU Throttling Protected).
Traite toutes les offres non enrichies par blocs de 5 avec temps de repos pour le CPU.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

# Support de l'exécution directe
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tqdm import tqdm
from src.database.db_session import SessionLocal
from src.database.sql_models import Offre
from src.parsers.job_enricher import JobEnricher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_batch_enrichment(
    batch_size: int = 5,
    sleep_per_job: float = 1.0,
    cooldown_between_batches: float = 5.0,
    auto_reindex: bool = True
) -> int:
    """
    Exécute l'enrichissement de toutes les offres en attente dans PostgreSQL.
    
    :param batch_size: Nombre d'offres par lot avant commit DB et pause thermique.
    :param sleep_per_job: Pause en secondes entre chaque appel LLM (1s par défaut).
    :param cooldown_between_batches: Pause thermique en secondes après chaque lot (5s par défaut).
    :param auto_reindex: Si True, ré-indexe automatiquement ChromaDB à la fin.
    :return: Nombre total d'offres enrichies avec succès.
    """
    db = SessionLocal()
    try:
        # Sélection de toutes les offres sans compétences renseignées
        pending_jobs = db.query(Offre).filter(
            (Offre.competences_necessaires == []) | (Offre.competences_necessaires == None)
        ).order_by(Offre.created_at.desc()).all()

        total_pending = len(pending_jobs)
        if total_pending == 0:
            logger.info("✅ Toutes les offres de la base sont déjà enrichies ! Rien à faire.")
            return 0

        total_batches = (total_pending + batch_size - 1) // batch_size
        logger.info(f"🚀 Début de l'enrichissement complet : {total_pending} offres à traiter ({total_batches} lots de {batch_size}).")
        logger.info(f"❄️ Protection CPU active : {sleep_per_job}s entre offres, {cooldown_between_batches}s de repos entre lots.\n")

        enricher = JobEnricher()
        enriched_count = 0
        failed_count = 0
        enriched_job_ids = []

        with tqdm(total=total_pending, desc="🧠 Enrichissement LLM", unit="offre") as pbar:
            for batch_idx in range(0, total_pending, batch_size):
                current_batch_num = (batch_idx // batch_size) + 1
                batch = pending_jobs[batch_idx : batch_idx + batch_size]

                for job in batch:
                    short_title = (job.poste or "Offre")[:25]
                    source_label = (job.source or "unknown").upper()
                    pbar.set_postfix_str(f"Lot {current_batch_num}/{total_batches} | [{source_label}] {short_title}")

                    try:
                        enricher.enrich_offre_instance(job)
                        db.commit()
                        enriched_count += 1
                        enriched_job_ids.append(job.id)
                        time.sleep(sleep_per_job)
                    except Exception as err:
                        db.rollback()
                        failed_count += 1
                        logger.warning(f"⚠️ Échec enrichissement offre {job.id}: {err}")

                    pbar.update(1)

                # Pause de refroidissement CPU après chaque lot de 5 (sauf au dernier lot)
                if batch_idx + batch_size < total_pending:
                    pbar.set_postfix_str(f"❄️ Repos CPU ({cooldown_between_batches}s)...")
                    time.sleep(cooldown_between_batches)

        logger.info(f"\n🎉 Enrichissement terminé : {enriched_count} offres enrichies, {failed_count} échecs.")

        # Vectorisation incrémentale ciblée de ChromaDB UNIQUEMENT pour les nouvelles offres enrichies
        if auto_reindex and enriched_job_ids:
            logger.info(f"🔄 Vectorisation incrémentale ciblée de {len(enriched_job_ids)} offre(s) enrichie(s) dans ChromaDB...")
            from src.pipeline.job_indexer import JobIndexer
            indexer = JobIndexer(batch_size=5)
            indexer.index_jobs_by_ids(enriched_job_ids)
            logger.info("✅ ChromaDB synchronisé avec succès !")

        return enriched_count

    finally:
        db.close()


if __name__ == "__main__":
    run_batch_enrichment(
        batch_size=5,
        sleep_per_job=1.0,
        cooldown_between_batches=5.0,
        auto_reindex=True
    )
