"""
src/pipeline/batch_enrich_jobs.py
Worker haute performance pour l'Enrichissement d'offres via GLiNER Zero-Shot.
Traite les offres par lots rapides sans surchauffe CPU ni besoin de throttle thermique lourd.
"""

import sys
import time
import logging
from pathlib import Path

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
    batch_size: int = 25,
    sleep_per_job: float = 0.0,
    cooldown_between_batches: float = 0.5,
    auto_reindex: bool = True,
    force_all: bool = False
) -> int:
    """
    Exécute l'enrichissement GLiNER de toutes les offres en attente dans PostgreSQL.
    
    :param batch_size: Nombre d'offres par lot avant commit DB.
    :param sleep_per_job: Pause entre offres (0s par défaut car GLiNER est ultra-léger).
    :param cooldown_between_batches: Pause légère après chaque lot (0.5s par défaut).
    :param auto_reindex: Si True, ré-indexe automatiquement ChromaDB à la fin.
    :param force_all: Si True, ré-enrichit l'intégralité des offres de la base.
    :return: Nombre total d'offres enrichies avec succès.
    """
    db = SessionLocal()
    try:
        if force_all:
            pending_jobs = db.query(Offre).filter(Offre.is_active == True).order_by(Offre.created_at.desc()).all()
            logger.info(f"🔄 Mode FORCE-ALL activé : Traitement intégral des {len(pending_jobs)} offres de la base.")
        else:
            # Sélection de toutes les offres sans compétences renseignées (vides ou 'Non spécifié')
            pending_jobs = db.query(Offre).filter(
                (Offre.competences_necessaires == []) |
                (Offre.competences_necessaires == None) |
                (Offre.competences_necessaires == ["Non spécifié"])
            ).order_by(Offre.created_at.desc()).all()

        total_pending = len(pending_jobs)
        if total_pending == 0:
            logger.info("✅ Toutes les offres de la base sont déjà enrichies ! Rien à faire.")
            return 0

        total_batches = (total_pending + batch_size - 1) // batch_size
        logger.info(f"🚀 Début de l'enrichissement GLiNER : {total_pending} offres à traiter ({total_batches} lots de {batch_size}).")

        enricher = JobEnricher()
        enriched_count = 0
        failed_count = 0
        enriched_job_ids = []

        with tqdm(total=total_pending, desc="⚡ Enrichissement GLiNER", unit="offre") as pbar:
            for batch_idx in range(0, total_pending, batch_size):
                current_batch_num = (batch_idx // batch_size) + 1
                batch = pending_jobs[batch_idx : batch_idx + batch_size]

                for job in batch:
                    short_title = (job.poste or "Offre")[:25]
                    source_label = (job.source or "unknown").upper()
                    secteur_label = (job.secteur or "general").upper()
                    pbar.set_postfix_str(f"Lot {current_batch_num}/{total_batches} | [{secteur_label}] {short_title}")

                    try:
                        enricher.enrich_offre_instance(job)
                        enriched_count += 1
                        enriched_job_ids.append(job.id)
                        if sleep_per_job > 0:
                            time.sleep(sleep_per_job)
                    except Exception as err:
                        failed_count += 1
                        logger.warning(f"⚠️ Échec enrichissement offre {job.id}: {err}")

                    pbar.update(1)

                # Commit à la fin de chaque lot
                try:
                    db.commit()
                except Exception as commit_err:
                    db.rollback()
                    logger.error(f"❌ Erreur lors du commit du lot {current_batch_num}: {commit_err}")

                if cooldown_between_batches > 0 and batch_idx + batch_size < total_pending:
                    time.sleep(cooldown_between_batches)

        logger.info(f"\n🎉 Enrichissement terminé : {enriched_count} offres traitées, {failed_count} échecs.")

        # Vectorisation incrémentale ciblée de ChromaDB UNIQUEMENT pour les nouvelles offres enrichies
        if auto_reindex and enriched_job_ids:
            logger.info(f"🔄 Vectorisation incrémentale ciblée de {len(enriched_job_ids)} offre(s) dans ChromaDB...")
            from src.pipeline.job_indexer import JobIndexer
            # FastEmbed ONNX C++ optimise le calcul par lots de 8
            indexer = JobIndexer(batch_size=8)
            indexer.index_jobs_by_ids(enriched_job_ids)
            logger.info("✅ ChromaDB synchronisé avec succès !")

        return enriched_count

    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Worker GLiNER Zero-Shot (Hard & Soft Skills)")
    parser.add_argument("--all", "--force-all", action="store_true", help="Ré-enrichir l'INTÉGRALITÉ des offres de la base")
    parser.add_argument("--batch-size", type=int, default=25, help="Nombre d'offres par lot")
    args = parser.parse_args()

    run_batch_enrichment(
        batch_size=args.batch_size,
        sleep_per_job=0.0,
        cooldown_between_batches=0.5,
        auto_reindex=True,
        force_all=args.all
    )
