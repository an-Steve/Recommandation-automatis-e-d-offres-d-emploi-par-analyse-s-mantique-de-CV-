"""
src/pipeline/job_indexer.py
Worker d'Indexation Vectorielle par Lots (PostgreSQL -> Ollama bge-m3 -> ChromaDB).
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from tqdm import tqdm

from src.database.db_session import SessionLocal
from src.database.sql_models import Offre
from src.database.vector_store import get_offres_collection
from src.pipeline.deduplication import filter_existing_jobs
from src.pipeline.job_normalizer import create_sql_offre_from_raw

logger = logging.getLogger(__name__)


def build_canonical_job_document(job: Offre) -> str:
    """
    Construit la représentation textuelle canonique structurée d'une offre d'emploi.
    Optimise la similarité cosinus dans ChromaDB en assurant une symétrie parfaite
    avec le profil candidat extrait du CV (Docling + Pydantic).
    """
    skills_list = job.competences_necessaires or []
    skills_str = ", ".join(skills_list) if skills_list else "Non spécifié"

    diploma_level = int(job.niveau_etude_min or 0)
    diploma_str = f"Bac+{diploma_level}" if diploma_level > 0 else "Non spécifié"

    exp_years = float(job.min_experience_years or 0.0)
    exp_str = f"{exp_years:.1f} an(s)" if exp_years > 0.0 else "Débutant / Non spécifié"

    secteur_str = str(job.secteur or "it_data").replace("_", " ").title()

    lines = [
        f"[INTITULÉ DU POSTE] : {job.poste}",
        f"[ENTREPRISE] : {job.company} | [LIEU] : {job.location} | [CONTRAT] : {job.type_de_contrat}",
        f"[SECTEUR D'ACTIVITÉ] : {secteur_str}",
        f"[COMPÉTENCES TECHNIQUES CLÉS] : {skills_str}",
        f"[NIVEAU D'ÉTUDES & EXPÉRIENCE] : Diplôme {diploma_str} | Expérience minimale {exp_str}",
    ]

    clean_mission = (job.mission or "").strip()
    clean_desc = (job.description or "").strip()

    if clean_mission:
        lines.append(f"[MISSIONS PRINCIPALES] :\n{clean_mission}")

    if clean_desc and clean_desc != clean_mission:
        lines.append(f"[PROFIL RECHERCHÉ & QUALIFICATIONS] :\n{clean_desc}")
    elif not clean_mission and clean_desc:
        lines.append(f"[DESCRIPTION DU POSTE] :\n{clean_desc}")

    return "\n".join(lines)


class JobIndexer:
    """
    Gère la persistance brute dans PostgreSQL et l'indexation vectorielle via Ollama.
    """

    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size
        self.collection = get_offres_collection()

    def save_raw_jobs_to_postgres(self, raw_jobs: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Étape 1 : Enregistre les offres brutes dans PostgreSQL avec is_indexed=False.
        """
        if not raw_jobs:
            return {"received": 0, "duplicates": 0, "saved": 0}

        db = SessionLocal()
        stats = {"received": len(raw_jobs), "duplicates": 0, "saved": 0}

        try:
            unique_jobs, duplicates_count = filter_existing_jobs(raw_jobs, db=db)
            stats["duplicates"] = duplicates_count

            if not unique_jobs:
                logger.info(f"ℹ️ Aucune nouvelle offre à insérer ({duplicates_count} doublons).")
                return stats

            for raw in tqdm(unique_jobs, desc="💾 Insertion PostgreSQL", unit="offre", leave=False):
                offre_sql = create_sql_offre_from_raw(raw)
                db.merge(offre_sql)

            db.commit()
            stats["saved"] = len(unique_jobs)
            logger.info(f"💾 {len(unique_jobs)} nouvelles offres enregistrées dans PostgreSQL.")
            return stats

        except Exception as e:
            db.rollback()
            logger.error(f"❌ Erreur lors de la sauvegarde PostgreSQL : {e}")
            raise e
        finally:
            db.close()

    def index_pending_jobs(self, limit: Optional[int] = None) -> int:
        """
        Étape 2 : Worker d'embedding qui prend les offres non indexées (is_indexed=False)
        dans PostgreSQL et les envoie par lots dans ChromaDB avec le format canonique structuré.
        """
        db = SessionLocal()
        total_indexed = 0

        try:
            query = db.query(Offre).filter(Offre.is_indexed == False)
            if limit:
                query = query.limit(limit)

            pending_jobs = query.all()
            if not pending_jobs:
                logger.info("✅ Toutes les offres de PostgreSQL sont déjà indexées dans ChromaDB.")
                return 0

            logger.info(f"⚡ {len(pending_jobs)} offre(s) en attente d'embedding Ollama...")

            with tqdm(total=len(pending_jobs), desc="⚡ Vectorisation ChromaDB", unit="offre") as pbar:
                for i in range(0, len(pending_jobs), self.batch_size):
                    batch = pending_jobs[i:i + self.batch_size]

                    ids = [job.id for job in batch]
                    documents = [build_canonical_job_document(job) for job in batch]
                    metadatas = [
                        {
                            "poste": job.poste,
                            "company": job.company,
                            "location": job.location,
                            "source": job.source or "unknown",
                            "secteur": job.secteur or "it_data",
                            "date_posted": str(job.date_posted) if job.date_posted else "",
                            "type_de_contrat": job.type_de_contrat,
                            "niveau_etude_min": int(job.niveau_etude_min),
                            "is_active": True,
                        }
                        for job in batch
                    ]

                    # ChromaDB délègue automatiquement le calcul du vecteur à Ollama
                    self.collection.upsert(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )

                    # Mise à jour du statut dans PostgreSQL
                    for job in batch:
                        job.is_indexed = True
                        job.indexed_at = datetime.now(timezone.utc)

                    db.commit()
                    total_indexed += len(batch)
                    pbar.update(len(batch))
                    pbar.set_postfix_str(f"Lot {i // self.batch_size + 1}/{(len(pending_jobs) + self.batch_size - 1) // self.batch_size}")

            logger.info(f"🎉 Indexation vectorielle terminée : {total_indexed} offres traitées.")
            return total_indexed

        except Exception as e:
            db.rollback()
            logger.error(f"❌ Erreur critique pendant l'indexation vectorielle : {e}")
            raise e
        finally:
            db.close()

    def index_jobs_by_ids(self, job_ids: List[str]) -> int:
        """
        Vectorise spécifiquement une liste d'offres ciblées par leurs IDs dans ChromaDB.
        Évite de recalculer inutilement les vecteurs de toute la base.
        """
        if not job_ids:
            return 0

        db = SessionLocal()
        try:
            target_jobs = db.query(Offre).filter(Offre.id.in_(job_ids)).all()
            if not target_jobs:
                return 0

            with tqdm(total=len(target_jobs), desc="⚡ Vectorisation Ciblée ChromaDB", unit="offre") as pbar:
                for i in range(0, len(target_jobs), self.batch_size):
                    batch = target_jobs[i:i + self.batch_size]
                    ids = [job.id for job in batch]
                    documents = [build_canonical_job_document(job) for job in batch]
                    metadatas = [
                        {
                            "poste": job.poste,
                            "company": job.company,
                            "location": job.location,
                            "source": job.source or "unknown",
                            "secteur": job.secteur or "it_data",
                            "date_posted": str(job.date_posted) if job.date_posted else "",
                            "type_de_contrat": job.type_de_contrat,
                            "niveau_etude_min": int(job.niveau_etude_min),
                            "is_active": True,
                        }
                        for job in batch
                    ]
                    self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                    for job in batch:
                        job.is_indexed = True
                        job.indexed_at = datetime.now(timezone.utc)
                    db.commit()
                    pbar.update(len(batch))

            logger.info(f"✅ Vectorisation ciblée terminée : {len(target_jobs)} offre(s) mise(s) à jour.")
            return len(target_jobs)
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Erreur vectorisation ciblée ChromaDB : {e}")
            raise e
        finally:
            db.close()

    def reindex_all_jobs(self) -> int:
        """
        Re-vectorise TOUTES les offres de PostgreSQL dans ChromaDB
        avec le format canonique structuré et les textes nettoyés.
        """
        db = SessionLocal()
        try:
            all_jobs = db.query(Offre).filter(Offre.is_active == True).all()
            if not all_jobs:
                return 0

            with tqdm(total=len(all_jobs), desc="🔄 Ré-indexation ChromaDB", unit="offre") as pbar:
                for i in range(0, len(all_jobs), self.batch_size):
                    batch = all_jobs[i:i + self.batch_size]
                    ids = [job.id for job in batch]
                    documents = [build_canonical_job_document(job) for job in batch]
                    metadatas = [
                        {
                            "poste": job.poste,
                            "company": job.company,
                            "location": job.location,
                            "source": job.source or "unknown",
                            "secteur": job.secteur or "it_data",
                            "date_posted": str(job.date_posted) if job.date_posted else "",
                            "type_de_contrat": job.type_de_contrat,
                            "niveau_etude_min": int(job.niveau_etude_min),
                            "is_active": True,
                        }
                        for job in batch
                    ]
                    self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                    for job in batch:
                        job.is_indexed = True
                        job.indexed_at = datetime.now(timezone.utc)
                    db.commit()
                    pbar.update(len(batch))

            logger.info(f"🎉 Ré-indexation ChromaDB terminée : {len(all_jobs)} offres actualisées.")
            return len(all_jobs)
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Erreur ré-indexation ChromaDB : {e}")
            raise e
        finally:
            db.close()
