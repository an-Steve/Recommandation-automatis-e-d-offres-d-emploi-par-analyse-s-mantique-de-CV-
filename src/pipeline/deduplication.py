"""
src/pipeline/deduplication.py
Gestion de l'empreinte unique des offres et filtrage des doublons SQL.
"""

import re
import hashlib
import unicodedata
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session

from src.database.sql_models import Offre


def clean_text_for_hash(text: str) -> str:
    """Nettoie une chaîne pour rendre le hash insensible aux accents, à la casse et aux espaces."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    no_accents = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return re.sub(r"[^a-zA-Z0-9]", "", no_accents).lower()


def compute_job_hash(company: str, poste: str, location: str) -> str:
    """
    Calcule un identifiant unique déterministe (MD5 sur 12 caractères).
    Format : job_a1b2c3d4e5f6
    """
    norm_company = clean_text_for_hash(company)
    norm_poste = clean_text_for_hash(poste)
    norm_location = clean_text_for_hash(location)

    raw_signature = f"{norm_company}_{norm_poste}_{norm_location}"
    md5_digest = hashlib.md5(raw_signature.encode("utf-8")).hexdigest()[:12]
    return f"job_{md5_digest}"


def filter_existing_jobs(
    jobs: List[Dict[str, Any]],
    db: Session
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Filtre les offres déjà présentes dans la table PostgreSQL 'offres'
    en une seule requête groupée (SELECT WHERE id IN (...)).
    """
    if not jobs:
        return [], 0

    # 1. Attribue le hash si non présent
    for job in jobs:
        if "id" not in job or not job["id"]:
            job["id"] = compute_job_hash(
                company=job.get("company", ""),
                poste=job.get("poste", ""),
                location=job.get("location", "")
            )

    # 2. Vérification d'existence groupée dans PostgreSQL
    candidate_ids = [j["id"] for j in jobs]
    existing_records = db.query(Offre.id).filter(Offre.id.in_(candidate_ids)).all()
    existing_ids = {rec[0] for rec in existing_records}

    # 3. Élimination des doublons
    new_jobs = []
    seen_in_batch = set()

    for job in jobs:
        jid = job["id"]
        if jid in existing_ids or jid in seen_in_batch:
            continue
        seen_in_batch.add(jid)
        new_jobs.append(job)

    duplicates_count = len(jobs) - len(new_jobs)
    return new_jobs, duplicates_count
