"""
src/pipeline/job_normalizer.py
Normalisation des offres d'emploi brutes vers SQLAlchemy 'Offre' avec assainissement textuel intelligent.
Élimine le bruit corporate (intros d'entreprises, avantages sociaux, tickets resto)
et sépare intelligemment 'missions' et 'profil' sans altérer les termes techniques.
"""

import re
import logging
from datetime import date, datetime
from typing import Dict, Any

from src.database.sql_models import Offre

logger = logging.getLogger(__name__)


class JobTextSanitizer:
    """
    Nettoyeur et découpeur textuel algorithmique déterministe (0 ms, 0 token, 0 surcharge CPU).
    Spécifiquement calibré pour les offres volumineuses Indeed et LinkedIn.
    """

    # Motifs d'introduction corporate à éliminer
    INTRO_PATTERNS = (
        r"^(?:\*\*|#{1,4}\s*)?(?:description de l['’]entreprise|à propos|about\s+(?:us|the\s+company)|"
        r"our\s+story|qui sommes[- ]nous|notre entreprise|l['’]entreprise|mieux nous conna[îi]tre|"
        r"notre histoire|who\s+we\s+are|l['’]enseigne|le groupe)[\s\S]*?"
        r"(?=(?:\*\*|#{1,4}\s*)?(?:description du poste|vos missions|missions?|descriptif|ce que vous ferez|"
        r"le poste|vos t[âa]ches|why\s+we\s+need\s+you|you\s+will\s+be|what\s+you\s+will\s+do|responsibilit(?:ies|y)))"
    )

    # Motifs de fin de page parasites à éliminer (avantages, salaire, légal)
    OUTRO_PATTERNS = (
        r"(?:\n|^)(?:\*\*|#{1,4}\s*)?(?:✔︎\s*des\s+avantages|avantages?|benefits?|package|r[eé]mun[eé]ration|"
        r"salaire|tickets restaurant|swile|mutuelle|cse|horaires|ce que nous offrons|what\s+we\s+offer|"
        r"equal\s+opportunity|inclusive\s+and\s+respectful|tous\s+nos\s+postes\s+sont\s+ouverts|"
        r"background\s+check|mentions l[eé]gales|additional\s+information|processus de recrutement)[\s\S]*$"
    )

    # Motifs de démarcation pour isoler la section Profil / Compétences
    PROFILE_SPLIT = (
        r"(?:\n|^)((?:\*\*|#{1,4}\s*)?(?:votre profil|profil recherch[eé]|comp[eé]tences?(?:\s+requises?)?|"
        r"qualifications?|pr[eé]requis|qui [êe]tes[- ]vous|ce que nous recherchons|vos atouts|"
        r"what\s+we\s+are\s+looking\s+for|your\s+profile|requirements?|skills|who\s+you\s+are)[\s\S]*)"
    )

    @classmethod
    def sanitize_and_split(cls, raw_text: str) -> Dict[str, str]:
        """
        Nettoie le bruit corporate et sépare 'mission' et 'description' (profil recherché).
        """
        if not raw_text or len(raw_text.strip()) < 150:
            clean = (raw_text or "").strip()
            return {"mission": clean, "description": clean, "cleaned_full": clean}

        text = raw_text.strip()

        # 1. Élimination de l'introduction d'entreprise
        intro_match = re.search(cls.INTRO_PATTERNS, text, re.IGNORECASE)
        if intro_match:
            text = text[intro_match.end():].strip()

        # 2. Élimination des avantages sociaux et mentions légales de fin de page
        outro_match = re.search(cls.OUTRO_PATTERNS, text, re.IGNORECASE)
        if outro_match:
            text = text[:outro_match.start()].strip()

        # 3. Découpage intelligent : Missions vs Profil
        match_profile = re.search(cls.PROFILE_SPLIT, text, re.IGNORECASE)
        if match_profile:
            missions_part = text[:match_profile.start()].strip()
            profile_part = match_profile.group(1).strip()
        else:
            missions_part = text
            profile_part = text

        # Nettoyage cosmétique des titres d'en-tête de mission
        missions_clean = re.sub(
            r"^(?:\*\*|#{1,4}\s*)?(?:vos missions|missions?|descriptif de l['’]offre|description du poste)[\s\S]*?\n",
            "",
            missions_part,
            flags=re.IGNORECASE
        ).strip()

        return {
            "mission": missions_clean or text,
            "description": profile_part or text,
            "cleaned_full": text
        }


def create_sql_offre_from_raw(raw: Dict[str, Any]) -> Offre:
    """
    Instancie un enregistrement SQLAlchemy 'Offre' avec assainissement textuel
    et statut is_indexed=False en attente d'embedding.
    """
    poste = str(raw.get("poste", "")).strip()
    company = str(raw.get("company", "Entreprise confidentielle")).strip()
    location = str(raw.get("location", "France")).strip()
    raw_desc = str(raw.get("description", "")).strip()

    # Assainissement textuel et découpage intelligent
    sanitized = JobTextSanitizer.sanitize_and_split(raw_desc)
    clean_mission = sanitized["mission"]
    clean_desc = sanitized["description"]

    # Si une mission distincte était déjà fournie par l'API source
    custom_mission = raw.get("mission")
    if custom_mission and str(custom_mission).strip() != raw_desc:
        clean_mission = str(custom_mission).strip()

    raw_date = raw.get("date_posted")
    parsed_date = date.today()
    if raw_date:
        if isinstance(raw_date, date):
            parsed_date = raw_date
        elif isinstance(raw_date, str):
            try:
                parsed_date = datetime.fromisoformat(raw_date[:10]).date()
            except Exception:
                pass

    return Offre(
        id=raw["id"],
        poste=poste,
        company=company,
        location=location,
        type_de_contrat=raw.get("type_de_contrat", "CDI"),
        niveau_etude_min=int(raw.get("niveau_etude_min", 0)),
        min_experience_years=float(raw.get("min_experience_years", 0.0)),
        competences_necessaires=raw.get("competences_necessaires", []),
        description=clean_desc,
        mission=clean_mission,
        source_url=raw.get("source_url", ""),
        source=str(raw.get("source", "unknown")).lower(),
        secteur=str(raw.get("secteur", "it_data")).lower(),
        date_posted=parsed_date,
        is_active=True,
        is_indexed=False,  # En attente de vectorisation ChromaDB
        indexed_at=None
    )
