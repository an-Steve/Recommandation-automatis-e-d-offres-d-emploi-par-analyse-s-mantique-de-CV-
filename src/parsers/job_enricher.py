"""
src/parsers/job_enricher.py
Module d'enrichissement d'offres d'emploi multi-secteurs via GLiNER (Zero-Shot NER extractif)
et extraction déterministe des prérequis (diplôme Bac+X, années d'expérience).
Remplace le prompting Few-Shot génératif : 0 hallucination, support natif des 8 secteurs.
"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from src.database.sql_models import Offre
from src.parsers.gliner_extractor import (
    GlinerSkillExtractor,
    JobRequirementsHeuristic,
    DEFAULT_MODEL_NAME
)

logger = logging.getLogger(__name__)


class JobEnrichmentOutput(BaseModel):
    """
    Schéma de validation Pydantic strict pour l'extraction des critères d'offre d'emploi.
    Tolérant aux valeurs null et types approximatifs.
    """
    competences_necessaires: Optional[List[str]] = Field(
        default_factory=list,
        description=(
            "Liste normalisée des compétences, outils, langages, logiciels ou habilitations "
            "extraits verbatim de l'annonce (ex: ['Python', 'Docker'] ou ['Sage 100', 'IFRS'])."
        )
    )
    niveau_etude_min: Optional[int] = Field(
        default=0,
        description=(
            "Niveau d'études minimum requis exprimé en années d'études supérieures post-bac : "
            "0 = non spécifié ou aucun diplôme exigé, "
            "2 = Bac+2 (BTS, DUT), "
            "3 = Bac+3 (Licence, Bachelor), "
            "5 = Bac+5 (Master, Diplôme d'ingénieur), "
            "8 = Doctorat / PhD."
        )
    )
    min_experience_years: Optional[float] = Field(
        default=0.0,
        description=(
            "Nombre d'années minimum d'expérience professionnelle requises. "
            "0.0 si débutant / junior / non spécifié, 1.0, 2.0, 3.0, 5.0 pour senior."
        )
    )

    @field_validator("competences_necessaires", mode="before")
    @classmethod
    def coerce_skills(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
        return []

    @field_validator("niveau_etude_min", mode="before")
    @classmethod
    def coerce_diploma(cls, v):
        if v is None:
            return 0
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    @field_validator("min_experience_years", mode="before")
    @classmethod
    def coerce_experience(cls, v):
        if v is None:
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0


class JobEnricher:
    """
    Enrichisseur d'offres d'emploi multi-secteurs.
    Exploite GLiNER (Zero-Shot NER) pour l'extraction fidèle de compétences
    et JobRequirementsHeuristic pour le diplôme et l'expérience.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.gliner_extractor = GlinerSkillExtractor.get_instance(
            model_name=model_name or DEFAULT_MODEL_NAME
        )

    def enrich_job_text(
        self,
        poste: str,
        description: str,
        secteur: Optional[str] = "it_data"
    ) -> JobEnrichmentOutput:
        """
        Extrait les compétences, diplôme et expérience sans aucun prompting Few-Shot.
        Garantit une extraction 100% verbatim et adaptée au secteur économique.
        """
        full_text = f"{poste}\n{description}"

        # 1. Extraction des compétences et outils par GLiNER (Zero-Shot multi-secteurs)
        skills = self.gliner_extractor.extract_skills(
            text=full_text,
            secteur=secteur
        )

        # 2. Extraction du diplôme et de l'expérience par heuristique déterministe
        diploma = JobRequirementsHeuristic.extract_diploma(full_text)
        experience = JobRequirementsHeuristic.extract_experience(full_text)

        return JobEnrichmentOutput(
            competences_necessaires=skills,
            niveau_etude_min=diploma,
            min_experience_years=experience,
        )

    def enrich_offre_instance(self, offre: Offre) -> Offre:
        """
        Met à jour in-place une instance SQLAlchemy Offre avec les champs extraits.
        S'adapte dynamiquement au secteur de l'offre (santé, BTP, finance, IT, etc.).
        """
        full_text = f"{offre.poste}\n{offre.description}\n{offre.mission or ''}"

        # 1. Extraction des compétences ciblées selon le secteur de l'offre
        skills = self.gliner_extractor.extract_skills(
            text=full_text,
            secteur=offre.secteur
        )

        # Si aucune compétence spécifique n'est mentionnée dans l'annonce
        if not skills:
            skills = ["Non spécifié"]

        offre.competences_necessaires = skills

        # 2. Détection du niveau d'études
        diploma = JobRequirementsHeuristic.extract_diploma(full_text)
        if diploma > 0:
            offre.niveau_etude_min = diploma

        # 3. Détection des années d'expérience requises
        exp = JobRequirementsHeuristic.extract_experience(full_text)
        if exp > 0.0:
            offre.min_experience_years = exp

        # Marque l'offre comme prête pour synchronisation dans ChromaDB
        offre.is_indexed = False

        return offre
