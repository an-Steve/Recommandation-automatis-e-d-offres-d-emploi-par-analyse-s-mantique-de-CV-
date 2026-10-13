"""
src/parsers/job_enricher.py
Module d'enrichissement IA des offres d'emploi via LLM local (Ollama + Pydantic).
Zéro regex, zéro dictionnaire : extraction sémantique contextuelle et structurée.
"""

import re
import logging
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from src.utils.ollama_client import OllamaClient
from src.database.sql_models import Offre

logger = logging.getLogger(__name__)


class JobEnrichmentOutput(BaseModel):
    """
    Schéma de validation Pydantic strict pour l'extraction des critères d'offre d'emploi.
    Tolérant aux valeurs null et types approximatifs des petits modèles locaux.
    """
    competences_necessaires: Optional[List[str]] = Field(
        default_factory=list,
        description=(
            "Liste normalisée des compétences techniques, outils, langages et bibliothèques "
            "exigés ou fortement souhaités dans l'offre (ex: ['Python', 'SQL', 'PyTorch', 'Docker']). "
            "Ne pas inclure de compétences comportementales génériques (ex: 'rigueur')."
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
    Enrichisseur d'offres utilisant OllamaClient et Structured Outputs Pydantic.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.client = OllamaClient(default_model=model_name)

    @staticmethod
    def extract_requirements_snippet(description: str, max_chars: int = 1200) -> str:
        """
        Extrait intelligemment la section la plus pertinente d'une offre d'emploi
        (Profil recherché, Compétences requises, Qualifications) pour minimiser la charge CPU du LLM.
        """
        clean_text = description.strip()
        if len(clean_text) <= max_chars:
            return clean_text

        anchor_patterns = [
            r"(profil\s+recherch[eé]|votre\s+profil|qui\s+êtes[- ]vous|ce\s+que\s+nous\s+recherchons)",
            r"(comp[eé]tences\s+requises?|pr[eé]requis|qualifications?|exigences?|pr[eé]alables?)",
            r"(vos\s+atouts|profil\s+souhait[eé]|exp[eé]rience\s+souhait[eé]e)",
            r"(skills|requirements|what\s+you\s+need|who\s+you\s+are)",
        ]

        lower_text = clean_text.lower()
        best_match_idx = -1

        for pat in anchor_patterns:
            match = re.search(pat, lower_text)
            if match:
                best_match_idx = match.start()
                break

        if best_match_idx != -1:
            start_idx = max(0, best_match_idx - 50)
            return clean_text[start_idx : start_idx + max_chars]

        # Si aucun en-tête repéré, on prend la fin de la description où se situe le profil
        return clean_text[-max_chars:]

    def enrich_job_text(self, poste: str, description: str) -> JobEnrichmentOutput:
        """
        Extrait les compétences, diplôme et expérience avec la stratégie gagnante Few-Shot Compact.
        """
        snippet = self.extract_requirements_snippet(description)

        system_prompt = (
            "Tu es un assistant RH technique. Extrais sous format JSON strict EXACTEMENT et UNIQUEMENT ces 3 champs :\n"
            "- \"competences_necessaires\": liste de chaînes des compétences/technologies/outils demandés\n"
            "- \"niveau_etude_min\": entier (0 si non spécifié, 2 pour Bac+2, 3 pour Bac+3, 5 pour Bac+5/Ingénieur)\n"
            "- \"min_experience_years\": float (ex: 0.0 pour débutant, 2.0 pour 2 ans)\n"
            "Règle stricte : n'ajoute AUCUN autre champ dans le JSON."
        )
        shot_user = "Poste : Analyste Data\nExtrait : Bac+5 ingénieur, maîtrise SQL et Tableau. Débutant."
        shot_assistant = '{"competences_necessaires": ["SQL", "Tableau"], "niveau_etude_min": 5, "min_experience_years": 0.0}'

        target_user = f"Poste : {poste}\nExtrait : {snippet}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": shot_user},
            {"role": "assistant", "content": shot_assistant},
            {"role": "user", "content": target_user}
        ]

        raw_output = self.client.chat(
            messages=messages,
            format="json",
            temperature=0.0,
            num_ctx=1024,
            num_predict=260,
            repeat_penalty=1.2,
        )

        return self.client._clean_and_parse(raw_output, JobEnrichmentOutput)

    def enrich_offre_instance(self, offre: Offre) -> Offre:
        """
        Met à jour in-place une instance SQLAlchemy Offre avec les champs extraits.
        """
        enriched = self.enrich_job_text(poste=offre.poste, description=offre.description)
        # Nettoyage et déduplication des compétences
        raw_skills = enriched.competences_necessaires or []
        skills = sorted(list({s.strip() for s in raw_skills if s and s.strip()}))

        # Si aucune compétence technique explicite n'a été trouvée (ex: offre purement managériale/produit),
        # on enregistre ['Non spécifié'] pour marquer définitivement l'offre comme traitée
        if not skills:
            skills = ["Non spécifié"]

        offre.competences_necessaires = skills
        if enriched.niveau_etude_min is not None and int(enriched.niveau_etude_min) > 0:
            offre.niveau_etude_min = int(enriched.niveau_etude_min)
        if enriched.min_experience_years is not None and float(enriched.min_experience_years) > 0.0:
            offre.min_experience_years = float(enriched.min_experience_years)

        # Les compétences ayant changé, l'offre nécessite une mise à jour dans ChromaDB
        offre.is_indexed = False

        return offre
