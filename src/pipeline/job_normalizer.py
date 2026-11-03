"""
src/pipeline/job_normalizer.py
Normalisation des offres d'emploi brutes vers SQLAlchemy 'Offre' avec assainissement textuel intelligent.
Élimine le bruit corporate (intros d'entreprises, avantages sociaux, tickets resto)
et sépare intelligemment 'missions' et 'description' via un pipeline en cascade à 2 niveaux :
- Niveau 1 : Regex Déterministe (Ancres structurelles de sections françaises & anglaises)
- Niveau 2 : spaCy NLP Syntaxique (fr_core_news_sm, verbes d'action et infinitifs)
Garantit 0 hallucination (approche 100% extractive verbatim) et élimine les doublons description == mission.
Intègre également la normalisation stricte des types de contrats (simples et composites).
"""

import re
import logging
from datetime import date, datetime
from typing import Dict, Any, Optional

from src.database.sql_models import Offre

logger = logging.getLogger(__name__)

# Modèle spaCy chargé paresseusement (lazy-loading) pour optimiser le temps de démarrage
_SPACY_NLP = None


def get_spacy_nlp():
    """Charge paresseusement le modèle linguistique français spaCy en cache mémoire."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            import spacy
            _SPACY_NLP = spacy.load("fr_core_news_sm")
            logger.info("Modèle spaCy fr_core_news_sm initialisé avec succès.")
        except Exception as err:
            logger.warning(f"Impossible de charger le modèle spaCy fr_core_news_sm: {err}")
            _SPACY_NLP = False
    return _SPACY_NLP if _SPACY_NLP is not False else None


class JobTextSanitizer:
    """
    Nettoyeur et découpeur textuel algorithmique en 2 niveaux déterministes.
    Sépare strictement 'description' (contexte de l'entreprise, profil recherché)
    et 'mission' (tâches opérationnelles à accomplir).
    """

    # 1. Motifs d'introduction corporate à éliminer
    INTRO_PATTERNS = (
        r"^(?:\*\*|#{1,4}\s*)?(?:description de l['’]entreprise|à propos|about\s+(?:us|the\s+company)|"
        r"our\s+story|qui sommes[- ]nous|notre entreprise|l['’]entreprise|mieux nous conna[îi]tre|"
        r"notre histoire|who\s+we\s+are|l['’]enseigne|le groupe)[\s\S]*?"
        r"(?=(?:\*\*|#{1,4}\s*)?(?:description du poste|vos missions|missions?|descriptif|ce que vous ferez|"
        r"le poste|vos t[âa]ches|why\s+we\s+need\s+you|you\s+will\s+be|what\s+you\s+will\s+do|responsibilit(?:ies|y)))"
    )

    # 2. Motifs de fin de page parasites à éliminer (avantages sociaux, salaire, mentions légales)
    OUTRO_PATTERNS = (
        r"(?:\n|^)(?:\*\*|#{1,4}\s*)?(?:✔︎\s*des\s+avantages|avantages?|benefits?|package|r[eé]mun[eé]ration|"
        r"salaire|tickets restaurant|swile|mutuelle|cse|horaires|ce que nous offrons|what\s+we\s+offer|"
        r"equal\s+opportunity|inclusive\s+and\s+respectful|tous\s+nos\s+postes\s+sont\s+ouverts|"
        r"background\s+check|mentions l[eé]gales|additional\s+information|processus de recrutement)[\s\S]*$"
    )

    # 3. Détection des sections Niveau 1 (Regex)
    HEADER_MISSIONS = re.compile(
        r"(?:\n|^|[.]\s+)\s*(?:[#*_\-]{0,4}\s*)?(?:vos missions|missions?(?:\s+principales?)?|descriptif(?: du poste)?|"
        r"ce que vous ferez|le poste au quotidien|vos activit[eé]s|principales activit[eé]s|"
        r"d[eé]tail des missions|responsabilit[eé]s?|votre r[oô]le|"
        r"responsibilities|what\s+you(?:['’]ll|\s+will)\s+do|the\s+role)\s*[:\n]",
        re.IGNORECASE
    )

    HEADER_PROFILE = re.compile(
        r"(?:\n|^|[.]\s+)\s*(?:[#*_\-]{0,4}\s*)?(?:votre profil|profil(?: recherch[eé])?|comp[eé]tences?(?:\s+requises?)?|"
        r"qualifications?|pr[eé]requis|qui [êe]tes[- ]vous|ce que nous recherchons|vos atouts|"
        r"what\s+we\s+are\s+looking\s+for|your\s+profile|requirements?|skills|who\s+you\s+are)\s*[:\n]",
        re.IGNORECASE
    )

    # Verbes / Tournures typiques pour le Niveau 2 (spaCy + anglais)
    ACTION_PATTERNS = re.compile(
        r"\b(vous\s+(?:intervenez|accompagnerez|participez|assurez|d[eé]velopperez|d[eé]veloppez|g[eé]rez|serez\s+amen[eé])|"
        r"la\s+mission\s+(?:vise|consiste|a\s+pour\s+objectif)|[àa]\s+ce\s+titre,\s+vos\s+missions|"
        r"en\s+charge\s+de|responsable\s+de|objectif\s+de\s+la\s+mission|"
        r"you\s+will(?:\s+be)?|responsible\s+for|in\s+charge\s+of|role\s+and\s+responsibilities)\b",
        re.IGNORECASE
    )

    @classmethod
    def normalize_contract_label(cls, raw_contract: str) -> str:
        """
        Harmonise et nettoie les libellés de contrat (anglo-saxons ou composites).
        Ex: 'Fulltime, contract' -> 'CDD / Freelance', 'internship' -> 'Stage'.
        """
        if not raw_contract:
            return "Non spécifié"
        clean = str(raw_contract).strip().lower()
        mapping = {
            "fulltime": "CDI",
            "fulltime, contract": "CDD / Freelance",
            "contract, fulltime": "CDD / Freelance",
            "contract": "CDD / Freelance",
            "parttime": "Temps partiel",
            "parttime, contract": "CDD Temps partiel",
            "internship": "Stage",
            "contract, internship": "Stage",
            "temporary": "Intérim",
            "cdd / mission": "CDD / Mission",
            "cdd / freelance": "CDD / Freelance",
            "cdi": "CDI",
            "stage": "Stage",
            "alternance": "Alternance",
            "temps partiel": "Temps partiel",
            "intérim": "Intérim",
            "interim": "Intérim",
            "non spécifié": "Non spécifié",
        }
        if clean in mapping:
            return mapping[clean]
        if "internship" in clean or "stage" in clean:
            return "Stage"
        if "alternance" in clean or "apprentissage" in clean:
            return "Alternance"
        if "contract" in clean or "freelance" in clean:
            return "CDD / Freelance"
        if "parttime" in clean or "temps partiel" in clean:
            return "Temps partiel"
        if "fulltime" in clean:
            return "CDI"
        return raw_contract.strip()

    @classmethod
    def sanitize_and_split(cls, raw_text: str) -> Dict[str, str]:
        """
        Nettoie le bruit corporate et sépare 'mission' et 'description' via la cascade 2 niveaux.
        Garantit que 'mission' != 'description'.
        """
        if not raw_text or len(raw_text.strip()) < 80:
            clean = (raw_text or "").strip()
            return {
                "mission": "Non spécifié dans l'annonce",
                "description": clean,
                "cleaned_full": clean,
                "tier_used": "court_fallback"
            }

        text = raw_text.strip()

        # Étape 1 : Élimination du bruit d'en-tête et de pied de page
        intro_match = re.search(cls.INTRO_PATTERNS, text, re.IGNORECASE)
        if intro_match:
            text = text[intro_match.end():].strip()

        outro_match = re.search(cls.OUTRO_PATTERNS, text, re.IGNORECASE)
        if outro_match:
            text = text[:outro_match.start()].strip()

        # =====================================================================
        # NIVEAU 1 : Découpage Structurel (Regex)
        # =====================================================================
        match_miss = cls.HEADER_MISSIONS.search(text)
        match_prof = cls.HEADER_PROFILE.search(text)

        # Cas 1.A : Missions PUIS Profil trouvés
        if match_miss and match_prof and match_prof.start() > match_miss.end():
            desc_intro = text[:match_miss.start()].strip()
            miss_part = text[match_miss.end():match_prof.start()].strip()
            prof_part = text[match_prof.end():].strip()

            final_desc = f"{desc_intro}\n\nProfil recherché :\n{prof_part}".strip() if desc_intro else prof_part
            if not final_desc:
                final_desc = text[:250]

            return {
                "mission": miss_part or "Non spécifié dans l'annonce",
                "description": final_desc,
                "cleaned_full": text,
                "tier_used": "Niveau 1 (Regex Structurelle)"
            }

        # Cas 1.B : Section Missions identifiée seule
        if match_miss and not match_prof:
            desc_part = text[:match_miss.start()].strip()
            miss_part = text[match_miss.end():].strip()
            final_desc = desc_part if len(desc_part) > 50 else text[:200]
            return {
                "mission": miss_part or "Non spécifié dans l'annonce",
                "description": final_desc,
                "cleaned_full": text,
                "tier_used": "Niveau 1 (Regex Missions Seules)"
            }

        # Cas 1.C : Profil trouvé avant Missions
        if match_miss and match_prof and match_miss.start() > match_prof.end():
            prof_part = text[match_prof.end():match_miss.start()].strip()
            miss_part = text[match_miss.end():].strip()
            return {
                "mission": miss_part or "Non spécifié dans l'annonce",
                "description": prof_part or text[:200],
                "cleaned_full": text,
                "tier_used": "Niveau 1 (Regex Profil Puis Missions)"
            }

        # =====================================================================
        # NIVEAU 2 : Analyse Syntaxique & Verbes d'Action (spaCy NLP)
        # =====================================================================
        nlp = get_spacy_nlp()
        if nlp:
            try:
                doc = nlp(text)
                mission_sentences = []
                desc_sentences = []

                for sent in doc.sents:
                    sent_text = sent.text.strip()
                    if not sent_text or len(sent_text) < 15:
                        continue

                    is_mission = False

                    # Règle A : Verbe à l'infinitif en début de proposition
                    first_token = None
                    for token in sent:
                        if not token.is_punct and not token.is_space:
                            first_token = token
                            break

                    if first_token and first_token.pos_ == "VERB" and "VerbForm=Inf" in first_token.morph:
                        is_mission = True

                    # Règle B : Tournures idiomatiques d'action
                    if not is_mission and cls.ACTION_PATTERNS.search(sent_text):
                        is_mission = True

                    if is_mission:
                        mission_sentences.append(sent_text)
                    else:
                        desc_sentences.append(sent_text)

                # Si spaCy a isolé des missions opérationnelles concrètes
                if mission_sentences:
                    desc_result = " ".join(desc_sentences).strip()
                    if not desc_result or len(desc_result) < 50:
                        desc_result = text[:200]

                    return {
                        "mission": "\n• " + "\n• ".join(mission_sentences),
                        "description": desc_result,
                        "cleaned_full": text,
                        "tier_used": "Niveau 2 (spaCy NLP Syntaxique)"
                    }
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse spaCy : {e}")

        # =====================================================================
        # FALLBACK HONNÊTE : Aucune tâche opérationnelle détectable
        # =====================================================================
        return {
            "mission": "Non spécifié dans l'annonce",
            "description": text,
            "cleaned_full": text,
            "tier_used": "Fallback Honnête (0 Mission Détectée)"
        }


def create_sql_offre_from_raw(raw: Dict[str, Any]) -> Offre:
    """
    Instancie un enregistrement SQLAlchemy 'Offre' avec assainissement textuel,
    normalisation robuste du type de contrat, et statut is_indexed=False en attente d'embedding.
    """
    poste = str(raw.get("poste", "")).strip()
    company = str(raw.get("company", "Entreprise confidentielle")).strip()
    location = str(raw.get("location", "France")).strip()
    raw_desc = str(raw.get("description", "")).strip()

    # Assainissement textuel et découpage intelligent
    sanitized = JobTextSanitizer.sanitize_and_split(raw_desc)
    clean_mission = sanitized["mission"]
    clean_desc = sanitized["description"]

    # Si une mission distincte et non vide était déjà fournie par l'API source
    custom_mission = raw.get("mission")
    if custom_mission and str(custom_mission).strip() and str(custom_mission).strip() != raw_desc:
        clean_mission = str(custom_mission).strip()

    # Normalisation propre du type de contrat
    raw_contract = raw.get("type_de_contrat", "CDI")
    clean_contract = JobTextSanitizer.normalize_contract_label(raw_contract)

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
        type_de_contrat=clean_contract,
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
