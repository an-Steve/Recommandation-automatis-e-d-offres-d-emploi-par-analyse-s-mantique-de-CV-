"""
src/parsers/gliner_extractor.py
Extracteur d'entités et compétences Zero-Shot universel multi-secteurs basé sur GLiNER.
Garantit 0 hallucination (extraction span-based verbatim) sur les 8 secteurs du projet.
Intègre également l'extraction déterministe (0 ms) du niveau d'études et de l'expérience.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)

# Modèle GLiNER multilingue officiel
DEFAULT_MODEL_NAME = "urchade/gliner_multi-v2.1"


def resolve_gliner_model_path(model_name: str = DEFAULT_MODEL_NAME) -> str:
    """
    Résout le chemin du modèle :
    1. Si un snapshot local complet est présent dans models/huggingface/ ou dans le cache Hugging Face,
       l'utilise directement (chargement instantané hors-ligne).
    2. Sinon, retourne l'identifiant distant Hugging Face.
    """
    if os.path.exists(model_name):
        return model_name

    from src.config import settings
    # Vérifier en priorité le dossier models/ local du projet, puis le cache utilisateur
    candidate_dirs = [
        settings.huggingface_cache_dir / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots",
        Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots",
    ]

    for hub_cache in candidate_dirs:
        if hub_cache.exists():
            snapshots = [d for d in hub_cache.iterdir() if d.is_dir()]
            if snapshots:
                for snap in snapshots:
                    if (snap / "model.safetensors").exists() or (snap / "pytorch_model.bin").exists() or (snap / "model.onnx").exists():
                        return str(snap)
    return model_name


class GlinerSkillExtractor:
    """
    Extracteur de compétences déterministe et universel via GLiNER.
    Charge paresseusement le modèle en mémoire (Singleton) pour optimiser l'empreinte RAM.
    """

    _INSTANCE: Optional["GlinerSkillExtractor"] = None
    _MODEL = None

    # Ontologie sectorielle dynamique pour les 8 secteurs (Hard Skills + Soft Skills métier)
    SECTOR_ONTOLOGY: Dict[str, List[str]] = {
        "it_data": [
            "langage de programmation", "framework", "outil cloud",
            "base de données", "logiciel", "technologie", "compétence technique",
            "rigueur", "autonomie", "résolution de problèmes", "esprit d'équipe"
        ],
        "sante_medical": [
            "acte de soin", "spécialité médicale", "logiciel médical",
            "protocole de soin", "certification médicale", "diplôme d'état",
            "écoute", "empathie", "sens relationnel", "patience", "gestion du stress", "bienveillance"
        ],
        "btp_construction": [
            "habilitation", "norme de sécurité", "logiciel CAO", "compétence chantier",
            "conduite de travaux", "engin de chantier",
            "rigueur", "sens des responsabilités", "réactivité", "organisation", "esprit d'équipe"
        ],
        "commerce_marketing": [
            "logiciel CRM", "technique de vente", "prospection", "outil marketing",
            "négociation", "compétence commerciale", "levier d'acquisition",
            "aisance relationnelle", "force de persuasion", "sens du contact", "dynamisme", "écoute active"
        ],
        "commercial_vente": [
            "logiciel CRM", "technique de vente", "prospection", "négociation",
            "aisance relationnelle", "force de persuasion", "sens commercial"
        ],
        "finance_gestion": [
            "logiciel comptable", "norme financière", "compétence fiscale", "outil financier",
            "audit", "méthode comptable", "contrôle de gestion", "tableur",
            "rigueur", "sens de la confidentialité", "esprit d'analyse", "précision", "organisation"
        ],
        "finance_compta": [
            "logiciel comptable", "norme financière", "compétence fiscale", "outil financier",
            "audit", "rigueur", "précision", "confidentialité"
        ],
        "industrie_ingenierie": [
            "technique de fabrication", "maintenance industrielle", "logiciel CAO", "logiciel FAO",
            "automatisme", "mécanique", "norme de qualité", "lean manufacturing",
            "rigueur", "résolution de problème", "vigilance", "esprit d'équipe"
        ],
        "industrie_btp": [
            "habilitation", "norme de sécurité", "logiciel CAO", "technique de fabrication",
            "rigueur", "sens des responsabilités"
        ],
        "rh_juridique": [
            "logiciel SIRH", "droit social", "domaine juridique", "processus RH",
            "recrutement", "réglementation", "droit du travail",
            "sens de la confidentialité", "écoute active", "aisance relationnelle", "diplomatie"
        ],
        "general_autre": [
            "compétence technique", "logiciel", "outil", "certification",
            "polyvalence", "sens du service", "ponctualité", "autonomie", "rigueur"
        ],
    }

    # Labels de repli si le secteur n'est pas spécifié
    GENERIC_LABELS: List[str] = [
        "compétence technique", "logiciel", "outil", "certification",
        "habilitation", "norme", "langage", "qualité relationnelle",
        "savoir-être", "soft skill", "autonomie", "rigueur"
    ]

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        if GlinerSkillExtractor._MODEL is None:
            from gliner import GLiNER
            target_path = resolve_gliner_model_path(model_name)
            logger.info(f"⏳ Chargement du modèle GLiNER multilingue depuis '{target_path}'...")
            # Si model.onnx est présent localement, on active l'accélération ONNX, sinon chargement direct PyTorch
            onnx_file = Path(target_path) / "model.onnx"
            if onnx_file.exists():
                try:
                    GlinerSkillExtractor._MODEL = GLiNER.from_pretrained(target_path, load_onnx_model=True)
                    logger.info("✅ Modèle GLiNER chargé en mode ONNX Runtime.")
                except Exception as e:
                    logger.debug(f"Erreur chargement ONNX ({e}), repli standard PyTorch...")
                    GlinerSkillExtractor._MODEL = GLiNER.from_pretrained(target_path)
            else:
                GlinerSkillExtractor._MODEL = GLiNER.from_pretrained(target_path)
                logger.info("✅ Modèle GLiNER prêt (PyTorch).")
        self.model = GlinerSkillExtractor._MODEL

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_MODEL_NAME) -> "GlinerSkillExtractor":
        """Garantit une instance unique en cache mémoire (Singleton)."""
        if cls._INSTANCE is None:
            cls._INSTANCE = cls(model_name=model_name)
        return cls._INSTANCE

    def extract_skills(
        self,
        text: str,
        secteur: Optional[str] = "it_data",
        threshold: float = 0.40,
        max_chars: int = 1400
    ) -> List[str]:
        """
        Extrait les compétences et outils en appliquant les étiquettes du secteur.

        Args:
            text (str): Texte de l'annonce (titre, profil recherché ou description).
            secteur (str, optionnel): Identifiant du secteur (ex: 'sante_medical').
            threshold (float): Seuil de confiance minimal (défaut : 0.45).
            max_chars (int): Fenêtre maximale de caractères analysée.

        Returns:
            List[str]: Liste dédupliquée et normalisée des compétences extraites verbatim.
        """
        if not text or len(text.strip()) < 15:
            return []

        # 1. Sélection des étiquettes sectorielles ciblées
        clean_secteur = (secteur or "").strip().lower()
        labels = self.SECTOR_ONTOLOGY.get(clean_secteur, self.GENERIC_LABELS)

        # 2. Rognage pour accélérer l'inférence
        snippet = text.strip()[:max_chars]

        # 3. Prédiction Zero-Shot des entités (extraction de spans stricts)
        entities = self.model.predict_entities(snippet, labels, threshold=threshold)

        # 4. Nettoyage, filtrage de longueur et déduplication
        extracted: Set[str] = set()
        for ent in entities:
            val = ent.get("text", "").strip()
            # On ignore les mots parasites trop courts ou purement numériques
            if len(val) >= 2 and not val.isdigit():
                clean_val = val.strip("•-–—:;,.()[]\"' \t\n")
                if len(clean_val) >= 2:
                    extracted.add(clean_val)

        return sorted(list(extracted))


class JobRequirementsHeuristic:
    """
    Extraction déterministe instantanée (0 ms, 0 token) du niveau d'études (Bac+X)
    et des années d'expérience requises par regex numériques strictes.
    """

    DIPLOMA_PATTERNS = [
        (r"\b(?:bac\s*\+\s*8|doctorat|ph\.?d)\b", 8),
        (r"\b(?:bac\s*\+\s*5|master(?:\s*2)?|ing[eé]nieur|dipl[oô]me d['’]ing[eé]nieur|m2|dea|dess)\b", 5),
        (r"\b(?:bac\s*\+\s*4|master\s*1|m1)\b", 4),
        (r"\b(?:bac\s*\+\s*3|licence(?:\s*pro)?|bachelor|but)\b", 3),
        (r"\b(?:bac\s*\+\s*2|bts|dut|deug)\b", 2),
        (r"\b(?:baccalaur[eé]at|niveau bac)\b", 1),
    ]

    @classmethod
    def extract_diploma(cls, text: str) -> int:
        """Détecte le niveau d'études Bac+X maximal mentionné dans l'offre."""
        if not text:
            return 0
        lower = text.lower()
        for pattern, level in cls.DIPLOMA_PATTERNS:
            if re.search(pattern, lower):
                return level
        return 0

    @classmethod
    def extract_experience(cls, text: str) -> float:
        """Détecte les années d'expérience requises (ex: '3 ans', 'au moins 5 ans')."""
        if not text:
            return 0.0
        lower = text.lower()
        if re.search(r"\b(?:d[eé]butant|premi[eè]re exp[eé]rience|junior)\b", lower):
            return 0.0

        # Patterns du type 'au moins 3 ans', 'minimum 2 ans d'expérience', '5 ans d'expérience'
        patterns = [
            r"(?:minimum|au\s+moins|[eé]tudiant|justifi(?:ez|er)\s+d['’]au\s+moins)\s+(\d+(?:[.,]\d+)?)\s*(?:ans?|ann[eé]es?)",
            r"(\d+(?:[.,]\d+)?)\s*(?:ans?|ann[eé]es?)\s+(?:d['’]exp[eé]rience|d['’]exp|minimum|souhait[eé]e)",
            r"(?:exp[eé]rience\s+de)\s+(\d+(?:[.,]\d+)?)\s*(?:ans?|ann[eé]es?)",
        ]
        for pat in patterns:
            match = re.search(pat, lower)
            if match:
                try:
                    return float(match.group(1).replace(",", "."))
                except ValueError:
                    pass
        return 0.0
