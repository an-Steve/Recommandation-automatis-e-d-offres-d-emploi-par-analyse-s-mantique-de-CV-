"""
src/scrapers/sector_mapping.py
Taxonomie des 8 secteurs d'activité universels et mapping pour Adzuna et JobSpy.
Aligne les catégories natives d'Adzuna et les requêtes parapluies pour Indeed/LinkedIn.
"""

from typing import Dict, Any, List

SECTOR_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "it_data": {
        "label": "Informatique, Data, IA & Web",
        "adzuna_category": "it-jobs",
        "jobspy_keywords": ["Informatique", "Data Scientist", "Développeur", "Intelligence Artificielle"],
        "default_query": "Informatique",
    },
    "sante_medical": {
        "label": "Santé, Soins, Pharmacie & Médical",
        "adzuna_category": "healthcare-nursing-jobs",
        "jobspy_keywords": ["Santé", "Infirmier", "Médecin", "Pharmacie", "Biologie"],
        "default_query": "Santé",
    },
    "btp_construction": {
        "label": "BTP, Bâtiment, Travaux & Architecture",
        "adzuna_category": "trade-construction-jobs",
        "jobspy_keywords": ["BTP", "Bâtiment", "Conducteur de travaux", "Architecture"],
        "default_query": "BTP",
    },
    "commerce_marketing": {
        "label": "Commerce, Vente & Marketing",
        "adzuna_category": "sales-jobs",
        "jobspy_keywords": ["Commercial", "Vente", "Marketing", "Business Developer"],
        "default_query": "Commercial",
    },
    "finance_gestion": {
        "label": "Finance, Comptabilité, Audit & Gestion",
        "adzuna_category": "accounting-finance-jobs",
        "jobspy_keywords": ["Comptabilité", "Contrôleur de gestion", "Audit", "Finance"],
        "default_query": "Comptabilité",
    },
    "industrie_ingenierie": {
        "label": "Industrie, Ingénierie & Logistique",
        "adzuna_category": "engineering-jobs",
        "jobspy_keywords": ["Ingénieur", "Industrie", "Logistique", "Production"],
        "default_query": "Industrie",
    },
    "rh_juridique": {
        "label": "Ressources Humaines & Juridique",
        "adzuna_category": "hr-jobs",
        "jobspy_keywords": ["Ressources Humaines", "Recrutement", "Juriste"],
        "default_query": "Ressources Humaines",
    },
    "general_autre": {
        "label": "Services Généraux & Polyvalent",
        "adzuna_category": "other-general-jobs",
        "jobspy_keywords": ["Services généraux", "Accueil", "Polyvalent"],
        "default_query": "Services généraux",
    },
}


def get_sector_config(sector_code: str) -> Dict[str, Any]:
    """Retourne la configuration d'un secteur ou celle de 'it_data' par défaut."""
    clean_code = str(sector_code).strip().lower()
    return SECTOR_TAXONOMY.get(clean_code, SECTOR_TAXONOMY["it_data"])


def list_supported_sectors() -> List[str]:
    """Retourne la liste des codes secteurs supportés."""
    return list(SECTOR_TAXONOMY.keys())
