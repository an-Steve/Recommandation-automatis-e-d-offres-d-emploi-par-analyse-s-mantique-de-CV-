"""
src/scrapers/adzuna_scraper.py
Connecteur officiel pour l'API REST Adzuna (France).
Spécification OpenAPI : docs/Adzuna.json
"""

import logging
from typing import List, Dict, Any, Optional
import requests

from src.config import settings
from src.scrapers.base_scraper import BaseJobScraper

logger = logging.getLogger(__name__)


class AdzunaScraper(BaseJobScraper):
    """
    Connecteur API REST pour l'agrégateur d'emploi Adzuna en France.
    """

    BASE_URL = "https://api.adzuna.com/v1/api/jobs"

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_key: Optional[str] = None,
        country: str = "fr"
    ):
        super().__init__(name="Adzuna")
        # Récupération depuis les paramètres ou depuis la configuration centrale
        self.app_id = app_id or settings.ADZUNA_APP_ID
        self.app_key = app_key or settings.ADZUNA_APP_KEY
        self.country = country.lower()

    def _normalize_contract(self, contract_type: Optional[str], contract_time: Optional[str]) -> str:
        """Convertit les critères contractuels Adzuna en types de contrats français."""
        if contract_type == "permanent":
            return "CDI"
        if contract_type == "contract":
            return "CDD / Mission"
        if contract_time == "part_time":
            return "Temps partiel"
        return "Non spécifié"

    def search_jobs(
        self,
        query: str,
        location: str = "Paris",
        limit: int = 20,
        page: int = 1,
        distance_km: int = 25,
        max_days_old: Optional[int] = 30,
        category: Optional[str] = "it-jobs",
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Interroge l'API Adzuna France pour rechercher des offres d'emploi.

        Args:
            query (str): Mots-clés (ex: "Data Scientist", "Machine Learning").
            location (str): Ville ou département (ex: "Paris", "Lyon").
            limit (int): Nombre d'offres souhaité par page (max 50 recommandé).
            page (int): Numéro de la page (commence à 1).
            distance_km (int): Rayon kilométrique autour de 'location'.
            max_days_old (int, optionnel): Fraîcheur des annonces en jours.
            category (str, optionnel): Filtre secteur ("it-jobs" par défaut pour la tech).

        Returns:
            List[Dict[str, Any]]: Liste d'offres standardisées.
        """
        if not self.app_id or not self.app_key:
            logger.error(
                "❌ Clés Adzuna manquantes ! Veuillez renseigner ADZUNA_APP_ID et ADZUNA_APP_KEY dans votre fichier .env"
            )
            return []

        endpoint = f"{self.BASE_URL}/{self.country}/search/{page}"
        params: Dict[str, Any] = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "results_per_page": min(limit, 50),
            "what": query,
            "where": location,
            "distance": distance_km,
            "sort_by": "relevance",
        }

        if max_days_old:
            params["max_days_old"] = max_days_old
        if category:
            params["category"] = category

        logger.info(
            f"🔍 Requête Adzuna | Country={self.country} | Query='{query}' | Where='{location}' | Page={page}"
        )

        try:
            response = requests.get(endpoint, params=params, timeout=12)

            if response.status_code == 410:
                logger.error("❌ Erreur 410 Adzuna : Vos identifiants ADZUNA_APP_ID ou ADZUNA_APP_KEY sont invalides.")
                return []

            if response.status_code != 200:
                logger.error(
                    f" Erreur API Adzuna (HTTP {response.status_code}) : {response.text}"
                )
                return []

            data = response.json()
            raw_jobs = data.get("results", [])
            total_count = data.get("count", 0)
            logger.info(f"📊 Adzuna indique {total_count} offres au total sur cette recherche.")

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur réseau lors de la communication avec Adzuna : {e}")
            return []
        except ValueError as e:
            logger.error(f"❌ Impossible de décoder la réponse JSON d'Adzuna : {e}")
            return []

        results: List[Dict[str, Any]] = []

        for item in raw_jobs:
            title = str(item.get("title", "")).strip()
            description = str(item.get("description", "")).strip()

            if not title or len(description) < 20:
                continue

            company_info = item.get("company", {})
            company_name = company_info.get("display_name", "Entreprise non spécifiée") if isinstance(company_info, dict) else "Entreprise non spécifiée"

            location_info = item.get("location", {})
            location_name = location_info.get("display_name", location) if isinstance(location_info, dict) else location

            contract = self._normalize_contract(
                contract_type=item.get("contract_type"),
                contract_time=item.get("contract_time")
            )

            job_id = f"adzuna_{item.get('id')}"
            salary_min = item.get("salary_min")
            salary_max = item.get("salary_max")

            clean_item = {
                "source": "adzuna",
                "external_id": job_id,
                "poste": title,
                "company": company_name,
                "location": location_name,
                "type_de_contrat": contract,
                "remote_policy": "Non spécifié",
                "description": description,
                "mission": description,
                "source_url": item.get("redirect_url", ""),
                "salary_min": salary_min,
                "salary_max": salary_max,
                "created_at": item.get("created"),
            }
            results.append(clean_item)

        logger.info(f" Adzuna : {len(results)} offres exploitables récupérées.")
        return results
