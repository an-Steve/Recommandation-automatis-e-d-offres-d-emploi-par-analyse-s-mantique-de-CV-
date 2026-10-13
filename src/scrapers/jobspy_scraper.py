"""
src/scrapers/jobspy_scraper.py
Connecteur multi-plateformes utilisant python-jobspy (Indeed, LinkedIn, HelloWork).
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from jobspy import scrape_jobs

from src.scrapers.base_scraper import BaseJobScraper

logger = logging.getLogger(__name__)


# Dictionnaire de correspondance des contrats anglo-saxons vers le droit français
CONTRACT_MAPPING = {
    "fulltime": "CDI",
    "parttime": "Temps partiel",
    "internship": "Stage",
    "contract": "CDD / Freelance",
    "temporary": "Intérim",
}


class JobSpyScraper(BaseJobScraper):
    """
    Scraper robuste exploitant python-jobspy pour agréger LinkedIn, Indeed et HelloWork.
    """

    def __init__(
        self,
        sites: Optional[List[str]] = None,
        country_indeed: str = "france",
    ):
        super().__init__(name="JobSpy")
        # Par défaut, on cible Indeed et LinkedIn (les deux plus riches en tech/IA)
        self.sites = sites or ["indeed", "linkedin"]
        self.country_indeed = country_indeed

    def _normalize_contract_type(self, raw_type: Optional[str]) -> str:
        """Convertit les types de contrats JobSpy en termes français standard."""
        if not raw_type or pd.isna(raw_type):
            return "Non spécifié"
        clean_type = str(raw_type).strip().lower()
        return CONTRACT_MAPPING.get(clean_type, clean_type.capitalize())

    def search_jobs(
        self,
        query: str,
        location: str = "France",
        limit: int = 15,
        hours_old: Optional[int] = 168,  # 7 derniers jours par défaut
        fetch_description: bool = True,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Récupère les offres d'emploi sur les plateformes configurées.

        Args:
            query (str): Intitulé du poste ou mots-clés (ex: "Machine Learning").
            location (str): Ville ou pays (ex: "Paris", "France").
            limit (int): Nombre d'offres souhaité par site.
            hours_old (int, optionnel): Offres publiées dans les N dernières heures.
            fetch_description (bool): Forcer la récupération du corps complet en Markdown.

        Returns:
            List[Dict[str, Any]]: Liste d'offres normalisées.
        """
        logger.info(
            f"🔍 Lancement scraping JobSpy | Sites={self.sites} | Query='{query}' | Location='{location}' | Limit={limit}"
        )

        try:
            jobs_df: pd.DataFrame = scrape_jobs(
                site_name=self.sites,
                search_term=query,
                location=location,
                country_indeed=self.country_indeed,
                results_wanted=limit,
                hours_old=hours_old,
                description_format="markdown",
                fetch_description=fetch_description,
                verbose=0,
                **kwargs
            )
        except Exception as e:
            logger.error(f"❌ Erreur critique lors de l'appel à scrape_jobs: {e}")
            return []

        if jobs_df is None or jobs_df.empty:
            logger.warning("⚠️ Aucune offre trouvée par JobSpy pour ces critères.")
            return []

        results: List[Dict[str, Any]] = []

        for _, row in jobs_df.iterrows():
            title = str(row.get("title", "")).strip()
            description = str(row.get("description", "")).strip()

            # On ignore les offres sans titre ou sans description exploitable
            if not title or len(description) < 30:
                continue

            company = str(row.get("company", "Entreprise confidentielle")).strip()
            job_location = str(row.get("location", location)).strip()
            contract_type = self._normalize_contract_type(row.get("job_type"))
            is_remote = bool(row.get("is_remote", False))
            job_url = str(row.get("job_url", "")).strip()
            site_source = str(row.get("site", "web")).strip()

            raw_item = {
                "source": site_source,
                "external_id": str(row.get("id", "")),
                "poste": title,
                "company": company,
                "location": job_location,
                "type_de_contrat": contract_type,
                "remote_policy": "Télétravail" if is_remote else "Sur site / Hybride",
                "description": description,
                "source_url": job_url,
                "date_posted": str(row.get("date_posted", "")),
            }
            results.append(raw_item)

        logger.info(f" Scraping terminé avec succès : {len(results)} offres exploitables retenues.")
        return results
