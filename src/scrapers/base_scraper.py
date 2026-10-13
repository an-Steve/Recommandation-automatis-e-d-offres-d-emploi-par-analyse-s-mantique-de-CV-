from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseJobScraper(ABC):
    """
    Interface abstraite pour la collecte d'offres d'emploi.
    Garantit l'interchangeabilité des sources dans le pipeline d'ingestion.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def search_jobs(
        self,
        query: str,
        location: str = "France",
        limit: int = 20,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Recherche et retourne une liste de dictionnaires représentant des offres brutes.

        Args:
            query (str): Mots-clés de recherche (ex: "Data Scientist", "Ingénieur IA").
            location (str): Localisation cible (ex: "Paris", "France", "Lyon").
            limit (int): Nombre maximum de résultats souhaités.
            **kwargs: Paramètres additionnels spécifiques au scraper.

        Returns:
            List[Dict[str, Any]]: Liste des offres au format brut standardisé.
        """
        pass
