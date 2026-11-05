"""
src/scrapers/jobscraper_connector.py
Connecteur unifié pour Welcome to the Jungle (WTTJ) et HelloWork.
Extrait des fiches complètes avec descriptions riches (via Algolia pour WTTJ et JSON-LD pour HelloWork)
et les formate selon le schéma standard du pipeline de recommandation de recrutement.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional
from datetime import date, datetime

import requests
from bs4 import BeautifulSoup

from src.scrapers.base_scraper import BaseJobScraper

logger = logging.getLogger(__name__)


def clean_html_text(raw_html: str) -> str:
    """Convertit du HTML brut en texte propre lisible."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    # Remplace les balises de fin de bloc par des sauts de ligne
    for br in soup.find_all(["br", "p", "div", "h1", "h2", "h3", "h4", "li"]):
        br.append("\n")
    text = soup.get_text()
    # Nettoyage des espaces et lignes vides consécutives
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


class JobScraperConnector(BaseJobScraper):
    """
    Connecteur pour interroger Welcome to the Jungle et HelloWork
    avec extraction exhaustive des descriptions et missions.
    """

    def __init__(self):
        super().__init__(name="JobScraperConnector")
        self.http_session = requests.Session()
        self.http_session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        })

    def search_jobs(
        self,
        query: str,
        location: str = "France",
        limit: int = 20,
        source: str = "wttj",
        target_date_iso: Optional[str] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Implémente l'interface BaseJobScraper en routant vers le bon scraper."""
        src = source.lower().strip()
        if src == "wttj":
            return self.search_wttj(query=query, location=location, limit=limit, target_date_iso=target_date_iso)
        elif src in ["hellowork", "hw"]:
            return self.search_hellowork(query=query, location=location, limit=limit, target_date_iso=target_date_iso)
        else:
            logger.warning(f"Source inconnue '{source}' pour JobScraperConnector.")
            return []

    # =========================================================================
    # 1. Welcome to the Jungle (WTTJ) via API Algolia
    # =========================================================================
    def search_wttj(
        self,
        query: str,
        location: str = "France",
        limit: int = 15,
        target_date_iso: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Interroge l'API Algolia publique de WTTJ pour extraire les offres,
        leurs missions clés structurées ('key_missions') et le profil recherché ('profile').
        """
        logger.info(f"🔎 [WTTJ] Recherche : query='{query}' | location='{location}' | limit={limit}")
        from jobscraper.scrapers.wttj import WTTJScraper

        scraper = WTTJScraper()
        results: List[Dict[str, Any]] = []
        date_iso = target_date_iso or date.today().isoformat()

        try:
            # Construction de la requête Algolia
            search_query = f"{query} {location}" if location.lower() != "france" else query
            algolia_res = scraper._fetch_algolia(
                query=search_query,
                filters="",
                page=0
            )

            hits = algolia_res.get("hits", [])
            for hit in hits:
                if len(results) >= limit:
                    break

                title = str(hit.get("name", "")).strip()
                if not title:
                    continue

                org = hit.get("organization") or {}
                company = str(org.get("name", "Entreprise confidentielle")).strip()

                # Localisation
                offices = hit.get("offices") or []
                if offices and isinstance(offices, list) and offices[0]:
                    city = offices[0].get("city", "")
                    country = offices[0].get("country", "")
                    loc = f"{city}, {country}" if city else country or location
                else:
                    loc = location

                # URL
                org_slug = org.get("slug", "")
                job_slug = hit.get("slug", "")
                url = f"https://www.welcometothejungle.com/fr/companies/{org_slug}/jobs/{job_slug}" if org_slug and job_slug else ""

                # Type de contrat
                contract_raw = hit.get("contract_type", "full_time")
                contract_type = "CDI"
                if contract_raw in ["fixed_term", "vie"]:
                    contract_type = "CDD"
                elif contract_raw in ["internship"]:
                    contract_type = "Stage"
                elif contract_raw in ["apprenticeship"]:
                    contract_type = "Alternance"
                elif contract_raw in ["freelance"]:
                    contract_type = "Freelance"

                # Remote
                remote_val = str(hit.get("remote", "")).lower()
                is_remote = remote_val in ["fulltime", "partial"]

                # Missions & Profil
                summary = str(hit.get("summary") or "").strip()
                raw_missions = hit.get("key_missions") or []
                mission_text = ""
                if isinstance(raw_missions, list) and raw_missions:
                    mission_text = "\n• " + "\n• ".join([str(m).strip() for m in raw_missions if str(m).strip()])

                raw_profile = hit.get("profile") or ""
                clean_profile = clean_html_text(raw_profile)

                # Description combinée riche
                desc_parts = []
                if summary:
                    desc_parts.append(summary)
                if clean_profile:
                    desc_parts.append(f"Profil recherché et compétences requises :\n{clean_profile}")

                full_desc = "\n\n".join(desc_parts) if desc_parts else (mission_text or title)

                item = {
                    "source": "wttj",
                    "external_id": f"wttj_{hit.get('reference') or hit.get('objectID')}",
                    "poste": title,
                    "company": company,
                    "location": loc,
                    "type_de_contrat": contract_type,
                    "remote_policy": "Télétravail" if is_remote else "Sur site / Hybride",
                    "description": full_desc,
                    "mission": mission_text,
                    "source_url": url,
                    "date_posted": str(hit.get("published_at_date") or date_iso),
                }
                results.append(item)

            logger.info(f"✅ [WTTJ] {len(results)} offres exploitables collectées.")
        except Exception as e:
            logger.error(f"❌ Erreur lors du scraping WTTJ : {e}")

        return results

    # =========================================================================
    # 2. HelloWork via Search + Parsing JSON-LD enrichi
    # =========================================================================
    def search_hellowork(
        self,
        query: str,
        location: str = "France",
        limit: int = 15,
        target_date_iso: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recherche des offres sur HelloWork et enrichit chaque annonce via sa balise
        JSON-LD ('JobPosting') pour obtenir 100% de descriptions complètes.
        """
        logger.info(f"🔎 [HelloWork] Recherche : query='{query}' | location='{location}' | limit={limit}")
        from jobscraper.scrapers.hellowork import HelloWorkScraper
        from jobscraper.models.job import SearchCriteria

        scraper = HelloWorkScraper()
        criteria = SearchCriteria(
            keywords=[q.strip() for q in query.replace("OR", " ").split() if q.strip()],
            location=location if location.lower() != "france" else "",
            max_results=limit
        )

        results: List[Dict[str, Any]] = []
        date_iso = target_date_iso or date.today().isoformat()

        try:
            raw_offers = list(scraper.search(criteria))
            for offer in raw_offers:
                if len(results) >= limit:
                    break

                job_url = str(offer.url or "").strip()
                desc = ""
                company = str(offer.company or "Entreprise confidentielle").strip()
                title = str(offer.title or "").strip()

                # Récupération détaillée via JSON-LD sur la page de l'offre
                if job_url:
                    try:
                        resp = self.http_session.get(job_url, timeout=8)
                        if resp.status_code == 200:
                            soup = BeautifulSoup(resp.text, "html.parser")
                            for script in soup.find_all("script", type="application/ld+json"):
                                try:
                                    ld_data = json.loads(script.string or "{}")
                                    if ld_data.get("@type") == "JobPosting":
                                        raw_ld_desc = ld_data.get("description", "")
                                        if raw_ld_desc:
                                            desc = clean_html_text(raw_ld_desc)
                                        ld_comp = ld_data.get("hiringOrganization", {})
                                        if isinstance(ld_comp, dict) and ld_comp.get("name"):
                                            company = ld_comp.get("name")
                                        if ld_data.get("title"):
                                            title = ld_data.get("title")
                                        break
                                except Exception:
                                    continue
                    except Exception as err:
                        logger.debug(f"Impossible de récupérer le JSON-LD de {job_url}: {err}")

                # Fallback si JSON-LD absent
                if not desc or len(desc) < 30:
                    desc = offer.description or f"Poste de {title} chez {company} à {offer.location}."

                contract = "CDI"
                if offer.contract_type:
                    contract = str(offer.contract_type.value if hasattr(offer.contract_type, "value") else offer.contract_type).upper()

                item = {
                    "source": "hellowork",
                    "external_id": str(offer.id),
                    "poste": title,
                    "company": company,
                    "location": offer.location or location,
                    "type_de_contrat": contract,
                    "remote_policy": "Télétravail" if offer.remote else "Sur site / Hybride",
                    "description": desc,
                    "mission": "",  # Le JobNormalizer s'occupe de la découper intelligemment
                    "source_url": job_url,
                    "date_posted": date_iso,
                }
                results.append(item)

            logger.info(f"✅ [HelloWork] {len(results)} offres exploitables collectées.")
        except Exception as e:
            logger.error(f"❌ Erreur lors du scraping HelloWork : {e}")

        return results
