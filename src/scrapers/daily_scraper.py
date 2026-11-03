"""
src/scrapers/daily_scraper.py
Orchestrateur de Scraping Quotidien par Date et par Secteur.
Coordonne Adzuna et JobSpy (Indeed/LinkedIn), applique un filtrage strict sur la date,
déduplique via MD5 dans PostgreSQL et indexe automatiquement dans ChromaDB.
"""

import sys
from pathlib import Path

# Résolution de la racine du projet pour exécution en CLI
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional

from src.scrapers.adzuna_scraper import AdzunaScraper
from src.scrapers.jobspy_scraper import JobSpyScraper
from src.scrapers.sector_mapping import get_sector_config, list_supported_sectors
from src.pipeline.job_indexer import JobIndexer

logger = logging.getLogger(__name__)


def parse_target_date(date_input: Optional[str] = None) -> date:
    """Parse une date au format YYYY-MM-DD, DD/MM/YYYY ou retourne la date du jour."""
    if not date_input or not str(date_input).strip():
        return date.today()

    clean_str = str(date_input).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(clean_str, fmt).date()
        except ValueError:
            continue

    logger.warning(f"Format de date non reconnu '{date_input}', utilisation de la date du jour.")
    return date.today()


class DailyJobScraper:
    """
    Coordonnateur de scraping quotidien multi-plateformes et multi-secteurs.
    """

    def __init__(self, batch_size_chroma: int = 8):
        self.indexer = JobIndexer(batch_size=batch_size_chroma)
        self.adzuna = AdzunaScraper(country="fr")

    def run_daily_pipeline(
        self,
        target_date_str: Optional[str] = None,
        sector_code: str = "it_data",
        sources: Optional[List[str]] = None,
        limit_per_source: int = 20,
        location: str = "France",
        auto_enrich: bool = True,
        enrich_limit: int = 0,  # 0 = enrichissement illimité (100% des nouvelles offres)
        auto_enrich_adzuna: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Exécute la collecte pour une date cible et un secteur donné.

        :param target_date_str: Date sous format "YYYY-MM-DD" ou "DD/MM/YYYY" (défaut: aujourd'hui)
        :param sector_code: Code secteur (ex: "it_data", "sante_medical", "btp_construction")
        :param sources: Liste des sources ["adzuna", "indeed", "linkedin"]
        :param limit_per_source: Plafond par plateforme (15-20 conseillé)
        :param location: Ville ou pays
        :return: Rapport statistique complet
        """
        target_date = parse_target_date(target_date_str)
        target_iso = target_date.isoformat()

        # Support du mode global multi-secteurs ("all", "tous", "*")
        if str(sector_code).strip().lower() in ["all", "tous", "*"]:
            all_sectors = list_supported_sectors()
            logger.info(f"🌐 Lancement du scraping global pour les {len(all_sectors)} secteurs...")
            sector_reports = []
            total_raw = 0
            total_saved = 0
            total_duplicates = 0
            total_indexed = 0
            total_enriched = 0

            for sec in all_sectors:
                rep = self.run_daily_pipeline(
                    target_date_str=target_date_str,
                    sector_code=sec,
                    sources=sources,
                    limit_per_source=limit_per_source,
                    location=location,
                    auto_enrich=auto_enrich,
                    enrich_limit=enrich_limit,
                    auto_enrich_adzuna=auto_enrich_adzuna,
                )
                sector_reports.append(rep)
                stats = rep.get("statistiques", {})
                total_raw += stats.get("total_collecte_brute", 0)
                total_saved += stats.get("nouvelles_offres_sql", 0)
                total_duplicates += stats.get("doublons_ignores", 0)
                total_indexed += stats.get("vecteurs_indexes_chroma", 0)
                total_enriched += stats.get("offres_enrichies_gliner", stats.get("offres_enrichies_llama", 0))

            return {
                "status": "success",
                "mode": "all_sectors",
                "target_date": target_iso,
                "secteurs_traites": all_sectors,
                "statistiques_globales": {
                    "total_collecte_brute": total_raw,
                    "nouvelles_offres_sql": total_saved,
                    "doublons_ignores": total_duplicates,
                    "vecteurs_indexes_chroma": total_indexed,
                    "offres_enrichies_gliner": total_enriched,
                },
                "details_par_secteur": sector_reports,
            }

        sector_cfg = get_sector_config(sector_code)
        target_sources = [s.lower() for s in (sources or ["adzuna", "indeed", "linkedin"])]

        # Calcul de la fraîcheur en jours par rapport à aujourd'hui
        today = date.today()
        delta_days = max(0, (today - target_date).days)
        max_days_old = delta_days + 1
        hours_old = max(24, (delta_days + 1) * 24)

        logger.info(
            f"📅 Lancement scraping journalier | Date: {target_iso} (delta: {delta_days}j) | "
            f"Secteur: '{sector_code}' ({sector_cfg['label']}) | Sources: {target_sources}"
        )

        from tqdm import tqdm

        collected_jobs: List[Dict[str, Any]] = []
        stats_per_source = {s: 0 for s in target_sources}

        # ----------------------------------------------------------------------
        # 1 & 2. Collecte Multi-Plateformes avec Barre de Progression
        # ----------------------------------------------------------------------
        with tqdm(total=len(target_sources), desc="🌐 Collecte plateformes", unit="source") as pbar_sources:
            # 1. Collecte ADZUNA
            if "adzuna" in target_sources:
                pbar_sources.set_postfix_str("Interrogation Adzuna...")
                try:
                    adz_jobs = self.adzuna.search_jobs(
                        query="",  # Toutes les offres sans restriction de titre
                        location=location,
                        limit=limit_per_source,
                        max_days_old=max_days_old,
                        category=sector_cfg["adzuna_category"]
                    )
                    filtered_adz = []
                    for j in adz_jobs:
                        j_created = str(j.get("created_at") or "")
                        if not j_created or j_created.startswith(target_iso) or delta_days == 0:
                            j["secteur"] = sector_code
                            j["date_posted"] = target_iso
                            j["source"] = "adzuna"
                            filtered_adz.append(j)

                    collected_jobs.extend(filtered_adz)
                    stats_per_source["adzuna"] = len(filtered_adz)
                    pbar_sources.set_postfix_str(f"Adzuna: {len(filtered_adz)} offres")
                except Exception as e:
                    logger.error(f"❌ Erreur lors de la collecte Adzuna : {e}")
                finally:
                    pbar_sources.update(1)

            # 2. Collecte JOBSPY (Indeed / LinkedIn)
            jobspy_sites = [s for s in target_sources if s in ["indeed", "linkedin"]]
            if jobspy_sites:
                pbar_sources.set_postfix_str(f"Interrogation {', '.join(jobspy_sites)}...")
                try:
                    jobspy_scraper = JobSpyScraper(sites=jobspy_sites, country_indeed="france")
                    query_term = sector_cfg["default_query"]
                    js_jobs = jobspy_scraper.search_jobs(
                        query=query_term,
                        location=location,
                        limit=limit_per_source,
                        hours_old=hours_old,
                    )
                    for j in js_jobs:
                        j["secteur"] = sector_code
                        j["date_posted"] = str(j.get("date_posted") or target_iso)
                        src = str(j.get("source", "indeed")).lower()
                        if src in stats_per_source:
                            stats_per_source[src] += 1
                    collected_jobs.extend(js_jobs)
                    pbar_sources.set_postfix_str(f"JobSpy: {len(js_jobs)} offres")
                except Exception as e:
                    logger.error(f"❌ Erreur lors de la collecte JobSpy : {e}")
                finally:
                    pbar_sources.update(len(jobspy_sites))

        # ----------------------------------------------------------------------
        # 3. Sauvegarde PostgreSQL & Déduplication Déterministe MD5
        # ----------------------------------------------------------------------
        save_stats = {"received": len(collected_jobs), "saved": 0, "duplicates": 0}
        if collected_jobs:
            save_stats = self.indexer.save_raw_jobs_to_postgres(collected_jobs)

        # ----------------------------------------------------------------------
        # 4. Enrichissement Zero-Shot GLiNER (Compétences, Diplôme, Expérience)
        # ----------------------------------------------------------------------
        should_enrich = auto_enrich if auto_enrich_adzuna is None else auto_enrich_adzuna
        enriched_count = 0
        if should_enrich and save_stats["saved"] > 0:
            from src.parsers.job_enricher import JobEnricher
            from src.database.db_session import SessionLocal
            from src.database.sql_models import Offre

            enricher = JobEnricher()
            db = SessionLocal()
            try:
                # Cible en priorité les offres non enrichies (vides ou 'Non spécifié')
                query_enrich = db.query(Offre).filter(
                    (Offre.competences_necessaires == []) |
                    (Offre.competences_necessaires == None) |
                    (Offre.competences_necessaires == ["Non spécifié"])
                ).order_by(Offre.created_at.desc())

                if enrich_limit > 0 and enrich_limit < save_stats["saved"]:
                    query_enrich = query_enrich.limit(enrich_limit)

                jobs_to_enrich = query_enrich.all()
                if jobs_to_enrich:
                    pbar_gliner = tqdm(jobs_to_enrich, desc="⚡ Enrichissement GLiNER (Zero-Shot)", unit="offre")
                    for o in pbar_gliner:
                        company_display = f"[{o.source.upper()}] {(o.company or 'Inconnue')[:10]}"
                        try:
                            enricher.enrich_offre_instance(o)
                            enriched_count += 1
                            skills_count = len(o.competences_necessaires or [])
                            pbar_gliner.set_postfix_str(f"OK {company_display} ({skills_count} comp, Bac+{o.niveau_etude_min})")
                        except Exception as err:
                            logger.warning(f"⚠️ Erreur enrichissement offre {o.id}: {err}")
                    db.commit()
                else:
                    logger.info("ℹ️ Aucune offre nécessitant d'enrichissement.")
            except Exception as err:
                db.rollback()
                logger.error(f"❌ Erreur lors de l'enrichissement GLiNER : {err}")
            finally:
                db.close()

        # ----------------------------------------------------------------------
        # 5. Indexation Vectorielle Canonique UNIQUE dans ChromaDB (FastEmbed ONNX C++)
        # ----------------------------------------------------------------------
        indexed_chroma_count = 0
        if save_stats["saved"] > 0:
            logger.info("⚡ Indexation vectorielle canonique UNIQUE dans ChromaDB (FastEmbed ONNX C++)...")
            indexed_chroma_count = self.indexer.index_pending_jobs()

        return {
            "status": "success",
            "target_date": target_iso,
            "secteur": sector_code,
            "secteur_label": sector_cfg["label"],
            "sources_cibles": target_sources,
            "statistiques": {
                "total_collecte_brute": len(collected_jobs),
                "par_plateforme": stats_per_source,
                "nouvelles_offres_sql": save_stats["saved"],
                "doublons_ignores": save_stats["duplicates"],
                "offres_enrichies_gliner": enriched_count,
                "vecteurs_indexes_chroma": indexed_chroma_count,
            }
        }


if __name__ == "__main__":
    import argparse
    import json

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Scraper Quotidien Multi-Plateformes & Multi-Secteurs (Master 2 IA)")
    parser.add_argument("--sector", "--secteur", default="all", help="Code secteur ou 'all' pour les 8 secteurs")
    parser.add_argument("--limit", type=int, default=10, help="Limite d'offres par plateforme (défaut: 10)")
    parser.add_argument("--sources", nargs="+", default=["adzuna", "indeed", "linkedin"], help="Sources à scraper")
    parser.add_argument("--date", default=None, help="Date cible YYYY-MM-DD (défaut: aujourd'hui)")

    args = parser.parse_args()

    print("=" * 70)
    print(f"🌍 COLLECTE QUOTIDIENNE (Secteur: '{args.sector}', Sources: {args.sources})")
    print("=" * 70)

    scraper = DailyJobScraper()
    rapport = scraper.run_daily_pipeline(
        target_date_str=args.date,
        sector_code=args.sector,
        sources=args.sources,
        limit_per_source=args.limit,
        auto_enrich=True
    )

    print("\n" + "=" * 70)
    print("📊 BILAN DE LA COLLECTE :")
    print("=" * 70)
    if "statistiques_globales" in rapport:
        stats = rapport["statistiques_globales"]
        print(f"• Total offres collectées : {stats['total_collecte_brute']}")
        print(f"• Nouvelles offres en base : {stats['nouvelles_offres_sql']}")
        print(f"• Doublons MD5 écartés     : {stats['doublons_ignores']}")
        print(f"• Vecteurs indexés ChromaDB: {stats['vecteurs_indexes_chroma']}")
    else:
        stats = rapport.get("statistiques", {})
        print(f"• Total offres collectées : {stats.get('total_collecte_brute', 0)}")
        print(f"• Nouvelles offres en base : {stats.get('nouvelles_offres_sql', 0)}")
        print(f"• Doublons MD5 écartés     : {stats.get('doublons_ignores', 0)}")
    print("=" * 70)
