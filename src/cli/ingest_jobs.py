"""
src/cli/ingest_jobs.py
Script CLI d'ingestion et d'indexation vectorielle avec Ollama.
"""

import sys
import argparse
from pathlib import Path

# Résolution des imports et de l'encodage Windows
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.scrapers.adzuna_scraper import AdzunaScraper
from src.scrapers.jobspy_scraper import JobSpyScraper
from src.pipeline.job_indexer import JobIndexer
from src.database.db_session import SessionLocal
from src.database.sql_models import Offre
from src.database.vector_store import get_offres_collection


def main():
    parser = argparse.ArgumentParser(description="Ingestion ELT et Indexation Vectorielle Ollama")
    parser.add_argument("--source", type=str, choices=["adzuna", "jobspy", "all"], default="all", help="Source de collecte (adzuna, jobspy, ou all)")
    parser.add_argument("--secteur", type=str, default="it_data", help="Code secteur (it_data, sante_medical, btp_construction...)")
    parser.add_argument("--query", type=str, default="Data Scientist", help="Mots-clés de recherche")
    parser.add_argument("--location", type=str, default="Paris", help="Lieu cible")
    parser.add_argument("--limit", type=int, default=10, help="Nombre d'offres à collecter par source")
    parser.add_argument("--batch-size", type=int, default=5, help="Taille des lots d'embedding Ollama (5 conseillé pour CPU)")
    parser.add_argument("--only-index", action="store_true", help="Ne pas scraper, indexer uniquement les offres en attente")

    args = parser.parse_args()

    print("=" * 70)
    print("🚀 PIPELINE D'INGESTION & INDEXATION ELT (POSTGRESQL + OLLAMA)")
    print("=" * 70)

    indexer = JobIndexer(batch_size=args.batch_size)

    # 1. Scraping & Sauvegarde SQL
    if not args.only_index:
        raw_jobs = []

        if args.source in ["adzuna", "all"]:
            print(f"\n[1/3] Collecte depuis ADZUNA (Secteur: {args.secteur})...")
            print(f"      Recherche : '{args.query}' à '{args.location}' (Limite: {args.limit})")
            scraper_adzuna = AdzunaScraper(country="fr")
            jobs_adzuna = scraper_adzuna.search_jobs(query=args.query, location=args.location, limit=args.limit)
            for j in jobs_adzuna:
                j["secteur"] = args.secteur
            print(f"      -> {len(jobs_adzuna)} offre(s) reçue(s) d'Adzuna.")
            raw_jobs.extend(jobs_adzuna)

        if args.source in ["jobspy", "all"]:
            print(f"\n[1/3] Collecte depuis JOBSPY (LinkedIn & Indeed - Secteur: {args.secteur})...")
            print(f"      Recherche : '{args.query}' à '{args.location}' (Limite: {args.limit})")
            scraper_jobspy = JobSpyScraper(sites=["indeed", "linkedin"], country_indeed="france")
            jobs_jobspy = scraper_jobspy.search_jobs(query=args.query, location=args.location, limit=args.limit)
            for j in jobs_jobspy:
                j["secteur"] = args.secteur
            print(f"      -> {len(jobs_jobspy)} offre(s) reçue(s) de JobSpy.")
            raw_jobs.extend(jobs_jobspy)

        print(f"\n📊 Total brut collecté : {len(raw_jobs)} offre(s).")
        if raw_jobs:
            print("\n[2/3] Sauvegarde brute dans PostgreSQL...")
            stats = indexer.save_raw_jobs_to_postgres(raw_jobs)
            print(f"      -> Nouvelles offres : {stats['saved']} | Doublons ignorés : {stats['duplicates']}")

    # 2. Worker d'embedding par lots Ollama
    print("\n[3/3] Traitement de la file d'attente d'embedding vectoriel Ollama bge-m3...")
    indexed_count = indexer.index_pending_jobs()

    # 3. Bilan final
    db = SessionLocal()
    try:
        total_sql = db.query(Offre).count()
        total_indexed_sql = db.query(Offre).filter(Offre.is_indexed == True).count()
        collection = get_offres_collection()
        total_chroma = collection.count()

        print("\n" + "=" * 70)
        print("🎉 BILAN D'ÉTAT DES BASES DE DONNÉES :")
        print(f"   • Offres vectorisées dans cette session : {indexed_count}")
        print(f"   • Total des offres dans PostgreSQL       : {total_sql} (dont {total_indexed_sql} indexées)")
        print(f"   • Total des vecteurs dans ChromaDB       : {total_chroma}")
        print("=" * 70)
    finally:
        db.close()


if __name__ == "__main__":
    main()
