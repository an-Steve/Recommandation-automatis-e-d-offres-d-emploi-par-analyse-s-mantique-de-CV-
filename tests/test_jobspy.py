"""
tests/test_jobspy.py
Script de vérification du scraper JobSpy en direct.
"""

import sys
from pathlib import Path

# Résolution des imports depuis la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Résolution de l'encodage pour le terminal Windows (PowerShell/CMD)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.scrapers.jobspy_scraper import JobSpyScraper


def main():
    print("=" * 65)
    print("🚀 TEST DE SCRAPING MULTI-SOURCES : JobSpy (Indeed & LinkedIn)")
    print("=" * 65)

    # 1. Initialisation du scraper
    scraper = JobSpyScraper(sites=["indeed", "linkedin"], country_indeed="france")

    # 2. Exécution d'une recherche test
    query = "Data Scientist"
    location = "Paris"
    limit = 3

    print(f"\n[1/2] Lancement de la recherche...")
    print(f"      Mot-clé   : '{query}'")
    print(f"      Lieu      : '{location}'")
    print(f"      Nb offres : {limit} par source")

    jobs = scraper.search_jobs(query=query, location=location, limit=limit)

    print(f"\n[2/2] Analyse des résultats récupérés : {len(jobs)} offres trouvées.")

    if not jobs:
        print("❌ Aucune offre récupérée. Vérifiez votre connexion Internet.")
        return

    # 3. Affichage détaillé des 3 premières offres
    for i, job in enumerate(jobs[:3], 1):
        print("\n" + "-" * 50)
        print(f"📄 Offre #{i} | Source : {job['source'].upper()}")
        print(f"📌 Poste      : {job['poste']}")
        print(f"🏢 Entreprise : {job['company']}")
        print(f"📍 Lieu       : {job['location']}")
        print(f"📋 Contrat    : {job['type_de_contrat']} ({job['remote_policy']})")
        print(f"🔗 URL        : {job['source_url']}")
        desc_preview = job['description'][:180].replace("\n", " ")
        print(f"📝 Extrait    : {desc_preview}...")

    print("\n" + "=" * 65)
    print("🎉 SUCCÈS : Le connecteur JobSpy fonctionne parfaitement !")
    print("=" * 65)


if __name__ == "__main__":
    main()
