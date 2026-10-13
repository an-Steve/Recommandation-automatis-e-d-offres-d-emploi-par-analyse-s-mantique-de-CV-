"""
tests/test_adzuna.py
Script de test pour valider le connecteur Adzuna France.
"""

import sys
from pathlib import Path

# Résolution des imports depuis la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Résolution d'encodage Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.scrapers.adzuna_scraper import AdzunaScraper
from src.config import settings


def main():
    print("=" * 65)
    print("🚀 TEST DU CONNECTEUR API ADZUNA (FRANCE)")
    print("=" * 65)

    # 1. Vérification de la configuration
    if not settings.ADZUNA_APP_ID or not settings.ADZUNA_APP_KEY:
        print("\n⚠️ AVERTISSEMENT : Clés Adzuna non détectées dans votre fichier .env !")
        print("Pour lancer ce test :")
        print("1. Inscrivez-vous sur : https://developer.adzuna.com")
        print("2. Renseignez ADZUNA_APP_ID et ADZUNA_APP_KEY dans votre fichier .env")
        return

    print(f"\n[1/3] Identifiants détectés : APP_ID={settings.ADZUNA_APP_ID[:4]}****")

    # 2. Instanciation et recherche
    scraper = AdzunaScraper(country="fr")
    query = "Data Scientist"
    location = "Paris"
    limit = 3

    print(f"\n[2/3] Recherche d'offres en France...")
    print(f"      Poste : '{query}'")
    print(f"      Ville : '{location}'")

    jobs = scraper.search_jobs(query=query, location=location, limit=limit)

    # 3. Affichage des résultats
    print(f"\n[3/3] Résultats récupérés : {len(jobs)} offre(s)")

    if not jobs:
        print("❌ Aucune offre retournée. Vérifiez vos clés ou votre connexion.")
        return

    for i, job in enumerate(jobs[:3], 1):
        print("\n" + "-" * 50)
        print(f"📄 Offre #{i} | ID : {job['external_id']}")
        print(f"📌 Poste      : {job['poste']}")
        print(f"🏢 Entreprise : {job['company']}")
        print(f"📍 Lieu       : {job['location']}")
        print(f"📋 Contrat    : {job['type_de_contrat']}")
        if job.get('salary_min'):
            print(f"💰 Salaire    : {job['salary_min']} € - {job.get('salary_max', '')} €")
        print(f"🔗 URL        : {job['source_url']}")
        desc_preview = job['description'][:160].replace("\n", " ")
        print(f"📝 Extrait    : {desc_preview}...")

    print("\n" + "=" * 65)
    print("🎉 SUCCÈS : Le connecteur Adzuna fonctionne à 100 % !")
    print("=" * 65)


if __name__ == "__main__":
    main()
