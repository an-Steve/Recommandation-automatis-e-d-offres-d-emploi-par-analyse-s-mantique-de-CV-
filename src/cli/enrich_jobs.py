"""
src/cli/enrich_jobs.py
Script CLI pour enrichir les offres PostgreSQL via le LLM local Ollama (Llama 3.2).
Extrait compétences, niveau d'études et années d'expérience sans dictionnaire en dur.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Résolution des imports et de l'encodage Windows
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import func
from src.config import settings
from src.database.db_session import SessionLocal
from src.database.sql_models import Offre
from src.parsers.job_enricher import JobEnricher
from src.utils.ollama_client import OllamaClient


def main():
    parser = argparse.ArgumentParser(
        description="Enrichissement IA des offres d'emploi (PostgreSQL -> Ollama LLM -> PostgreSQL)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Nombre d'offres à traiter (par défaut: 5 pour tester)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1000,
        help="Longueur minimale de la description (défaut: 1000 pour cibler les offres riches de JobSpy)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["all", "adzuna", "indeed", "linkedin"],
        default="all",
        help="Filtrer par plateforme source (adzuna, indeed, linkedin, all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Traiter toutes les offres non enrichies",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.LLM_EXTRACTION_MODEL,
        help=f"Modèle Ollama à utiliser (par défaut: {settings.LLM_EXTRACTION_MODEL})",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Ré-enrichir même les offres ayant déjà des compétences renseignées",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🧠 ENRICHISSEMENT IA DES OFFRES PAR LLM LOCAL (OLLAMA + PYDANTIC)")
    print("=" * 70)
    print(f"• Modèle d'extraction : {args.model}")
    print(f"• Source ciblée       : {args.source.upper()}")
    print(f"• URL Ollama          : {settings.OLLAMA_BASE_URL}")

    # Vérification de la disponibilité d'Ollama
    client = OllamaClient(default_model=args.model)
    if not client.is_available():
        print(f"\n❌ ERREUR : Le serveur Ollama n'est pas joignable sur {settings.OLLAMA_BASE_URL}.")
        print("   Vérifiez qu'Ollama est bien lancé ('ollama serve' ou icône Windows).")
        sys.exit(1)

    enricher = JobEnricher(model_name=args.model)
    db = SessionLocal()

    try:
        query = db.query(Offre)
        if args.source != "all":
            query = query.filter(Offre.source == args.source)

        if args.min_length > 0:
            query = query.filter(func.length(Offre.description) >= args.min_length)

        if not args.force_all:
            # Sélectionne les offres sans compétences extraites
            query = query.filter(
                (Offre.competences_necessaires == []) | (Offre.competences_necessaires == None)
            )

        total_pending = query.count()
        if total_pending == 0:
            print("\n✅ Toutes les offres en base possèdent déjà des compétences extraites !")
            return

        limit = None if args.all else args.limit
        if limit:
            query = query.limit(limit)

        offres_to_process = query.all()
        print(f"\n⚡ {len(offres_to_process)} offre(s) sélectionnée(s) sur un total de {total_pending} en attente.")
        print("-" * 70)

        success_count = 0
        for idx, offre in enumerate(offres_to_process, 1):
            start_time = datetime.now()
            print(f"\n[{idx}/{len(offres_to_process)}] 🔍 Analyse de l'offre : '{offre.poste}' ({offre.company})")

            try:
                enricher.enrich_offre_instance(offre)
                db.commit()

                duration = (datetime.now() - start_time).total_seconds()
                success_count += 1

                print(f"      ⏱️ Durée : {duration:.1f}s")
                print(f"      🎓 Diplôme min (Bac+) : {offre.niveau_etude_min}")
                print(f"      💼 Expérience min      : {float(offre.min_experience_years)} an(s)")
                skills_preview = ", ".join(offre.competences_necessaires[:8])
                if len(offre.competences_necessaires) > 8:
                    skills_preview += f" ... (+{len(offre.competences_necessaires) - 8} autres)"
                print(f"      🛠️ Compétences ({len(offre.competences_necessaires)}) : [{skills_preview}]")

            except Exception as e:
                db.rollback()
                print(f"      ❌ Erreur lors de l'extraction de l'offre ID {offre.id} : {e}")

        print("\n" + "=" * 70)
        print("🎉 BILAN DE L'ENRICHISSEMENT IA :")
        print(f"   • Offres enrichies avec succès : {success_count}/{len(offres_to_process)}")
        remaining = db.query(Offre).filter(
            (Offre.competences_necessaires == []) | (Offre.competences_necessaires == None)
        ).count()
        print(f"   • Offres restantes à enrichir  : {remaining}")
        print("=" * 70)

    finally:
        db.close()


if __name__ == "__main__":
    main()
