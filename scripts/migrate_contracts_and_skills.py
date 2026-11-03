"""
scripts/migrate_contracts_and_skills.py
Script de maintenance et migration PostgreSQL :
1. Normalisation des contrats anglo-saxons composites ('Fulltime, contract' -> 'CDD / Freelance', etc.)
2. Nettoyage déterministe verbatim des compétences déjà enrichies (élimine les hallucinations de type 'SQL' / 'Tableau')

Usage :
    # Simulation (aucun commit en base) :
    python scripts/migrate_contracts_and_skills.py --dry-run

    # Exécution réelle :
    python scripts/migrate_contracts_and_skills.py
"""

import sys
import argparse
from pathlib import Path

# Encodage Windows UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ajout de la racine du projet au PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tqdm import tqdm
from src.database.db_session import SessionLocal
from src.database.sql_models import Offre
from src.pipeline.job_normalizer import JobTextSanitizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migration de normalisation des contrats et assainissement des compétences."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simule l'exécution sans enregistrer dans PostgreSQL.",
    )
    return parser.parse_args()


def run_migration(dry_run: bool = False):
    print("=" * 75)
    print("🛠️  MAINTENANCE POSTGRESQL : NORMALISATION CONTRATS & NETTOYAGE COMPÉTENCES")
    if dry_run:
        print("⚠️  MODE SIMULATION (--dry-run) : Aucune modification ne sera enregistrée.")
    print("=" * 75)

    db = SessionLocal()
    try:
        offres = db.query(Offre).all()
        print(f"📦 Total des offres analysées en base : {len(offres)}\n")

        contracts_fixed = 0
        skills_cleaned_count = 0
        pruned_hallucinations_total = 0

        preview_contract_fixes = []
        preview_skill_fixes = []

        for o in tqdm(offres, desc="Audit & Nettoyage", unit="offre"):
            # --- 1. Normalisation du type de contrat ---
            old_contract = o.type_de_contrat or "Non spécifié"
            new_contract = JobTextSanitizer.normalize_contract_label(old_contract)

            if new_contract != old_contract:
                contracts_fixed += 1
                if len(preview_contract_fixes) < 5:
                    preview_contract_fixes.append((o.id, o.poste[:35], old_contract, new_contract))
                if not dry_run:
                    o.type_de_contrat = new_contract

            # --- 2. Vérification verbatim des compétences existantes ---
            current_skills = o.competences_necessaires or []
            if current_skills and current_skills != ["Non spécifié"]:
                full_text = f"{o.poste} {o.description} {o.mission}".lower()
                verified = [s for s in current_skills if s.strip().lower() in full_text]

                if len(verified) != len(current_skills):
                    pruned = [s for s in current_skills if s.strip().lower() not in full_text]
                    skills_cleaned_count += 1
                    pruned_hallucinations_total += len(pruned)

                    if len(preview_skill_fixes) < 5:
                        preview_skill_fixes.append((o.id, o.poste[:35], current_skills, verified or ["Non spécifié"], pruned))

                    if not dry_run:
                        o.competences_necessaires = verified if verified else ["Non spécifié"]

        if not dry_run:
            print("\n💾 Validation des modifications dans PostgreSQL (commit)...")
            db.commit()
            print("✅ Commit réussi !")
        else:
            db.rollback()
            print("\n↩️  Rollback effectué (mode simulation).")

        print("\n" + "=" * 75)
        print("📊 BILAN DE L'OPÉRATION :")
        print("=" * 75)
        print(f"• Contrats composites normalisés           : {contracts_fixed} offres")
        print(f"• Offres assainies (compétences filtrées) : {skills_cleaned_count} offres")
        print(f"• Compétences hallucinées supprimées       : {pruned_hallucinations_total} occurrences")
        print("=" * 75)

        if preview_contract_fixes:
            print("\n🔍 EXEMPLES DE NORMALISATION DE CONTRATS :")
            for jid, title, old_c, new_c in preview_contract_fixes:
                print(f"  • [{jid}] {title} : '{old_c}' ➔ '{new_c}'")

        if preview_skill_fixes:
            print("\n🔍 EXEMPLES D'ÉLIMINATION D'HALLUCINATIONS :")
            for jid, title, orig, after, pruned in preview_skill_fixes:
                print(f"  • [{jid}] {title}")
                print(f"    - Avant       : {orig}")
                print(f"    - Après       : {after}")
                print(f"    - Supprimé(s) : {pruned}")
        print("=" * 75)

    except Exception as err:
        db.rollback()
        print(f"\n❌ Erreur pendant la maintenance : {err}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    args = parse_args()
    run_migration(dry_run=args.dry_run)
