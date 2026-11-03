"""
scripts/migrate_split_missions.py
Migration PostgreSQL rétroactive : séparation intelligente des descriptions et missions
pour les offres ayant 'description == mission' (notamment Adzuna et annonces non segmentées).

Utilise le pipeline déterministe en cascade à 2 niveaux (sans LLM génératif) :
- Niveau 1 : Regex Déterministe (Découpage par ancres structurelles)
- Niveau 2 : spaCy NLP Syntaxique (Extraction de verbes d'action et propositions opérationnelles)
- Fallback honnête : Mention explicite 'Non spécifié dans l'annonce' (0 duplication)

Usage :
    # Mode réel (applique les modifications en base) :
    python scripts/migrate_split_missions.py

    # Mode simulation (affiche les changements sans modifier PostgreSQL) :
    python scripts/migrate_split_missions.py --dry-run
"""

import sys
import argparse
from pathlib import Path

# Encodage Windows UTF-8 pour la console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ajout de la racine du projet au PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tqdm import tqdm
from sqlalchemy import func
from src.database.db_session import SessionLocal
from src.database.sql_models import Offre
from src.pipeline.job_normalizer import JobTextSanitizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migration de découpage rétroactif 'description' vs 'mission' dans PostgreSQL."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simule l'exécution sans valider les modifications dans PostgreSQL.",
    )
    return parser.parse_args()


def run_migration(dry_run: bool = False):
    print("=" * 75)
    print("🛠️  MIGRATION POSTGRESQL : SÉPARATION MISSIONS VS DESCRIPTIONS")
    if dry_run:
        print("⚠️  MODE SIMULATION (--dry-run) : Aucune modification ne sera enregistrée.")
    print("=" * 75)

    db = SessionLocal()
    try:
        # Récupération de toutes les offres où description == mission
        query = db.query(Offre).filter(
            (Offre.description == Offre.mission) | (Offre.mission == "") | (Offre.mission.is_(None))
        )
        offres = query.all()
        total_targets = len(offres)

        print(f"📦 Offres ciblées à corriger (doublons ou missions vides) : {total_targets}")

        if total_targets == 0:
            print("✅ Aucun doublon détecté. La base de données est déjà parfaitement propre !")
            return

        stats = {
            "Niveau 1 (Regex Structurelle)": 0,
            "Niveau 2 (spaCy NLP Syntaxique)": 0,
            "Fallback Honnête (0 Mission Détectée)": 0,
            "court_fallback": 0,
            "total_updated": 0,
        }

        preview_examples = []

        for o in tqdm(offres, desc="Migration en cours", unit="offre"):
            # On prend la description existante comme source complète
            raw_content = o.description or o.mission or ""
            if not raw_content.strip():
                continue

            res = JobTextSanitizer.sanitize_and_split(raw_content)
            tier = res.get("tier_used", "Inconnu")
            new_miss = res["mission"]
            new_desc = res["description"]

            # Comptabilisation des métriques
            matched_tier = False
            for k in stats.keys():
                if k in tier:
                    stats[k] += 1
                    matched_tier = True
                    break
            if not matched_tier:
                stats["Fallback Honnête (0 Mission Détectée)"] += 1

            # Conservation de quelques exemples pour la prévisualisation finale
            if len(preview_examples) < 3 and new_miss != "Non spécifié dans l'annonce":
                preview_examples.append({
                    "id": o.id,
                    "poste": o.poste,
                    "source": o.source,
                    "tier": tier,
                    "desc_extrait": new_desc[:120] + "...",
                    "miss_extrait": new_miss[:150] + "..."
                })

            # Mise à jour des colonnes dans le modèle SQLAlchemy
            if not dry_run:
                o.mission = new_miss
                o.description = new_desc
                # Note : l'offre reste valide pour la recherche hybride
                # et conserve son index vectoriel ChromaDB
            stats["total_updated"] += 1

        if not dry_run:
            print("\n💾 Validation des transactions dans PostgreSQL (commit)...")
            db.commit()
            print("✅ Commit réussi !")
        else:
            db.rollback()
            print("\n↩️  Rollback effectué (mode simulation).")

        print("\n" + "=" * 75)
        print("📊 BILAN DE LA MIGRATION :")
        print("=" * 75)
        print(f"• Offres traitées avec succès        : {stats['total_updated']} / {total_targets}")
        print(f"  ├─ Niveau 1 (Regex Structurelle)  : {stats['Niveau 1 (Regex Structurelle)']} offres")
        print(f"  ├─ Niveau 2 (spaCy NLP)           : {stats['Niveau 2 (spaCy NLP Syntaxique)']} offres")
        print(f"  └─ Fallback Honnête (0 Mission)   : {stats['Fallback Honnête (0 Mission Détectée)'] + stats['court_fallback']} offres")
        print("=" * 75)

        if preview_examples:
            print("\n🔍 EXEMPLES DE SÉPARATION APPLIQUÉE :")
            for i, ex in enumerate(preview_examples, 1):
                print(f"\n[{i}] Poste : {ex['poste']} ({ex['source']}) — via {ex['tier']}")
                print(f"    • Description : {ex['desc_extrait']}")
                print(f"    • Mission     : {ex['miss_extrait']}")
            print("=" * 75)

    except Exception as err:
        db.rollback()
        print(f"\n❌ Erreur inattendue pendant la migration : {err}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    args = parse_args()
    run_migration(dry_run=args.dry_run)
