"""
src/database/migrate_sanitize_jobs.py
Migration de nettoyage pour les offres existantes dans PostgreSQL.
Applique 'JobTextSanitizer' sur les offres volumineuses Indeed et LinkedIn :
- Élimine le bruit corporate (intros) et les avantages sociaux (tickets resto, salaires)
- Sépare proprement 'mission' et 'description' (profil recherché)
"""

import sys
from pathlib import Path

# Encodage Windows UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from tqdm import tqdm
from src.database.db_session import SessionLocal
from src.database.sql_models import Offre
from src.pipeline.job_normalizer import JobTextSanitizer


def main():
    print("=" * 75)
    print("🧹 MIGRATION & NETTOYAGE DES OFFRES EXISTANTES DANS POSTGRESQL")
    print("=" * 75)

    db = SessionLocal()
    try:
        # Sélectionner les offres volumineuses (> 500 caractères)
        offres = db.query(Offre).filter(Offre.source.in_(["indeed", "linkedin"])).all()
        print(f"📦 {len(offres)} offres Indeed et LinkedIn trouvées à assainir.")

        modified_count = 0
        total_chars_saved = 0

        for o in tqdm(offres, desc="Nettoyage du texte", unit="offre"):
            init_desc_len = len(o.description or "")
            if init_desc_len < 150:
                continue

            res = JobTextSanitizer.sanitize_and_split(o.description)
            clean_mission = res["mission"]
            clean_desc = res["description"]
            cleaned_full = res["cleaned_full"]

            chars_saved = init_desc_len - len(cleaned_full)
            if chars_saved > 0 or clean_mission != o.mission:
                o.mission = clean_mission
                o.description = clean_desc
                modified_count += 1
                total_chars_saved += max(0, chars_saved)

        db.commit()
        print("\n" + "=" * 75)
        print("✅ MIGRATION TERMINÉE AVEC SUCCÈS !")
        print(f"• Offres nettoyées et mises à jour : {modified_count}/{len(offres)}")
        print(f"• Caractères de bruit supprimés    : {total_chars_saved:,} caractères !")
        print("• Découpage 'mission' vs 'profil' appliqué en base de données.")
        print("=" * 75)

    except Exception as e:
        db.rollback()
        print(f"❌ Erreur lors de la migration : {e}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
