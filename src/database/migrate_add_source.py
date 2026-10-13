"""
src/database/migrate_add_source.py
Migration de schéma PostgreSQL : ajout de la colonne 'source' (adzuna, indeed, linkedin)
et peuplement automatique basé sur les URLs des 60 offres existantes.
"""

import sys
from pathlib import Path

# Résolution des imports et de l'encodage Windows
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import text
from src.database.db_session import engine


def migrate():
    print("=" * 65)
    print("🔄 MIGRATION POSTGRESQL : AJOUT DE LA COLONNE 'source'")
    print("=" * 65)

    with engine.begin() as conn:
        # 1. Ajout de la colonne si non existante
        conn.execute(text("ALTER TABLE offres ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'unknown';"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_offres_source ON offres(source);"))
        print("✅ Colonne 'source' et index 'idx_offres_source' créés avec succès.")

        # 2. Peuplement rétroactif des sources
        res_adzuna = conn.execute(text("UPDATE offres SET source = 'adzuna' WHERE source_url LIKE '%adzuna%';"))
        res_indeed = conn.execute(text("UPDATE offres SET source = 'indeed' WHERE source_url LIKE '%indeed%';"))
        res_linkedin = conn.execute(text("UPDATE offres SET source = 'linkedin' WHERE source_url LIKE '%linkedin%';"))

        print(f"   • Offres Adzuna   mises à jour : {res_adzuna.rowcount}")
        print(f"   • Offres Indeed   mises à jour : {res_indeed.rowcount}")
        print(f"   • Offres LinkedIn mises à jour : {res_linkedin.rowcount}")

        # 3. Répartition globale
        summary = conn.execute(text("SELECT source, count(*) FROM offres GROUP BY source ORDER BY count(*) DESC;")).fetchall()
        print("\n📊 Répartition actuelle des offres en base :")
        for row in summary:
            print(f"   • Source '{row[0]}' : {row[1]} offres")

    print("=" * 65)
    print("🎉 MIGRATION TERMINÉE AVEC SUCCÈS !")
    print("=" * 65)


if __name__ == "__main__":
    migrate()
