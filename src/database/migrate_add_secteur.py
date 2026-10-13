"""
src/database/migrate_add_secteur.py
Migration PostgreSQL : ajout des colonnes 'secteur' et 'date_posted' dans la table 'offres',
création des index d'accélération et rétro-affectation des 60 offres existantes vers 'it_data'.
"""

import sys
from pathlib import Path

# Résolution des imports et encodage Windows
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import text
from src.database.db_session import engine


def migrate():
    print("=" * 70)
    print("🔄 MIGRATION POSTGRESQL : AJOUT DES COLONNES 'secteur' & 'date_posted'")
    print("=" * 70)

    with engine.begin() as conn:
        # 1. Ajout de la colonne secteur et date_posted
        conn.execute(text("ALTER TABLE offres ADD COLUMN IF NOT EXISTS secteur VARCHAR(50) DEFAULT 'it_data';"))
        conn.execute(text("ALTER TABLE offres ADD COLUMN IF NOT EXISTS date_posted DATE DEFAULT CURRENT_DATE;"))

        # 2. Création des index
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_offres_secteur ON offres(secteur);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_offres_date_posted ON offres(date_posted);"))
        print("✅ Colonnes 'secteur', 'date_posted' et index créés avec succès.")

        # 3. Rétro-affectation des offres existantes vers le secteur it_data (offres Data/IA actuelles)
        res_secteur = conn.execute(text("UPDATE offres SET secteur = 'it_data' WHERE secteur IS NULL OR secteur = '';"))
        conn.execute(text("UPDATE offres SET date_posted = CURRENT_DATE WHERE date_posted IS NULL;"))
        print(f"   • Offres rétro-affectées au secteur 'it_data' : {res_secteur.rowcount}")

        # 4. Résumé
        summary = conn.execute(text("SELECT secteur, count(*) FROM offres GROUP BY secteur;")).fetchall()
        print("\n📊 Répartition actuelle par secteur dans PostgreSQL :")
        for row in summary:
            print(f"   • Secteur '{row[0]}' : {row[1]} offres")

    print("=" * 70)
    print("🎉 MIGRATION SECTEUR TERMINÉE AVEC SUCCÈS !")
    print("=" * 70)


if __name__ == "__main__":
    migrate()
