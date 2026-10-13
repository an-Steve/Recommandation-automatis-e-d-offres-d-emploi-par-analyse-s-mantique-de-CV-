"""
src/database/init_db.py
Script d'Initialisation Automatisée du Socle de Données (PostgreSQL 16 + ChromaDB + Ollama).
Crée automatiquement la base de données, les tables relationnelles et la collection vectorielle.
"""

import sys
import os
import logging
from pathlib import Path

# Résolution automatique du chemin racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from urllib.parse import urlparse
import requests
from sqlalchemy import create_engine, text
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ensure_postgres_database_exists() -> bool:
    """
    Vérifie si la base de données 'projet_cv_db' existe dans PostgreSQL.
    Si elle n'existe pas, se connecte à la base système 'postgres' et la crée automatiquement.
    """
    raw_url = settings.DATABASE_URL
    if not raw_url:
        logger.error("❌ DATABASE_URL est vide dans le fichier .env !")
        return False

    parsed = urlparse(raw_url)
    target_dbname = parsed.path.lstrip("/") or "projet_cv_db"
    user = parsed.username or "postgres"
    password = parsed.password or ""
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 5432

    # URL vers la base par défaut 'postgres'
    default_db_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/postgres"

    try:
        engine_default = create_engine(default_db_url, isolation_level="AUTOCOMMIT")
        with engine_default.connect() as conn:
            result = conn.execute(
                text(f"SELECT 1 FROM pg_database WHERE datname = '{target_dbname}'")
            ).fetchone()

            if not result:
                logger.info(f"🔨 La base de données '{target_dbname}' n'existe pas encore. Création en cours...")
                conn.execute(text(f"CREATE DATABASE {target_dbname}"))
                logger.info(f"✅ Base de données '{target_dbname}' créée avec succès dans PostgreSQL !")
            else:
                logger.info(f"ℹ️ La base de données PostgreSQL '{target_dbname}' existe déjà.")
        return True
    except Exception as e:
        logger.warning(
            f"⚠️ Impossible de vérifier/créer la base automatiquement via 'postgres' ({e}).\n"
            f"   Assurez-vous que le serveur PostgreSQL tourne et que le mot de passe dans .env est correct."
        )
        return False


def init_relational_tables() -> bool:
    """Crée l'ensemble des tables déclaratives SQLAlchemy (offres, cv_users, evaluations)."""
    try:
        from src.database.db_session import engine
        from src.database.sql_models import Base

        logger.info("📦 Création des tables relationnelles dans PostgreSQL...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables relationnelles prêtes : 'offres', 'cv_users', 'evaluations'.")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création des tables PostgreSQL : {e}")
        return False


def init_chromadb_collection() -> bool:
    """Initialise le répertoire persistant ChromaDB et la collection 'offres' (distance cosinus)."""
    try:
        from src.database.vector_store import get_offres_collection

        logger.info(f"🧭 Initialisation de la base vectorielle ChromaDB ({settings.CHROMA_PERSIST_DIR})...")
        collection = get_offres_collection()
        count = collection.count()
        logger.info(f"✅ Collection ChromaDB '{collection.name}' opérationnelle ({count} offre(s) actuellement indexée(s)).")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation de ChromaDB : {e}")
        return False


def verify_ollama_runtime() -> bool:
    """Vérifie la disponibilité du serveur Ollama et la présence des modèles requis."""
    base_url = settings.OLLAMA_BASE_URL.rstrip("/")
    logger.info(f"🤖 Vérification du runtime Ollama sur {base_url}...")
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            models_data = resp.json().get("models", [])
            installed_models = [m.get("name", "") for m in models_data]
            logger.info(f"✅ Serveur Ollama actif. Modèles détectés : {installed_models}")

            req_embed = settings.EMBEDDING_MODEL_NAME
            req_llm = settings.LLM_EXTRACTION_MODEL

            has_embed = any(req_embed in name for name in installed_models)
            has_llm = any(req_llm in name for name in installed_models)

            if not has_embed:
                logger.warning(f"⚠️ Modèle d'embedding '{req_embed}' manquant. Exécutez : ollama pull {req_embed}")
            if not has_llm:
                logger.warning(f"⚠️ Modèle LLM '{req_llm}' manquant. Exécutez : ollama pull {req_llm}")

            return True
        else:
            logger.warning(f"⚠️ Ollama a répondu avec le statut HTTP {resp.status_code}")
            return False
    except Exception as e:
        logger.warning(
            f"⚠️ Impossible de contacter Ollama sur {base_url} ({e}).\n"
            f"   Pensez à lancer l'application Ollama ou exécuter 'ollama serve'."
        )
        return False


def run_full_initialization():
    """Point d'entrée principal pour initialiser tout le socle de données du projet."""
    print("=" * 70)
    print("  🚀 INITIALISATION DU SOCLE DE DONNÉES - PROJET MASTER 2 IA")
    print("=" * 70)

    # 1. Base PostgreSQL
    ensure_postgres_database_exists()
    pg_ok = init_relational_tables()

    # 2. Base Vectorielle ChromaDB
    chroma_ok = init_chromadb_collection()

    # 3. Vérification Ollama
    ollama_ok = verify_ollama_runtime()

    print("\n" + "=" * 70)
    if pg_ok and chroma_ok:
        print("  🎉 SUCCÈS : PostgreSQL et ChromaDB sont configurés et prêts !")
        if not ollama_ok:
            print("  ℹ️ Note : Pensez à démarrer Ollama avant de lancer l'enrichissement.")
    else:
        print("  ⚠️ Des erreurs ont été rencontrées. Vérifiez vos paramètres dans .env.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_full_initialization()
