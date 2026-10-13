"""
Gestionnaire de connexion et de sessions pour PostgreSQL.
"""
import logging
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from src.config import settings

logger = logging.getLogger(__name__)

# Normalisation automatique du protocole pour SQLAlchemy 2.0 (utilise psycopg2)
db_url = settings.DATABASE_URL
if db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(
    db_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False  # Mettre à True si vous souhaitez afficher toutes les requêtes SQL dans la console
)

# Fabrique de sessions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db() -> Generator[Session, None, None]:
    """
    Générateur de session sécurisé (Context Manager & FastAPI Dependency).
    Garantit la fermeture propre de la session même en cas d'exception.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_postgres_connection() -> bool:
    """Vérifie si le serveur PostgreSQL est joignable et actif."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Connexion à PostgreSQL établie avec succès.")
        return True
    except Exception as e:
        logger.error(f"Impossible de se connecter à PostgreSQL : {e}")
        return False
