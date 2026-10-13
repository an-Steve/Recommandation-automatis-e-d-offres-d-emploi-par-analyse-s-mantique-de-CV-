# Implémentation du Socle de Base de Données (`src/database/`)
> Projet Master 2 IA — Code Robuste pour PostgreSQL (SQLAlchemy 2.0) et ChromaDB

Ce document contient l'**implémentation de référence complète, robuste et sécurisée** pour les fichiers de configuration et de persistance du projet :
1. [`src/config.py`](#1-srcconfigpy--configuration-sécurisée--miroir-11-du-env) : Miroir exact de votre `.env`, zéro mot de passe en dur, détection dynamique CUDA/CPU.
2. [`src/database/db_session.py`](#2-srcdatabasedb_sessionpy--connexion--sessions-postgresql) : Moteur résilient SQLAlchemy 2.0 avec `pool_pre_ping`.
3. [`src/database/sql_models.py`](#3-srcdatabasesql_modelspy--modèles-orm-offres-cv_users-evaluations) : Tables PostgreSQL avec types `JSONB`, `ARRAY(Text)` et contraintes.
4. [`src/database/vector_store.py`](#4-srcdatabasevector_storepy--client-chromadb--recherche-cosinus) : Gestionnaire ChromaDB avec distance cosinus HNSW et pré-filtrage.
5. [`tests/test_database.py`](#5-script-de-validation-immédiate-teststest_databasepy) : Script de test d'intégration pour valider la double écriture en 5 secondes.

---

## 1. `src/config.py` : Configuration Sécurisée & Miroir 1:1 du `.env`

Ce fichier est le **reflet exact et sécurisé** de votre fichier `.env`.
* **Sécurité** : Aucun mot de passe ni clé secrète n'apparaît en dur dans ce code (`""`). Tout est injecté dynamiquement depuis votre `.env` en mémoire RAM.
* **Détection matérielle** : La propriété `@property device` bascule automatiquement sur `cuda` si une carte NVIDIA est disponible, sinon sur `cpu` sans aucun plantage.

```python
"""
Module de configuration centralisé de l'application.
Miroir exact 1:1 des variables définies dans .env.
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


# Chemin absolu racine du projet (C:/MASTER2_IA/Projet_CV)
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Paramètres globaux validés automatiquement par Pydantic.
    Les noms correspondent exactement aux 14 variables du fichier .env.
    """
    
    # 1. Configuration Générale
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # 2. Base de Données PostgreSQL (Aucun mot de passe en dur : injecté depuis .env)
    DATABASE_URL: str = ""

    # 3. Base Vectorielle ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chromadb"
    CHROMA_COLLECTION_OFFRES: str = "offres"

    # 4. Modèles IA & Matériel
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    FORCE_DEVICE: Optional[str] = None

    # 5. Scrapers (Clés d'APIs injectées depuis .env)
    FRANCE_TRAVAIL_CLIENT_ID: str = ""
    FRANCE_TRAVAIL_CLIENT_SECRET: str = ""
    ADZUNA_APP_ID: str = ""
    ADZUNA_APP_KEY: str = ""

    # 6. Modèles LLM (Injectés depuis .env)
    MISTRAL_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # Chargement automatique du fichier .env
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def device(self) -> str:
        """
        Détection automatique et dynamique du matériel :
        1. Si FORCE_DEVICE est défini dans .env -> respecte le choix forcé.
        2. Si un GPU NVIDIA CUDA est détecté -> 'cuda'.
        3. Si puce Apple Silicon -> 'mps'.
        4. Sinon -> bascule silencieusement sur 'cpu' sans plantage.
        """
        if self.FORCE_DEVICE:
            return self.FORCE_DEVICE.lower()
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
            
        return "cpu"


# Instance unique (Singleton) importable partout dans le projet
settings = Settings()

# Création automatique du répertoire ChromaDB s'il n'existe pas
Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
```

---

## 2. `src/database/db_session.py` : Connexion & Sessions PostgreSQL

Utilise **SQLAlchemy 2.0** avec gestion de pool résiliente (`pool_pre_ping=True` pour éviter les déconnexions inattendues).

```python
"""
Gestionnaire de connexion et de sessions pour PostgreSQL.
"""
import logging
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from src.config import settings

logger = logging.getLogger(__name__)

# Moteur SQLAlchemy :
# - pool_pre_ping=True : teste la connexion avant chaque requête pour éviter les erreurs "server closed connection"
# - pool_size=10 : garde 10 connexions réutilisables ouvertes
# - max_overflow=20 : pic autorisé jusqu'à 30 connexions simultanées
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False  # Mettre à True pour afficher toutes les requêtes SQL dans la console de débug
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
    Garantit la fermeture propre de la session même en cas d'erreur.
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
```

---

## 3. `src/database/sql_models.py` : Modèles ORM (Offres, CVUsers, Evaluations)

Implémente le schéma relationnel complet avec les types natifs avancés de PostgreSQL (`JSONB`, `ARRAY(Text)`).

```python
"""
Modèles de tables relationnelles SQLAlchemy 2.0 pour PostgreSQL.
Correspondance exacte avec le schéma DDL du projet.
"""
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Numeric,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    func
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Offre(Base):
    """
    Table des Offres d'Emploi.
    Source de vérité pour toutes les données scrapées et normalisées.
    """
    __tablename__ = "offres"

    id = Column(String(64), primary_key=True, index=True, comment="Hash unique job_hash")
    poste = Column(String(255), nullable=False, index=True)
    company = Column(String(255), nullable=False, index=True)
    type_de_contrat = Column(String(50), nullable=False, index=True)  # CDI, CDD, Stage...
    niveau_etude_min = Column(Integer, default=0, index=True)        # 0 (indifférent), 3 (Bac+3), 5 (Bac+5)
    min_experience_years = Column(Numeric(4, 1), default=0.0)
    
    # Compétences requises stockées sous forme de tableau PostgreSQL requêtable avec @>
    competences_necessaires = Column(ARRAY(Text), nullable=False, default=list)
    
    description = Column(Text, nullable=False)
    mission = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relations
    evaluations = relationship("Evaluation", back_populates="offre", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Offre(id='{self.id}', poste='{self.poste}', company='{self.company}', contrat='{self.type_de_contrat}')>"


class CVUser(Base):
    """
    Table des Candidats (Profils de CVs extraits par Docling et Pydantic).
    """
    __tablename__ = "cv_users"

    id = Column(String(64), primary_key=True, index=True, comment="Identifiant unique cand_xxx")
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True)
    
    desired_contract_types = Column(ARRAY(String(50)), nullable=False, default=list)
    parcours_academique = Column(Integer, nullable=False, default=5)  # 3 (Bac+3), 5 (Bac+5)
    total_experience_years = Column(Numeric(4, 1), default=0.0)
    
    competences = Column(ARRAY(Text), nullable=False, default=list)
    experiences_professionnelles = Column(JSONB, default=list)
    experiences_universitaires = Column(JSONB, default=list)
    
    raw_markdown = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relations
    evaluations = relationship("Evaluation", back_populates="cv_user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<CVUser(id='{self.id}', name='{self.full_name}', bac='{self.parcours_academique}')>"


class Evaluation(Base):
    """
    Table d'association Many-to-Many entre Candidats et Offres.
    Conserve les scores décomposés (Dense, BM25, Reranker) et l'explicabilité LLM.
    """
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cv_user_id = Column(String(64), ForeignKey("cv_users.id", ondelete="CASCADE"), nullable=False, index=True)
    offre_id = Column(String(64), ForeignKey("offres.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Décomposition des scores pour l'évaluation et l'auditabilité
    dense_score = Column(Float, nullable=True)
    bm25_score = Column(Float, nullable=True)
    reranker_score = Column(Float, nullable=True)
    final_score = Column(Integer, nullable=False, index=True)  # Score consolidé (0 à 100)
    
    # Explicabilité
    matched_skills = Column(ARRAY(Text), default=list)
    missing_skills = Column(ARRAY(Text), default=list)
    justification = Column(Text, nullable=True)
    conseils_candidature = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Contrainte d'unicité : 1 seule évaluation par couple (CV, Offre)
    __table_args__ = (
        UniqueConstraint("cv_user_id", "offre_id", name="uq_eval_cv_offre"),
    )

    # Relations inverses
    cv_user = relationship("CVUser", back_populates="evaluations")
    offre = relationship("Offre", back_populates="evaluations")

    def __repr__(self) -> str:
        return f"<Evaluation(cv='{self.cv_user_id}', offre='{self.offre_id}', score={self.final_score})>"
```

---

## 4. `src/database/vector_store.py` : Client ChromaDB & Recherche Cosinus

Ce module encapsule ChromaDB en mode local persistant avec distance cosinus HNSW et pré-filtrage strict.

```python
"""
Gestionnaire de la base vectorielle ChromaDB.
"""
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.api.models.Collection import Collection
from src.config import settings

logger = logging.getLogger(__name__)

# Initialisation du client persistant local (stocké dans le chemin CHROMA_PERSIST_DIR)
chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)


def get_offres_collection() -> Collection:
    """
    Récupère ou crée la collection vectorielle pour les offres d'emploi.
    Configure explicitement la distance cosinus HNSW pour un matching standard.
    """
    return chroma_client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_OFFRES,
        metadata={"hnsw:space": "cosine"}  # Similarité cosinus
    )


def upsert_job_vector(
    job_id: str,
    document: str,
    embedding: List[float],
    metadata: Dict[str, Any]
) -> None:
    """
    Insère ou met à jour le vecteur d'une offre dans ChromaDB.
    
    :param job_id: Identifiant exact correspondant à la clé primaire de PostgreSQL.
    :param document: Texte représentatif de l'offre (Poste + Compétences + Missions).
    :param embedding: Vecteur dense calculé par BGE-M3 (1024 floats).
    :param metadata: Critères de filtrage strict (type_de_contrat, niveau_etude_min, etc.).
    """
    collection = get_offres_collection()
    
    # Nettoyage des métadonnées (ChromaDB n'accepte que str, int, float, bool)
    clean_meta = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            clean_meta[k] = v
        elif isinstance(v, list):
            clean_meta[k] = ", ".join(map(str, v))
        elif v is not None:
            clean_meta[k] = str(v)

    collection.upsert(
        ids=[job_id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[clean_meta]
    )


def query_similar_jobs(
    query_embedding: List[float],
    n_results: int = 30,
    where_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Effectue une recherche par similarité cosinus avec pré-filtrage strict sur métadonnées.
    
    :param query_embedding: Vecteur du profil candidat.
    :param n_results: Nombre d'offres à remonter pour le reranking (Top 30 par défaut).
    :param where_filter: Filtre ChromaDB (ex: {"type_de_contrat": "CDI"}).
    :return: Liste de dictionnaires contenant job_id, score cosinus et métadonnées.
    """
    collection = get_offres_collection()
    
    kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": n_results
    }
    if where_filter:
        kwargs["where"] = where_filter

    results = collection.query(**kwargs)
    
    formatted_results = []
    if results and results.get("ids") and len(results["ids"][0]) > 0:
        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
        documents = results["documents"][0] if results.get("documents") else [""] * len(ids)
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        
        for job_id, dist, doc, meta in zip(ids, distances, documents, metadatas):
            # En distance cosinus ChromaDB : similarité = 1.0 - distance
            cosine_similarity = round(max(0.0, 1.0 - dist), 4)
            formatted_results.append({
                "job_id": job_id,
                "cosine_similarity": cosine_similarity,
                "document": doc,
                "metadata": meta
            })
            
    return formatted_results


def count_indexed_jobs() -> int:
    """Retourne le nombre total d'offres indexées dans ChromaDB."""
    collection = get_offres_collection()
    return collection.count()


def delete_job_vector(job_id: str) -> None:
    """Supprime un vecteur de ChromaDB par son ID."""
    collection = get_offres_collection()
    collection.delete(ids=[job_id])
```

---

## 5. Script de Validation Immédiate (`tests/test_database.py`)

Ce script de test s'exécute en terminal pour vérifier que PostgreSQL et ChromaDB fonctionnent ensemble :

```python
"""
Script de test d'intégration pour le socle de base de données.
Valide la lecture/écriture synchronisée dans PostgreSQL et ChromaDB.
"""
from pathlib import Path
import sys

# Ajoute automatiquement la racine du projet (C:/MASTER2_IA/Projet_CV) au sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import settings
from src.database.db_session import check_postgres_connection, get_db
from src.database.sql_models import Offre
from src.database.vector_store import (
    upsert_job_vector,
    query_similar_jobs,
    delete_job_vector,
    count_indexed_jobs
)


def run_integration_test():
    print("=" * 60)
    print("🚀 TEST D'INTÉGRATION : PostgreSQL + ChromaDB")
    print(f"🔧 Device détecté pour l'IA : {settings.device.upper()}")
    print("=" * 60)
    
    # 1. Vérification de la connexion PostgreSQL
    print("\n[1/4] Test de connexion PostgreSQL...")
    if not check_postgres_connection():
        print("❌ ÉCHEC : Impossible de joindre PostgreSQL. Vérifiez DATABASE_URL dans votre .env.")
        sys.exit(1)
    print("✅ Connexion PostgreSQL réussie !")

    # 2. Test d'insertion dans PostgreSQL
    print("\n[2/4] Test d'insertion d'une offre dans PostgreSQL...")
    db_gen = get_db()
    db = next(db_gen)
    
    test_job_id = "test_job_999"
    try:
        # Nettoyage préventif si le test avait déjà tourné
        existing = db.query(Offre).filter(Offre.id == test_job_id).first()
        if existing:
            db.delete(existing)
            db.commit()

        # Insertion de test
        offre_test = Offre(
            id=test_job_id,
            poste="Data Scientist NLP Test",
            company="Lab IA Test",
            type_de_contrat="CDI",
            niveau_etude_min=5,
            competences_necessaires=["Python", "PyTorch", "NLP", "ChromaDB"],
            description="Conception de systèmes RAG pour projet Master 2.",
            mission="Développement et évaluation des modèles.",
            is_active=True
        )
        db.add(offre_test)
        db.commit()
        print(f"✅ Offre insérée dans PostgreSQL : {offre_test.id} - {offre_test.poste}")
    except Exception as e:
        db.rollback()
        print(f"❌ Erreur lors de l'insertion SQL : {e}")
        sys.exit(1)

    # 3. Test d'insertion et de recherche dans ChromaDB
    print("\n[3/4] Test vectoriel ChromaDB...")
    try:
        # Vecteur factice de 1024 dimensions (comme BGE-M3)
        fake_vector = [0.05] * 1024
        fake_document = "Poste: Data Scientist NLP | Compétences: Python, PyTorch | Contrat: CDI"
        fake_metadata = {"type_de_contrat": "CDI", "niveau_etude_min": 5, "is_active": True}
        
        upsert_job_vector(test_job_id, fake_document, fake_vector, fake_metadata)
        print(f"✅ Vecteur inséré dans ChromaDB. Total indexé : {count_indexed_jobs()} offre(s)")
        
        # Test de recherche cosinus avec pré-filtrage
        results = query_similar_jobs(fake_vector, n_results=1, where_filter={"type_de_contrat": "CDI"})
        assert len(results) > 0, "Aucun résultat trouvé dans ChromaDB"
        print(f"✅ Recherche vectorielle réussie ! ID trouvé : {results[0]['job_id']} (Score cosinus: {results[0]['cosine_similarity']})")
    except Exception as e:
        print(f"❌ Erreur ChromaDB : {e}")
        sys.exit(1)

    # 4. Nettoyage de l'offre de test
    print("\n[4/4] Nettoyage des données de test...")
    db.delete(db.query(Offre).filter(Offre.id == test_job_id).first())
    db.commit()
    delete_job_vector(test_job_id)
    db.close()
    print("✅ Données de test nettoyées proprement.")

    print("\n" + "=" * 60)
    print("🎉 SUCCÈS TOTAL : Votre socle PostgreSQL + ChromaDB est 100% opérationnel !")
    print("=" * 60)


if __name__ == "__main__":
    run_integration_test()
```

---

## 6. Comment valider votre socle ?

1. Copiez les codes dans vos fichiers respectifs :
   * [`src/config.py`](file:///C:/MASTER2_IA/Projet_CV/src/config.py)
   * [`src/database/db_session.py`](file:///C:/MASTER2_IA/Projet_CV/src/database/db_session.py)
   * [`src/database/sql_models.py`](file:///C:/MASTER2_IA/Projet_CV/src/database/sql_models.py)
   * [`src/database/vector_store.py`](file:///C:/MASTER2_IA/Projet_CV/src/database/vector_store.py)
   * [`tests/test_database.py`](file:///C:/MASTER2_IA/Projet_CV/tests/test_database.py)
2. Lancez le test d'intégration dans votre terminal :
   ```powershell
   python tests/test_database.py
   ```
