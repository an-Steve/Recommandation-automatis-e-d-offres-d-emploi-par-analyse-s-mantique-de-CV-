import os
from pathlib import Path
from typing import Optional
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # 1. Configuration Générale
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # 2. Base de Données PostgreSQL
    DATABASE_URL: str = ""

    # 3. Base Vectorielle ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chromadb"
    CHROMA_COLLECTION_OFFRES: str = "offres"

    # 4. Modèles IA, Embeddings & Matériel (Dossier local models/ exclu de Git)
    MODELS_DIR: str = "./models"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_PROVIDER: str = "fastembed"  # "fastembed" (ONNX C++ local), "ollama" ou "sentence-transformers"
    EMBEDDING_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"
    LLM_EXTRACTION_MODEL: str = "llama3.2:1b"
    LLM_EXPLANATION_MODEL: str = "mistral:latest"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    FORCE_DEVICE: Optional[str] = None

    # 5. Scrapers (Clés d'APIs)
    FRANCE_TRAVAIL_CLIENT_ID: str = ""
    FRANCE_TRAVAIL_CLIENT_SECRET: str = ""
    ADZUNA_APP_ID: str = ""
    ADZUNA_APP_KEY: str = ""

    # 6. Modèles LLM
    MISTRAL_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # Chargement automatique du fichier .env
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def fastembed_cache_dir(self) -> Path:
        """Dossier local dans le projet pour les poids FastEmbed ONNX."""
        p = (BASE_DIR / self.MODELS_DIR / "fastembed").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def huggingface_cache_dir(self) -> Path:
        """Dossier local dans le projet pour les poids Hugging Face (GLiNER, etc.)."""
        p = (BASE_DIR / self.MODELS_DIR / "huggingface").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def device(self) -> str:
        """Détecte CUDA GPU si présent, sinon bascule sur le CPU."""
        if self.FORCE_DEVICE:
            return self.FORCE_DEVICE.lower()
        return "cuda" if torch.cuda.is_available() else "cpu"


# Instance unique accessible partout
settings = Settings()

# Synchronise l'environnement Hugging Face avec le dossier models local du projet
os.environ.setdefault("HF_HOME", str(settings.huggingface_cache_dir))
