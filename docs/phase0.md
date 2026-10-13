# Guide de Réalisation — Phase 0 : Setup, Environnement Virtuel & Architecture
> Projet Master 2 IA — Système de Recommandation d'Offres d'Emploi par Analyse de CV

Ce document est le **guide pas-à-pas d'initialisation technique**. Il détaille la création de l'environnement virtuel Python sous Windows, l'installation des dépendances, la création complète de l'arborescence du projet, et la configuration des variables d'environnement.

---

## 1. Objectifs de la Phase 0

1. **Isolation de l'environnement** : Créer un `venv` propre sous Python 3.10+ (compatible PyTorch, Docling et ChromaDB).
2. **Installation des dépendances** : Installer et vérifier les librairies listées dans `requirements.txt`.
3. **Structure du code** : Générer l'arborescence modulaire (`src/`, `data/`, `tests/`, `notebooks/`) avec les fichiers `__init__.py`.
4. **Configuration sécurisée** : Mettre en place `.gitignore`, `.env.example`, et `.env`.
5. **Validation (Smoke Test)** : Exécuter un script de vérification validant les imports clés (PyTorch, ChromaDB, SQLAlchemy, Pydantic).

---

## 2. Étape 1 : Création et Activation du `venv` (Windows PowerShell)

Ouvrez un terminal PowerShell à la racine du projet (`C:\MASTER2_IA\Projet_CV`) :

### 2.1 Vérification de la version de Python
Sur votre machine Windows, plusieurs versions sont installées (3.9, 3.11, 3.13) :
* **Python 3.9** est trop ancien pour Docling (qui exige `>=3.10`).
* **Python 3.13** est trop récent pour les wheels précompilés de PyTorch et ChromaDB.
* **Python 3.11 est la version idéale (Sweet Spot IA)**.

Vérifiez la liste de vos interpréteurs avec le launcher Windows `py` :
```powershell
py --list
```

### 2.2 Création de l'environnement virtuel avec Python 3.11
Ciblez explicitement la version 3.11 via l'option `-3.11` :
```powershell
py -3.11 -m venv .venv
```

### 2.3 Autorisation d'exécution des scripts (si bloqué par Windows)
Si PowerShell bloque l'activation avec une erreur `Execution_Policies` :
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2.4 Activation du `venv`
```powershell
.\.venv\Scripts\Activate.ps1
```
*(Le préfixe `(.venv)` doit apparaître devant votre invite de commande PowerShell).*

### 2.5 Mise à jour de `pip`
```powershell
python -m pip install --upgrade pip setuptools wheel
```

---

## 3. Étape 2 : Installation des Dépendances

Installez les packages listés dans `requirements.txt` :

```powershell
pip install -r requirements.txt
```

> [!NOTE]
> Si vous disposez d'un GPU NVIDIA et souhaitez activer l'accélération CUDA pour Docling et BGE-M3 :
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 4. Étape 3 : Création de l'Arborescence du Projet

Exécutez ce bloc PowerShell pour créer tous les dossiers et sous-packages nécessaires d'un seul coup :

```powershell
# 1. Création des répertoires
New-Item -ItemType Directory -Force -Path `
    "data\raw_cvs", `
    "data\raw_jobs", `
    "data\processed", `
    "data\chromadb", `
    "notebooks", `
    "tests", `
    "src\scrapers", `
    "src\database", `
    "src\parsers", `
    "src\retrieval", `
    "src\agents", `
    "src\pipeline", `
    "src\api", `
    "src\ui", `
    "src\cli"

# 2. Création des fichiers __init__.py pour transformer les dossiers en packages Python
@(
    "src\__init__.py",
    "src\scrapers\__init__.py",
    "src\database\__init__.py",
    "src\parsers\__init__.py",
    "src\retrieval\__init__.py",
    "src\agents\__init__.py",
    "src\pipeline\__init__.py",
    "src\api\__init__.py",
    "src\ui\__init__.py",
    "src\cli\__init__.py",
    "tests\__init__.py"
) | ForEach-Object { 
    if (-not (Test-Path $_)) { New-Item -ItemType File -Force -Path $_ } 
}
```

### Vue de l'arborescence obtenue :
```
Projet_CV/
├── .venv/                         # Environnement virtuel (ignoré par Git)
├── docs/                          # Documentation complète du projet
├── data/                          # Données locales (ignoré par Git)
│   ├── raw_cvs/                   # PDFs de CVs de test
│   ├── raw_jobs/                  # Offres brutes téléchargées
│   ├── processed/                 # Fichiers Markdown extraits
│   └── chromadb/                  # Persistance de l'index vectoriel
├── notebooks/                     # Expérimentations & benchmarks M2
├── tests/                         # Tests unitaires et d'intégration
├── src/                           # Code source modulaire de l'application
│   ├── __init__.py
│   ├── config.py                  # Configuration et variables d'environnement
│   ├── scrapers/                  # Connecteurs d'offres (France Travail, JobSpy, Adzuna)
│   ├── database/                  # Modèles SQLAlchemy (Postgres) & client ChromaDB
│   ├── parsers/                   # Docling parser et extracteur Pydantic de CV
│   ├── retrieval/                 # Moteur hybride (BGE-M3, BM25, Reranker)
│   ├── agents/                    # LLM d'explicabilité et prompts
│   ├── pipeline/                  # Déduplication, orchestration et matching
│   ├── api/                       # API REST FastAPI
│   ├── ui/                        # Interface utilisateur Streamlit
│   └── cli/                       # Commandes d'ingestion et de test en ligne de commande
├── .env                           # Variables secrètes locales (ignoré par Git)
├── .env.example                   # Modèle de variables d'environnement
├── .gitignore                     # Exclusion des fichiers temporaires/données
├── requirements.txt               # Liste des dépendances
└── README.md                      # Accueil du projet
```

---

## 5. Étape 4 : Fichiers de Configuration

### 5.1 Fichier `.gitignore`
À placer à la racine pour ne pas versionner les données lourdes ni les secrets :
```gitignore
# Environnements virtuels
.venv/
venv/
ENV/

# Fichiers Python compilés
__pycache__/
*.py[cod]
*$py.class
*.so

# Variables d'environnement & Secrets
.env
.env.local

# Données locales et bases de données
data/
*.sqlite3
*.db

# Caches de bibliothèques IA
.cache/
models/
huggingface/
docling/

# IDE & OS
.vscode/
.idea/
.DS_Store
Thumbs.db
```

### 5.2 Fichier `.env.example`
Modèle à commiter pour l'équipe / le jury :
```env
# Configuration Application
APP_ENV=development
LOG_LEVEL=INFO

# Base de Données Relationnelle (PostgreSQL)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/projet_cv_db

# Base de Données Vectorielle (ChromaDB)
CHROMA_PERSIST_DIR=./data/chromadb
CHROMA_COLLECTION_OFFRES=offres

# Modèle d'Embedding & Reranker
EMBEDDING_MODEL_NAME=BAAI/bge-m3
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3

# Option 1 : API France Travail (francetravail.io)
FRANCE_TRAVAIL_CLIENT_ID=
FRANCE_TRAVAIL_CLIENT_SECRET=

# Option 3 : Adzuna (developer.adzuna.com)
ADZUNA_APP_ID=
ADZUNA_APP_KEY=

# LLM (Mistral / OpenAI / Ollama)
MISTRAL_API_KEY=
OPENAI_API_KEY=
```

### 5.3 Fichier `.env`
Copiez `.env.example` vers `.env` et ajustez les paramètres locaux :
```powershell
Copy-Item .env.example .env
```

---

## 6. Étape 5 : Module de Configuration Centralisé (`src/config.py`)

Créez `src/config.py` avec **Pydantic Settings** pour valider automatiquement la configuration au démarrage :

```python
"""Module de configuration centralisé de l'application."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    
    # Chemins locaux
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    chroma_persist_dir: Path = BASE_DIR / "data" / "chromadb"
    
    # Base PostgreSQL
    database_url: str = "postgresql://postgres:postgres@localhost:5432/projet_cv_db"
    
    # ChromaDB
    chroma_collection_offres: str = "offres"
    
    # Modèles IA
    embedding_model_name: str = "BAAI/bge-m3"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    
    # Clés Scrapers
    france_travail_client_id: str = ""
    france_travail_client_secret: str = ""
    adzuna_app_id: str = ""
    adzuna_app_key: str = ""
    
    # Clés LLM
    mistral_api_key: str = ""
    openai_api_key: str = ""

    # Détection dynamique du matériel (GPU CUDA vs CPU)
    force_device: str | None = None  # Permet de forcer 'cpu' ou 'cuda' dans .env si besoin

    @property
    def device(self) -> str:
        """Détecte automatiquement si un GPU NVIDIA CUDA est disponible, sinon bascule sur CPU."""
        if self.force_device:
            return self.force_device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Support pour puce Apple Silicon si testé sur macOS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
```

---

## 7. Étape 6 : Smoke Test de Validation

Créez un script `tests/smoke_test.py` pour vérifier que tout l'environnement est opérationnel :

```python
"""Smoke test pour valider l'environnement virtuel et les packages clés."""
import sys

def run_smoke_test():
    print(f"🐍 Version Python : {sys.version.split()[0]}")
    
    # 1. Pydantic
    import pydantic
    print(f"✅ Pydantic v{pydantic.__version__} OK")
    
    # 2. SQLAlchemy
    import sqlalchemy
    print(f"✅ SQLAlchemy v{sqlalchemy.__version__} OK")
    
    # 3. ChromaDB
    import chromadb
    client = chromadb.EphemeralClient()
    print(f"✅ ChromaDB v{chromadb.__version__} (In-Memory client) OK")
    
    # 4. PyTorch & GPU
    import torch
    device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    print(f"✅ PyTorch v{torch.__version__} (Device: {device}) OK")
    
    # 5. JobSpy
    try:
        import jobspy
        print("✅ python-jobspy OK")
    except ImportError:
        print("⚠️ python-jobspy non installé")
        
    print("\n🎉 PHASE 0 VALIDÉE AVEC SUCCÈS ! Vous êtes prêt pour la Phase 1.")

if __name__ == "__main__":
    run_smoke_test()
```

Exécution :
```powershell
python tests/smoke_test.py
```

---

## 8. Résumé des Commandes à Exécuter

| # | Action | Commande PowerShell |
|---|---|---|
| 1 | Créer le venv (Python 3.11) | `py -3.11 -m venv .venv` |
| 2 | Activer le venv | `.\.venv\Scripts\Activate.ps1` |
| 3 | Installer packages | `pip install -r requirements.txt` |
| 4 | Créer dossiers/packages | *(Script New-Item de la section 4)* |
| 5 | Créer `.env` | `Copy-Item .env.example .env` |
| 6 | Valider | `python tests/smoke_test.py` |
