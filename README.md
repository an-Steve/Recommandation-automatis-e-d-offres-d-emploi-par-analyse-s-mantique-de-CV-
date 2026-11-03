# <u>Recommandation Sémantique d'Offres d'Emploi par Analyse de CV</u>

> **Projet Master 2 Intelligence Artificielle - Université Paris 8**  

Ce projet implémente un système complet de matching sémantique et explicable entre des CVs de candidats (PDF) et des offres d'emploi collectées sur le marché.>
Ce document est le **guide complet d'installation et d'exécution** destiné aux membres de l'équipe pour reproduire intégralement la **Phase 0 (Setup & Socle)** et la **Phase 1 (Scraping, Nettoyage, Enrichissement LLM & Double Indexation)**.

##  Membres du projet :

| Nom et prénom |
|---|
| <u>ANTON NELCON Steve</u> &nbsp; \| &nbsp; <u>LIMACHE Adrien</u> &nbsp; \| &nbsp; <u>MOULAI HACENE Rania</u> &nbsp; \| &nbsp; <u>ZAID Ibtissam</u> &nbsp; \| &nbsp; <u>ZHENG Caroline</u> |

## 1. Architecture Globale (Phase 0 & 1) 
> réalisé du 10 au  13 octobre 2026

```mermaid
flowchart TD
    subgraph SOURCES["Sources d'Offres d'Emploi"]
        IND["Indeed (JobSpy)"]
        LNK["LinkedIn (JobSpy)"]
        ADZ["Adzuna (API REST)"]
    end

    subgraph BACKEND["Backend Flask (main.py :5000)"]
        API["Routes REST (/api/jobs/*)"]
        DEDUP["Déduplication MD5 (deduplication.py)"]
        
        subgraph CASCADE["Pipeline Normaliseur en Cascade 2 Niveaux (JobTextSanitizer)"]
            T1["Niveau 1 : Regex Déterministe\n(Ancres structurelles : Missions / Profil)"]
            T2["Niveau 2 : spaCy NLP Syntaxique\n(fr_core_news_sm : Verbes d'action & infinitifs)"]
            FALL["Fallback Honnête Anti-Hallucination\n('Non spécifié dans l'annonce')"]
            T1 -->|Échec ancre| T2 -->|0 action trouvée| FALL
        end
        
        subgraph EXTRACT["Extraction de Compétences & Critères (0 Hallucination)"]
            GLIN["GLiNER Zero-Shot (gliner_multi-v2.1)\nOntologie Dynamique 8 Secteurs\n(Span Classification Verbatim)"]
            HEUR["Heuristique Déterministe Regex\n(Niveau Bac+X & Années d'Expérience)"]
        end
        
        CANON["Gabarit Canonique Structuré\n(clean_job_text_for_embedding)"]
    end

    subgraph STORAGE["Socle de Stockage Découplé"]
        PG[("PostgreSQL 16 (projet_cv_db)\nTable relationnelle 'offres'")]
        CHROMA[("ChromaDB Persistant\nCollection dense 'offres' (Cosine)")]
        OLLAMA["Ollama Runtime (localhost:11434)\nall-minilm / nomic-embed-text"]
    end

    SOURCES --> API
    API --> DEDUP --> CASCADE --> PG
    PG --> EXTRACT
    GLIN --> CANON
    HEUR --> CANON
    CANON --> CHROMA
```

>Explication  de l'architecture :
>
>•Architecture du pipeline
>
>Ce projet collecte des offres d'emploi depuis plusieurs plateformes, les nettoie, les enrichit grâce à une IA locale, puis les indexe pour permettre une recherche sémantique. L'architecture se décompose en trois >blocs.
>
>•Sources d'offres d'emploi
>Les offres proviennent de trois sources : **Indeed** et **LinkedIn**, récupérées via la bibliothèque JobSpy, et **Adzuna**, interrogée via son API REST.
>
>• Backend Flask (`main.py`, port 5000)
>Les offres collectées sont reçues par les routes REST (`/api/jobs/*`), puis traitées en plusieurs étapes:
>
>1. **Déduplication** (`deduplication.py`) : un hash MD5 identifie les doublons afin de ne conserver qu'une seule version de chaque annonce.
>2. **Nettoyage** (`JobTextSanitizer`) : un nettoyeur algorithmique supprime le bruit du texte (balises, éléments inutiles).
>3. **Stockage relationnel** : les offres nettoyées sont enregistrées dans PostgreSQL.
>4. **Enrichissement par IA locale** (`JobEnricher`) : le modèle Llama 3.2 1B, exécuté en local via Ollama, extrait et structure les informations clés de chaque offre.
>5. **Gabarit canonique** (`job_indexer.py`) : chaque offre enrichie est mise sous un format uniforme et structuré, prêt à être indexé.
>
>•Socle de stockage découplé
>Les données sont réparties entre trois composants, chacun dédié à un usage précis :
>
>- **PostgreSQL 16** (base `projet_cv_db`, table `offres`) : stockage relationnel des offres.
>- **ChromaDB** (collection `offres`, distance cosinus) : base vectorielle persistante pour la recherche sémantique.
>- **Ollama** (`localhost:11434`) : runtime local qui fournit les modèles `all-minilm` (embeddings) et `llama3.2:1b` (enrichissement).
>
>•En résumé
>Les offres sont collectées, dédupliquées, nettoyées et stockées dans PostgreSQL, puis enrichies par une IA exécutée en local et indexées dans ChromaDB. Cette séparation entre données relationnelles et données >vectorielles permet à la fois des requêtes SQL classiques et une recherche sémantique (par exemple pour faire correspondre des offres à un CV), sans qu'aucune donnée ne soit envoyée à un service externe d'IA.
>**Composants** : Document AI (Docling), Ingestion Multi-Sources (JobSpy, Adzuna), Embeddings Denses (ChromaDB + Ollama), IA Générative Locale (Llama 3.2 1B), Recherche Hybride & Reranking.
---
## 2. Prérequis Système Obligatoires

Avant de commencer l'installation, assurez-vous d'avoir installé sur votre machine :

| Logo | Outil | Version requise | Rôle |
| :---: | :--- | :--- | :--- |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="35" alt="Python"> | **Python** | **3.11.x** | Compatibilité avec PyTorch, ChromaDB et Docling. |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/postgresql/postgresql-original.svg" width="35" alt="PostgreSQL"> | **PostgreSQL** | **16.x** + pgAdmin 4 | Base de données relationnelle (`projet_cv_db`). |
| <img src="https://cdn.jsdelivr.net/gh/homarr-labs/dashboard-icons/svg/ollama.svg" width="35" alt="Ollama"> | **Ollama** | Dernière version | Exécution locale des modèles LLM et d'embeddings. |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" width="35" alt="Git"> | **Git** | **2.x+** | Gestion de versions du projet. |


| Technologie | Technologie | Technologie |
| :---: | :---: | :---: |
| Python 3.11 | Flask | PostgreSQL 16 |
| SQLAlchemy | ChromaDB | Ollama |
| all-MiniLM | Llama 3.2 1B | Pydantic |
| JobSpy | API Adzuna | Docling |
---

## 3. Guide d'Installation Pas-à-Pas (Phase 0 & 1)

### Étape 1 : Cloner le dépôt et récupérer la branche Phase 1

```powershell
git clone -b phase1_scrappe_offre <URL_DU_REPO_GITHUB>
cd Projet_CV
```

### Étape 2 : Créer et activer l'environnement virtuel Python 3.11

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Étape 3 : Installer les dépendances du projet

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Étape 3.bis : Télécharger le modèle linguistique spaCy

Le pipeline normaliseur utilise le modèle français compact de spaCy pour l'extraction syntaxique des verbes d'action :
```powershell
python -m spacy download fr_core_news_sm
```

### Étape 4 : Démarrer et configurer Ollama

1. Assurez-vous qu'Ollama est démarré sur votre machine.
2. Téléchargez le modèle d'embedding utilisé par ChromaDB :
   ```powershell
   ollama pull nomic-embed-text
   ```

### Étape 5 : Configurer les variables d'environnement (`.env`)

Copiez le modèle `.env.example` vers `.env` et adaptez les identifiants :
```powershell
Copy-Item .env.example .env
```

Vérifiez notamment votre chaîne de connexion PostgreSQL :
```env
DATABASE_URL="postgresql://postgres:VOTRE_MOT_DE_PASSE@127.0.0.1:5432/projet_cv_db"
OLLAMA_BASE_URL="http://localhost:11434"
EMBEDDING_PROVIDER="ollama"
EMBEDDING_MODEL_NAME="nomic-embed-text"
```

---

## 4. Initialisation Automatisée du Socle de Données

Lancez le script d'initialisation du socle :
```powershell
python src/database/init_db.py
```
* **PostgreSQL** : Crée la base de données `projet_cv_db` si nécessaire et génère les tables (`offres`, `cv_users`, `evaluations`).
* **ChromaDB** : Initialise la collection vectorielle persistante `offres` avec distance cosinus.
* **Ollama** : Valide la connectivité locale.

---

## 5. Exécution des Pipelines Opérationnels (Phase 1)

### 1. Démarrer le Serveur Backend Flask
Dans un terminal dédié :
```powershell
python main.py
```
* Le serveur écoute sur `http://127.0.0.1:5000`.
* Test de santé : `Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/health"`

---

### 2. Collecte Quotidienne & Normalisation Multi-Sources
Déclenchez la collecte sur les 3 sources (Adzuna, LinkedIn, Indeed) pour un secteur ciblé :
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/jobs/scrape-daily" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"secteur": "it_data", "hours_old": 24, "indeed_results": 15, "linkedin_results": 15, "adzuna_results": 15}'
```

---

### 3. Scripts de Maintenance & Migration de Données

#### A. Migration Rétroactive des Descriptions et Missions Dupliquées :
```powershell
# Simulation préalable :
python scripts/migrate_split_missions.py --dry-run

# Exécution réelle :
python scripts/migrate_split_missions.py
```

#### B. Normalisation des Contrats & Éradication des Hallucinations :
```powershell
# Simulation préalable :
python scripts/migrate_contracts_and_skills.py --dry-run

# Exécution réelle :
python scripts/migrate_contracts_and_skills.py
```

---

### 4. Enrichissement Zero-Shot par Lots (GLiNER Multi-Secteurs)
Pour extraire les compétences verbatim sans aucune hallucination et détecter le niveau d'études/expérience :
```powershell
python src/pipeline/batch_enrich_jobs.py
```
* **Caractéristiques** : Inférence locale Zero-Shot sur 8 secteurs, classification par spans stricts, 0 token LLM consommé.

---

### 5. Indexation Vectorielle Canonique dans ChromaDB (CPU-Protected)
Pour vectoriser les offres enrichies en lots de 1 par 1 (évitant tout timeout HTTP) :
```powershell
python src/pipeline/job_indexer.py
```

---

### 6. Tester la Recherche Sémantique Vectorielle
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/jobs/search-vector" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query": "Data Scientist Python Machine Learning PyTorch Bac+5", "n_results": 3}'
```

---

## 6. Structure Complète du Code Source

```
Projet_CV/
├── main.py                             # Point d'entrée du serveur Backend Flask (:5000)
├── requirements.txt                     # Dépendances Python (spaCy, GLiNER, ChromaDB...)
├── .env.example                         # Modèle de variables d'environnement
├── .gitignore                           # Fichiers exclus du versionnement Git
├── README.md                            # Ce guide complet d'installation et de référence
│
├── docs/                                # Documentation technique et rapports d'architecture
│   ├── ARCHITECTURE.md                  # Schémas d'architecture globale et flux de données
│   ├── CANONICAL_VECTORIZATION_SPEC.md  # Spécification du gabarit canonique pour ChromaDB
│   ├── FLASK_BACKEND_SPEC.md            # Spécification de l'API REST Flask
│   ├── GLINER_IMPLEMENTATION_GUIDE.md   # Guide exhaustif de l'extracteur GLiNER Zero-Shot
│   ├── EXPLICATION_PIPELINE_PHASE1.md   # Guide pédagogique complet du pipeline Phase 1
│   ├── EXPLICATION_OPTIMISATION_VECTORISATION.md # Analyse de l'optimisation ChromaDB & CPU
│   ├── RAPPORT_ANALYSE_DESCRIPTION_VS_MISSION.md # Audit des doublons description / mission
│   ├── RAPPORT_STRATEGIE_EXTRACTION_MISSIONS_ET_ANTI_HALLUCINATION.md # Conception cascade 2 niveaux
│   ├── phase0.md                        # Documentation historique Phase 0 (Setup)
│   └── phase1.md                        # Documentation historique Phase 1 (Ingestion)
│
├── scripts/                             # Scripts de maintenance et migrations de base
│   ├── migrate_split_missions.py        # Migration de découpage 'description' vs 'mission'
│   ├── migrate_contracts_and_skills.py  # Normalisation contrats composites & nettoyage compétences
│   └── test_spacy_split_preview.py      # Prévisualisation visuelle du normaliseur spaCy
│
├── src/
│   ├── config.py                        # Configuration Pydantic Settings & variables .env
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py                    # Endpoints REST (/api/jobs/*, /api/health)
│   ├── database/
│   │   ├── db_session.py                # Session & Engine SQLAlchemy (PostgreSQL)
│   │   ├── sql_models.py                # Modèles relationnels (Offre, CVUser, Evaluation)
│   │   ├── vector_store.py              # Client persistant ChromaDB (Collection 'offres')
│   │   └── export_chroma.py             # Script d'export JSON pour audit qualité
│   ├── parsers/
│   │   ├── gliner_extractor.py          # Extracteur Zero-Shot GLiNER multi-secteurs (0 hallucination)
│   │   ├── job_enricher.py              # Façade d'enrichissement unifiée
│   │   └── docling_parser.py            # [Phase 2] Parseur de CVs PDF (Docling IBM)
│   ├── pipeline/
│   │   ├── deduplication.py             # Hachage MD5 et détection de doublons
│   │   ├── job_normalizer.py            # Normaliseur cascade 2 niveaux (Regex + spaCy NLP)
│   │   ├── job_indexer.py               # Constructeur du template canonique & indexeur ChromaDB (batch=1)
│   │   └── batch_enrich_jobs.py         # Worker batch d'enrichissement GLiNER
│   ├── scrapers/
│   │   ├── adzuna_scraper.py            # Connecteur API REST Adzuna
│   │   ├── jobspy_scraper.py            # Scraper Indeed & LinkedIn via JobSpy
│   │   └── daily_scraper.py             # Orchestrateur de collecte quotidienne multi-sources
│   └── utils/
│       └── ollama_client.py             # Client HTTP Ollama résilient
│
├── tests/
│   ├── test_gliner_multisector.py       # Benchmark multi-secteurs & anti-hallucination GLiNER
│   └── ...                              # Autres suites de tests unitaires
│
└── data/                                # Données locales (ignorées par Git)
    ├── chromadb/                        # Données persistantes ChromaDB
    └── chroma_dump.json                 # Export JSON d'audit des offres vectorisées
```
