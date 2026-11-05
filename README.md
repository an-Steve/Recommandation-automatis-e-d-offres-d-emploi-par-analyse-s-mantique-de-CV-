# <u>Recommandation Sémantique d'Offres d'Emploi par Analyse de CV</u>

> **Projet Master 2 Intelligence Artificielle - Université Paris 8**  

Ce projet implémente un système complet de matching sémantique et explicable entre des CVs de candidats (PDF) et des offres d'emploi collectées sur le marché.
Ce document est le **guide complet d'installation et d'exécution** destiné aux membres de l'équipe pour reproduire intégralement la **Phase 0 (Setup & Socle)** et la **Phase 1 (Scraping, Nettoyage, Enrichissement Zero-Shot & Double Indexation Haute Performance CPU)**.

##  Membres du projet :

| Nom et prénom |
|---|
| <u>ANTON NELCON Steve</u> &nbsp; \| &nbsp; <u>LIMACHE Adrien</u> &nbsp; \| &nbsp; <u>MOULAI HACENE Rania</u> &nbsp; \| &nbsp; <u>ZAID Ibtissam</u> &nbsp; \| &nbsp; <u>ZHENG Caroline</u> |

## 1. Architecture Globale (Phase 0 & 1) 
> Réalisé en octobre 2026 - Optimisé CPU & C++ en novembre 2026

```mermaid
flowchart TD
    subgraph SOURCES["Sources d'Offres d'Emploi Multi-Plateformes"]
        IND["Indeed (JobSpy)"]
        LNK["LinkedIn (JobSpy)"]
        WTTJ["Welcome to the Jungle\n(API Algolia)"]
        HW["HelloWork\n(JSON-LD enrichi)"]
        ADZ["Adzuna (Optionnel/Déprécié)"]
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
            GLIN["GLiNER Zero-Shot (gliner_multi-v2.1)\nOntologie Dynamique 8 Secteurs\n(Hard & Soft Skills Verbatim)"]
            HEUR["Heuristique Déterministe Regex\n(Niveau Bac+X & Années d'Expérience)"]
        end
        
        CANON["Gabarit Canonique Structuré\n(clean_job_text_for_embedding)"]
    end

    subgraph STORAGE["Socle de Stockage Découplé & Moteur In-Process"]
        PG[("PostgreSQL 16 (projet_cv_db)\nTable relationnelle 'offres'")]
        CHROMA[("ChromaDB Persistant\nCollection dense 'offres' (Cosine)")]
        FASTEMBED["FastEmbed ONNX C++ (In-Process)\nnomic-ai/nomic-embed-text-v1.5 (768d)\n~500 ms/doc (0 timeout HTTP)"]
        OLLAMA["Ollama Runtime (localhost:11434)\nllama3.2:1b (Synthèse & Matching LLM)"]
    end

    subgraph AUDIT["Évaluation & Robustesse (.agents/skills)"]
        AUDITOR["Pipeline Auditor (scripts/audit_pipeline_robustness.py)\nMRR, Top-K, Tokens DeBERTa, Joignabilité URLs"]
    end

    SOURCES --> API
    API --> DEDUP --> CASCADE --> PG
    PG --> EXTRACT
    GLIN --> CANON
    HEUR --> CANON
    CANON --> FASTEMBED --> CHROMA
    PG -.-> AUDITOR
    CHROMA -.-> AUDITOR
```

> **Explication de l'architecture :**
>
> • **Architecture du pipeline** :
> Ce projet collecte des offres d'emploi depuis plusieurs grandes plateformes françaises et internationales, les nettoie, les enrichit grâce à une IA locale déterministe (GLiNER), puis les indexe pour permettre une recherche sémantique ultra-rapide sur CPU.
>
> • **Sources d'offres d'emploi** :
> Les annonces proviennent désormais de 4 sources majeures : **Indeed** et **LinkedIn** via JobSpy, **Welcome to the Jungle (WTTJ)** via l'API Algolia publique, et **HelloWork** via extraction de balises structurées JSON-LD (`JobPosting`). L'API Adzuna (dont les descriptions étaient tronquées à 400 caractères) a été remplacée par ce mix plus qualitatif.
>
> • **Backend Flask (`main.py`, port 5000)** :
> 1. **Déduplication** (`deduplication.py`) : un hash MD5 déterministe identifie les doublons pour ne conserver qu'une seule version de chaque annonce.
> 2. **Normalisation en cascade** (`JobTextSanitizer`) : dissociation rigoureuse entre `description` et `mission` via Regex + spaCy NLP syntaxique.
> 3. **Stockage relationnel** : enregistrement des offres brutes et nettoyées dans PostgreSQL.
> 4. **Enrichissement Zero-Shot 0 hallucination** (`gliner_extractor.py`) : le modèle **GLiNER multi-v2.1** extrait les compétences techniques et comportementales (Hard & Soft Skills) verbatim selon une ontologie de 8 secteurs d'activité, complété par une heuristique regex pour le diplôme (Bac+X) et l'expérience.
> 5. **Gabarit canonique & Indexation ONNX C++** (`job_indexer.py`) : vectorisation directe en mémoire via **FastEmbed C++** (`nomic-ai/nomic-embed-text-v1.5`, 768 dimensions), garantissant une similarité cosinus optimale avec les futurs profils CV.
>
> • **Socle de stockage découplé** :
> - **PostgreSQL 16** (base `projet_cv_db`, table `offres`) : stockage relationnel, filtres booléens, audit qualité.
> - **ChromaDB** (collection `offres`, distance cosinus) : base vectorielle persistante.
> - **FastEmbed ONNX Runtime** : moteur C++ in-process remplaçant les requêtes HTTP, divisant le temps de calcul par 50 sans saturer la RAM.
> - **Ollama** (`localhost:11434`) : runtime LLM local dédié aux synthèses et explications (Phase 2 & 3).

---

## 2. Prérequis Système Obligatoires

Avant de commencer l'installation, assurez-vous d'avoir installé sur votre machine :

| Logo | Outil | Version requise | Rôle |
| :---: | :--- | :--- | :--- |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="35" alt="Python"> | **Python** | **3.11.x** | Compatibilité garantie avec PyTorch, FastEmbed, ONNX Runtime et Docling. |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/postgresql/postgresql-original.svg" width="35" alt="PostgreSQL"> | **PostgreSQL** | **16.x** + pgAdmin 4 | Base de données relationnelle (`projet_cv_db`). |
| <img src="https://cdn.jsdelivr.net/gh/homarr-labs/dashboard-icons/svg/ollama.svg" width="35" alt="Ollama"> | **Ollama** | Dernière version | Inférence des modèles LLM de génération (Llama 3.2). |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" width="35" alt="Git"> | **Git** | **2.x+** | Gestion de versions du projet. |

| Technologie | Technologie | Technologie |
| :---: | :---: | :---: |
| Python 3.11 | Flask 3.0+ | PostgreSQL 16 |
| FastEmbed (ONNX C++) | ChromaDB (Vector Store) | GLiNER Zero-Shot |
| spaCy (NLP Syntaxique) | SQLAlchemy 2.0+ | Pydantic v2 |
| JobSpy & JobScraper | WTTJ (Algolia) & HelloWork (JSON-LD) | Docling (Phase 2) |

---

## 3. Guide d'Installation Pas-à-Pas (Phase 0 & 1)

### Étape 1 : Cloner le dépôt et se placer sur la branche

```powershell
git clone -b phase1_scrappe_offre <URL_DU_REPO_GITHUB>
cd Projet_CV
```

### Étape 2 : Créer et activer l'environnement virtuel Python 3.11

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Étape 3 : Installer les dépendances du projet & les scrapers

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install git+https://github.com/zeffut/jobscraper.git
```

### Étape 3.bis : Télécharger le modèle linguistique spaCy

Le pipeline normaliseur utilise le modèle français compact de spaCy pour l'extraction syntaxique des verbes d'action :
```powershell
python -m spacy download fr_core_news_sm
```

### Étape 3.ter : Initialiser le dossier local des modèles NLP (`models/`)

Pour inspecter les modèles directement dans VS Code tout en protégeant le dépôt GitHub grâce au `.gitignore` :
```powershell
python scripts/sync_models_local.py
```
> [!NOTE]
> Ce script synchronise ou télécharge les poids ONNX de FastEmbed (`nomic-ai/nomic-embed-text-v1.5`) et GLiNER dans le dossier `./models/` de votre projet. Les fichiers binaires volumineux sont automatiquement exclus de Git via `.gitignore`.

### Étape 4 : Démarrer et configurer Ollama (pour les LLM)

1. Assurez-vous qu'Ollama est démarré sur votre machine.
2. Téléchargez le modèle LLM pour la Phase 2 :
   ```powershell
   ollama pull llama3.2:1b
   ```

### Étape 5 : Configurer les variables d'environnement (`.env`)

Copiez le modèle `.env.example` vers `.env` et adaptez vos identifiants :
```powershell
Copy-Item .env.example .env
```

Vérifiez les paramètres clés dans votre `.env` :
```env
DATABASE_URL="postgresql://postgres:VOTRE_MOT_DE_PASSE@127.0.0.1:5432/projet_cv_db"
MODELS_DIR="./models"
EMBEDDING_PROVIDER="fastembed"
EMBEDDING_MODEL_NAME="nomic-ai/nomic-embed-text-v1.5"
OLLAMA_BASE_URL="http://localhost:11434"
LLM_EXTRACTION_MODEL="llama3.2:1b"
```

---

## 4. Initialisation Automatisée du Socle de Données

Lancez le script d'initialisation du socle :
```powershell
python src/database/init_db.py
```
* **PostgreSQL** : Crée la base de données `projet_cv_db` si nécessaire et génère les tables relationnelles (`offres`, `cv_users`, `evaluations`).
* **ChromaDB** : Initialise la collection vectorielle persistante `offres` avec métrique de similarité cosinus.
* **FastEmbed** : Vérifie la présence du moteur C++ ONNX local.

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
Déclenchez la collecte sur l'ensemble des 4 plateformes pour tous les secteurs (ou un secteur ciblé) :
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/jobs/scrape-daily" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"secteur": "all", "limit_per_source": 10, "sources": ["wttj", "hellowork", "indeed", "linkedin"]}'
```

* **Paramètres de la requête** :
  * `secteur` : `"all"` pour balayer automatiquement les 8 secteurs d'activité de l'ontologie, ou le code d'un secteur unique (`"it_data"`, `"sante_medical"`, `"btp_construction"`, `"commerce_marketing"`, `"finance_gestion"`, `"industrie_ingenierie"`, `"rh_juridique"`, `"general_autre"`).
  * `limit_per_source` : Nombre d'offres ciblées par source et par secteur (ex: `10`). Pour `all`, 10 offres $\times$ 8 secteurs = **80 offres max par plateforme**.
  * `sources` : Liste des plateformes à interroger : `["wttj", "hellowork", "indeed", "linkedin"]` (l'ancienne source `"adzuna"` reste supportée optionnellement).
  * `auto_enrich` : `true` (par défaut) pour enchaîner immédiatement la déduplication MD5, l'extraction Zero-Shot GLiNER et l'indexation ChromaDB.

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

### 5. Indexation Vectorielle Haute Performance (FastEmbed ONNX C++)
Pour vectoriser et indexer les offres enrichies dans ChromaDB par lots ultra-rapides :
```powershell
python src/pipeline/job_indexer.py
```
* **Performance** : ~500 ms par offre sur CPU (contre 25 à 30 secondes via Ollama HTTP).
* Zéro timeout réseau, synchronisation automatique de l'état `is_indexed = True` dans PostgreSQL.

---

### 6. Tester la Recherche Sémantique Vectorielle
Vous pouvez tester directement la recherche sémantique via le script de test :
```powershell
python tests/test_fastembed_vector_store.py
```

Ou via l'API REST Flask :
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/jobs/search-vector" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query": "Data Scientist Python Machine Learning PyTorch Bac+5", "n_results": 3}'
```

---

### 7. Audit de Robustesse du Pipeline & Métriques NLP / Vectorielles (Skill Gemini CLI)

Un outil complet de benchmark et d'audit a été développé pour évaluer de bout en bout l'intégrité des données, la joignabilité réelle des annonces et les performances IA :
```powershell
python scripts/audit_pipeline_robustness.py --csv data-1793867529460.csv --output RAPPORT_AUDIT_PIPELINE.md
```

* **Indicateurs mesurés automatiquement** :
  1. **Complétude du Dataset** : Vérification du taux de remplissage de 100% des colonnes et de l'isolation missions vs descriptions sur les 829 offres.
  2. **Joignabilité URLs en direct** : Envoi de requêtes HTTP réelles sur un échantillon représentatif d'annonces (Indeed, LinkedIn, WTTJ, HelloWork) avec mesure de la latence réseau.
  3. **Profilage GLiNER** : Temps d'inférence (ms/offre), comptage réel des tokens DeBERTa pour valider le respect du plafond de 384 tokens (0% de troncature), score de confiance moyen et répartition Hard Skills vs Soft Skills.
  4. **Recherche Vectorielle ChromaDB** : Évaluation du **MRR (Mean Reciprocal Rank)**, de la **Top-1 / Top-3 / Top-5 Accuracy** sectorielle et de la latence cosinus sur des requêtes métiers.
  5. **Score Global de Robustesse** : Note de synthèse sur 100 points enregistrée dans [`RAPPORT_AUDIT_PIPELINE.md`](./RAPPORT_AUDIT_PIPELINE.md).
* **Skill Gemini CLI / Antigravity** : La définition du skill est disponible dans [`.agents/skills/pipeline-auditor/SKILL.md`](./.agents/skills/pipeline-auditor/SKILL.md).

---

## 6. Structure Complète du Code Source

```
Projet_CV/
├── main.py                             # Point d'entrée du serveur Backend Flask (:5000)
├── requirements.txt                     # Dépendances Python (FastEmbed, GLiNER, ChromaDB...)
├── .env.example                         # Modèle de variables d'environnement
├── .gitignore                           # Fichiers exclus du versionnement Git (models/, data/, .env...)
├── README.md                            # Ce guide complet d'installation et de référence
├── RAPPORT_AUDIT_PIPELINE.md            # Rapport d'audit de robustesse et benchmark de performance
│
├── .agents/                             # Configuration des extensions & compétences Gemini CLI / Antigravity
│   └── skills/
│       └── pipeline-auditor/
│           └── SKILL.md                 # Skill d'audit automatisé et d'intégrité du pipeline
│
├── models/                              # Modèles NLP locaux (exclus de Git, visibles dans VS Code)
│   ├── .gitkeep                         # Maintient le dossier dans le dépôt sans les poids
│   ├── fastembed/                       # Modèles ONNX C++ (nomic-embed-text-v1.5)
│   └── huggingface/                     # Poids GLiNER Zero-Shot multi-v2.1
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
├── scripts/                             # Scripts de maintenance, migration et utilitaires
│   ├── audit_pipeline_robustness.py     # Moteur d'évaluation, test d'URLs en direct & MRR ChromaDB
│   ├── sync_models_local.py             # Synchronisation locale des modèles NLP dans ./models/
│   ├── migrate_split_missions.py        # Migration de découpage 'description' vs 'mission'
│   ├── migrate_contracts_and_skills.py  # Normalisation contrats composites & nettoyage compétences
│   └── test_spacy_split_preview.py      # Prévisualisation visuelle du normaliseur spaCy
│
├── src/
│   ├── config.py                        # Configuration Pydantic Settings & gestion du dossier models/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py                    # Endpoints REST (/api/jobs/*, /api/health)
│   ├── database/
│   │   ├── db_session.py                # Session & Engine SQLAlchemy (PostgreSQL)
│   │   ├── sql_models.py                # Modèles relationnels (Offre, CVUser, Evaluation)
│   │   ├── vector_store.py              # Intégration FastEmbed ONNX C++ & ChromaDB
│   │   └── export_chroma.py             # Script d'export JSON pour audit qualité
│   ├── parsers/
│   │   ├── gliner_extractor.py          # Extracteur Zero-Shot GLiNER multi-secteurs (Hard & Soft Skills)
│   │   ├── job_enricher.py              # Façade d'enrichissement unifiée & condensation ciblée
│   │   └── docling_parser.py            # [Phase 2] Parseur de CVs PDF (Docling IBM)
│   ├── pipeline/
│   │   ├── deduplication.py             # Hachage MD5 et détection de doublons
│   │   ├── job_normalizer.py            # Normaliseur cascade 2 niveaux (Regex + spaCy NLP)
│   │   ├── job_indexer.py               # Constructeur canonique & indexeur vectoriel par lots
│   │   └── batch_enrich_jobs.py         # Worker batch d'enrichissement GLiNER
│   ├── scrapers/
│   │   ├── jobscraper_connector.py      # Connecteur unifié Welcome to the Jungle (Algolia) & HelloWork (JSON-LD)
│   │   ├── jobspy_scraper.py            # Scraper Indeed & LinkedIn via JobSpy
│   │   ├── daily_scraper.py             # Orchestrateur de collecte multi-sources (WTTJ, HelloWork, Indeed, LinkedIn)
│   │   └── sector_mapping.py            # Ontologie des 8 secteurs et requêtes ciblées
│   └── utils/
│       └── ollama_client.py             # Client HTTP Ollama résilient
│
├── tests/
│   ├── test_fastembed_vector_store.py   # Test unitaire et benchmark similarité cosinus FastEmbed
│   ├── test_gliner_multisector.py       # Benchmark multi-secteurs & anti-hallucination GLiNER
│   └── ...                              # Autres suites de tests unitaires
│
└── data/                                # Données locales (ignorées par Git)
    ├── chromadb/                        # Données persistantes ChromaDB
    └── chroma_dump.json                 # Export JSON d'audit des offres vectorisées
```
