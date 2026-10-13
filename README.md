# Recommandation Sémantique d'Offres d'Emploi par Analyse de CV

> **Projet Master 2 Intelligence Artificielle**  
> **Composants** : Document AI (Docling), Ingestion Multi-Sources (JobSpy, Adzuna), Embeddings Denses (ChromaDB + Ollama), IA Générative Locale (Llama 3.2 1B), Recherche Hybride & Reranking.


Ce projet implémente un système complet de matching sémantique et explicable entre des CVs de candidats (PDF) et des offres d'emploi collectées sur le marché.

Ce document est le **guide complet d'installation et d'exécution** destiné aux membres de l'équipe pour reproduire intégralement la **Phase 0 (Setup & Socle)** et la **Phase 1 (Scraping, Nettoyage, Enrichissement LLM & Double Indexation)**.

##  Membres du projet

| Nom et prénom |
|---|
| ANTON NELCON Steve |
| LIMACHE Adrien |
| MOULAI HACENE Rania |
| Ibtissam Zaid |
| Caroline Zheng |---

## 1. Architecture Globale (Phase 0 & 1)

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
        SAN["Nettoyeur Algorithmique (JobTextSanitizer)"]
        ENRICH["Enrichisseur IA Local (JobEnricher - Llama 3.2 1B)"]
        CANON["Gabarit Canonique Structuré (job_indexer.py)"]
    end

    subgraph STORAGE["Socle de Stockage Découplé"]
        PG[("PostgreSQL 16 (projet_cv_db)\nTable relationnelle 'offres'")]
        CHROMA[("ChromaDB Persistant\nCollection dense 'offres' (Cosine)")]
        OLLAMA["Ollama Runtime (localhost:11434)\nall-minilm + llama3.2:1b"]
    end

    SOURCES --> API
    API --> DEDUP --> SAN --> PG
    PG --> ENRICH --> OLLAMA
    ENRICH --> CANON --> CHROMA
```

---

## 2. Prérequis Système Obligatoires

Avant de commencer l'installation, assurez-vous d'avoir installé sur votre machine :

| Logo | Outil | Version requise | Rôle |
| :---: | :--- | :--- | :--- |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="35" alt="Python"> | **Python** | **3.11.x** | Compatibilité avec PyTorch, ChromaDB et Docling. |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/postgresql/postgresql-original.svg" width="35" alt="PostgreSQL"> | **PostgreSQL** | **16.x** + pgAdmin 4 | Base de données relationnelle (`projet_cv_db`). |
| <img src="https://cdn.jsdelivr.net/gh/homarr-labs/dashboard-icons/svg/ollama.svg" width="35" alt="Ollama"> | **Ollama** | Dernière version | Exécution locale des modèles LLM et d'embeddings. |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" width="35" alt="Git"> | **Git** | **2.x+** | Gestion de versions du projet. |

---

## 3. Guide d'Installation Pas-à-Pas (Phase 0)

### Étape 1 : Cloner le dépôt et récupérer la branche Phase 1

#### Cas A — Si vous clonez le projet pour la première fois :
Clonez directement la branche de travail `phase1_scrappe_offre` :
```powershell
git clone -b phase1_scrappe_offre <URL_DU_REPO_GITHUB>
cd Projet_CV
```

#### Cas B — Si vous avez déjà cloné le dépôt sur une autre branche :
Récupérez les branches distantes et basculez sur `phase1_scrappe_offre` :
```powershell
git fetch origin
git checkout phase1_scrappe_offre
git pull origin phase1_scrappe_offre
```

### Étape 2 : Créer l'environnement virtuel avec Python 3.11
Sous Windows, utilisez le launcher `py` pour cibler spécifiquement la version 3.11 :
```powershell
py -3.11 -m venv .venv
```
*(Sur Linux/macOS : `python3.11 -m venv .venv`)*

### Étape 3 : Activer l'environnement virtuel
* **Windows (PowerShell)** :
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(Si PowerShell bloque l'activation, exécutez d'abord : `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`)*
* **Linux / macOS** :
  ```bash
  source .venv/bin/activate
  ```

Le préfixe `(.venv)` doit impérativement apparaître devant votre invite de commande.

### Étape 4 : Mettre à jour pip et installer les dépendances
```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Étape 5 : Installer et démarrer Ollama
1. Téléchargez et installez Ollama depuis [ollama.com](https://ollama.com/).
2. Ouvrez un terminal et téléchargez les deux modèles nécessaires au projet :
   ```bash
   # Modèle d'embedding vectoriel (rapide, 384 dimensions)
   ollama pull all-minilm

   # Modèle LLM d'extraction structurée (ultra-rapide sur CPU)
   ollama pull llama3.2:1b
   ```
3. Vérifiez qu'Ollama est actif en visitant `http://localhost:11434` dans votre navigateur (doit afficher *"Ollama is running"*).

### Étape 6 : Configuration du fichier `.env`
Placez le fichier `.env` (fourni pour le projet) à la racine de votre dossier `Projet_CV`, et adaptez si besoin la variable `DATABASE_URL` avec votre mot de passe PostgreSQL local :
```ini
DATABASE_URL="postgresql://postgres:VOTRE_MOT_DE_PASSE@127.0.0.1:5432/projet_cv_db"
```
*(Pour les scrapers, les clés Adzuna publiques sont déjà renseignées ou configurables dans `.env`)*.

---

## 4. Initialisation Automatisée du Socle de Données (PostgreSQL & ChromaDB)

Pour éviter toute manipulation manuelle dans pgAdmin, lancez simplement le script d'initialisation automatisé :

```powershell
python src/database/init_db.py
```

Ce script configure l'intégralité du socle technique en une seule commande :
* **PostgreSQL** : Vérifie la connexion, crée la base de données `projet_cv_db` si elle n'existe pas encore, et génère l'ensemble des tables relationnelles (`offres`, `cv_users`, `evaluations`).
* **ChromaDB** : Initialise le dossier de persistance `data/chromadb` et la collection vectorielle `offres` avec la métrique de similarité cosinus (`hnsw:space: cosine`).
* **Ollama** : Vérifie que le runtime local est actif et confirme la présence des modèles `all-minilm` (embedding) et `llama3.2:1b` (LLM).

---

## 5. Utilisation & Workflow Opérationnel (Phase 1)

### Étape 1 : Démarrer le Serveur Backend Flask (Terminal 1)
```powershell
python main.py
```
Le serveur démarre sur `http://127.0.0.1:5000`.

---

### Étape 2 : Collecte Multi-Sources des Offres (Scraping Quotidien)
Dans un **second terminal**, déclenchez la collecte multi-sources (Indeed, LinkedIn, Adzuna) via l'API REST :
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/jobs/scrape-daily" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"secteur": "it_data", "hours_old": 24, "indeed_results": 15, "linkedin_results": 15, "adzuna_results": 15}'
```
* Ce module télécharge les offres, élimine les doublons via hash MD5, nettoie le bruit textuel (intros corporate, tickets resto, mutuelle) via `JobTextSanitizer`, et insère les offres saines dans PostgreSQL.

---

### Étape 3 : Enrichissement LLM par Lots (Llama 3.2 1B)
Pour les offres Indeed et LinkedIn dont les compétences ne sont pas fournies par l'API, lancez le worker batch d'enrichissement.  
Il intègre une **protection thermique CPU** (traitement par lots de 5 offres + pause de repos) :
```powershell
python src/pipeline/batch_enrich_jobs.py
```
* **Ce qu'il fait** : Le LLM local extrait sous schéma Pydantic strict les compétences techniques clés, le niveau d'études minimal (`Bac+X`) et les années d'expérience requises.
* **Auto-synchronisation** : À la fin du lot, il vectorise automatiquement les nouvelles offres dans ChromaDB.

---

### Étape 4 : Ré-indexation Vectorielle Canonique dans ChromaDB (Optionnel)
Si vous souhaitez régénérer l'intégralité des vecteurs de toutes les offres de PostgreSQL dans ChromaDB avec le template structuré canonique :
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/jobs/reindex" -Method Post -TimeoutSec 600
```

---

### Étape 5 : Audit Qualité & Export des Données ChromaDB
Pour inspecter la structure exacte des offres stockées dans ChromaDB :
```powershell
python src/database/export_chroma.py
```
Cette commande génère le fichier `data/chroma_dump.json` contenant tous les documents canoniques et métadonnées indexés.

---

### Étape 6 : Tester la Recherche Sémantique Vectorielle
Testez la similarité cosinus dense directement via l'API :
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
├── requirements.txt                     # Dépendances Python du projet
├── .env.example                         # Modèle de variables d'environnement
├── .gitignore                           # Fichiers exclus du versionnement Git
├── README.md                            # Ce guide d'installation et de référence
│
├── docs/                                # Documentation technique détaillée
│   ├── ARCHITECTURE.md                  # Schémas d'architecture globale et flux de données
│   ├── CANONICAL_VECTORIZATION_SPEC.md  # Spécification du gabarit canonique pour ChromaDB
│   ├── FLASK_BACKEND_SPEC.md            # Spécification de l'API REST Flask
│   ├── phase0.md                        # Documentation historique Phase 0 (Setup)
│   └── phase1.md                        # Documentation historique Phase 1 (Ingestion)
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
│   │   ├── job_enricher.py              # Extracteur IA (Llama 3.2 1B -> Pydantic Output)
│   │   └── docling_parser.py            # [Phase 2] Parseur de CVs PDF (Docling IBM)
│   ├── pipeline/
│   │   ├── deduplication.py             # Hachage MD5 et détection de doublons
│   │   ├── job_normalizer.py            # Nettoyeur Regex (JobTextSanitizer) et schéma Offre
│   │   ├── job_indexer.py               # Constructeur du template canonique & indexeur ChromaDB
│   │   └── batch_enrich_jobs.py         # Worker batch d'enrichissement LLM avec repos CPU
│   ├── scrapers/
│   │   ├── adzuna_scraper.py            # Connecteur API REST Adzuna
│   │   ├── jobspy_scraper.py            # Scraper Indeed & LinkedIn via JobSpy
│   │   └── daily_scraper.py             # Orchestrateur de collecte quotidienne multi-sources
│   └── utils/
│       └── ollama_client.py             # Client HTTP Ollama résilient avec auto-réparation JSON
│
└── data/                                # Stockage local (ignoré par Git)
    ├── chromadb/                        # Données persistantes ChromaDB (fichiers HNSW SQLite)
    └── chroma_dump.json                 # Export JSON d'audit des offres vectorisées
```
