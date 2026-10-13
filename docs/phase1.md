# Guide de Réalisation — Phase 1 : Base de Données, Scraping & Double Indexation
> Projet Master 2 IA — Système de Recommandation d'Offres d'Emploi par Analyse de CV

Ce document est le **plan d'action pas-à-pas** pour implémenter la première phase opérationnelle :
1. La création de la base de données relationnelle (**PostgreSQL via pgAdmin**) et vectorielle (**ChromaDB**).
2. La collecte automatisée d'offres d'emploi via **3 options développées une par une**.
3. La déduplication, la normalisation des critères et la double indexation synchronisée.

---

## 1. Architecture des Fichiers de la Phase 1

Tous ces fichiers ont été créés vides à la racine du projet et sont prêts à être implémentés :

```
Projet_CV/
├── .env                             # Vos variables secrètes locales (DATABASE_URL, clés API)
├── .env.example                     # Modèle de configuration
├── requirements.txt                 # Dépendances Python (installées en Phase 0)
├── src/
│   ├── __init__.py
│   ├── config.py                    # Settings Pydantic (device CUDA/CPU, URLs, clés)
│   ├── database/
│   │   ├── __init__.py
│   │   ├── db_session.py            # Moteur & session SQLAlchemy vers PostgreSQL
│   │   ├── sql_models.py            # Modèles déclaratifs SQLAlchemy (Offre, CVUser, Evaluation)
│   │   └── vector_store.py          # Client ChromaDB persistant & collection 'offres'
│   ├── scrapers/
│   │   ├── __init__.py
│   │   ├── base_scraper.py          # Classe abstraite unifiant les scrapers
│   │   ├── france_travail_api.py    # [Option 1] Connecteur officiel API France Travail
│   │   ├── jobspy_scraper.py        # [Option 2] Scraper LinkedIn / Indeed
│   │   └── adzuna_scraper.py        # [Option 3] Connecteur Adzuna
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── deduplication.py         # Calcul du hash MD5 et détection des doublons
│   │   ├── job_normalizer.py        # Normalisation vers le schéma Pydantic JobOffer
│   │   └── job_indexer.py           # Orchestrateur d'ingestion (Scrape -> PostgreSQL + ChromaDB)
│   └── cli/
│       ├── __init__.py
│       └── ingest_jobs.py           # Script CLI pour lancer la collecte en terminal
└── tests/
    ├── __init__.py
    ├── test_france_travail.py       # Test unitaire de l'Option 1
    ├── test_jobspy.py               # Test unitaire de l'Option 2
    └── test_indexer.py              # Test du double stockage
```

---

## 2. Déroulement de la Réalisation : Étape par Étape

```mermaid
flowchart TD
    E0["Étape 0 : Socle Base de Données\n1. Création projet_cv_db dans pgAdmin\n2. Exécution du script SQL DDL\n3. Modèles Python dans src/database/"]
    E1["Étape 1 : Option 1 (France Travail)\n1. OAuth2 Client Credentials\n2. Requête /offres/search\n3. Test unitaire"]
    E2["Étape 2 : Option 2 (JobSpy)\n1. Scraping LinkedIn & Indeed\n2. Gestion des filtres France\n3. Test unitaire"]
    E3["Étape 3 : Option 3 (Adzuna)\n1. Client REST Adzuna\n2. Récupération des offres tech"]
    E4["Étape 4 : Déduplication & Normalisation\n1. Calcul job_hash MD5\n2. Extraction niveau d'études & compétences\n3. Validation JobOffer Pydantic"]
    E5["Étape 5 : Double Indexation & CLI\n1. INSERT PostgreSQL (table offres)\n2. Embedding BGE-M3 (CUDA/CPU)\n3. UPSERT ChromaDB (collection offres)\n4. Commande CLI ingest_jobs.py"]

    E0 --> E1
    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> E5
```

---

### Étape 0 : Socle de Base de Données (pgAdmin, PostgreSQL & ChromaDB)

Avant d'écrire la moindre ligne de scraper, la base de données doit exister pour accueillir les données.

#### 0.1 Création de la base dans pgAdmin
1. Ouvrez **pgAdmin**.
2. Clic droit sur **Databases** > **Create** > **Database...**
3. Nom : **`projet_cv_db`** > **Save**.

#### 0.2 Création des Tables (Script SQL DDL)
Dans pgAdmin, faites un clic droit sur `projet_cv_db` > **Query Tool**, collez ce script et appuyez sur **F5** :

```sql
-- Extension pour UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Table des Offres d'emploi
CREATE TABLE offres (
    id VARCHAR(64) PRIMARY KEY,
    poste VARCHAR(255) NOT NULL,
    company VARCHAR(255) NOT NULL,
    type_de_contrat VARCHAR(50) NOT NULL,
    niveau_etude_min INTEGER DEFAULT 0,
    min_experience_years NUMERIC(4, 1) DEFAULT 0.0,
    competences_necessaires TEXT[] NOT NULL DEFAULT '{}',
    description TEXT NOT NULL,
    mission TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Table des Candidats (CVs)
CREATE TABLE cv_users (
    id VARCHAR(64) PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    desired_contract_types VARCHAR(50)[] NOT NULL DEFAULT '{}',
    parcours_academique INTEGER NOT NULL DEFAULT 5,
    total_experience_years NUMERIC(4, 1) DEFAULT 0.0,
    competences TEXT[] NOT NULL DEFAULT '{}',
    experiences_professionnelles JSONB DEFAULT '[]'::jsonb,
    experiences_universitaires JSONB DEFAULT '[]'::jsonb,
    raw_markdown TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Table des Évaluations de Matching (Scores)
CREATE TABLE evaluations (
    id SERIAL PRIMARY KEY,
    cv_user_id VARCHAR(64) NOT NULL REFERENCES cv_users(id) ON DELETE CASCADE,
    offre_id VARCHAR(64) NOT NULL REFERENCES offres(id) ON DELETE CASCADE,
    dense_score FLOAT,
    bm25_score FLOAT,
    reranker_score FLOAT,
    final_score INTEGER NOT NULL,
    matched_skills TEXT[] DEFAULT '{}',
    missing_skills TEXT[] DEFAULT '{}',
    justification TEXT,
    conseils_candidature TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_cv_offre UNIQUE (cv_user_id, offre_id)
);

-- Index pour accélérer les recherches et tris
CREATE INDEX idx_offres_competences ON offres USING GIN(competences_necessaires);
CREATE INDEX idx_cv_competences ON cv_users USING GIN(competences);
CREATE INDEX idx_eval_user_score ON evaluations (cv_user_id, final_score DESC);
```

#### 0.3 Configuration du fichier `.env`
Assurez-vous que votre fichier [`.env`](file:///C:/MASTER2_IA/Projet_CV/.env) contient votre mot de passe pgAdmin :
```env
DATABASE_URL=postgresql://postgres:VOTRE_MOT_DE_PASSE@localhost:5432/projet_cv_db
CHROMA_PERSIST_DIR=./data/chromadb
CHROMA_COLLECTION_OFFRES=offres
```

#### 0.4 Implémentation du module Python `src/database/`
* **`src/database/db_session.py`** :
  - Créer `engine = create_engine(settings.database_url)`.
  - Créer `SessionLocal = sessionmaker(bind=engine, autoflush=False)`.
  - Fournir un générateur de session `get_db()`.
* **`src/database/sql_models.py`** :
  - Définir `Base = declarative_base()`.
  - Définir le modèle `class Offre(Base): __tablename__ = "offres" ...` avec les colonnes correspondantes.
* **`src/database/vector_store.py`** :
  - Initialiser `chromadb.PersistentClient(path=settings.chroma_persist_dir)`.
  - Fournir une fonction `get_or_create_offres_collection()`.

---

### Étape 1 : Option 1 — Connecteur API France Travail (Priorité 1)

L'API France Travail est l'API officielle Open Data en France.

#### 1. Fichiers à implémenter :
* **`src/scrapers/base_scraper.py`** :
  - Classe abstraite `BaseJobScraper` définissant la méthode `scrape(self, query: str, location: str, limit: int) -> List[Dict]`.
* **`src/scrapers/france_travail_api.py`** :
  - Hérite de `BaseJobScraper`.
  - Gère l'authentification OAuth2 (`client_credentials`) sur `https://entreprise.francetravail.fr/connexion/oauth2/access_token`.
  - Interroge l'endpoint `https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search`.
  - Transforme le JSON retourné en dictionnaire standardisé.

#### 2. Test unitaire :
* **`tests/test_france_travail.py`** : Tester la récupération de 5 offres avec la requête `"Data Scientist"` et vérifier que les champs `titre`, `description` et `entreprise` sont bien présents.

---

### Étape 2 : Option 2 — Scraper Multi-Plateformes avec JobSpy (Priorité 2)

Utilisation de la bibliothèque `python-jobspy` pour requêter directement **LinkedIn** et **Indeed**.

#### 1. Fichier à implémenter :
* **`src/scrapers/jobspy_scraper.py`** :
  - Hérite de `BaseJobScraper`.
  - Appelle `jobspy.scrape_jobs(site_name=["linkedin", "indeed"], search_term=query, location="France", results_wanted=limit)`.
  - Convertit le DataFrame Pandas retourné en liste de dictionnaires normalisés.

#### 2. Test unitaire :
* **`tests/test_jobspy.py`** : Tester le scraping de 5 offres sur LinkedIn/Indeed et vérifier la présence de l'intitulé et du lien source.

---

### Étape 3 : Option 3 — Connecteur APIs Agrégatrices (Priorité 3)

#### 1. Fichier à implémenter :
* **`src/scrapers/adzuna_scraper.py`** :
  - Hérite de `BaseJobScraper`.
  - Interroge l'API REST Adzuna (`https://api.adzuna.com/v1/api/jobs/fr/search/1`).
  - Gère la pagination et le parsing des salaires et localisations.

---

### Étape 4 : Déduplication & Normalisation Pydantic

Une offre peut être récupérée à la fois sur France Travail, LinkedIn et Indeed.

#### 1. Déduplication (`src/pipeline/deduplication.py`) :
* Calcul d'une clé de hachage unique :
  ```python
  import hashlib

  def compute_job_hash(company: str, poste: str, location: str) -> str:
      raw = f"{company.strip().lower()}_{poste.strip().lower()}_{location.strip().lower()}"
      return f"job_{hashlib.md5(raw.encode('utf-8')).hexdigest()[:12]}"
  ```
* Vérification de l'existence de l'`id` dans la table PostgreSQL `offres`.

#### 2. Normalisation Légère (`src/pipeline/job_normalizer.py`) :
* Normalisation des champs bruts vers le modèle SQLAlchemy `Offre` :
  * Le type de contrat (`CDI`, `CDD`, `Stage`, `Alternance`).
  * Les champs textuels (poste, entreprise, lieu, description complète).
  * `is_indexed = False` (en attente du worker d'embedding Ollama bge-m3).
  * **Zéro dictionnaire en dur** : Le texte intégral est préservé pour le modèle BGE-M3 et le LLM.

---

### Étape 5 : Double Indexation Synchronisée & Commande CLI

#### 1. Indexeur (`src/pipeline/job_indexer.py`) :
Pour chaque offre normalisée :
1. **Sauvegarde PostgreSQL** : Insertion ou mise à jour dans la table `offres` via SQLAlchemy.
2. **Vectorisation sémantique avec Ollama (`bge-m3`)** :
   ChromaDB est configuré avec `OllamaEmbeddingFunction` (modèle `bge-m3` sur `http://localhost:11434`).
   Le texte enrichi de l'offre est vectorisé automatiquement en dimension 1024 par lots sans saturer le CPU.
3. **Insertion ChromaDB** : Upsert du vecteur avec le **même `id`** que PostgreSQL et les métadonnées de pré-filtrage (`type_de_contrat`, `niveau_etude_min`, `is_active`).

#### 2. Script CLI (`src/cli/ingest_jobs.py`) :
Permet de déclencher l'ingestion directement en terminal :
```bash
python -m src.cli.ingest_jobs --source france_travail --query "Data Scientist" --limit 50
```

---

## 3. Critères de Validation du Livrable Phase 1

À la fin de la Phase 1, vous devez pouvoir :
1. ✅ Ouvrir **pgAdmin** et voir au moins 100 offres d'emploi réelles dans la table `offres`.
2. ✅ Ouvrir un terminal Python et vérifier que ChromaDB contient le même nombre de vecteurs :
   ```python
   import chromadb
   client = chromadb.PersistentClient(path="./data/chromadb")
   col = client.get_collection("offres")
   print("Nombre d'offres indexées :", col.count())
   ```
3. ✅ Vérifier qu'une recherche vectorielle sur ChromaDB renvoie des IDs qui correspondent exactement aux offres dans PostgreSQL.
