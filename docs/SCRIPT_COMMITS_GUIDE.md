# Guide des Commits Individuels par Script Python (`.py`)

> **Projet Master 2 IA** — Système de Recommandation d'Offres d'Emploi par Analyse de CV  
> **Branche Git cible** : `phase1_scrappe_offre`  
> **Objectif** : Exécuter un `git add <fichier>` et un `git commit` individuel pour **chaque script Python** afin d'afficher clairement sa fonction dans l'historique Git.

---

## 📌 Préalable : Créer la branche (si ce n'est pas encore fait)

```powershell
git init
git remote add origin <URL_DU_REPO_GITHUB>
git checkout -b phase1_scrappe_offre
```

---

## 0. Fichiers de Configuration & Socle Initial (Non-Python)

Avant de commiter les scripts Python, enregistrez les fichiers de configuration de base :

### 0.1 Règle d'exclusion Git
```powershell
git add .gitignore
git commit -m "chore(config): ajout du fichier .gitignore pour exclure venv, cache et bases locales"
```

### 0.2 Dépendances du projet
```powershell
git add requirements.txt
git commit -m "chore(deps): ajout du fichier requirements.txt listant les dependances du projet"
```

---

## 1. Scripts de Configuration & Utilitaires de Base

### 1.1 `src/config.py`
* **Fonction** : Charge et valide les variables d'environnement (`.env`) via Pydantic Settings (URLs, clés API, device CUDA/CPU, noms des modèles Ollama).
```powershell
git add src/config.py
git commit -m "feat(config): gestion centralisee des parametres du projet via pydantic settings"
```

### 1.2 `src/utils/ollama_client.py`
* **Fonction** : Client HTTP local pour Ollama (`/api/chat`), Structured Outputs Pydantic, pénalité anti-répétition (`repeat_penalty`) et auto-réparation chirurgicale des JSON tronqués par limite de tokens.
```powershell
git add src/utils/ollama_client.py
git commit -m "feat(utils): client http ollama avec support repeat_penalty et auto-reparation json"
```

---

## 2. Scripts de Base de Données (PostgreSQL & ChromaDB)

### 2.1 `src/database/db_session.py`
* **Fonction** : Configure le moteur SQLAlchemy, le pool de connexions vers PostgreSQL et fournit le générateur de session `get_db()`.
```powershell
git add src/database/db_session.py
git commit -m "feat(db): configuration de la session sqlalchemy et du pool de connexion postgresql"
```

### 2.2 `src/database/sql_models.py`
* **Fonction** : Définit les modèles relationnels déclaratifs SQLAlchemy 2.0 (`Offre`, `CVUser`, `Evaluation`) avec types PostgreSQL natifs (`ARRAY`, `JSONB`).
```powershell
git add src/database/sql_models.py
git commit -m "feat(db): modeles relationnels sqlalchemy pour les offres, cvs et evaluations"
```

### 2.3 `src/database/vector_store.py`
* **Fonction** : Initialise le client persistant ChromaDB, la collection vectorielle `offres` et la fonction d'embedding Ollama avec similarité cosinus (`hnsw:space: cosine`).
```powershell
git add src/database/vector_store.py
git commit -m "feat(vector): gestionnaire de collection vectorielle chromadb avec distance cosinus"
```

### 2.4 `src/database/init_db.py`
* **Fonction** : Script d'initialisation en une commande qui crée automatiquement la base `projet_cv_db`, les tables relationnelles, la collection ChromaDB et teste Ollama.
```powershell
git add src/database/init_db.py
git commit -m "feat(db): script d'initialisation automatisee du socle postgresql et chromadb"
```

### 2.5 `src/database/export_chroma.py`
* **Fonction** : Extrait l'ensemble des documents textuels canoniques et métadonnées indexés dans ChromaDB vers un fichier JSON (`data/chroma_dump.json`) pour audit qualité.
```powershell
git add src/database/export_chroma.py
git commit -m "feat(vector): script d'exportation json des offres de chromadb pour audit qualite"
```

### 2.6 `src/database/migrate_add_secteur.py`
* **Fonction** : Script de migration SQL ajoutant la colonne `secteur` à la table `offres` pour le filtrage multi-domaines.
```powershell
git add src/database/migrate_add_secteur.py
git commit -m "feat(migration): migration sql pour l'ajout de la colonne secteur dans offres"
```

### 2.7 `src/database/migrate_add_source.py`
* **Fonction** : Script de migration SQL ajoutant la colonne `source` pour tracer l'origine de l'offre (Indeed, LinkedIn, Adzuna).
```powershell
git add src/database/migrate_add_source.py
git commit -m "feat(migration): migration sql pour l'ajout de la colonne source dans offres"
```

### 2.8 `src/database/migrate_sanitize_jobs.py`
* **Fonction** : Script de migration nettoyant rétroactivement les descriptions et séparant missions/profils pour réduire de 33% le bruit textuel en base.
```powershell
git add src/database/migrate_sanitize_jobs.py
git commit -m "feat(migration): migration d'assainissement textuel retroactif des offres sql"
```

---

## 3. Scripts de Collecte d'Offres (Scrapers)

### 3.1 `src/scrapers/base_scraper.py`
* **Fonction** : Classe abstraite définissant l'interface standardisée que chaque connecteur de scraping doit respecter (`fetch_jobs`).
```powershell
git add src/scrapers/base_scraper.py
git commit -m "feat(scrapers): classe abstraite definissant le contrat d'interface des scrapers"
```

### 3.2 `src/scrapers/sector_mapping.py`
* **Fonction** : Dictionnaire de correspondance et mots-clés de recherche par secteur métier (IT/Data, Santé, BTP, etc.).
```powershell
git add src/scrapers/sector_mapping.py
git commit -m "feat(scrapers): dictionnaire de mapping sectoriel et mots-cles de recherche"
```

### 3.3 `src/scrapers/adzuna_scraper.py`
* **Fonction** : Connecteur vers l'API REST officielle Adzuna France avec pagination, récupération du salaire et des métadonnées brutes.
```powershell
git add src/scrapers/adzuna_scraper.py
git commit -m "feat(scrapers): connecteur api rest adzuna pour la collecte d'offres"
```

### 3.4 `src/scrapers/jobspy_scraper.py`
* **Fonction** : Scraper multi-plateformes (LinkedIn, Indeed) via JobSpy avec gestion du périmètre géographique France et du formatage de date.
```powershell
git add src/scrapers/jobspy_scraper.py
git commit -m "feat(scrapers): connecteur jobspy pour le scraping d'offres indeed et linkedin"
```

### 3.5 `src/scrapers/france_travail_api.py`
* **Fonction** : Connecteur OAuth2 vers l'API officielle France Travail (Pôle Emploi).
```powershell
git add src/scrapers/france_travail_api.py
git commit -m "feat(scrapers): connecteur oauth2 pour l'api officielle france travail"
```

### 3.6 `src/scrapers/daily_scraper.py`
* **Fonction** : Chef d'orchestre de la collecte quotidienne multi-sources, coordonne Adzuna et JobSpy avec filtrage temporel strict sur 24h.
```powershell
git add src/scrapers/daily_scraper.py
git commit -m "feat(scrapers): orchestrateur de scraping quotidien multi-sources et multi-secteurs"
```

---

## 4. Scripts de Pipeline de Traitement & Indexation

### 4.1 `src/pipeline/deduplication.py`
* **Fonction** : Algorithme de déduplication par hachage MD5 (Poste + Entreprise + Localisation) éliminant les offres redondantes entre plateformes.
```powershell
git add src/pipeline/deduplication.py
git commit -m "feat(pipeline): algorithme de deduplication inter-plateformes par hachage md5"
```

### 4.2 `src/pipeline/job_normalizer.py`
* **Fonction** : Nettoyeur textuel Regex `JobTextSanitizer` (suppression des intros RH, packages salariaux, tickets resto) et convertisseur d'offres brutes vers SQLAlchemy.
```powershell
git add src/pipeline/job_normalizer.py
git commit -m "feat(pipeline): nettoyeur textuel regex jobtextsanitizer et normalisation d'offres"
```

### 4.3 `src/pipeline/job_indexer.py`
* **Fonction** : Générateur du gabarit canonique structuré (`[INTITULÉ]`, `[COMPÉTENCES]`, `[MISSIONS]`), indexeur par lots ChromaDB et méthode ciblée `index_jobs_by_ids()`.
```powershell
git add src/pipeline/job_indexer.py
git commit -m "feat(pipeline): gabarit canonique structuré et indexation vectorielle dans chromadb"
```

### 4.4 `src/pipeline/batch_enrich_jobs.py`
* **Fonction** : Worker de traitement par lots d'enrichissement LLM (Llama 3.2 1B) par blocs de 5 avec temporisation thermique CPU et indexation ciblée.
```powershell
git add src/pipeline/batch_enrich_jobs.py
git commit -m "feat(pipeline): worker batch d'enrichissement llama avec protection thermique cpu"
```

---

## 5. Scripts d'Enrichissement Sémantique & IA (Parsers)

### 5.1 `src/parsers/job_enricher.py`
* **Fonction** : Extracteur IA sémantique sous Pydantic utilisant Ollama `llama3.2:1b` (compétences techniques, niveau d'études Bac+X, années d'expérience) avec tolérance aux valeurs `null`.
```powershell
git add src/parsers/job_enricher.py
git commit -m "feat(parsers): extracteur semantique d'offres via llama 3.2 1b et schema pydantic"
```

---

## 6. Scripts d'API REST & Point d'Entrée Serveur

### 6.1 `src/api/routes.py`
* **Fonction** : Blueprint REST Flask exposant les routes de diagnostic (`/health`), de scraping (`/jobs/scrape-daily`), d'enrichissement (`/jobs/enrich-pending`), d'indexation (`/jobs/reindex`) et de recherche cosinus (`/jobs/search-vector`).
```powershell
git add src/api/routes.py
git commit -m "feat(api): routes rest flask pour le scraping, l'enrichissement et la recherche vectorielle"
```

### 6.2 `main.py`
* **Fonction** : Point d'entrée exécutable du serveur backend Flask (`http://127.0.0.1:5000`) avec activation CORS et gestion de configuration.
```powershell
git add main.py
git commit -m "feat(main): point d'entree executable du serveur backend flask avec support cors"
```

---

## 7. Scripts CLI & Outils de Benchmark

### 7.1 `src/cli/ingest_jobs.py`
* **Fonction** : Script CLI de collecte et d'ingestion directe d'offres en ligne de commande.
```powershell
git add src/cli/ingest_jobs.py
git commit -m "feat(cli): script d'ingestion d'offres en ligne de commande"
```

### 7.2 `src/cli/enrich_jobs.py`
* **Fonction** : Script CLI pour tester l'enrichissement d'offres unitaires via invite de commande.
```powershell
git add src/cli/enrich_jobs.py
git commit -m "feat(cli): outil de test d'enrichissement sémantique en ligne de commande"
```

### 7.3 `src/cli/benchmark_prompts.py`
* **Fonction** : Banc d'essai comparatif de différentes stratégies de prompt engineering pour l'extraction de critères.
```powershell
git add src/cli/benchmark_prompts.py
git commit -m "feat(cli): benchmark de prompt engineering pour l'extraction d'attributs d'offres"
```

### 7.4 `src/cli/test_cleaning_benchmark.py`
* **Fonction** : Benchmark de mesure de l'efficacité de nettoyage de `JobTextSanitizer` (taux de réduction du bruit).
```powershell
git add src/cli/test_cleaning_benchmark.py
git commit -m "feat(cli): benchmark de validation du nettoyage algorithmique de texte"
```

---

## 8. Packages Initiaux (`__init__.py`)

Pour enregistrer les modules Python du projet :
```powershell
git add src/__init__.py src/api/__init__.py src/database/__init__.py src/scrapers/__init__.py src/pipeline/__init__.py src/parsers/__init__.py src/utils/__init__.py src/cli/__init__.py src/agents/__init__.py src/retrieval/__init__.py src/ui/__init__.py tests/__init__.py
git commit -m "chore(modules): initialisation des packages python __init__.py du projet"
```

---

## 9. Scripts de Tests Unitaires (`tests/`)

### 9.1 `tests/smoke_test.py`
```powershell
git add tests/smoke_test.py
git commit -m "test: test de fumee validant l'environnement python et les librairies cles"
```

### 9.2 `tests/test_database.py`
```powershell
git add tests/test_database.py
git commit -m "test: tests unitaires de la connexion postgresql et des sessions sqlalchemy"
```

### 9.3 `tests/test_adzuna.py`
```powershell
git add tests/test_adzuna.py
git commit -m "test: tests unitaires du connecteur api adzuna"
```

### 9.4 `tests/test_jobspy.py`
```powershell
git add tests/test_jobspy.py
git commit -m "test: tests unitaires du scraper jobspy sur indeed et linkedin"
```

### 9.5 `tests/test_france_travail.py`
```powershell
git add tests/test_france_travail.py
git commit -m "test: tests unitaires du connecteur api france travail"
```

### 9.6 `tests/test_indexer.py`
```powershell
git add tests/test_indexer.py
git commit -m "test: tests unitaires de la double persistance postgresql et chromadb"
```

### 9.7 `tests/test_prompt_strategies.py`
```powershell
git add tests/test_prompt_strategies.py
git commit -m "test: tests comparatifs de strategies de prompt pour extracteur llama"
```

---

## 10. Documentation du Projet & README

### 10.1 Documentation technique (`docs/`)
```powershell
git add docs/
git commit -m "docs: ensemble de la documentation technique, architectures et specifications m2 ia"
```

### 10.2 README principal d'installation
```powershell
git add README.md
git commit -m "docs: guide complet d'installation et d'utilisation du projet pour l'equipe"
```

---

## 🚀 Étape Finale : Pousser la Branche sur GitHub

Une fois que vous avez exécuté ces commits, envoyez votre historique propre sur GitHub :

```powershell
git push -u origin phase1_scrappe_offre
```

Vos collègues et enseignants verront un historique GitHub **irréprochable**, où chaque script dispose de son propre commit expliquant sa fonction exacte !
