# Guide des Commits Atomiques — Branche `phase1_scrappe_offre`

> **Projet Master 2 IA** — Système de Recommandation d'Offres d'Emploi par Analyse de CV  
> **Branche de travail** : `phase1_scrappe_offre`  
> **Méthodologie** : Commits Atomiques (Convention [Conventional Commits](https://www.conventionalcommits.org/))  

Ce document fournit la liste exacte des commandes `git add` et `git commit` à exécuter pour envoyer votre travail étape par étape sur GitHub. Chaque commit représente une brique logique bien définie, ce qui valorise la qualité de votre code lors de la relecture par vos collègues et évaluateurs.

---

## Préalable : Créer la branche (si pas encore fait)

```powershell
git init
git remote add origin <URL_DU_REPO_GITHUB>
git checkout -b phase1_scrappe_offre
```

---

## 1. Commit 1 — Socle & Configuration du Projet (Phase 0)

* **Rôle** : Met en place les paramètres généraux Pydantic Settings, la gestion des secrets via `.env.example`, la liste des dépendances Python et les règles d'exclusion `.gitignore`.
* **Fichiers** : `.gitignore`, `.env.example`, `requirements.txt`, `src/config.py`, `src/__init__.py`.

```powershell
git add .gitignore .env.example requirements.txt src/config.py src/__init__.py
git commit -m "chore(setup): socle de configuration, dépendances et règles gitignore"
```

---

## 2. Commit 2 — Socle de Base de Données (PostgreSQL & ChromaDB)

* **Rôle** : Implémente la connexion SQLAlchemy, la modélisation relationnelle de la table `offres`, le client persistant ChromaDB (distance cosinus HNSW) et le script d'initialisation automatisé `init_db.py`.
* **Fichiers** : `src/database/db_session.py`, `src/database/sql_models.py`, `src/database/vector_store.py`, `src/database/init_db.py`, `src/database/__init__.py`.

```powershell
git add src/database/db_session.py src/database/sql_models.py src/database/vector_store.py src/database/init_db.py src/database/__init__.py
git commit -m "feat(database): modeles relationnels postgresql, collection chromadb et script init_db"
```

---

## 3. Commit 3 — Client LLM Résilient (Ollama Client)

* **Rôle** : Fournit le connecteur HTTP Ollama local optimisé pour CPU, avec Structured Outputs Pydantic, pénalité anti-répétition (`repeat_penalty`) et auto-réparation chirurgicale des JSON tronqués.
* **Fichiers** : `src/utils/ollama_client.py`, `src/utils/__init__.py`.

```powershell
git add src/utils/ollama_client.py src/utils/__init__.py
git commit -m "feat(utils): client http ollama avec repeat_penalty et auto-reparation json"
```

---

## 4. Commit 4 — Connecteurs de Scraping Multi-Sources

* **Rôle** : Implémente la collecte multi-plateformes unifiée (Adzuna REST API, JobSpy Indeed & LinkedIn, France Travail), le mapping sectoriel et l'orchestrateur de scraping quotidien par date.
* **Fichiers** : `src/scrapers/base_scraper.py`, `src/scrapers/jobspy_scraper.py`, `src/scrapers/adzuna_scraper.py`, `src/scrapers/france_travail_api.py`, `src/scrapers/sector_mapping.py`, `src/scrapers/daily_scraper.py`, `src/scrapers/__init__.py`.

```powershell
git add src/scrapers/base_scraper.py src/scrapers/jobspy_scraper.py src/scrapers/adzuna_scraper.py src/scrapers/france_travail_api.py src/scrapers/sector_mapping.py src/scrapers/daily_scraper.py src/scrapers/__init__.py
git commit -m "feat(scrapers): connecteurs multi-sources jobspy, adzuna et orchestrateur quotidien"
```

---

## 5. Commit 5 — Pipeline de Déduplication, Nettoyage & Gabarit Canonique

* **Rôle** : Ajoute la déduplication MD5, le nettoyeur textuel algorithmique `JobTextSanitizer` (-33% de bruit marketing/RH), le template canonique structuré et l'indexeur ChromaDB incrémental (`index_jobs_by_ids`).
* **Fichiers** : `src/pipeline/deduplication.py`, `src/pipeline/job_normalizer.py`, `src/pipeline/job_indexer.py`, `src/pipeline/__init__.py`.

```powershell
git add src/pipeline/deduplication.py src/pipeline/job_normalizer.py src/pipeline/job_indexer.py src/pipeline/__init__.py
git commit -m "feat(pipeline): deduplication md5, nettoyeur regex et template canonique chromadb"
```

---

## 6. Commit 6 — Enrichissement LLM par Lots & Repos CPU

* **Rôle** : Extraction sémantique Pydantic via Llama 3.2 1B (compétences techniques, Bac+X, expérience), tolérance aux valeurs `null`, et worker batch par blocs de 5 avec pauses thermiques CPU.
* **Fichiers** : `src/parsers/job_enricher.py`, `src/pipeline/batch_enrich_jobs.py`, `src/parsers/__init__.py`.

```powershell
git add src/parsers/job_enricher.py src/pipeline/batch_enrich_jobs.py src/parsers/__init__.py
git commit -m "feat(enrichment): extracteur llama 3.2 1b pydantic et worker batch avec repos thermique cpu"
```

---

## 7. Commit 7 — API REST Flask & Endpoints Opérationnels

* **Rôle** : Point d'entrée de l'application backend (`main.py`), Blueprint REST avec routes de diagnostic, collecte quotidienne, enrichissement par lot, ré-indexation et export d'audit ChromaDB.
* **Fichiers** : `main.py`, `src/api/routes.py`, `src/api/__init__.py`, `src/database/export_chroma.py`.

```powershell
git add main.py src/api/routes.py src/api/__init__.py src/database/export_chroma.py
git commit -m "feat(api): serveur backend flask et routes rest pour scraping et recherche vectorielle"
```

---

## 8. Commit 8 — Scripts de Migration & Benchmarks

* **Rôle** : Scripts de migration de base de données (ajout de colonnes, assainissement rétroactif des descriptions existantes) et scripts de diagnostic CLI.
* **Fichiers** : `src/database/migrate_*.py`, `src/cli/*.py`, `src/agents/__init__.py`, `src/retrieval/__init__.py`, `src/ui/__init__.py`.

```powershell
git add src/database/migrate_add_secteur.py src/database/migrate_add_source.py src/database/migrate_sanitize_jobs.py src/cli/benchmark_prompts.py src/cli/enrich_jobs.py src/cli/ingest_jobs.py src/cli/test_cleaning_benchmark.py src/cli/__init__.py src/agents/__init__.py src/retrieval/__init__.py src/ui/__init__.py
git commit -m "chore(scripts): scripts de migration sql, benchmarks et utilitaires cli"
```

---

## 9. Commit 9 — Tests Unitaires

* **Rôle** : Suites de tests unitaires pour valider les scrapers et la double persistance PostgreSQL / ChromaDB.
* **Fichiers** : `tests/*.py`.

```powershell
git add tests/test_france_travail.py tests/test_jobspy.py tests/test_indexer.py tests/__init__.py
git commit -m "test: suites de tests unitaires pour scrapers et double indexation"
```

---

## 10. Commit 10 — Documentation Exhaustive & README Équipe

* **Rôle** : Spécifications techniques d'architecture, guide de vectorisation canonique structurée, guide d'installation pour les collègues et documentation de Phase 0/1.
* **Fichiers** : `README.md`, `docs/*`.

```powershell
git add README.md docs/
git commit -m "docs: guide complet d'installation equipe, specifications techniques et architectures"
```

---

## 11. Envoi Final sur GitHub

Une fois les 10 commits réalisés, poussez l'ensemble de votre historique propre sur GitHub :

```powershell
git push -u origin phase1_scrappe_offre
```

Sur GitHub, vos collègues verront un historique propre, lisible et professionnel découpé en 10 étapes claires !
