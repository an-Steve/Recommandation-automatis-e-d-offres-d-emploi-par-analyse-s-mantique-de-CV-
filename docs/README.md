# Recommandation Automatisée d'Offres d'Emploi par Analyse Sémantique de CV
> Projet Master 2 IA — Spécifications, Architecture & Pipeline Technique

Ce dossier contient l'ensemble de la documentation de conception et d'architecture pour le projet de matching sémantique intelligent entre Curriculum Vitae (CV) et Offres d'emploi.

---

## 📚 Sommaire de la Documentation

1. [**Architecture Globale (`ARCHITECTURE.md`)**](file:///C:/MASTER2_IA/Projet_CV/docs/ARCHITECTURE.md)
   - Schémas d'architecture (Mermaid)
   - Flux de données (Dataflow) de bout en bout
   - Découpage modulaire et responsabilités

2. [**Choix Technologiques & Benchmark (`TECH_STACK.md`)**](file:///C:/MASTER2_IA/Projet_CV/docs/TECH_STACK.md)
   - Justification de **Docling** pour l'extraction de documents
   - Pourquoi et comment utiliser **ChromaDB** & bases relationnelles
   - Place et rôle de **LangChain**
   - Sélection des modèles d'Embedding (Multilingue/Français) et Cross-Encoders (Reranking)
   - Analyse comparative des solutions alternatives

3. [**Spécification Détaillée du Pipeline (`PIPELINE_SPEC.md`)**](file:///C:/MASTER2_IA/Projet_CV/docs/PIPELINE_SPEC.md)
   - Phase 1 : Ingestion et parsing de CV (Docling + Chunking sémantique)
   - Phase 2 : Extraction d'entités structurées (Pydantic / LLM Information Extraction)
   - Phase 3 : Indexation hybride des offres d'emploi (Dense + BM25)
   - Phase 4 : Stratégie de Matching en 3 étapes (Filtre dur -> Recherche hybride -> Reranker)
   - Phase 5 : Synthèse explicative & calcul d'affinité (Explainable AI)

4. [**Modèle de Données & Schémas (`DATA_MODEL.md`)**](file:///C:/MASTER2_IA/Projet_CV/docs/DATA_MODEL.md)
   - Schémas Pydantic pour le CV et l'Offre
   - Métadonnées vectorielles pour ChromaDB
   - Schéma relationnel (SQLite / PostgreSQL) pour les utilisateurs et l'historique de matching

5. [**Feuille de Route & Jalons (`ROADMAP.md`)**](file:///C:/MASTER2_IA/Projet_CV/docs/ROADMAP.md)
   - Découpage par étapes (Sprints / Jalons M2 IA)
   - Métriques d'évaluation (Recall@K, MRR, nDCG, latence, précision des extractions)
   - Démonstrateur attendu (API FastAPI + Interface Streamlit)

6. [**Guide de Réalisation Phase 0 (`phase0.md`)**](file:///C:/MASTER2_IA/Projet_CV/docs/phase0.md)
   - Création et activation de l'environnement virtuel (`venv` sous Windows)
   - Installation des dépendances (`requirements.txt`)
   - Initialisation de l'arborescence modulaire (`src/`, `data/`, `tests/`)
   - Configuration (`.gitignore`, `.env.example`, `config.py`) et Smoke Test

7. [**Guide de Réalisation Phase 1 (`phase1.md`)**](file:///C:/MASTER2_IA/Projet_CV/docs/phase1.md)
   - Scraping multi-sources : France Travail (Option 1), JobSpy LinkedIn/Indeed (Option 2), Adzuna (Option 3)
   - Déduplication & normalisation vers Pydantic `JobOffer`
   - Double indexation synchronisée dans PostgreSQL (`offres`) et ChromaDB (`offres`)

8. [**Code de Référence du Socle Base de Données (`DATABASE_IMPLEMENTATION.md`)**](file:///C:/MASTER2_IA/Projet_CV/docs/DATABASE_IMPLEMENTATION.md)
   - Implémentation robuste de `src/config.py`, `db_session.py`, `sql_models.py`, `vector_store.py`
   - Script de test d'intégration `tests/test_database.py`

---

## 🎯 Problématique & Objectifs du Projet

Les systèmes traditionnels de recrutement (ATS classiques) reposent principalement sur la correspondance exacte de mots-clés (regex/keyword matching). Cela engendre de forts biais et des manques :
- Synonymes non reconnus (ex. *Machine Learning Engineer* vs *Data Scientist NLP*).
- Absence de prise en compte du contexte (un candidat peut mentionner "Java" dans une formation passée sans vouloir en faire son métier).
- Structure complexe des CV (colonnes multiples, tableaux, encadrés) souvent détruite par les parseurs PDF basiques (`pypdf`, `pdfminer`).

**La solution proposée** allie :
- L'analyse de document par vision/layout avancée (**Docling**).
- La recherche sémantique dense (**ChromaDB** + Embeddings multilingues).
- La recherche lexicale fine (**BM25**) pour garantir la présence des compétences clés requises.
- Un **Reranker** neuronal pour affiner le top-K.
- Une couche **LLM d'explicabilité** pour justifier le score et guider le candidat.
