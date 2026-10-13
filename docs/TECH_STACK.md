# Choix Technologiques & Analyse d'Expertise
> Projet Master 2 IA — Analyse critique de la stack technique

Ce document détaille et justifie les choix technologiques pour la chaîne de traitement, du document brut au score de recommandation explicable.

---

## 1. Extraction et Parsing de Documents : Pourquoi Docling ?

### 1.1 Le défi des CVs modernes
Les CVs sont parmi les documents PDF les plus difficiles à analyser :
- **Mise en page multi-colonnes** : Les parseurs naïfs (`pypdf`, `pdfminer`, `PyMuPDF`) lisent souvent horizontalement d'un bord à l'autre de la page, mélangeant la colonne latérale (compétences, contacts) avec le corps principal (expériences professionnelles).
- **Éléments graphiques et tableaux** : Les compétences sont souvent réparties sous forme de grilles, jauges ou tableaux sans bordures.
- **Formattage hétérogène** : PDF vectoriels, DOCX, scans OCR.

### 1.2 Benchmark des solutions

| Outil | Maintien du Layout | Vitesse | Multi-colonnes | Sortie Markdown propre | Coût / Licence |
|---|---|---|---|---|---|
| **Docling (IBM Research)** | **Excellente (LayoutLM/Vision)** | Rapide (CPU/GPU) | **Oui (Ordre de lecture restauré)** | **Oui (Markdown hiérarchique)** | Open Source (MIT) |
| **Unstructured.io** | Bonne | Moyenne | Partiel | Partiel | Open Source / API payante |
| **PyMuPDF / pdfplumber** | Basique (Boîtes de texte) | Très rapide | Non (mélange les colonnes) | Non (texte brut ou HTML basique) | Open Source (AGPL/LGPL) |
| **Marker (VikParuchuri)** | Très bonne | Lente (GPU requis) | Oui | Oui | Open Source (GPL-3.0) |
| **OCR Tesseract brut** | Nulle | Lente | Non | Non | Open Source |

### 1.3 Avantages clés de Docling pour ce projet
- **Modèle de vision spécialisé dans le Document Layout Analysis** : Docling détecte nativement les sections, titres (`#`, `##`), listes à puces et tableaux.
- **Export Markdown structuré** : Le Markdown est le format idéal pour alimenter directement un LLM ou un chunker sémantique car il conserve les délimitations logiques sans le bruit du HTML ou du XML.
- **Support multiformat** : Traite nativement les PDF, DOCX et images.
- **Intégration LangChain** : Docling possède un loader officiel (`langchain-docling`) facilitant l'injection dans le pipeline.

---

## 2. Bases de Données : Approche Hybride Vectorielle + Relationnelle

Dans un système de recommandation de postes, **une base purement vectorielle ne suffit pas**. Il faut combiner une base vectorielle et une base relationnelle (**Dual Storage**).

```mermaid
flowchart TD
    subgraph INTAKE["1. Documents Sources"]
        CV_IN["CV Candidat (PDF/DOCX)"]
        JOB_IN["Offres d'emploi (JSON/Scraper)"]
    end

    subgraph STRUCT["2. Parsing & Structuration"]
        CV_IN --> DOCLING_PARSER["Docling + Pydantic"]
        JOB_IN --> JOB_PARSER["Nettoyeur & Validateur Pydantic"]
    end

    subgraph POSTGRES["3. PostgreSQL (Source de Vérité)"]
        DOCLING_PARSER -->|Insert/Update| TBL_CV[("Table: cv_users\n- id PK\n- desired_contract_types TEXT[]\n- parcours_academique INT (3, 4, 5)\n- competences TEXT[]\n- experiences_pro JSONB\n- experiences_univ JSONB\n- raw_markdown TEXT")]
        JOB_PARSER -->|Insert/Update| TBL_JOB[("Table: offres\n- id PK (ex: job_042)\n- poste VARCHAR\n- type_de_contrat VARCHAR\n- niveau_etude_min INT\n- competences_necessaires TEXT[]\n- description & mission TEXT")]
    end

    subgraph EMBED_PROC["4. Vectorisation & Indexation"]
        JOB_PARSER --> TEXT_BUILDER["Concaténation sémantique :\nPoste + Missions + Compétences"]
        TEXT_BUILDER --> EMBEDDER["Modèle d'Embedding (BGE-M3)"]
        EMBEDDER -->|Vector + Metadata| CHROMA_DB[("ChromaDB (Index Vectoriel)\n- id = job_042\n- document: texte concaténé\n- metadata: contrat, niveau_min, xp_min")]
    end

    subgraph MATCH_FLOW["5. Requête & Réconciliation"]
        TBL_CV -.->|Critères stricts & Vecteur requête| QUERY_ENGINE["Moteur de Matching"]
        CHROMA_DB -.->|Top-30 job_ids| QUERY_ENGINE
        QUERY_ENGINE -->|SELECT * WHERE id IN (top_ids)| TBL_JOB
        QUERY_ENGINE --> TBL_MATCH[("Table: evaluations\n- cv_user_id FK\n- offre_id FK\n- scores & justifications")]
    end
```

### 2.1 Base Vectorielle : ChromaDB
- **Rôle exclusif** : Moteur de recherche par proximité sémantique (similarité cosinus dans un espace vectoriel dense à 1024 dimensions).
- **Pourquoi ChromaDB ?**
  - **Légère et embarquable** : S'exécute in-process en mode local (`PersistentClient`) sans nécessiter un cluster dédié (idéal pour prototypage, TP, démonstrateur M2 IA).
  - **Filtres sur métadonnées** : Permet de combiner recherche par similarité cosinus avec des filtres WHERE stricts (ex. `{"type_de_contrat": {"$in": ["CDI", "Stage"]}, "niveau_etude_min": {"$lte": 5}}`).
  - **Compatibilité LangChain native** : Intégration directe avec `langchain-chroma`.
- *Ce que ChromaDB ne fait pas* : Il ne stocke pas les données brutes lourdes, ne gère pas l'intégrité référentielle, et ne permet pas d'historiser les candidatures.

### 2.2 Base Relationnelle : PostgreSQL
- **Rôle exclusif** : Source unique de vérité (Single Source of Truth), intégrité relationnelle, stockage structuré et auditabilité.
- **Pourquoi PostgreSQL ?**
  - **Support natif du JSONB** : Permet de stocker les listes d'expériences professionnelles et académiques tout en conservant la capacité d'interrogation et d'indexation GIN.
  - **Types Array (`TEXT[]`)** : Parfait pour les compétences (`competences` et `competences_necessaires`), permettant des requêtes d'intersection ultra-rapides (`@>`).
  - **Normalisation numérique** : Stockage du parcours académique en entier (`3` pour Bac+3, `5` pour Bac+5) pour des filtres mathématiques (`niveau_etude_min <= parcours_academique`).
  - **ORM** : **SQLAlchemy 2.0** pour mapper directement les modèles Pydantic aux tables Postgres.

---

## 3. Orchestration & LLM : LangChain & LangGraph

### 3.1 Rôle de LangChain dans ce projet
Plutôt que d'utiliser LangChain comme une "boîte noire", nous l'utilisons pour ses abstractions à forte valeur ajoutée :
1. **Document Loaders & Splitters** : Connexion fluide avec Docling et découpage de texte conscient des en-têtes Markdown (`MarkdownHeaderTextSplitter`).
2. **LCEL (LangChain Expression Language)** : Chaînage déclaratif et asynchrone des étapes (Prompt -> LLM -> OutputParser).
3. **Structured Outputs (PydanticOutputParser / `with_structured_output`)** : Extraction fiable et typée du profil candidat depuis le Markdown brut du CV.
4. **LangGraph (Optionnel mais recommandé pour le M2)** : Modélisation du pipeline sous forme de graphe d'états (StateGraph) :
   - Noeud 1 : *Extract Profile*
   - Noeud 2 : *Query Expansion / Reformulation*
   - Noeud 3 : *Hybrid Retrieval*
   - Noeud 4 : *Reranking*
   - Noeud 5 : *Explainability Agent*

---

## 4. Modèles d'Embedding & Stratégie de Matching

### 4.1 Modèles d'Embedding (Dense Retrieval)
Le marché de l'emploi en France nécessite des modèles performants sur le **français technique** et le vocabulaire IT/IA.

| Modèle | Dimensions | Multilingue | Rendu sur le français | Contexte max | Empreinte / Déploiement |
|---|---|---|---|---|---|
| **`BAAI/bge-m3`** (Recommandé) | 1024 | Oui (100+ langues) | **Exceptionnel** (Dense + Sparse + ColBERT) | 8192 tokens | Modéré (~2.2 Go GPU/CPU) |
| **`intfloat/multilingual-e5-large`** | 1024 | Oui | Très bon | 512 tokens | Modéré (~1.5 Go) |
| **`text-embedding-3-small` (OpenAI)** | 1536 | Oui | Bon | 8191 tokens | API propriétaire payante |
| **`mistral-embed` (Mistral AI)** | 1024 | Oui (Excellence FR) | Excellent | 8192 tokens | API propriétaire |

> **Recommandation projet :**
> - **Option Open-Source locale :** `BAAI/bge-m3` via HuggingFace / SentenceTransformers. C'est l'état de l'art pour capturer des textes longs de CV tout en gérant les acronymes techniques.
> - **Option API économique/rapide :** Mistral Embed ou OpenAI text-embedding-3-small.

### 4.2 Nécessité de la Recherche Hybride (Dense + Sparse/BM25)
Un embedding dense seul souffre du "problème des compétences précises" :
- Pour un modèle d'embedding, "Angular" et "React" sont sémantiquement très proches (frameworks frontend JavaScript).
- Or, si une offre exige impérativement **React** et 3 ans d'expérience, un profil 100% **Angular** risque d'avoir un score cosinus élevé alors qu'il ne coche pas le prérequis strict.
- **Solution : Recherche Hybride (Ensemble Retriever dans LangChain)** :
  - `Score = α * Score_Dense(BGE-M3) + (1 - α) * Score_Sparse(BM25)` avec typiquement $\alpha = 0.65$ ou $0.7$.

### 4.3 Le Reranker (Cross-Encoder) : La botte secrète
Dans un Bi-Encoder (recherche vectorielle classique), le CV et l'Offre sont encodés séparément : $cos(E(CV), E(Job))$.
Un **Cross-Encoder** prend en entrée la paire complète $(CV, Job)$ et applique l'attention croisée token par token.
- Modèle recommandé : **`BAAI/bge-reranker-v2-m3`** ou **Cohere Rerank**.
- Permet de faire passer le Top-30 retourné par ChromaDB au crible fin pour ressortir le Top-5 ultra-pertinent.

---

## 5. Synthèse de la Stack Technique

| Rôle | Outil Retenu | Alternatives considérées |
|---|---|---|
| **Parser CV (Document AI)** | **Docling (IBM Research)** | PyMuPDF, Unstructured, Marker |
| **Runtime IA Local** | **Ollama (`localhost:11434`)** | vLLM, HuggingFace Local |
| **Modèle d'Embedding** | **BGE-M3 (via Ollama local)** | Sentence-Transformers, OpenAI |
| **Base Vectorielle** | **ChromaDB (Index HNSW)** | Qdrant, Milvus, pgvector |
| **Base Relationnelle** | **PostgreSQL (`projet_cv_db`)** | SQLite, MySQL |
| **Framework LLM & Pipeline** | **LangChain + LangGraph** | LlamaIndex, Haystack, DSPy |
| **Validation de données** | **Pydantic v2** | Marshmallow, dataclasses |
| **Moteur Lexical** | **Rank-BM25** | Elasticsearch, Meilisearch |
| **Reranker** | **BAAI/bge-reranker-v2-m3** | Cross-Encoder MS-Marco |
| **LLMs Locaux (Extraction / XAI)** | **Ollama (`llama3.2:3b` & `mistral:latest`)** | APIs Cloud payantes |
| **Backend & Interface** | **FastAPI + React (`npm run dev`)** | Streamlit, Gradio |
