# Spécification Détaillée du Pipeline Technique
> Projet Master 2 IA — Pipeline de Recommandation Sémantique

Ce document spécifie chaque phase de la chaîne de traitement, les algorithmes sous-jacents, les formats de données échangés et la formule de scoring hybride.

---

## Vue d'ensemble du Pipeline

Le système fonctionne selon deux cycles complémentaires :
1. **Cycle Asynchrone / Batch (Phase 0)** : Collecte continue, déduplication, normalisation et indexation des offres d'emploi (PostgreSQL + ChromaDB).
2. **Cycle Temps Réel / Online (Phases 1 à 6)** : Analyse du CV candidat, recherche sémantique, reranking et explicabilité.

```
+----------------------------------------------------------------------------------------------------+
|                      CYCLE BATCH : COLLECTE & INDEXATION DES OFFRES                               |
+----------------------------------------------------------------------------------------------------+
  [Plateformes & APIs d'emploi (France Travail, LinkedIn/Indeed via JobSpy, Adzuna)]
        |
        v
  Phase 0 : INGESTION & DÉDUPLICATION
        |-----> Option 1 : API France Travail (Officielle Open Data JSON)
        |-----> Option 2 : Scraper Multi-plateformes JobSpy (LinkedIn, Indeed)
        |-----> Option 3 : APIs Agrégatrices Ouvertes (Adzuna, Remotive)
        |-----> Déduplication par empreinte MD5(company + poste + lieu)
        v
  NORMALISATION & DOUBLE STOCKAGE (Dual Storage)
        |-----> PostgreSQL : Données exhaustives (table 'offres')
        |-----> ChromaDB : Vecteur sémantique dense + Métadonnées de pré-filtrage

+----------------------------------------------------------------------------------------------------+
|                      CYCLE ONLINE : RECOMMANDATION & EXPLICABILITÉ DU CV                           |
+----------------------------------------------------------------------------------------------------+
  [CV Candidat PDF/DOCX]
        |
        v
  Phase 1 : EXTRACTION & LAYOUT ANALYSIS (Docling)
        |-----> Sortie : Markdown hiérarchique préservant l'ordre de lecture
        v
  Phase 2 : EXTRACTION STRUCTURÉE (LLM + Pydantic)
        |-----> Profil normalisé (Compétences techniques, Années d'exp, Formations, Contrat)
        |-----> Stockage dans PostgreSQL (table 'cv_users')
        v
  Phase 3 : PRÉ-FILTRAGE STRICT (Métadonnées ChromaDB)
        |-----> Élimination des offres incompatibles (Type contrat, Niveau académique min)
        v
  Phase 4 : RECHERCHE HYBRIDE (ChromaDB Dense + BM25 Sparse)
        |-----> Fusion par Reciprocal Rank Fusion (RRF) ou combinaison linéaire -> Top 30
        v
  Phase 5 : RERANKING NEURONAL (Cross-Encoder BAAI/bge-reranker-v2-m3)
        |-----> Attention croisée fine Token-à-Token -> Top 5
        v
  Phase 6 : AGENT D'EXPLICABILITÉ (LLM LangChain)
        |-----> Score consolidé, compétences validées, gaps et conseils
        |-----> Sauvegarde dans PostgreSQL (table 'evaluations')
        v
  [Tableau de bord Recommandations & Restitution]
```

---

## Phase 0 : Ingestion, Scraping & Indexation des Offres d'Emploi

Cette phase alimente le catalogue d'offres de façon modulaire via 3 options progressives.

### 0.1 Les 3 Stratégies de Collecte d'Offres (1 par 1)

#### Option 1 : API France Travail (Officielle Open Data — Priorité 1)
* **Accès** : Inscription sur [francetravail.io](https://francetravail.io) (Client ID / Client Secret gratuits).
* **Protocole** : Authentification OAuth2 (`client_credentials`) -> Requête REST `/offres/search`.
* **Avantages** : Flux JSON officiel, très haute disponibilité, aucun risque de blocage d'IP, champs standardisés (ROME, compétences, contrat, géolocalisation).

#### Option 2 : Scraper Multi-Plateformes avec `JobSpy` (LinkedIn & Indeed — Priorité 2)
* **Outil** : Librairie Python `python-jobspy` (`jobspy.scrape_jobs`).
* **Fonctionnement** : Requêtes HTTP optimisées contournant les sessions fermées de LinkedIn et Indeed sans infrastructure lourde Selenium/Playwright.
* **Paramètres** : Termes cibles (`"Data Scientist"`, `"Machine Learning"`, `"Ingénieur IA"`), géolocalisation (`"Paris, France"` ou `"France"`), et type de contrat (`"fulltime"`, `"internship"`).

#### Option 3 : APIs Agrégatrices Ouvertes (Adzuna & Remotive — Priorité 3)
* **Adzuna API** : API REST couvrant des millions d'offres en France avec un quota gratuit généreux.
* **Remotive API** : API publique spécialisée dans les postes tech / IA 100% Remote.

### 0.2 Pipeline de Déduplication & Normalisation

Une même offre étant souvent multi-diffusée sur plusieurs canaux, un filtre de déduplication strict est appliqué avant tout stockage :

1. **Calcul du Hash Unique d'Offre** :
   ```python
   import hashlib

   def compute_job_hash(company: str, poste: str, location: str) -> str:
       raw = f"{company.strip().lower()}_{poste.strip().lower()}_{location.strip().lower()}"
       return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
   ```
2. **Vérification d'existence** :
   - Si `id` (ou `job_hash`) existe déjà dans la table `offres` de PostgreSQL, on met simplement à jour le champ `created_at` / `is_active` sans recalculer le vecteur dans ChromaDB.
   - Si l'offre est nouvelle, on procède à l'extraction structurée et au double stockage.

### 0.3 Double Stockage (Dual Storage)
* **PostgreSQL (`offres`)** :
  - Clé primaire : `id` (`job_hash`)
  - Données complètes : `poste`, `company`, `type_de_contrat`, `niveau_etude_min`, `min_experience_years`, `competences_necessaires` (`TEXT[]`), `description`, `mission`.
* **ChromaDB (`collection: offres`)** :
  - Identifiant vecteur : `id` (strictement identique au `job_hash` de PostgreSQL).
  - Document pour embedding : Texte synthétique optimisé (`Poste + Missions + Compétences requises`).
  - Métadonnées pour pré-filtrage : `{"job_id": id, "type_de_contrat": contrat, "niveau_etude_min": int, "is_active": True}`.

---

## Phase 1 : Extraction Documentaire avec Docling

### 1.1 Fonctionnement
Docling traite le document PDF/DOCX en utilisant un modèle de détection d'objets (layout analysis) qui identifie :
- Les en-têtes hiérarchiques (`H1`, `H2`, `H3`)
- Les colonnes distinctes (lecture séquentielle correcte gauche/droite)
- Les listes à puces et tableaux de compétences

### 1.2 Exemple de conversion Docling

**Entrée (PDF 2 colonnes) :**
- Colonne de gauche : Contact, Compétences (Python, PyTorch, Docker)
- Colonne de droite : Expérience (2024 - Data Scientist chez X : conception de RAG...)

**Sortie Docling (Markdown généré) :**
```markdown
# Jean DUPONT
*Ingénieur Machine Learning & NLP*
Contact : jean.dupont@email.com | Paris, France

## Compétences Techniques
- **Langages** : Python, SQL, C++
- **Frameworks & Libs** : PyTorch, Hugging Face, LangChain, FastAPI
- **Cloud & MLOps** : Docker, MLflow, AWS, ChromaDB

## Expériences Professionnelles
### Data Scientist NLP | TechCorp (2024 - Présent)
- Développement d'un moteur RAG pour l'analyse de documents juridiques.
- Fine-tuning de modèles Mistral 7B sur données métiers.

### Stagiaire IA | Lab Recherche (2023 - 6 mois)
- Implémentation d'architectures Transformers pour l'extraction d'entités nommées.

## Formation
- **Master 2 Intelligence Artificielle** | Université Paris-Saclay (2023 - 2024)
```

---

## Phase 2 : Extraction Structurée par LLM (Information Extraction)

Bien que le Markdown soit exploitable pour l'embedding global, une recommandation précise exige de connaître les critères discriminants (années d'expérience, niveau d'études, hard skills).

### 2.1 Invocation avec Pydantic Output Parser
Un prompt dédié extrait le schéma Pydantic `CandidateProfile` :
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class CandidateProfile(BaseModel):
    full_name: str = Field(description="Nom complet du candidat")
    target_role: str = Field(description="Poste ciblé ou intitulé principal")
    total_experience_years: float = Field(description="Nombre estimé d'années d'expérience professionnelle")
    hard_skills: List[str] = Field(description="Compétences techniques clés (ex: Python, PyTorch, Docker)")
    soft_skills: List[str] = Field(description="Compétences relationnelles ou managériales")
    education_level: str = Field(description="Niveau d'études le plus élevé (ex: Bac+5, Doctorat)")
    spoken_languages: List[str] = Field(description="Langues maîtrisées")
    desired_contract: Optional[str] = Field(description="Type de contrat recherché (CDI, Stage, Alternance)")
    location: Optional[str] = Field(description="Localisation ou mobilité géographique")
```

---

## Phase 3 : Indexation et Représentation des Offres d'Emploi

Pour que la recherche vectorielle soit performante, chaque offre d'emploi est découpée et représentée sous deux formes dans le système :

1. **Représentation Synthétique pour l'Embedding (Dense Vector)** :
   Une concaténation ciblée des éléments clés pour former un texte à forte densité sémantique :
   ```
   Poste : {title} | Entreprise : {company} | Contrat : {contract_type} | Lieu : {location}
   Missions principales : {missions}
   Compétences requises : {required_skills}
   Profil recherché : {profile_description}
   ```
2. **Métadonnées de Filtrage (ChromaDB Metadata)** :
   ```json
   {
     "job_id": "job_042",
     "contract_type": "CDI",
     "location": "Paris",
     "remote_policy": "Hybrid",
     "min_experience_years": 2,
     "category": "Data / AI"
   }
   ```
3. **Index Lexical (BM25)** :
   L'ensemble des documents textuels est tokenisé et indexé dans une structure de type fréquence inversée de documents (TF-IDF / BM25).

---

## Phase 4 : Stratégie de Matching en 3 Étapes

Le matching naïf (uniquement similarité cosinus) est insuffisant. Nous déployons une stratégie éprouvée en 3 étapes :

```
[Espace des Offres (ex: 5 000 offres)]
          |
          |  Étape 1 : Pré-filtrage strict (Contrat, Lieu si spécifié)
          v
[Offres éligibles (ex: 800 offres)]
          |
          |  Étape 2 : Recherche Hybride (BGE-M3 Dense + BM25 Sparse)
          v
[Top-30 Offres candidates]
          |
          |  Étape 3 : Reranking par Cross-Encoder (bge-reranker-v2-m3)
          v
[Top-5 Recommandations Finales]
```

### 4.1 Étape 1 : Pré-filtrage Métadonnées (Hard Filtering)
Si le candidat recherche expressément un **Stage** ou une **Alternance**, il est absurde de lui proposer un poste senior en CDI de 8 ans d'expérience.
- Requête ChromaDB / SQL :
  `where = {"contract_type": {"$in": ["Stage", "Indifférent"]}}`

### 4.2 Étape 2 : Recherche Hybride (Dense + Sparse)
Deux requêtes sont formulées à partir du profil candidat :
1. **Requête Dense ($Q_{dense}$)** : Le résumé sémantique du profil (titre cible + principales expériences + compétences).
   $$S_{dense} = \cos(E(Q_{dense}), E(\text{Job})) = \frac{E(Q_{dense}) \cdot E(\text{Job})}{\|E(Q_{dense})\| \|E(\text{Job})\|}$$
2. **Requête Sparse ($Q_{sparse}$)** : La liste exhaustive des Hard Skills extraits (ex: `"Python PyTorch LangChain Docker MLOps NLP"`).
   $$S_{BM25} = \text{Score}_{BM25}(Q_{sparse}, \text{Job})$$

**Méthode de Fusion : Reciprocal Rank Fusion (RRF)**
Pour éviter de normaliser des échelles de scores hétérogènes :
$$RRF(d) = \frac{\alpha}{k + \text{rank}_{dense}(d)} + \frac{1 - \alpha}{k + \text{rank}_{sparse}(d)}$$
*(avec $k = 60$ et $\alpha = 0.65$)*.
On conserve le **Top 30**.

### 4.3 Étape 3 : Reranking Neuronal (Cross-Encoder)
Le modèle de Reranking `BAAI/bge-reranker-v2-m3` prend en entrée :
- **Texte A** : Synthèse du CV du candidat (Expériences clés + Compétences).
- **Texte B** : Description complète de l'offre (Missions + Profil exigé).

Le Cross-Encoder applique une attention complète bidirectionnelle ($L \times L$) entre tous les mots du CV et tous les mots de l'offre.
Il en résulte un score de pertinence $S_{rerank} \in [0, 1]$ bien plus fin que le simple produit scalaire d'embeddings.
On extrait le **Top 5**.

---

## Phase 5 : Agent d'Explicabilité & Scoring Final

Un système de recommandation en recrutement doit impérativement être **transparent et explicable** (IA explicable / XAI) pour aider le candidat à comprendre la décision.

### 5.1 Structure de la Restitution
Pour chaque offre du Top 5, le LLM évalue le match et retourne :
1. **Score Global d'Affinité (0 à 100%)** : Combinaison pondérée du score de Reranking et de la complétion des exigences.
2. **Forces du Candidat (Skills Matchés)** : Les compétences et expériences du CV qui correspondent directement aux besoins de l'offre.
3. **Points de Vigilance (Skill Gaps)** : Les prérequis mentionnés dans l'offre mais absents ou faibles dans le CV.
4. **Conseil Personnalisé** : Recommandation sur les projets ou arguments à mettre en avant dans une lettre de motivation ou un entretien.

### 5.2 Schéma JSON de Sortie de l'Agent d'Explicabilité
```json
{
  "job_id": "job_042",
  "job_title": "Ingénieur RAG & NLP",
  "company": "TechSolutions",
  "match_score": 88,
  "match_status": "Forte adéquation",
  "matched_skills": [
    "Maîtrise de Python et des bibliothèques NLP (Transformers, LangChain)",
    "Expérience démontrée sur les architectures RAG et vector stores (ChromaDB)",
    "Niveau de formation Master 2 IA en adéquation avec les attentes"
  ],
  "missing_skills": [
    "Expérience requise sur Kubernetes non mentionnée dans le CV",
    "Certification Cloud (AWS/GCP) appréciée mais non présente"
  ],
  "application_tips": "Mettez en valeur votre projet de M2 sur les moteurs RAG lors de votre entretien, et soulignez votre capacité d'apprentissage rapide sur les outils de conteneurisation comme Kubernetes."
}
```

---

### 5.3 Persistance & Télémétrie dans PostgreSQL (`evaluations`)

Chaque recommandation générée pour un couple `(cv_user_id, offre_id)` est sauvegardée dans la table relationnelle `evaluations` :

1. **Affichage Frontend (UI)** : L'utilisateur ne voit **que** le `final_score` (ex: 88%) et la restitution textuelle pour une clarté optimale.
2. **Conservation des 3 scores intermédiaires (`dense_score`, `bm25_score`, `reranker_score`) en base** :
   - **Débogage immédiat** : Identifier instantanément si une mauvaise recommandation provient de l'embedding, du manque de mots-clés ou du Reranker.
   - **Ablation Study (Soutenance M2)** : Permet d'extraire en une requête SQL la distribution des scores pour prouver scientifiquement l'apport de chaque brique.
   - **Recalibrage des poids à chaud** : Possibilité de modifier la pondération finale directement en SQL sans re-lancer les modèles GPU.
