# 🛡️ Rapport d'Audit de Robustesse & Performance du Pipeline Recrutement
**Master 2 Intelligence Artificielle** — *Date d'évaluation : 05/11/2026 10:34*

---

## 🏆 Score Global de Robustesse : `76.8 / 100`

```
  Qualité Dataset  : 25.0 / 25 pts
  Joignabilité URLs: 12.5 / 25 pts
  Performance GLiNER: 17.2 / 25 pts
  ChromaDB MRR/Top-K: 22.2 / 25 pts
```

---

## 1. Audit Qualité du Dataset & Complétude
* **Volume total en base** : **829 offres d'emploi**
* **Répartition multi-sources** :
  * `indeed` : 401 offres (48.4%)
  * `linkedin` : 284 offres (34.3%)
  * `wttj` : 77 offres (9.3%)
  * `hellowork` : 67 offres (8.1%)
* **Taux de complétude par colonne** :
  * `poste` : **100.0%**
  * `company` : **100.0%**
  * `location` : **100.0%**
  * `type_de_contrat` : **100.0%**
  * `description` : **99.76%**
  * `mission` : **99.88%**
  * `competences_necessaires` : **100.0%**
  * `source_url` : **100.0%**
* **Isolation Missions vs Description** : **99.88%** des offres ont une section missions explicitement séparée sans redondance.

---

## 2. Test en Direct de Joignabilité des URLs Sources
* **Échantillon vérifié** : 16 URLs (Indeed, LinkedIn, WTTJ, HelloWork)
* **Taux de joignabilité** : **50.0%** (HTTP 200 / Redirections actives)
* **Latence réseau moyenne** : `282.89 ms`

---

## 3. Benchmark d'Inférence & Qualité GLiNER (Zero-Shot)
* **Temps d'inférence moyen** : **`7383.33 ms`** par annonce (p95 : `9723.75 ms`)
* **Nombre moyen de tokens DeBERTa** : **`316.9`** (fenêtre max autorisée : 384 tokens)
* **Taux de troncature** : **`33.33%`** *(0 warning Sentence truncated)*
* **Score de confiance moyen** : **`64.17%`**
* **Répartition des compétences** :
  * Hard Skills techniques : **54**
  * Soft Skills comportementales : **5**

---

## 4. Benchmark de Recherche Vectorielle ChromaDB (FastEmbed BGE-M3)
* **Mean Reciprocal Rank (MRR)** : **`0.812`**
* **Précision Top-1** : **`62.5%`**
* **Précision Top-3** : **`100.0%`**
* **Précision Top-5** : **`100.0%`**
* **Latence de recherche cosinus** : **`463.02 ms`**

---

## 5. Synthèse & Recommandations Architecturales
1. **Zéro Hallucination** : L'approche purement extractive GLiNER garantit 0% d'invention de compétences fantômes.
2. **Symétrie Canonique** : Les documents stockés dans ChromaDB sont calibrés pour s'aligner parfaitement avec les profils candidats générés par Docling en Phase 2.
3. **Pérennité du Scraping** : Le couplage JobSpy (Indeed/LinkedIn) + WTTJ (Algolia) + HelloWork (JSON-LD) assure une résilience optimale sans dépendance à des clés API payantes.
