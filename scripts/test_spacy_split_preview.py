"""
scripts/test_spacy_split_preview.py
Validation visuelle "Avant / Après" du Pipeline d'Extraction en 2 Niveaux :
Niveau 1 : Regex Déterministe (Ancres de sections)
Niveau 2 : spaCy fr_core_news_sm (Analyse grammaticale et verbes d'action)
"""

import sys
import re
from pathlib import Path
import pandas as pd
import spacy

# Force l'encodage UTF-8 pour le terminal Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Chargement du modèle linguistique français spaCy
print("⏳ Chargement du modèle spaCy français (fr_core_news_sm)...")
nlp = spacy.load("fr_core_news_sm")
print("✅ Modèle spaCy prêt.\n")

# Motifs de démarcation formelle pour le Niveau 1 (Regex)
HEADER_MISSIONS = re.compile(
    r"(?:\n|^)\s*(?:[#*_-]{0,4}\s*)?(?:vos missions|missions? principales?|descriptif du poste|ce que vous ferez|le poste au quotidien|vos activit[eé]s)\s*[:\n]",
    re.IGNORECASE
)
HEADER_PROFILE = re.compile(
    r"(?:\n|^)\s*(?:[#*_-]{0,4}\s*)?(?:votre profil|profil recherch[eé]|comp[eé]tences?(?:\s+requises?)?|qualifications?|pr[eé]requis|qui [êe]tes[- ]vous|ce que nous recherchons)\s*[:\n]",
    re.IGNORECASE
)


def split_job_2tiers(raw_text: str) -> dict:
    """
    Sépare 'description' (contexte/rôle) et 'mission' (tâches opérationnelles)
    en 2 niveaux 100% déterministes, sans AUCUNE génération LLM.
    """
    if not raw_text or len(raw_text.strip()) < 80:
        return {
            "tier_used": "court_fallback",
            "description": (raw_text or "").strip(),
            "mission": "Non spécifié dans l'annonce"
        }

    text = raw_text.strip()

    # =========================================================================
    # NIVEAU 1 : Découpage Structurel (Regex)
    # =========================================================================
    match_miss = HEADER_MISSIONS.search(text)
    match_prof = HEADER_PROFILE.search(text)

    if match_miss and match_prof and match_prof.start() > match_miss.end():
        desc_part = text[:match_miss.start()].strip()
        miss_part = text[match_miss.end():match_prof.start()].strip()
        # Si la description initiale était vide, on prend l'en-tête de mission
        final_desc = desc_part if len(desc_part) > 50 else text[:150]
        return {
            "tier_used": "Niveau 1 (Regex Structurelle)",
            "description": final_desc,
            "mission": miss_part
        }

    if match_miss and not match_prof:
        desc_part = text[:match_miss.start()].strip()
        miss_part = text[match_miss.end():].strip()
        return {
            "tier_used": "Niveau 1 (Regex Missions Seules)",
            "description": desc_part if len(desc_part) > 50 else text[:150],
            "mission": miss_part
        }

    # =========================================================================
    # NIVEAU 2 : Analyse Syntaxique & Verbes d'Action (spaCy)
    # =========================================================================
    # Découpage du texte en phrases / propositions
    doc = nlp(text)

    mission_sentences = []
    desc_sentences = []

    # Verbes / Tournures typiques de mission
    action_cues = ["intervenir", "développer", "concevoir", "assurer", "gérer", "participer",
                   "piloter", "animer", "réaliser", "accompagner", "mettre", "contribuer",
                   "coordonner", "veiller", "superviser", "déployer", "analyser", "rédiger"]

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text or len(sent_text) < 15:
            continue

        sent_lower = sent_text.lower()
        is_mission = False

        # Règle A : Ligne commençant par un verbe à l'infinitif (ex: "• Développer les modèles...")
        first_token = None
        for token in sent:
            if not token.is_punct and not token.is_space:
                first_token = token
                break

        if first_token:
            # Vérifie si le premier token est un verbe à l'infinitif
            if first_token.pos_ == "VERB" and "VerbForm=Inf" in first_token.morph:
                is_mission = True

        # Règle B : Tournures idiomatiques d'action ("Vous intervenez", "La mission consiste à", etc.)
        if not is_mission:
            if re.search(r"\b(vous\s+(?:intervenez|accompagnerez|participez|assurez|d[eé]velopperez|g[eé]rez|serez\s+amen[eé]))\b", sent_lower):
                is_mission = True
            elif re.search(r"\b(la\s+mission\s+(?:vise|consiste|a\s+pour\s+objectif)|a\s+ce\s+titre,\s+vos\s+missions)\b", sent_lower):
                is_mission = True
            elif re.search(r"\b(en\s+charge\s+de|responsable\s+de|objectif\s+de\s+la\s+mission)\b", sent_lower):
                is_mission = True

        if is_mission:
            mission_sentences.append(sent_text)
        else:
            desc_sentences.append(sent_text)

    # Si spaCy a isolé des missions d'action concrètes
    if mission_sentences:
        return {
            "tier_used": "Niveau 2 (spaCy NLP Syntaxique)",
            "description": " ".join(desc_sentences).strip() or text[:200],
            "mission": "\n• " + "\n• ".join(mission_sentences)
        }

    # =========================================================================
    # FALLBACK HONNÊTE : Aucune mission opérationnelle dans le texte
    # (ex: pur blabla marketing ou présentation corporate sans tâches)
    # =========================================================================
    return {
        "tier_used": "Fallback Honnête (0 Mission Détectée)",
        "description": text,
        "mission": "Non spécifié dans l'annonce"
    }


def main():
    csv_path = "data-1793707303020.csv"
    if not Path(csv_path).exists():
        print(f"❌ Fichier {csv_path} introuvable.")
        return

    df = pd.read_csv(csv_path)
    # Filtrer les offres qui étaient dupliquées (description == mission)
    duplicated_df = df[df['description'].fillna('') == df['mission'].fillna('')]

    print("=" * 80)
    print(f"🔬 TEST DE DÉMONSTRATION SUR LES OFFRES DUPLIQUÉES ({len(duplicated_df)} au total)")
    print("=" * 80)

    # Prendre un échantillon de 2 Adzuna, 2 Indeed, 2 LinkedIn
    samples = []
    for src in ["adzuna", "indeed", "linkedin"]:
        subset = duplicated_df[duplicated_df['source'] == src]
        if not subset.empty:
            samples.extend(subset.head(2).to_dict(orient="records"))

    for idx, row in enumerate(samples, 1):
        print(f"\n{'='*80}")
        print(f"📌 CAS #{idx} | Source: {row['source'].upper()} | Poste: {row['poste']}")
        print(f"{'='*80}")

        raw_desc = str(row['description'])
        print(f"[AVANT] Longueur texte dupliqué : {len(raw_desc)} caractères")
        print(f"[AVANT] Extrait : {raw_desc[:160]}...\n")

        result = split_job_2tiers(raw_desc)

        print(f"⚙️ MÉTHODE UTILISÉE : {result['tier_used']}")
        print("\n🔹 [APRÈS] COLONNE 'description' (Contexte & Rôle) :")
        desc_preview = result['description'][:300] + ("..." if len(result['description']) > 300 else "")
        print(f"   {desc_preview.replace(chr(10), ' ')}")

        print("\n🎯 [APRÈS] COLONNE 'mission' (Tâches Opérationnelles Extraites) :")
        mission_preview = result['mission'][:400] + ("..." if len(result['mission']) > 400 else "")
        print(f"   {mission_preview}")

    print("\n" + "=" * 80)
    print("🎉 DÉMONSTRATION TERMINÉE : Zéro hallucination, 100% Verbatim d'origine.")
    print("=" * 80)


if __name__ == "__main__":
    main()
