"""
tests/test_gliner_multisector.py
Validation de l'extracteur GLiNER multi-secteurs sur des offres réelles :
- IT & Data
- Santé & Médical
- Finance & Comptabilité
- Industrie & BTP
Vérifie le respect du principe 0 hallucination (termes présents verbatim dans le texte).
"""

import sys
import logging
from pathlib import Path

# Encodage console Windows UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ajout de la racine au PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def test_gliner_multisector():
    from src.parsers.gliner_extractor import GlinerSkillExtractor, JobRequirementsHeuristic
    
    print("\n" + "="*70)
    print("TEST DE CHARGEMENT DU SINGLETON GLiNER")
    print("="*70)
    extractor = GlinerSkillExtractor.get_instance()
    assert extractor.model is not None, "Le modèle GLiNER n'a pas pu être chargé."
    print("--> GLiNER Model chargé avec succès.")

    samples = [
        {
            "secteur": "it_data",
            "titre": "Data Engineer Senior",
            "text": """
            Nous recherchons un Data Engineer (H/F).
            Vous maîtrisez Python, SQL et l'écosystème AWS (S3, Redshift).
            Une expérience solide avec Apache Spark, Docker et Kubernetes est indispensable.
            Diplôme : Bac +5 (école d'ingénieur ou Master) avec minimum 3 ans d'expérience.
            """,
            "expected_skills": ["Python", "SQL", "AWS", "Spark", "Docker", "Kubernetes"],
            "expected_bac": 5,
            "expected_exp": 3.0
        },
        {
            "secteur": "sante_medical",
            "titre": "Infirmier Diplômé d'État (IDE)",
            "text": """
            Établissement hospitalier recrute un Infirmier en soins généraux (IDE).
            Missions : administration des perfusions, pansements complexes, pose de cathéter
            et surveillance des constantes vitales via le logiciel Dossier Patient Informatisé (DPI).
            Profil : Diplôme d'État d'Infirmier requis (Bac+3), première expérience acceptée.
            """,
            "expected_skills_contain": ["soins", "perfusion", "cathéter"],
            "expected_bac": 3,
            "expected_exp": 0.0
        },
        {
            "secteur": "finance_compta",
            "titre": "Comptable Général",
            "text": """
            Cabinet d'expertise recrute un Comptable Général confirmé.
            Maîtrise indispensable de Sage 100, Cegid et Excel avancé.
            Connaissance approfondie des normes IFRS, des déclarations fiscales (TVA, liasse fiscale)
            et du bilan comptable. Bac+2 (BTS Comptabilité ou DUT GEA) requis, au moins 4 ans d'expérience.
            """,
            "expected_skills_contain": ["Sage", "Cegid", "IFRS", "TVA"],
            "expected_bac": 2,
            "expected_exp": 4.0
        },
        {
            "secteur": "industrie_btp",
            "titre": "Technicien de Maintenance Industrielle",
            "text": """
            Site industriel recherche un Technicien de Maintenance.
            Missions de diagnostic électromécanique, lecture de schémas électriques et programmation d'automates Siemens.
            Habilitation électrique BR/BC et CACES 3 indispensables.
            Niveau Bac+2 exigé avec 2 ans d'expérience souhaitée.
            """,
            "expected_skills_contain": ["Siemens", "CACES", "BR/BC", "automates"],
            "expected_bac": 2,
            "expected_exp": 2.0
        }
    ]

    print("\n" + "="*70)
    print("BENCHMARK MULTI-SECTEURS ET VERIFICATION D'ANTI-HALLUCINATION")
    print("="*70)

    for i, s in enumerate(samples, 1):
        secteur = s["secteur"]
        print(f"\n[{i}/4] Secteur: {secteur.upper()} - {s['titre']}")
        
        # Test extraction compétences GLiNER
        skills = extractor.extract_skills(s["text"], secteur=secteur)
        print(f"  * Compétences extraites ({len(skills)}) : {skills}")
        
        # Vérification 0 hallucination (chaque compétence doit être une sous-chaîne du texte original)
        text_lower = s["text"].lower()
        for sk in skills:
            assert sk.lower() in text_lower, f"HALLUCINATION DÉTECTÉE ! '{sk}' n'est pas dans le texte original."
        print("  * Garantie 0 hallucination : VALIDÉ (toutes les compétences sont des sous-chaînes verbatim)")

        # Test Heuristique Bac & Expérience
        bac = JobRequirementsHeuristic.extract_diploma(s["text"])
        exp = JobRequirementsHeuristic.extract_experience(s["text"])
        print(f"  * Niveau Bac détecté : Bac+{bac} (attendu Bac+{s['expected_bac']})")
        print(f"  * Expérience détectée: {exp} ans (attendu {s['expected_exp']} ans)")
        assert bac == s["expected_bac"], f"Erreur Bac: {bac} != {s['expected_bac']}"
        assert exp == s["expected_exp"], f"Erreur Exp: {exp} != {s['expected_exp']}"

    print("\n" + "="*70)
    print("TOUS LES TESTS MULTI-SECTEURS ONT REUSSI SANS AUCUNE HALLUCINATION !")
    print("="*70)

if __name__ == "__main__":
    test_gliner_multisector()
