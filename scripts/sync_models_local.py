"""
scripts/sync_models_local.py
Copie ou initialise les modèles NLP (FastEmbed nomic-embed et GLiNER multi-v2.1)
dans le dossier local 'models/' du projet (exclu de Git par .gitignore).
"""

import sys
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import settings

def sync_models():
    print("=" * 70)
    print("📦 SYNCHRONISATION DES MODÈLES NLP DANS LE DOSSIER LOCAL ./models/")
    print("=" * 70)

    models_dir = ROOT_DIR / "models"
    fastembed_dst = settings.fastembed_cache_dir
    hf_dst = settings.huggingface_cache_dir

    fastembed_dst.mkdir(parents=True, exist_ok=True)
    hf_dst.mkdir(parents=True, exist_ok=True)

    user_home = Path.home()
    
    # 1. FastEmbed (nomic-embed-text-v1.5)
    src_fastembed = user_home / ".cache" / "fastembed"
    print(f"\n1. Vérification FastEmbed -> {fastembed_dst}")
    if src_fastembed.exists():
        for item in src_fastembed.iterdir():
            target = fastembed_dst / item.name
            if not target.exists():
                print(f"   📋 Copie de {item.name} vers ./models/fastembed/...")
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
            else:
                print(f"   ✅ Déjà présent dans ./models/fastembed/ : {item.name}")
    else:
        print("   ⏳ Cache FastEmbed introuvable dans ~/.cache. Téléchargement direct...")
        from fastembed import TextEmbedding
        TextEmbedding(model_name=settings.EMBEDDING_MODEL_NAME, cache_dir=str(fastembed_dst))

    # 2. GLiNER (urchade/gliner_multi-v2.1)
    src_hf_hub = user_home / ".cache" / "huggingface" / "hub"
    dst_hf_hub = hf_dst / "hub"
    dst_hf_hub.mkdir(parents=True, exist_ok=True)

    print(f"\n2. Vérification GLiNER -> {hf_dst}")
    gliner_folder_name = "models--urchade--gliner_multi-v2.1"
    src_gliner = src_hf_hub / gliner_folder_name
    dst_gliner = dst_hf_hub / gliner_folder_name

    if src_gliner.exists():
        if not dst_gliner.exists():
            print(f"   📋 Copie de GLiNER multi-v2.1 vers ./models/huggingface/hub/...")
            shutil.copytree(src_gliner, dst_gliner)
            print("   ✅ Copie terminée avec succès !")
        else:
            print("   ✅ Déjà présent dans ./models/huggingface/hub/.")
    else:
        print("   ℹ️ Modèle GLiNER non présent dans le cache ~/.cache. Il se téléchargera lors de l'appel.")

    print("\n" + "=" * 70)
    print("✅ DOSSIER LOCAL ./models/ CONFIGURÉ ET PRÊT DANS VS CODE !")
    print("💡 Les poids sont protégés par le .gitignore (aucun risque pour GitHub).")
    print("=" * 70)

if __name__ == "__main__":
    sync_models()
