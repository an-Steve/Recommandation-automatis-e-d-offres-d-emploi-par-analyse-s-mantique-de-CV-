"""Smoke test pour valider l'environnement virtuel et les packages clés."""
import sys

def run_smoke_test():
    print(f"🐍 Version Python : {sys.version.split()[0]}")
    
    # 1. Pydantic
    import pydantic
    print(f"✅ Pydantic v{pydantic.__version__} OK")
    
    # 2. SQLAlchemy
    import sqlalchemy
    print(f"✅ SQLAlchemy v{sqlalchemy.__version__} OK")
    
    # 3. ChromaDB
    import chromadb
    client = chromadb.EphemeralClient()
    print(f"✅ ChromaDB v{chromadb.__version__} (In-Memory client) OK")
    
    # 4. PyTorch & GPU
    import torch
    device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    print(f"✅ PyTorch v{torch.__version__} (Device: {device}) OK")
    
    # 5. JobSpy
    try:
        import jobspy
        print("✅ python-jobspy OK")
    except ImportError:
        print("⚠️ python-jobspy non installé")
        
    print("\n🎉 PHASE 0 VALIDÉE AVEC SUCCÈS ! Vous êtes prêt pour la Phase 1.")

if __name__ == "__main__":
    run_smoke_test()
