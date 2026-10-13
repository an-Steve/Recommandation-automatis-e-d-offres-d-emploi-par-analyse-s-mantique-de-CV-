"""
main.py
Point d'entrée principal du serveur Backend Flask (Master 2 IA).
Système de Recommandation d'Offres d'Emploi par Analyse Sémantique de CV.
"""

import sys
from pathlib import Path

# Résolution des imports et de l'encodage Windows
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from flask import Flask, jsonify
from flask_cors import CORS
from src.config import settings
from src.api.routes import api_bp

app = Flask(__name__)

# Activation du CORS pour autoriser les interfaces web (React, Streamlit, etc.) et mobiles
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Enregistrement du Blueprint des routes API
app.register_blueprint(api_bp)


@app.route("/", methods=["GET"])
def index():
    """Route d'accueil API."""
    return jsonify({
        "app": "Projet CV Sémantique IA - Backend API",
        "version": "1.0.0",
        "status": "running",
        "health": "/api/health",
        "endpoints": {
            "health": "/api/health",
            "jobs": "/api/jobs",
            "search_vector": "/api/jobs/search-vector",
            "enrich_job": "/api/jobs/enrich/<job_id>",
        }
    }), 200


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 DÉMARRAGE DU SERVEUR BACKEND FLASK (MASTER 2 IA)")
    print("=" * 70)
    print("• Serveur API        : http://127.0.0.1:5000")
    print("• Diagnostic Health  : http://127.0.0.1:5000/api/health")
    print("• Liste des offres   : http://127.0.0.1:5000/api/jobs")
    print("=" * 70)
    # use_reloader=False est impératif pour éviter que Flask ne redémarre quand ChromaDB écrit sur le disque
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
