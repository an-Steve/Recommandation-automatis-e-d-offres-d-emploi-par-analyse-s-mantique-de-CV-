"""
src/api/routes.py
Endpoints REST Flask pour le système universel de recommandation de recrutement (Master 2 IA).
Connecte PostgreSQL, ChromaDB, Ollama, le scraping quotidien multi-secteurs et l'enrichissement IA.
"""

import logging
from flask import Blueprint, request, jsonify
from sqlalchemy import func

from src.config import settings
from src.database.db_session import SessionLocal
from src.database.sql_models import Offre, CVUser, Evaluation
from src.database.vector_store import get_offres_collection, count_indexed_jobs
from src.parsers.job_enricher import JobEnricher
from src.utils.ollama_client import OllamaClient
from src.scrapers.daily_scraper import DailyJobScraper
from src.scrapers.sector_mapping import SECTOR_TAXONOMY, list_supported_sectors

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


# ------------------------------------------------------------------------------
# 1. Diagnostic Système & Healthcheck
# ------------------------------------------------------------------------------
@api_bp.route("/health", methods=["GET"])
def healthcheck():
    """Vérifie l'état opérationnel de PostgreSQL, ChromaDB et Ollama."""
    db = SessionLocal()
    pg_status = False
    total_jobs_sql = 0
    sources_summary = {}
    secteurs_summary = {}
    try:
        total_jobs_sql = db.query(Offre).count()
        # Répartition par source
        rows_src = db.query(Offre.source, func.count(Offre.id)).group_by(Offre.source).all()
        sources_summary = {str(r[0]): r[1] for r in rows_src}
        # Répartition par secteur
        rows_sec = db.query(Offre.secteur, func.count(Offre.id)).group_by(Offre.secteur).all()
        secteurs_summary = {str(r[0]): r[1] for r in rows_sec}
        pg_status = True
    except Exception as e:
        logger.error(f"Erreur PostgreSQL: {e}")
    finally:
        db.close()

    chroma_status = False
    chroma_count = 0
    try:
        chroma_count = count_indexed_jobs()
        chroma_status = True
    except Exception as e:
        logger.error(f"Erreur ChromaDB: {e}")

    ollama_client = OllamaClient()
    ollama_status = ollama_client.is_available()
    ollama_models = ollama_client.list_models() if ollama_status else []

    return jsonify({
        "status": "ok" if (pg_status and chroma_status) else "degraded",
        "environment": settings.APP_ENV,
        "database": {
            "postgres": pg_status,
            "total_offres_sql": total_jobs_sql,
            "repartition_sources": sources_summary,
            "repartition_secteurs": secteurs_summary,
        },
        "vector_store": {
            "chroma": chroma_status,
            "collection": settings.CHROMA_COLLECTION_OFFRES,
            "total_vectors": chroma_count,
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
        },
        "ollama": {
            "connected": ollama_status,
            "base_url": settings.OLLAMA_BASE_URL,
            "models_available": ollama_models,
        }
    }), 200


# ------------------------------------------------------------------------------
# 2. Taxonomie des Secteurs Supportés
# ------------------------------------------------------------------------------
@api_bp.route("/sectors", methods=["GET"])
def get_supported_sectors():
    """Retourne la liste des 8 secteurs universels supportés et leurs configurations."""
    return jsonify({
        "total_sectors": len(SECTOR_TAXONOMY),
        "sectors": SECTOR_TAXONOMY
    }), 200


# ------------------------------------------------------------------------------
# 3. Gestion des Offres d'Emploi
# ------------------------------------------------------------------------------
@api_bp.route("/jobs", methods=["GET"])
def list_jobs():
    """
    Retourne la liste paginée des offres avec filtres optionnels :
    ?source=indeed|adzuna|linkedin
    ?secteur=it_data|sante_medical|...
    ?date_posted=2026-10-13
    ?limit=20
    ?offset=0
    ?contract=CDI
    """
    source = request.args.get("source")
    secteur = request.args.get("secteur")
    date_posted = request.args.get("date_posted")
    contract = request.args.get("contract")
    limit = request.args.get("limit", default=20, type=int)
    offset = request.args.get("offset", default=0, type=int)

    db = SessionLocal()
    try:
        query = db.query(Offre)
        if source and source != "all":
            query = query.filter(Offre.source == source.lower())
        if secteur and secteur != "all":
            query = query.filter(Offre.secteur == secteur.lower())
        if date_posted:
            query = query.filter(Offre.date_posted == date_posted)
        if contract:
            query = query.filter(Offre.type_de_contrat.ilike(f"%{contract}%"))

        total_count = query.count()
        offres = query.order_by(Offre.created_at.desc()).offset(offset).limit(limit).all()

        results = []
        for o in offres:
            results.append({
                "id": o.id,
                "poste": o.poste,
                "company": o.company,
                "location": o.location,
                "source": o.source,
                "secteur": o.secteur,
                "date_posted": str(o.date_posted) if o.date_posted else None,
                "type_de_contrat": o.type_de_contrat,
                "niveau_etude_min": o.niveau_etude_min,
                "min_experience_years": float(o.min_experience_years),
                "competences_necessaires": o.competences_necessaires,
                "is_indexed": o.is_indexed,
                "source_url": o.source_url,
                "created_at": o.created_at.isoformat() if o.created_at else None,
            })

        return jsonify({
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "jobs": results
        }), 200
    finally:
        db.close()


@api_bp.route("/jobs/<job_id>", methods=["GET"])
def get_job_detail(job_id: str):
    """Retourne la fiche détaillée d'une offre d'emploi."""
    db = SessionLocal()
    try:
        offre = db.query(Offre).filter(Offre.id == job_id).first()
        if not offre:
            return jsonify({"error": "Offre non trouvée"}), 404

        return jsonify({
            "id": offre.id,
            "poste": offre.poste,
            "company": offre.company,
            "location": offre.location,
            "source": offre.source,
            "secteur": offre.secteur,
            "date_posted": str(offre.date_posted) if offre.date_posted else None,
            "type_de_contrat": offre.type_de_contrat,
            "niveau_etude_min": offre.niveau_etude_min,
            "min_experience_years": float(offre.min_experience_years),
            "competences_necessaires": offre.competences_necessaires,
            "description": offre.description,
            "mission": offre.mission,
            "source_url": offre.source_url,
            "is_indexed": offre.is_indexed,
            "created_at": offre.created_at.isoformat() if offre.created_at else None,
        }), 200
    finally:
        db.close()


# ------------------------------------------------------------------------------
# 4. Scraping Quotidien par Date & par Secteur (Multi-Platform)
# ------------------------------------------------------------------------------
@api_bp.route("/jobs/scrape-daily", methods=["POST"])
def scrape_daily_jobs():
    """
    Déclenche le scraping quotidien des offres du jour (ou d'une date spécifique)
    avec filtrage sectoriel et limites sécurisées par plateforme.

    JSON Body:
    {
      "date": "2026-10-13",             (optionnel, défaut: aujourd'hui)
      "secteur": "it_data",             (optionnel: it_data, sante_medical, btp_construction...)
      "sources": ["adzuna", "indeed"], (optionnel: adzuna, indeed, linkedin)
      "limit_per_source": 15,           (optionnel: défaut 20)
      "location": "France"              (optionnel)
    }
    """
    data = request.get_json() or {}
    date_str = data.get("date")
    sector_code = data.get("secteur", "it_data")
    sources = data.get("sources", ["adzuna", "indeed", "linkedin"])
    limit_per_source = data.get("limit_per_source", 20)
    location = data.get("location", "France")
    auto_enrich = data.get("auto_enrich", data.get("auto_enrich_adzuna", True))
    enrich_limit = data.get("enrich_limit", 3)

    try:
        daily_scraper = DailyJobScraper()
        result = daily_scraper.run_daily_pipeline(
            target_date_str=date_str,
            sector_code=sector_code,
            sources=sources,
            limit_per_source=limit_per_source,
            location=location,
            auto_enrich=auto_enrich,
            enrich_limit=enrich_limit,
        )
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"❌ Erreur lors du scraping quotidien : {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


# ------------------------------------------------------------------------------
# 5. Recherche Sémantique & Vectorielle avec Routage Sectoriel (ChromaDB)
# ------------------------------------------------------------------------------
@api_bp.route("/jobs/search-vector", methods=["POST"])
def search_vector():
    """
    Recherche sémantique dense dans ChromaDB avec pré-filtrage sectoriel HNSW.
    JSON Body: {"query": "Data Scientist NLP", "top_k": 5, "secteur": "it_data", "source": "indeed"}
    """
    data = request.get_json() or {}
    query_text = data.get("query")
    if not query_text:
        return jsonify({"error": "Le champ 'query' est requis"}), 400

    top_k = data.get("top_k", 5)
    source_filter = data.get("source")
    sector_filter = data.get("secteur")

    collection = get_offres_collection()

    # Construction du filtre composite ChromaDB
    conditions = []
    if source_filter and source_filter != "all":
        conditions.append({"source": source_filter.lower()})
    if sector_filter and sector_filter != "all":
        conditions.append({"secteur": sector_filter.lower()})

    where_meta = None
    if len(conditions) == 1:
        where_meta = conditions[0]
    elif len(conditions) > 1:
        where_meta = {"$and": conditions}

    # ChromaDB délègue automatiquement l'embedding à Ollama (all-minilm)
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        where=where_meta
    )

    formatted_results = []
    if results and results.get("ids") and len(results["ids"][0]) > 0:
        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)

        db = SessionLocal()
        try:
            for job_id, dist, meta in zip(ids, distances, metadatas):
                sim = round(max(0.0, 1.0 - dist), 4)
                offre_sql = db.query(Offre).filter(Offre.id == job_id).first()
                formatted_results.append({
                    "job_id": job_id,
                    "similarity": sim,
                    "poste": offre_sql.poste if offre_sql else meta.get("poste"),
                    "company": offre_sql.company if offre_sql else meta.get("company"),
                    "location": offre_sql.location if offre_sql else meta.get("location"),
                    "source": offre_sql.source if offre_sql else meta.get("source", "unknown"),
                    "secteur": offre_sql.secteur if offre_sql else meta.get("secteur", "it_data"),
                    "competences": offre_sql.competences_necessaires if offre_sql else [],
                    "niveau_etude_min": offre_sql.niveau_etude_min if offre_sql else 0,
                    "source_url": offre_sql.source_url if offre_sql else "",
                })
        finally:
            db.close()

    return jsonify({
        "query": query_text,
        "secteur_filtre": sector_filter or "tous",
        "count": len(formatted_results),
        "results": formatted_results
    }), 200


# ------------------------------------------------------------------------------
# 6. Enrichissement IA d'une Offre (Worker LLM Llama 3.2 Few-Shot Compact)
# ------------------------------------------------------------------------------
@api_bp.route("/jobs/enrich/<job_id>", methods=["POST"])
def enrich_single_job(job_id: str):
    """Déclenche l'enrichissement LLM (Llama 3.2 Few-Shot Compact) sur une offre spécifique."""
    db = SessionLocal()
    try:
        offre = db.query(Offre).filter(Offre.id == job_id).first()
        if not offre:
            return jsonify({"error": "Offre non trouvée"}), 404

        enricher = JobEnricher()
        enricher.enrich_offre_instance(offre)
        db.commit()

        return jsonify({
            "success": True,
            "job_id": offre.id,
            "poste": offre.poste,
            "secteur": offre.secteur,
            "competences_necessaires": offre.competences_necessaires,
            "niveau_etude_min": offre.niveau_etude_min,
            "min_experience_years": float(offre.min_experience_years)
        }), 200
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur lors de l'enrichissement de l'offre {job_id}: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


# ------------------------------------------------------------------------------
# 7. Ré-indexation Globale dans ChromaDB (Texte Nettoyé)
# ------------------------------------------------------------------------------
@api_bp.route("/jobs/reindex", methods=["POST"])
def reindex_vector_store():
    """Ré-indexe toutes les offres de PostgreSQL dans ChromaDB avec les textes assainis."""
    try:
        from src.pipeline.job_indexer import JobIndexer
        indexer = JobIndexer(batch_size=5)
        total = indexer.reindex_all_jobs()
        return jsonify({
            "status": "success",
            "message": f"{total} offres ré-indexées avec succès dans ChromaDB.",
            "total_indexed": total
        }), 200
    except Exception as e:
        logger.error(f"Erreur ré-indexation ChromaDB: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


# ------------------------------------------------------------------------------
# 7b. Exportation des Données ChromaDB (Audit Qualité)
# ------------------------------------------------------------------------------
@api_bp.route("/jobs/export-chroma", methods=["GET"])
def export_chroma_store():
    """Exporte les documents canoniques et métadonnées indexés dans ChromaDB."""
    try:
        from src.database.vector_store import get_offres_collection
        collection = get_offres_collection()
        total_count = collection.count()
        if total_count == 0:
            return jsonify({
                "status": "warning",
                "message": "ChromaDB est actuellement vide.",
                "total_count": 0,
                "items": []
            }), 200

        data = collection.get(include=["documents", "metadatas"])
        items = [
            {
                "id": job_id,
                "metadata": meta,
                "canonical_document": doc
            }
            for job_id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
        ]

        return jsonify({
            "status": "success",
            "total_count": total_count,
            "exported_count": len(items),
            "items": items
        }), 200
    except Exception as e:
        logger.error(f"Erreur export ChromaDB: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


# ------------------------------------------------------------------------------
# 8. Enrichissement par Lot des Offres en Attente (Indeed, LinkedIn, Adzuna)
# ------------------------------------------------------------------------------
@api_bp.route("/jobs/enrich-pending", methods=["POST"])
def enrich_pending_jobs():
    """Enrichit par lot les offres dont les compétences ne sont pas encore renseignées avec protection CPU."""
    data = request.get_json() or {}
    limit = data.get("limit", 5)
    batch_size = data.get("batch_size", 5)
    cooldown = float(data.get("cooldown", 5.0))
    sleep_per_job = float(data.get("sleep_per_job", 1.0))

    from src.parsers.job_enricher import JobEnricher
    from tqdm import tqdm
    import time

    db = SessionLocal()
    try:
        query = db.query(Offre).filter(
            (Offre.competences_necessaires == []) | (Offre.competences_necessaires == None)
        ).order_by(Offre.created_at.desc())

        if limit and limit > 0:
            query = query.limit(limit)

        jobs = query.all()
        if not jobs:
            return jsonify({
                "status": "ok",
                "message": "Toutes les offres sont déjà enrichies.",
                "enriched_count": 0,
                "jobs": []
            }), 200

        enricher = JobEnricher()
        enriched_results = []
        pbar = tqdm(jobs, desc="🧠 Enrichissement Lot", unit="offre")

        for idx, o in enumerate(pbar):
            comp_name = f"[{o.source.upper()}] {(o.company or 'Inconnue')[:10]}"
            pbar.set_postfix_str(f"Extraction {comp_name}...")
            try:
                enricher.enrich_offre_instance(o)
                db.commit()
                enriched_results.append({
                    "id": o.id,
                    "poste": o.poste,
                    "company": o.company,
                    "source": o.source,
                    "competences": o.competences_necessaires,
                    "niveau_etude_min": o.niveau_etude_min,
                    "min_experience_years": float(o.min_experience_years)
                })
                pbar.set_postfix_str(f"OK {comp_name} ({len(o.competences_necessaires)} skills)")
                time.sleep(sleep_per_job)
            except Exception as err:
                db.rollback()
                logger.warning(f"Erreur enrichissement {o.id}: {err}")

            # Cooldown thermique CPU après chaque lot de 'batch_size'
            if (idx + 1) % batch_size == 0 and (idx + 1) < len(jobs):
                pbar.set_postfix_str(f"❄️ Cooldown CPU ({cooldown}s)...")
                time.sleep(cooldown)

        # Indexation ciblée UNIQUEMENT des offres qui viennent d'être enrichies
        if enriched_results:
            from src.pipeline.job_indexer import JobIndexer
            indexer = JobIndexer(batch_size=batch_size)
            indexer.index_jobs_by_ids([r["id"] for r in enriched_results])

        return jsonify({
            "status": "success",
            "enriched_count": len(enriched_results),
            "jobs": enriched_results
        }), 200
    except Exception as e:
        logger.error(f"Erreur enrichissement par lot: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500
    finally:
        db.close()
