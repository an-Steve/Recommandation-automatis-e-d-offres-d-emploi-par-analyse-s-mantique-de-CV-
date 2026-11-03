"""
Gestionnaire de la base vectorielle ChromaDB.
Supporte FastEmbed (ONNX Runtime C++ local, haute performance CPU) et Ollama.
"""
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions
from src.config import settings

logger = logging.getLogger(__name__)

# Initialisation du client persistant local (stocké dans le chemin CHROMA_PERSIST_DIR)
chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)


class FastEmbedEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Fonction d'embedding haute performance ONNX C++ basée sur FastEmbed.
    Exécution locale directe en C++ (0 serveur HTTP, 0 overhead PyTorch, 0 timeout).
    """
    _INSTANCE = None
    _MODEL = None

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", **kwargs):
        self.model_name = model_name
        if FastEmbedEmbeddingFunction._MODEL is None:
            from fastembed import TextEmbedding
            logger.info(f"⏳ Initialisation de FastEmbed ONNX ({model_name})...")
            FastEmbedEmbeddingFunction._MODEL = TextEmbedding(model_name=model_name, **kwargs)
            logger.info("✅ Modèle FastEmbed prêt.")
        self._model = FastEmbedEmbeddingFunction._MODEL

    def __call__(self, input: Documents) -> Embeddings:
        # Conversion du générateur numpy en liste de floats standard pour ChromaDB
        return [list(map(float, e)) for e in self._model.embed(input)]

    def name(self) -> str:
        # Rétrocompatibilité stricte avec la signature persistée dans ChromaDB
        return "ollama"


def get_embedding_function():
    """
    Retourne la fonction d'embedding selon le provider configuré dans .env.
    Par défaut : FastEmbed ONNX C++ local (nomic-ai/nomic-embed-text-v1.5).
    """
    provider = getattr(settings, "EMBEDDING_PROVIDER", "fastembed").lower()
    if provider == "fastembed":
        return FastEmbedEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL_NAME,
            cache_dir=str(settings.fastembed_cache_dir)
        )
    elif provider == "ollama":
        return embedding_functions.OllamaEmbeddingFunction(
            url=settings.OLLAMA_BASE_URL,
            model_name=settings.EMBEDDING_MODEL_NAME,
            timeout=300
        )
    return None


def get_offres_collection() -> Collection:
    """
    Récupère ou crée la collection vectorielle pour les offres d'emploi.
    Configure explicitement la distance cosinus HNSW et l'embedding FastEmbed / Ollama.
    """
    return chroma_client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_OFFRES,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"}  # Similarité cosinus
    )


def upsert_job_vector(
    job_id: str,
    document: str,
    embedding: List[float],
    metadata: Dict[str, Any]
) -> None:
    """
    Insère ou met à jour le vecteur d'une offre dans ChromaDB.
    
    :param job_id: Identifiant exact correspondant à la clé primaire de PostgreSQL.
    :param document: Texte représentatif de l'offre (Poste + Compétences + Missions).
    :param embedding: Vecteur dense calculé (768 floats).
    :param metadata: Critères de filtrage strict (type_de_contrat, niveau_etude_min, etc.).
    """
    collection = get_offres_collection()
    
    # Nettoyage des métadonnées (ChromaDB n'accepte que str, int, float, bool)
    clean_meta = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            clean_meta[k] = v
        elif isinstance(v, list):
            clean_meta[k] = ", ".join(map(str, v))
        elif v is not None:
            clean_meta[k] = str(v)

    collection.upsert(
        ids=[job_id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[clean_meta]
    )


def query_similar_jobs(
    query_embedding: List[float],
    n_results: int = 30,
    where_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Effectue une recherche par similarité cosinus avec pré-filtrage strict sur métadonnées.
    
    :param query_embedding: Vecteur du profil candidat.
    :param n_results: Nombre d'offres à remonter pour le reranking (Top 30 par défaut).
    :param where_filter: Filtre ChromaDB (ex: {"type_de_contrat": "CDI"}).
    :return: Liste de dictionnaires contenant job_id, score cosinus et métadonnées.
    """
    collection = get_offres_collection()
    
    kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": n_results
    }
    if where_filter:
        kwargs["where"] = where_filter

    results = collection.query(**kwargs)
    
    formatted_results = []
    if results and results.get("ids") and len(results["ids"][0]) > 0:
        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
        documents = results["documents"][0] if results.get("documents") else [""] * len(ids)
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        
        for job_id, dist, doc, meta in zip(ids, distances, documents, metadatas):
            # En distance cosinus ChromaDB : similarité = 1.0 - distance
            cosine_similarity = round(max(0.0, 1.0 - dist), 4)
            formatted_results.append({
                "job_id": job_id,
                "cosine_similarity": cosine_similarity,
                "document": doc,
                "metadata": meta
            })
            
    return formatted_results


def count_indexed_jobs() -> int:
    """Retourne le nombre total d'offres indexées dans ChromaDB."""
    collection = get_offres_collection()
    return collection.count()


def delete_job_vector(job_id: str) -> None:
    """Supprime un vecteur de ChromaDB par son ID."""
    collection = get_offres_collection()
    collection.delete(ids=[job_id])
