"""
src/utils/ollama_client.py
Client HTTP générique pour le runtime local Ollama avec support des Structured Outputs Pydantic.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Type, TypeVar
import requests
from pydantic import BaseModel, ValidationError

from src.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaClient:
    """
    Client wrapper pour interagir avec le runtime local Ollama (http://localhost:11434).
    Gère l'inférence texte standard et l'extraction structurée garantie par schéma Pydantic.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: int = 300,
    ):
        self.base_url = (base_url or getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.default_model = default_model or getattr(settings, "LLM_EXTRACTION_MODEL", "llama3.2:1b")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Vérifie si le démon Ollama répond sur l'URL configurée."""
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return res.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Retourne la liste des noms de modèles disponibles localement dans Ollama."""
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if res.status_code == 200:
                data = res.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Impossible de lister les modèles Ollama : {e}")
        return []

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        format: Optional[Any] = None,
        temperature: float = 0.0,
        num_ctx: int = 1024,
        num_predict: int = 256,
        repeat_penalty: float = 1.15,
        stream: bool = False,
    ) -> str:
        """
        Envoie une requête de chat completion à Ollama.

        :param messages: Liste de messages [{"role": "system"|"user", "content": "..."}]
        :param model: Modèle cible (ex: "llama3.2:3b"), prend default_model si non précisé
        :param format: "json" ou un schéma JSON dict pour les Structured Outputs
        :param temperature: Température de génération (0.0 pour déterminisme absolu)
        :param num_ctx: Taille de la fenêtre de contexte (1024 optimisé pour CPU)
        :param num_predict: Nombre maximal de tokens générés (256 suffisant pour du JSON)
        :param repeat_penalty: Pénalité de répétition pour éviter les boucles infinies de tokens
        :param stream: Streaming désactivé par défaut
        :return: Texte brut généré par le modèle
        """
        target_model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "repeat_penalty": repeat_penalty,
            },
        }

        if format is not None:
            payload["format"] = format

        endpoint = f"{self.base_url}/api/chat"
        try:
            response = requests.post(endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur lors de l'appel à Ollama ({target_model}) : {e}")
            raise RuntimeError(f"Échec de l'appel Ollama : {e}") from e

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        num_ctx: int = 1024,
        num_predict: int = 256,
        use_json_mode: bool = True,
    ) -> T:
        """
        Génère une réponse strictement conforme à un modèle Pydantic.
        Par défaut, utilise format='json' (optimisé pour petits LLMs 1B/3B) et valide via Pydantic.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # format="json" évite la surcharge de grammaire complexe pour les petits modèles
        format_param = "json" if use_json_mode else schema.model_json_schema()

        raw_output = self.chat(
            messages=messages,
            model=model,
            format=format_param,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )

        return self._clean_and_parse(raw_output, schema)

    def _clean_and_parse(self, raw_output: str, schema: Type[T]) -> T:
        """Nettoie le texte généré et instancie l'objet Pydantic validé avec auto-réparation."""
        cleaned = raw_output.strip()

        # 1. Extraction si le JSON est dans un bloc markdown ```json ... ```
        json_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if json_block:
            cleaned = json_block.group(1).strip()
        else:
            # 2. Extraction du premier objet JSON délimité par { ... }
            curly_match = re.search(r"(\{[\s\S]*\})", cleaned)
            if curly_match:
                cleaned = curly_match.group(1).strip()
            elif "{" in cleaned:
                # Si le JSON commence par { mais a été coupé sans }
                cleaned = cleaned[cleaned.find("{"):].strip()

        # 3. Auto-réparation robuste si la fin du JSON a été tronquée par coupure de tokens (EOF / unterminated string)
        if "{" in cleaned:
            # Si le texte se termine au milieu d'un guillemet non fermé (ex: "Py)
            quotes = re.findall(r'(?<!\\)"', cleaned)
            if len(quotes) % 2 != 0:
                last_quote_idx = cleaned.rfind('"')
                if last_quote_idx != -1:
                    cleaned = cleaned[:last_quote_idx].rstrip()

            # Nettoyage des virgules pendantes
            cleaned = cleaned.rstrip().rstrip(",")

            # Fermeture des crochets de tableaux non fermés
            open_brackets = cleaned.count("[") - cleaned.count("]")
            if open_brackets > 0:
                cleaned += "]" * open_brackets

            # Fermeture des accolades d'objets non fermées
            cleaned = cleaned.rstrip().rstrip(",")
            open_braces = cleaned.count("{") - cleaned.count("}")
            if open_braces > 0:
                cleaned += "}" * open_braces

        try:
            return schema.model_validate_json(cleaned)
        except (ValidationError, Exception) as e:
            logger.warning(f"⚠️ Échec parsing JSON direct ({e}), tentative de fallback avec nettoyage...")
            try:
                # Nettoyage des virgules pendantes avant accolades ou crochets fermants
                repaired = re.sub(r",\s*([\}\]])", r"\1", cleaned)
                data = json.loads(repaired)
                return schema.model_validate(data)
            except Exception as e2:
                logger.error(f"❌ Impossible de désérialiser la sortie en {schema.__name__} : {raw_output}")
                raise ValueError(
                    f"La réponse d'Ollama n'est pas conforme au schéma Pydantic {schema.__name__} : {e2}"
                ) from e2
