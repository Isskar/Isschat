"""
Service d'embedding centralisé pour Isschat.
Remplace tous les embeddings dispersés dans data_pipeline, ingestion et core.
"""

from .service import EmbeddingService, get_embedding_service
from .models import SUPPORTED_MODELS, get_model_info

__all__ = ["EmbeddingService", "get_embedding_service", "SUPPORTED_MODELS", "get_model_info"]
