"""
Centralized embedding service for Isschat.
Single shared service between ingestion and RAG to avoid duplication.
"""

import logging
from typing import List, Dict, Any, Optional

import torch
from sentence_transformers import SentenceTransformer
import numpy as np

from ..config import get_config
from .models import get_model_dimension, get_model_info


class EmbeddingService:
    """Centralized embedding service with model cache"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
    ):
        """
        Initialize the embedding service

        Args:
            model_name: Model name, otherwise uses config
            device: Torch device, otherwise uses config
            batch_size: Batch size, otherwise uses config
            normalize: Normalize embeddings, otherwise uses config
        """
        self.config = get_config()

        # Use config if parameters not provided
        self.model_name = model_name or self.config.embeddings_model
        self.device = device or self.config.embeddings_device
        self.batch_size = batch_size or self.config.embeddings_batch_size
        self.normalize = normalize if normalize is not None else self.config.embeddings_normalize

        # Model cache
        self._model: Optional[SentenceTransformer] = None
        self._model_dimension: Optional[int] = None

        # Logs
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def model(self) -> SentenceTransformer:
        """Get the model (lazy loading with cache)"""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding dimension of the model"""
        if self._model_dimension is None:
            self._model_dimension = get_model_dimension(self.model_name)
        return self._model_dimension

    def _load_model(self) -> None:
        """Load the model with error handling"""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)

            # Check actual vs expected dimension
            test_embedding = self._model.encode(["test"], convert_to_numpy=True)
            actual_dim = test_embedding.shape[1]
            expected_dim = get_model_dimension(self.model_name)

            if actual_dim != expected_dim:
                self.logger.warning(f"Model dimension different: expected={expected_dim}, actual={actual_dim}")
                self._model_dimension = actual_dim
            else:
                self._model_dimension = expected_dim

            self.logger.info(f"Model loaded, dimension: {self._model_dimension}, device: {self.device}")

        except Exception as e:
            self.logger.error(f"Model loading error {self.model_name}: {e}")
            raise RuntimeError(f"Unable to load embedding model: {e}")

    def encode_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of texts to embeddings

        Args:
            texts: List of texts to encode
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings [n_texts, dimension]
        """
        if not texts:
            return np.empty((0, self.dimension))

        try:
            self.logger.debug(f"Encoding {len(texts)} texts...")

            # Encode with the model
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
            )

            self.logger.debug(f"Embeddings generated: shape={embeddings.shape}")
            return embeddings

        except Exception as e:
            self.logger.error(f"Text encoding error: {e}")
            raise RuntimeError(f"Encoding error: {e}")

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text (convenient for queries)"""
        embeddings = self.encode_texts([text])
        return embeddings[0] if len(embeddings) > 0 else np.zeros(self.dimension)

    def encode_query(self, query: str) -> List[float]:
        """Encode a query and return list (vector DB format)"""
        embedding = self.encode_single(query)
        return embedding.tolist()

    def get_info(self) -> Dict[str, Any]:
        """Information about the embedding service"""
        model_info = get_model_info(self.model_name)

        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "model_loaded": self._model is not None,
            "model_info": model_info.__dict__ if model_info else None,
            "torch_available": torch.cuda.is_available() if self.device == "cuda" else True,
        }

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Ensure embeddings are normalized
            if not self.normalize:
                embedding1 = embedding1 / np.linalg.norm(embedding1)
                embedding2 = embedding2 / np.linalg.norm(embedding2)

            # Cosine similarity (dot product if normalized)
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)

        except Exception as e:
            self.logger.error(f"Similarity calculation error: {e}")
            return 0.0


# Global instance to avoid multiple reloads
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the global instance of the embedding service"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def reset_embedding_service():
    """Reset for tests"""
    global _embedding_service
    _embedding_service = None
