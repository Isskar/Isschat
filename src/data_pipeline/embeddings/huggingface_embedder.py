"""
HuggingFace embedding implementation.
"""

from typing import List, Dict, Any
from .base_embedder import BaseEmbedder

from src.core.interfaces import Document


class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace-based embedding generator."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace embedder.

        Args:
            config: Configuration containing model settings
        """
        self.config = config
        self.model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.device = config.get("device", "cpu")
        self.batch_size = config.get("batch_size", 32)
        self._model = None
        self._dimension = None

    def _load_model(self):
        """Load the HuggingFace model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
                # Get dimension from a test embedding
                test_embedding = self._model.encode("test")
                self._dimension = len(test_embedding)
            except ImportError:
                raise ImportError("sentence-transformers library is required for HuggingFaceEmbedder")

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for documents.

        Args:
            documents: List of documents to embed

        Returns:
            List[List[float]]: Document embeddings
        """
        self._load_model()

        # Extract text content from documents
        texts = [doc.page_content for doc in documents]

        # Generate embeddings in batches
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_embeddings = self._model.encode(batch_texts, convert_to_tensor=False)
            embeddings.extend(batch_embeddings.tolist())

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            query: Query text to embed

        Returns:
            List[float]: Query embedding
        """
        self._load_model()
        embedding = self._model.encode(query, convert_to_tensor=False)
        return embedding.tolist()

    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension.

        Returns:
            int: Embedding dimension
        """
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dict: Model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.get_embedding_dimension(),
            "batch_size": self.batch_size,
        }
