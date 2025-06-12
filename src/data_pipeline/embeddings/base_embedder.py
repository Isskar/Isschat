"""
Abstract base class for embedding generation.
"""

from abc import ABC, abstractmethod
from typing import List

from src.core.interfaces import Document


class BaseEmbedder(ABC):
    """Abstract base class for document embedding generation."""

    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of documents to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Query text to embed

        Returns:
            List[float]: Embedding vector
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            int: Embedding dimension
        """
        pass

    def validate_embeddings(self, embeddings: List[List[float]]) -> bool:
        """
        Validate that embeddings have consistent dimensions.

        Args:
            embeddings: List of embedding vectors

        Returns:
            bool: True if embeddings are valid
        """
        if not embeddings:
            return True

        expected_dim = self.get_embedding_dimension()
        return all(len(emb) == expected_dim for emb in embeddings)
