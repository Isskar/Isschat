"""
Base retriever interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from src.core.interfaces import RetrievalResult


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            RetrievalResult with documents and metadata
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        pass
