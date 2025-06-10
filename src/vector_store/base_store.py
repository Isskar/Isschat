"""
Base vector store interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from src.core.interfaces import Document


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
        """
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform similarity search.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of similar documents
        """
        pass

    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the vector store.

        Args:
            document_ids: List of document IDs to delete
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        pass

    @abstractmethod
    def persist(self) -> None:
        """Persist the vector store to disk."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load the vector store from disk."""
        pass
