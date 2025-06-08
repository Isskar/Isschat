"""
Factory for creating vector stores.
"""

from typing import Dict, Any
from .base_store import BaseVectorStore
from .faiss_store import FAISSVectorStore

# Use absolute imports with fallbacks
try:
    from core.exceptions import ConfigurationError
except ImportError:
    try:
        from src.core.exceptions import ConfigurationError
    except ImportError:
        from ..core.exceptions import ConfigurationError


class VectorStoreFactory:
    """Factory for creating vector store instances."""

    @staticmethod
    def create_store(store_type: str = "faiss", **kwargs) -> BaseVectorStore:
        """
        Create a vector store instance.

        Args:
            store_type: Type of vector store to create
            **kwargs: Configuration parameters for the store

        Returns:
            Configured vector store instance

        Raises:
            ConfigurationError: If store type is unknown
        """
        store_type = store_type.lower()

        if store_type == "faiss":
            return FAISSVectorStore(**kwargs)
        else:
            raise ConfigurationError(f"Unknown vector store type: {store_type}")

    @staticmethod
    def get_available_stores() -> Dict[str, str]:
        """
        Get list of available vector store types.

        Returns:
            Dictionary mapping store names to descriptions
        """
        return {"faiss": "FAISS-based vector store for similarity search"}

    @staticmethod
    def get_default_config(store_type: str = "faiss") -> Dict[str, Any]:
        """
        Get default configuration for a vector store type.

        Args:
            store_type: Type of vector store

        Returns:
            Default configuration dictionary
        """
        store_type = store_type.lower()

        if store_type == "faiss":
            return {"persist_directory": "./data/vector_db"}
        else:
            raise ConfigurationError(f"Unknown vector store type: {store_type}")
