"""
Factory for creating different types of retrievers.
Updated to work with the new centralized architecture.
"""

from typing import Dict, Any, Optional
from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.vector_store_retriever import VectorStoreRetriever
from src.core.exceptions import ConfigurationError


class RetrieverFactory:
    """Factory for creating retriever instances with centralized architecture."""

    @staticmethod
    def create_retriever_from_vector_store(
        vector_store, search_kwargs: Optional[Dict[str, Any]] = None
    ) -> BaseRetriever:
        """
        Create a retriever from a pre-loaded vector store.
        This is the preferred method in the new architecture.

        Args:
            vector_store: Pre-loaded vector store instance
            search_kwargs: Search configuration

        Returns:
            Configured retriever instance
        """
        return VectorStoreRetriever(vector_store, search_kwargs)

    @staticmethod
    def create_retriever(retriever_type: str, **kwargs) -> BaseRetriever:
        """
        Create a retriever instance (legacy method - deprecated).

        Args:
            retriever_type: Type of retriever to create
            **kwargs: Configuration parameters for the retriever

        Returns:
            Configured retriever instance

        Raises:
            ConfigurationError: If retriever type is unknown or deprecated
        """
        print("⚠️  Warning: create_retriever is deprecated. Use create_retriever_from_vector_store instead.")

        retriever_type = retriever_type.lower()

        if retriever_type == "faiss":
            # For backward compatibility, try to create a VectorStoreRetriever
            # but this requires a vector_store to be passed in kwargs
            vector_store = kwargs.get("vector_store")
            if not vector_store:
                raise ConfigurationError(
                    "vector_store is required in kwargs for FAISS retriever. "
                    "Use create_retriever_from_vector_store instead."
                )

            search_kwargs = kwargs.get("search_kwargs", {"k": 3, "fetch_k": 5})
            return VectorStoreRetriever(vector_store, search_kwargs)
        else:
            raise ConfigurationError(f"Unknown retriever type: {retriever_type}")

    @staticmethod
    def get_available_retrievers() -> Dict[str, str]:
        """
        Get list of available retriever types.

        Returns:
            Dictionary mapping retriever names to descriptions
        """
        return {
            "vector_store": "VectorStoreRetriever - Works with pre-loaded vector stores",
        }

    @staticmethod
    def get_default_config(retriever_type: str = "vector_store") -> Dict[str, Any]:
        """
        Get default configuration for a retriever type.

        Args:
            retriever_type: Type of retriever

        Returns:
            Default configuration dictionary
        """
        return {"search_kwargs": {"k": 3, "fetch_k": 5}}
