"""
Factory for creating different types of retrievers.
"""

from typing import Dict, Any
from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.simple_retriever import SimpleRetriever

from src.core.exceptions import ConfigurationError


class RetrieverFactory:
    """Factory for creating retriever instances."""

    @staticmethod
    def create_retriever(retriever_type: str = "simple", **kwargs) -> BaseRetriever:
        """
        Create a retriever instance.

        Args:
            retriever_type: Type of retriever to create
            **kwargs: Configuration parameters for the retriever

        Returns:
            Configured retriever instance

        Raises:
            ConfigurationError: If retriever type is unknown
        """
        retriever_type = retriever_type.lower()

        if retriever_type == "simple":
            return SimpleRetriever(**kwargs)
        elif retriever_type == "faiss":
            # Alias for simple retriever
            return SimpleRetriever(**kwargs)
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
            "simple": "Simple FAISS-based retriever (current implementation)",
            "faiss": "Alias for simple retriever",
        }

    @staticmethod
    def get_default_config(retriever_type: str = "simple") -> Dict[str, Any]:
        """
        Get default configuration for a retriever type.

        Args:
            retriever_type: Type of retriever

        Returns:
            Default configuration dictionary
        """
        retriever_type = retriever_type.lower()

        if retriever_type in ["simple", "faiss"]:
            return {"search_kwargs": {"k": 3, "fetch_k": 5}}
        else:
            raise ConfigurationError(f"Unknown retriever type: {retriever_type}")
