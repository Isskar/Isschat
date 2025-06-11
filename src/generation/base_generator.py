"""
Base generator interface and implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


from src.core.interfaces import GenerationResult, RetrievalResult


class BaseGenerator(ABC):
    """Abstract base class for all generators."""

    @abstractmethod
    def generate(self, query: str, retrieval_result: RetrievalResult) -> GenerationResult:
        """
        Generate answer based on query and retrieved documents.

        Args:
            query: User query
            retrieval_result: Result from retrieval step

        Returns:
            GenerationResult with answer and metadata
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        pass
