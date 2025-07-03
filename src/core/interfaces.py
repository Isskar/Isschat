"""
Core interfaces for the RAG system.
These abstract base classes define the contracts for all components.
"""

# Import centralized document classes
from .documents import Document, VectorDocument, SearchResult, RetrievalDocument, RetrievalResult, GenerationResult  # noqa: F401

__all__ = ["Document", "VectorDocument", "SearchResult", "RetrievalDocument", "RetrievalResult", "GenerationResult"]
