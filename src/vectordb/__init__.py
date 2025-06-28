"""
Vector database unifiée pour Isschat.
Qdrant avec HNSW comme backend principal.
"""

from .interface import VectorDatabase, Document, SearchResult
from .qdrant_client import QdrantVectorDB
from .factory import VectorDBFactory

__all__ = ["VectorDatabase", "Document", "SearchResult", "QdrantVectorDB", "VectorDBFactory"]
