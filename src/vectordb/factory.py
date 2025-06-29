"""
Vector database factory for creating the appropriate implementation.
Intégré avec la configuration unifiée.
"""

from typing import Optional
from .interface import VectorDatabase
from .qdrant_client import QdrantVectorDB


class VectorDBFactory:
    """Factory for creating vector database instances"""

    @staticmethod
    def create_qdrant(collection_name: Optional[str] = None, embedding_dim: Optional[int] = None) -> QdrantVectorDB:
        """Create Qdrant database instance with config unifiée"""
        return QdrantVectorDB(collection_name=collection_name, embedding_dim=embedding_dim)

    @staticmethod
    def create_from_config() -> VectorDatabase:
        """Create vector DB depuis configuration unifiée"""
        from ..config import get_config

        config = get_config()

        if config.vectordb_index_type in ["hnsw"]:
            return VectorDBFactory.create_qdrant()
        else:
            raise ValueError(f"Type vector DB non supporté: {config.vectordb_index_type}")

    @staticmethod
    def get_available_types() -> list[str]:
        """Liste des types de vector DB disponibles"""
        return ["qdrant"]
