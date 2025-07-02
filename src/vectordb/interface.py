"""
Vector database interface for clean abstractions.
Supports both Qdrant and FAISS implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """Simple document representation"""

    id: Optional[str] = None
    content: str = ""
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Search result with document and score"""

    document: Document
    score: float


class VectorDatabase(ABC):
    """Vector database interface"""

    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """Add documents with their embeddings to the database"""
        pass

    @abstractmethod
    def search(
        self, query_embedding: List[float], k: int = 3, filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents"""
        pass

    @abstractmethod
    def exists(self) -> bool:
        """Check if the database exists and is ready"""
        pass

    @abstractmethod
    def count(self) -> int:
        """Get total number of documents in database"""
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection"""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get database information"""
        pass
