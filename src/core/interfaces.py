"""
Core interfaces for the RAG system.
These abstract base classes define the contracts for all components.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Document:
    """Document representation"""

    page_content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary format for vector store."""
        return {"page_content": self.page_content, "metadata": self.metadata}

    @property
    def content(self) -> str:
        """Alias for page_content to maintain backward compatibility."""
        return self.page_content


@dataclass
class RetrievalResult:
    """Result from retrieval operation"""

    documents: List[Document]
    scores: List[float]
    query: str
    retrieval_time: float


@dataclass
class GenerationResult:
    """Result from generation operation"""

    answer: str
    sources: str
    generation_time: float
    token_count: int
