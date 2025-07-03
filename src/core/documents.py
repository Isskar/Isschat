"""
Document hierarchy for the RAG system.
Centralized document classes with clear inheritance and purpose.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC


@dataclass
class BaseDocument(ABC):
    """Base document class with common functionality"""

    content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary format"""
        return {"content": self.content, "metadata": self.metadata}

    @property
    def title(self) -> str:
        """Get document title from metadata"""
        return self.metadata.get("title", "Untitled Document")

    @property
    def url(self) -> Optional[str]:
        """Get document URL from metadata"""
        return self.metadata.get("url")


@dataclass
class Document(BaseDocument):
    """Standard document for RAG pipeline operations"""

    pass


@dataclass
class VectorDocument(BaseDocument):
    """Document for vector database operations"""

    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with id field"""
        base_dict = super().to_dict()
        if self.id:
            base_dict["id"] = self.id
        return base_dict


@dataclass
class SearchResult:
    """Search result with document and score"""

    document: VectorDocument
    score: float


@dataclass
class RetrievalDocument(BaseDocument):
    """Document with retrieval score for RAG responses"""

    score: float

    def to_context_section(self, max_content_length: int = 800) -> str:
        """Format document as a context section for generation"""
        truncated_content = self.content[:max_content_length]
        if len(self.content) > max_content_length:
            truncated_content += "..."

        score_info = f" (score: {self.score:.3f})" if self.score > 0 else ""
        return f"## {self.title}{score_info}\n{truncated_content}"

    def to_source_link(self) -> str:
        """Format document as a source link"""
        if self.url:
            return f"[{self.title}]({self.url})"
        return self.title


@dataclass
class RetrievalResult:
    """Result from retrieval operation"""

    documents: list[Document]
    scores: list[float]
    query: str
    retrieval_time: float


@dataclass
class GenerationResult:
    """Result from generation operation"""

    answer: str
    sources: str
    generation_time: float
    token_count: int
