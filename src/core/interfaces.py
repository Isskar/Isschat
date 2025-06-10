"""
Core interfaces for the RAG system.
These abstract base classes define the contracts for all components.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """Document representation"""

    page_content: str
    metadata: Dict[str, Any]


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
