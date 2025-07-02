"""
Data models for document ingestion and processing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class RawDocument:
    """Raw document before processing"""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class TextChunk:
    """Text chunk from document processing"""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    chunk_hash: Optional[str] = None
    document_id: Optional[str] = None


@dataclass
class ProcessedDocument:
    """Processed document with chunks"""

    raw_document: RawDocument
    chunks: List[TextChunk] = field(default_factory=list)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_timestamp: Optional[datetime] = None
