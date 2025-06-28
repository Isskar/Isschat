"""
LangGraph state definition for RAG workflow.
Tracks the flow of data through the RAG pipeline.
"""

from typing import TypedDict, List, Optional, Dict, Any
from ...vectordb.interface import Document


class RAGState(TypedDict):
    """State object for RAG workflow in LangGraph"""

    # Input
    query: str
    history: Optional[str]

    # Query processing
    processed_query: Optional[str]
    query_metadata: Dict[str, Any]

    # Retrieval
    retrieved_documents: List[Document]
    retrieval_scores: List[float]
    context: str

    # Generation
    answer: str
    sources: str
    generation_metadata: Dict[str, Any]

    # Control flow
    should_retrieve: bool
    should_generate: bool

    # Error handling
    error: Optional[str]
    step: str
