"""
Custom exceptions for the RAG system.
"""


class RAGSystemError(Exception):
    """Base exception for RAG system errors"""

    pass


class RAGError(RAGSystemError):
    """General RAG pipeline error"""

    pass


class ExtractionError(RAGSystemError):
    """Raised when data extraction fails"""

    pass


class ProcessingError(RAGSystemError):
    """Raised when document processing fails"""

    pass


class EmbeddingError(RAGSystemError):
    """Raised when embedding generation fails"""

    pass


class VectorStoreError(RAGSystemError):
    """Raised when vector store operations fail"""

    pass


class RetrievalError(RAGSystemError):
    """Raised when document retrieval fails"""

    pass


class GenerationError(RAGSystemError):
    """Raised when answer generation fails"""

    pass


class ConfigurationError(RAGSystemError):
    """Raised when configuration is invalid"""

    pass


class MigrationError(RAGSystemError):
    """Raised when migration operations fail"""

    pass


class EvaluationError(RAGSystemError):
    """Raised when evaluation operations fail"""

    pass
