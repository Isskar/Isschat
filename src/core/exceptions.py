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


class StorageAccessError(RAGSystemError):
    """Raised when storage access fails due to authentication or permission issues"""

    def __init__(self, message: str, storage_type: str = None, original_error: Exception = None):
        super().__init__(message)
        self.storage_type = storage_type
        self.original_error = original_error


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
