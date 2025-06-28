"""
Custom exceptions for the RAG system.
"""

from typing import Optional


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

    def __init__(self, message: str, storage_type: Optional[str] = None, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.storage_type = storage_type
        self.original_error = original_error


def create_azure_access_error(
    error_type: str, resource_name: str, error: Exception, help_message: str
) -> StorageAccessError:
    """
    Create a standardized Azure access error

    Args:
        error_type: Type of error (STORAGE, CONTAINER, etc.)
        resource_name: Name of the resource that failed
        error: Original exception
        help_message: Specific help message for this error type

    Returns:
        StorageAccessError with standardized message format
    """
    return StorageAccessError(
        f"❌ IMPOSSIBLE D'ACCÉDER AU {error_type} AZURE: "
        f"Cannot access {resource_name}:\n"
        f"Error: {str(error)}\n"
        f"Vous avez demandé d'utiliser Azure Storage mais l'accès est impossible.\n"
        f"{help_message}",
        storage_type="azure",
        original_error=error,
    )


class RebuildError(RAGSystemError):
    """Raised when rebuild operations fail due to storage access issues"""

    def __init__(self, message: str, storage_type: Optional[str] = None, requested_storage: Optional[str] = None):
        super().__init__(message)
        self.storage_type = storage_type
        self.requested_storage = requested_storage


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
