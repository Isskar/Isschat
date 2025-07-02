"""
Base connector interface for data sources with sync capabilities.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

from ...core.interfaces import Document


class SyncMode(Enum):
    """Synchronization modes."""

    FULL = "full"  # Retrieve all documents
    INCREMENTAL = "incremental"  # Retrieve only changes since last sync
    SPECIFIC = "specific"  # Retrieve specific documents by ID


class SyncResult:
    """Result of a sync operation."""

    def __init__(self):
        self.success = False
        self.documents_retrieved = 0
        self.documents_new = 0
        self.documents_updated = 0
        self.documents_deleted = 0
        self.errors: List[str] = []
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get sync duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class BaseConnector(ABC):
    """Abstract base class for all data source connectors."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize connector with configuration."""
        self.config = config

    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Validate connection to the data source.

        Returns:
            True if connection is valid
        """
        pass

    @abstractmethod
    def extract(self) -> List[Document]:
        """
        Extract all documents from the source (full sync).

        Returns:
            List of extracted documents
        """
        pass

    def sync(self, mode: SyncMode = SyncMode.FULL, **kwargs) -> SyncResult:
        """
        Synchronize documents from the source.

        Args:
            mode: Synchronization mode
            **kwargs: Additional parameters for specific sync modes

        Returns:
            SyncResult with operation details
        """
        result = SyncResult()
        result.started_at = datetime.now()

        try:
            if mode == SyncMode.FULL:
                documents = self.extract()
                result.documents_retrieved = len(documents)
                result.documents_new = len(documents)  # Assume all are new in full sync

            elif mode == SyncMode.INCREMENTAL:
                since = kwargs.get("since")
                if not since:
                    raise ValueError("Incremental sync requires 'since' parameter")
                documents = self.get_changed_documents(since)
                result.documents_retrieved = len(documents)
                result.documents_updated = len(documents)

            elif mode == SyncMode.SPECIFIC:
                document_ids = kwargs.get("document_ids", [])
                if not document_ids:
                    raise ValueError("Specific sync requires 'document_ids' parameter")
                documents = self.get_documents_by_ids(document_ids)
                result.documents_retrieved = len(documents)
                result.documents_updated = len(documents)

            else:
                raise ValueError(f"Unsupported sync mode: {mode}")

            result.success = True
            result.completed_at = datetime.now()

        except Exception as e:
            result.errors.append(str(e))
            result.completed_at = datetime.now()

        return result

    def get_changed_documents(self, since: datetime) -> List[Document]:
        """
        Get documents that have changed since a given timestamp.

        Default implementation falls back to full extraction.
        Subclasses should override for efficient incremental sync.

        Args:
            since: Timestamp to check changes from

        Returns:
            List of changed documents
        """
        # Default implementation - subclasses should override
        return self.extract()

    def get_documents_by_ids(self, document_ids: List[str]) -> List[Document]:
        """
        Get specific documents by their IDs.

        Default implementation filters from full extraction.
        Subclasses should override for efficiency.

        Args:
            document_ids: List of document identifiers

        Returns:
            List of requested documents
        """
        # Default implementation - subclasses should override
        all_docs = self.extract()
        id_set = set(document_ids)

        # Try to match by page_id in metadata
        return [doc for doc in all_docs if doc.metadata.get("page_id") in id_set]

    def get_sync_capabilities(self) -> Dict[str, bool]:
        """
        Get connector sync capabilities.

        Returns:
            Dictionary of supported sync features
        """
        return {
            "full_sync": True,
            "incremental_sync": hasattr(self, "_supports_incremental") and self._supports_incremental(),
            "specific_sync": hasattr(self, "_supports_specific") and self._supports_specific(),
            "change_detection": hasattr(self, "get_changed_documents"),
            "bulk_retrieval": hasattr(self, "get_documents_by_ids"),
        }
