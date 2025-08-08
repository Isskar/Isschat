"""
SharePoint ingestion pipeline using Microsoft Graph API.
"""

from typing import List, Dict, Any

from .base_pipeline import BaseIngestionPipeline
from .connectors.sharepoint_connector import SharePointConnector
from ..core.interfaces import Document


class SharePointPipeline(BaseIngestionPipeline):
    """Ingestion pipeline for SharePoint Online documents."""

    def __init__(self, sharepoint_config: Dict[str, Any]):
        """
        Initialize SharePoint pipeline.

        Args:
            sharepoint_config: SharePoint connection configuration
                Required fields:
                - tenant_id: Azure AD tenant ID
                - client_id: Azure AD app client ID
                - client_secret: Azure AD app client secret
                - site_url: SharePoint site URL (e.g., "https://inergie.sharepoint.com/sites/ISSKAR")
        """
        super().__init__()

        self.sharepoint_config = sharepoint_config
        self.connector = SharePointConnector(sharepoint_config)

        self.logger.info(f"SharePoint pipeline initialized for site: {sharepoint_config.get('site_url')}")

    def extract_documents(self) -> List[Document]:
        """Extract documents from SharePoint."""
        self.logger.info("Starting SharePoint document extraction")

        # Validate connection first
        if not self.connector.validate_connection():
            raise ConnectionError("Failed to connect to SharePoint site")

        # Extract all documents
        documents = self.connector.extract()

        if not documents:
            self.logger.warning("No documents were extracted from SharePoint")
        else:
            self.logger.info(f"Successfully extracted {len(documents)} documents from SharePoint")

            # Log some statistics
            total_size = sum(doc.metadata.get("size", 0) for doc in documents)
            avg_size = total_size / len(documents) if documents else 0

            libraries = set(doc.metadata.get("library_name") for doc in documents)
            self.logger.info(f"Documents from {len(libraries)} libraries, average size: {avg_size:.0f} bytes")

        return documents

    def get_source_name(self) -> str:
        """Get the source name for this pipeline."""
        return "sharepoint"

    def sync_documents(self, mode: str = "full", **kwargs) -> Dict[str, Any]:
        """
        Synchronize documents from SharePoint.

        Args:
            mode: Sync mode ('full', 'incremental', or 'specific')
            **kwargs: Additional parameters for sync

        Returns:
            Sync result with statistics
        """
        from .connectors.base_connector import SyncMode

        self.logger.info(f"Starting SharePoint sync in {mode} mode")

        try:
            # Map string mode to enum
            sync_mode_map = {"full": SyncMode.FULL, "incremental": SyncMode.INCREMENTAL, "specific": SyncMode.SPECIFIC}

            if mode not in sync_mode_map:
                raise ValueError(f"Invalid sync mode: {mode}. Use 'full', 'incremental', or 'specific'")

            sync_mode = sync_mode_map[mode]

            # Perform sync
            result = self.connector.sync(sync_mode, **kwargs)

            if result.success:
                self.logger.info(
                    f"Sync completed: {result.documents_retrieved} documents retrieved "
                    f"in {result.duration_seconds:.1f}s"
                )
            else:
                self.logger.error(f"Sync failed: {result.errors}")

            return {
                "success": result.success,
                "mode": mode,
                "documents_retrieved": result.documents_retrieved,
                "documents_new": result.documents_new,
                "documents_updated": result.documents_updated,
                "documents_deleted": result.documents_deleted,
                "duration_seconds": result.duration_seconds,
                "errors": result.errors,
                "started_at": result.started_at.isoformat() if result.started_at else None,
                "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            }

        except Exception as e:
            self.logger.error(f"Sync operation failed: {e}")
            return {"success": False, "mode": mode, "error": str(e), "documents_retrieved": 0}

    def get_connector_capabilities(self) -> Dict[str, bool]:
        """Get SharePoint connector capabilities."""
        return self.connector.get_sync_capabilities()

    def test_connection(self) -> Dict[str, Any]:
        """Test connection to SharePoint."""
        try:
            success = self.connector.validate_connection()

            return {
                "success": success,
                "site_url": self.sharepoint_config.get("site_url"),
                "message": "Connection successful" if success else "Connection failed",
            }

        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return {"success": False, "site_url": self.sharepoint_config.get("site_url"), "error": str(e)}

    def get_site_info(self) -> Dict[str, Any]:
        """Get information about the SharePoint site."""
        try:
            # This would make a call to get site information
            # For now, return basic config info
            return {
                "site_url": self.sharepoint_config.get("site_url"),
                "tenant_id": self.sharepoint_config.get("tenant_id"),
                "client_id": self.sharepoint_config.get("client_id"),
                "capabilities": self.get_connector_capabilities(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get site info: {e}")
            return {"error": str(e)}


def create_sharepoint_pipeline(tenant_id: str, client_id: str, client_secret: str, site_url: str) -> SharePointPipeline:
    """
    Convenience function to create a SharePoint pipeline.

    Args:
        tenant_id: Azure AD tenant ID
        client_id: Azure AD app client ID
        client_secret: Azure AD app client secret
        site_url: SharePoint site URL

    Returns:
        Configured SharePoint pipeline
    """
    config = {"tenant_id": tenant_id, "client_id": client_id, "client_secret": client_secret, "site_url": site_url}

    return SharePointPipeline(config)
