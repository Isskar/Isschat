"""
Confluence-specific ingestion pipeline.
"""

from typing import List
from ..core.interfaces import Document
from .base_pipeline import BaseIngestionPipeline
from .connectors.confluence_connector import ConfluenceConnector
from .processors.chunker import ConfluenceChunker


class ConfluenceIngestionPipeline(BaseIngestionPipeline):
    """Pipeline spécialisée pour l'ingestion Confluence."""

    def __init__(self):
        """Initialize Confluence pipeline."""
        super().__init__()
        self._extractor = None

        # Use Confluence-specific chunker with confluence_sections strategy
        self.chunker = ConfluenceChunker(
            {"chunk_size": self.config.chunk_size, "chunk_overlap": self.config.chunk_overlap}
        )

    def get_source_name(self) -> str:
        """Return source name."""
        return "confluence"

    def _get_extractor(self) -> ConfluenceConnector:
        """Get or create Confluence extractor."""
        if self._extractor is None:
            # Validate configuration first
            if not self._validate_confluence_config():
                raise ValueError("Invalid Confluence configuration")

            # Create extractor with configuration
            extractor_config = {
                "confluence_private_api_key": self.config.confluence_api_key,
                "confluence_space_key": self.config.confluence_space_key,
                "confluence_space_name": self.config.confluence_space_name,
                "confluence_email_address": self.config.confluence_email,
                "confluence_url": self.config.confluence_url,
            }

            self._extractor = ConfluenceConnector(extractor_config)

            # Validate connection
            if not self._extractor.validate_connection():
                raise ValueError("Unable to connect to Confluence")

        return self._extractor

    def extract_documents(self) -> List[Document]:
        """Extract documents from Confluence."""
        extractor = self._get_extractor()

        self.logger.info("📥 Extracting documents from Confluence")
        documents = extractor.extract()

        if not documents:
            raise ValueError("No documents extracted from Confluence")

        self.logger.info(f"📄 {len(documents)} documents extracted from Confluence")
        return documents

    def _validate_confluence_config(self) -> bool:
        """Validate Confluence configuration."""
        required_configs = [
            ("confluence_api_key", "CONFLUENCE_PRIVATE_API_KEY"),
            ("confluence_space_key", "CONFLUENCE_SPACE_KEY"),
            ("confluence_space_name", "CONFLUENCE_SPACE_NAME"),
            ("confluence_email", "CONFLUENCE_EMAIL_ADDRESS"),
        ]

        missing = []
        for config_attr, env_name in required_configs:
            if not getattr(self.config, config_attr, None):
                missing.append(env_name)

        if missing:
            self.logger.error(f"Missing Confluence configuration: {missing}")
            return False

        return True

    def check_connection(self) -> bool:
        """Check if Confluence connection is working."""
        try:
            extractor = self._get_extractor()
            return extractor.validate_connection()
        except Exception as e:
            self.logger.error(f"Confluence connection check failed: {e}")
            return False


def create_confluence_pipeline() -> ConfluenceIngestionPipeline:
    """Factory function to create Confluence pipeline."""
    return ConfluenceIngestionPipeline()
