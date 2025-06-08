"""
Post-processing for documents after chunking.
"""

from typing import List, Dict, Any, Optional

# Use absolute imports with fallbacks
try:
    from data_pipeline.extractors.base_extractor import Document
except ImportError:
    try:
        from src.data_pipeline.extractors.base_extractor import Document
    except ImportError:
        from ..extractors.base_extractor import Document


class PostProcessor:
    """Post-processes documents after chunking."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the post-processor.

        Args:
            config: Post-processing configuration
        """
        self.config = config or {}
        self.clean_whitespace = self.config.get("clean_whitespace", True)
        self.normalize_text = self.config.get("normalize_text", True)
        self.add_metadata = self.config.get("add_metadata", True)

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Post-process documents.

        Args:
            documents: List of documents to process

        Returns:
            List[Document]: Post-processed documents
        """
        processed_docs = []

        for doc in documents:
            processed_doc = self._process_document(doc)
            processed_docs.append(processed_doc)

        return processed_docs

    def _process_document(self, document: Document) -> Document:
        """
        Process a single document.

        Args:
            document: Document to process

        Returns:
            Document: Processed document
        """
        content = document.content
        metadata = document.metadata.copy()

        # Clean whitespace
        if self.clean_whitespace:
            content = self._clean_whitespace(content)

        # Normalize text
        if self.normalize_text:
            content = self._normalize_text(content)

        # Add processing metadata
        if self.add_metadata:
            metadata.update({"processed": True, "processing_steps": ["whitespace_cleaning", "text_normalization"]})

        return Document(content=content, metadata=metadata)

    def _clean_whitespace(self, text: str) -> str:
        """
        Clean excessive whitespace from text.

        Args:
            text: Text to clean

        Returns:
            str: Cleaned text
        """
        # Remove excessive whitespace
        import re

        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text formatting.

        Args:
            text: Text to normalize

        Returns:
            str: Normalized text
        """
        # Basic text normalization
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove excessive newlines
        import re

        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def get_processing_stats(self, original_docs: List[Document], processed_docs: List[Document]) -> Dict[str, Any]:
        """
        Get processing statistics.

        Args:
            original_docs: Original documents
            processed_docs: Processed documents

        Returns:
            Dict: Processing statistics
        """
        original_total_length = sum(len(doc.content) for doc in original_docs)
        processed_total_length = sum(len(doc.content) for doc in processed_docs)

        return {
            "original_count": len(original_docs),
            "processed_count": len(processed_docs),
            "original_total_length": original_total_length,
            "processed_total_length": processed_total_length,
            "compression_ratio": processed_total_length / original_total_length if original_total_length > 0 else 0,
        }
