"""
Post-processing for documents after chunking.
"""

from typing import List, Dict, Any, Optional

from src.core.interfaces import Document


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
        self.enrich_content = self.config.get("enrich_content", True)

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
        content = document.page_content
        metadata = document.metadata.copy()

        # Clean whitespace
        if self.clean_whitespace:
            content = self._clean_whitespace(content)

        # Normalize text
        if self.normalize_text:
            content = self._normalize_text(content)

        # Enrich content with metadata for better search
        if self.enrich_content:
            content = self._enrich_content_with_metadata(content, metadata)

        # Add processing metadata
        if self.add_metadata:
            processing_steps = ["whitespace_cleaning", "text_normalization"]
            if self.enrich_content:
                processing_steps.append("content_enrichment")
            metadata.update({"processed": True, "processing_steps": processing_steps})

        return Document(page_content=content, metadata=metadata)

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

    def _enrich_content_with_metadata(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Enrich chunk content with important metadata for better search.

        Args:
            content: Original chunk content
            metadata: Document metadata

        Returns:
            str: Enriched content
        """
        enrichment_parts = []

        # Add title if available
        title = metadata.get("title", "").strip()
        if title:
            enrichment_parts.append(f"Title: {title}")

        # Add filename/page name if available
        filename = metadata.get("filename", metadata.get("page_name", metadata.get("source", ""))).strip()
        if filename and filename != title:
            enrichment_parts.append(f"Document: {filename}")

        # Add URL if available (for Confluence)
        url = metadata.get("url", "").strip()
        if url:
            enrichment_parts.append(f"URL: {url}")

        # Build enriched content
        if enrichment_parts:
            enrichment_header = " | ".join(enrichment_parts)
            return f"[{enrichment_header}]\n\n{content}"

        return content

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
