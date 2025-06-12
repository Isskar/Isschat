"""
Document quality filtering.
"""

from typing import List, Dict, Any, Optional
from src.core.interfaces import Document


class DocumentFilter:
    """Filter documents based on quality criteria."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the document filter.

        Args:
            config: Filtering configuration
        """
        self.config = config or {}
        self.min_length = self.config.get("min_length", 50)
        self.max_length = self.config.get("max_length", 10000)
        self.excluded_patterns = self.config.get("excluded_patterns", [])

    def filter_documents(self, documents: List[Document]) -> List[Document]:
        """
        Filter documents based on quality criteria.

        Args:
            documents: List of documents to filter

        Returns:
            List[Document]: Filtered documents
        """
        filtered_docs = []

        for doc in documents:
            if self._is_valid_document(doc):
                filtered_docs.append(doc)

        return filtered_docs

    def _is_valid_document(self, document: Document) -> bool:
        """
        Check if a document meets quality criteria.

        Args:
            document: Document to verify

        Returns:
            bool: True if document is valid
        """
        content = document.page_content.strip()

        # Check length
        if len(content) < self.min_length or len(content) > self.max_length:
            return False

        # Check excluded patterns
        for pattern in self.excluded_patterns:
            if pattern.lower() in content.lower():
                return False

        # Check that content is not empty or only whitespace
        if not content or content.isspace():
            return False

        return True

    def get_filter_stats(self, original_docs: List[Document], filtered_docs: List[Document]) -> Dict[str, Any]:
        """
        Return filtering statistics.

        Args:
            original_docs: Original documents
            filtered_docs: Filtered documents

        Returns:
            Dict: Filtering statistics
        """
        return {
            "original_count": len(original_docs),
            "filtered_count": len(filtered_docs),
            "removed_count": len(original_docs) - len(filtered_docs),
            "retention_rate": len(filtered_docs) / len(original_docs) if original_docs else 0,
        }
