"""
Query preprocessing for intent analysis and keyword extraction.
"""

import re
from dataclasses import dataclass


@dataclass
class QueryAnalysis:
    """Result of query analysis."""

    original_query: str
    normalized_query: str
    processed_query: str


class QueryProcessor:
    """
    Processes user queries to extract intent and optimize for search.
    """

    def __init__(self):
        """Initialize the query processor."""
        return

    def process_query(self, query: str) -> QueryAnalysis:
        """
        Process a user query to extract intent and optimize for search.

        Args:
            query: Original user query

        Returns:
            QueryAnalysis with processed information
        """
        # Normalize query
        normalized_query = self._normalize_query(query)

        processed_query = normalized_query

        return QueryAnalysis(original_query=query, normalized_query=normalized_query, processed_query=processed_query)

    def _normalize_query(self, query: str) -> str:
        """Normalize the query (lowercase, clean punctuation)."""
        # Convert to lowercase
        normalized = query.lower().strip()

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Clean punctuation but keep important chars
        normalized = re.sub(r"[^\w\s\'-]", " ", normalized)

        return normalized
