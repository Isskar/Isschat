"""
Query processor for semantic understanding and intent detection.
Handles query expansion, synonym matching, and intent classification.
"""

import logging
from typing import List
import re
from dataclasses import dataclass

from ..config import get_config
from ..embeddings import get_embedding_service


@dataclass
class QueryProcessingResult:
    """Result of query processing with expanded queries and intent"""

    original_query: str
    expanded_queries: List[str]
    intent: str
    keywords: List[str]
    semantic_variations: List[str]
    confidence: float


class QueryProcessor:
    """
    Advanced query processor with semantic understanding capabilities.
    Handles misleading keywords by expanding queries with semantic variations.
    """

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._embedding_service = None

        # Note: Semantic mappings and intent patterns removed in favor of LLM-based reformulation

    @property
    def embedding_service(self):
        """Lazy loading of embedding service"""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def process_query(self, query: str) -> QueryProcessingResult:
        """
        Process a query with basic normalization and keyword extraction.
        Legacy method kept for backward compatibility - semantic expansion now handled by ReformulationService.

        Args:
            query: Original user query

        Returns:
            QueryProcessingResult with basic processing
        """
        try:
            # Clean and normalize query
            normalized_query = self._normalize_query(query)

            # Extract keywords
            keywords = self._extract_keywords(normalized_query)

            # Simplified result - no intent classification or semantic variations
            result = QueryProcessingResult(
                original_query=query,
                expanded_queries=[normalized_query],
                intent="general",  # Legacy compatibility
                keywords=keywords,
                semantic_variations=[],  # No longer generated here
                confidence=0.8,  # Higher confidence for simpler processing
            )

            self.logger.debug(f"Query processed: '{query}' -> normalized and keywords extracted")
            return result

        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            # Return fallback result
            return QueryProcessingResult(
                original_query=query,
                expanded_queries=[query],
                intent="general",
                keywords=query.split(),
                semantic_variations=[],
                confidence=0.5,
            )

    def _normalize_query(self, query: str) -> str:
        """Normalize query text"""
        # Convert to lowercase
        query = query.lower().strip()

        # Remove extra whitespace
        query = re.sub(r"\s+", " ", query)

        # Remove punctuation but keep accents
        query = re.sub(r"[^\w\s\-àâäéèêëïîôöùûüÿñç]", " ", query)

        return query.strip()

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        # Stop words in French and English
        stop_words = {
            "le",
            "la",
            "les",
            "un",
            "une",
            "des",
            "du",
            "de",
            "da",
            "et",
            "ou",
            "est",
            "sont",
            "avec",
            "sur",
            "dans",
            "pour",
            "par",
            "qui",
            "que",
            "quoi",
            "comment",
            "où",
            "quand",
            "pourquoi",
            "the",
            "a",
            "an",
            "and",
            "or",
            "is",
            "are",
            "with",
            "on",
            "in",
            "for",
            "by",
            "who",
            "what",
            "where",
            "when",
            "why",
            "how",
            "this",
            "that",
            "these",
            "those",
            "can",
            "could",
            "should",
            "nous",
            "vous",
            "ils",
            "elles",
            "je",
            "tu",
            "il",
            "elle",
            "me",
            "te",
            "se",
            "nous",
            "vous",
            "moi",
            "toi",
            "lui",
            "elle",
            "eux",
            "elles",
            "mon",
            "ton",
            "son",
            "ma",
            "ta",
            "sa",
            "mes",
            "tes",
            "ses",
            "notre",
            "votre",
            "leur",
            "nos",
            "vos",
            "leurs",
        }

        words = query.split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        return keywords

    # All legacy semantic processing methods removed - functionality replaced by ReformulationService
