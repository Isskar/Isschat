"""
Query preprocessing for intent analysis and keyword extraction.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class QueryAnalysis:
    """Result of query analysis."""

    original_query: str
    processed_query: str
    intent: str
    entities: List[str]
    keywords: List[str]
    confidence: float


class QueryProcessor:
    """
    Processes user queries to extract intent and optimize for search.
    """

    def __init__(self):
        """Initialize the query processor."""
        # Common project/entity patterns
        self.project_patterns = [
            r"(?:projet|project)\s+([A-Za-z0-9_-]+)",
            r"(?:parle(?:\s+moi)?(?:\s+d[eu])?|tell\s+me\s+about)\s+(?:le\s+projet\s+)?([A-Za-z0-9_-]+)",
            r"(?:qu'est-ce que|what\s+is)\s+(?:le\s+projet\s+)?([A-Za-z0-9_-]+)",
            r"(?:information(?:s)?|info(?:s)?)\s+(?:sur|about)\s+(?:le\s+projet\s+)?([A-Za-z0-9_-]+)",
        ]

        # Noise words to remove
        self.noise_words = {
            "french": [
                "peux",
                "tu",
                "me",
                "parler",
                "de",
                "du",
                "le",
                "la",
                "les",
                "un",
                "une",
                "des",
                "pouvez",
                "vous",
                "pourriez",
                "hello",
                "salut",
                "bonjour",
                "bonsoir",
                "stp",
                "s'il",
                "te",
                "plaît",
                "plait",
                "merci",
                "svp",
                "que",
                "qu'est-ce",
                "comment",
                "où",
                "quand",
                "pourquoi",
                "est",
                "sont",
                "était",
                "étaient",
                "sera",
                "seront",
                "avoir",
                "être",
                "faire",
                "aller",
                "venir",
                "voir",
                "savoir",
                "donner",
                "prendre",
                "vouloir",
                "pouvoir",
                "devoir",
            ],
            "english": [
                "can",
                "you",
                "me",
                "tell",
                "about",
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "hello",
                "hi",
                "please",
                "thanks",
                "thank",
                "what",
                "how",
                "where",
                "when",
                "why",
                "is",
                "are",
                "was",
                "were",
                "will",
                "would",
                "could",
                "should",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "get",
                "go",
                "come",
                "see",
                "know",
                "give",
                "take",
                "want",
                "need",
            ],
        }

        # Intent patterns
        self.intent_patterns = {
            "project_info": [
                r"(?:parle(?:\s+moi)?(?:\s+d[eu])?|tell\s+me\s+about|information(?:s)?|info(?:s)?)",
                r"(?:qu\'est-ce\s+que|what\s+is|c\'est\s+quoi)",
                r"(?:présent(?:er|ation)|describe|explain)",
            ],
            "troubleshooting": [
                r"(?:problème|problem|erreur|error|bug|issue)",
                r"(?:ne\s+fonctionne\s+pas|doesn\'t\s+work|not\s+working)",
                r"(?:comment\s+résoudre|how\s+to\s+fix|solution)",
            ],
            "how_to": [
                r"(?:comment|how\s+to|how\s+do\s+i)",
                r"(?:étapes|steps|procédure|procedure)",
                r"(?:tutoriel|tutorial|guide)",
            ],
        }

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

        # Extract entities (project names, etc.)
        entities = self._extract_entities(normalized_query)

        # Detect intent
        intent = self._detect_intent(normalized_query)

        # Extract keywords
        keywords = self._extract_keywords(normalized_query)

        # Build optimized search query
        processed_query = self._build_search_query(entities, keywords, intent)

        # Calculate confidence
        confidence = self._calculate_confidence(entities, keywords, intent)

        return QueryAnalysis(
            original_query=query,
            processed_query=processed_query,
            intent=intent,
            entities=entities,
            keywords=keywords,
            confidence=confidence,
        )

    def _normalize_query(self, query: str) -> str:
        """Normalize the query (lowercase, clean punctuation)."""
        # Convert to lowercase
        normalized = query.lower().strip()

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Clean punctuation but keep important chars
        normalized = re.sub(r"[^\w\s\'-]", " ", normalized)

        return normalized

    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities (project names, etc.) from query."""
        entities = []

        # Try project patterns
        for pattern in self.project_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entity = match.group(1).strip()
                if entity and len(entity) > 1:
                    entities.append(entity)

        # Look for capitalized words that might be project names
        words = query.split()
        for word in words:
            # Check if word looks like a project name (contains caps or numbers)
            if (
                len(word) > 2
                and (any(c.isupper() for c in word) or any(c.isdigit() for c in word))
                and word.lower() not in self.noise_words["french"]
                and word.lower() not in self.noise_words["english"]
            ):
                entities.append(word)

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower not in seen:
                seen.add(entity_lower)
                unique_entities.append(entity)

        return unique_entities

    def _detect_intent(self, query: str) -> str:
        """Detect the intent of the query."""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent

        return "general_info"

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        words = query.split()
        keywords = []

        for word in words:
            # Skip noise words
            if (
                word.lower() not in self.noise_words["french"]
                and word.lower() not in self.noise_words["english"]
                and len(word) > 2
            ):
                keywords.append(word)

        return keywords

    def _build_search_query(self, entities: List[str], keywords: List[str], intent: str) -> str:
        """Build an optimized search query."""
        # Start with entities (highest priority)
        query_parts = entities.copy()

        # Add relevant keywords
        for keyword in keywords:
            if keyword.lower() not in [e.lower() for e in entities]:
                query_parts.append(keyword)

        # Add intent-specific terms
        if intent == "project_info":
            # For project info, prioritize the entity names
            pass
        elif intent == "troubleshooting":
            query_parts.append("problème")
        elif intent == "how_to":
            query_parts.append("comment")

        # Join and clean
        search_query = " ".join(query_parts).strip()

        # If no meaningful terms found, return original keywords
        if not search_query:
            search_query = " ".join(keywords)

        return search_query

    def _calculate_confidence(self, entities: List[str], keywords: List[str], intent: str) -> float:
        """Calculate confidence in the query analysis."""
        confidence = 0.5  # Base confidence

        # Boost confidence for found entities
        if entities:
            confidence += 0.3

        # Boost for clear intent
        if intent != "general_info":
            confidence += 0.2

        # Boost for meaningful keywords
        if len(keywords) >= 2:
            confidence += 0.1

        return min(confidence, 1.0)

    def get_debug_info(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Get debug information about query processing."""
        return {
            "original_query": analysis.original_query,
            "processed_query": analysis.processed_query,
            "intent": analysis.intent,
            "entities": analysis.entities,
            "keywords": analysis.keywords,
            "confidence": analysis.confidence,
            "transformation": f"'{analysis.original_query}' → '{analysis.processed_query}'",
        }
