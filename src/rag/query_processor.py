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

        # Domain-specific synonym mapping for Isschat context
        self.semantic_mappings = {
            # Team/collaboration terms
            "collaborateurs": ["équipe", "team", "membres", "développeurs", "participants"],
            "équipe": ["collaborateurs", "team", "membres", "développeurs", "participants"],
            "team": ["équipe", "collaborateurs", "membres", "développeurs", "participants"],
            "membres": ["équipe", "collaborateurs", "team", "développeurs", "participants"],
            "développeurs": ["équipe", "collaborateurs", "team", "membres", "participants"],
            # Project/product terms
            "projet": ["application", "produit", "système", "plateforme", "solution"],
            "application": ["projet", "produit", "système", "plateforme", "solution"],
            "produit": ["projet", "application", "système", "plateforme", "solution"],
            "système": ["projet", "application", "produit", "plateforme", "solution"],
            "plateforme": ["projet", "application", "produit", "système", "solution"],
            # Technical terms
            "configuration": ["config", "paramètres", "réglages", "settings"],
            "config": ["configuration", "paramètres", "réglages", "settings"],
            "paramètres": ["configuration", "config", "réglages", "settings"],
            "réglages": ["configuration", "config", "paramètres", "settings"],
            # Documentation terms
            "documentation": ["docs", "guide", "manuel", "aide"],
            "docs": ["documentation", "guide", "manuel", "aide"],
            "guide": ["documentation", "docs", "manuel", "aide"],
            "manuel": ["documentation", "docs", "guide", "aide"],
            # Common French/English variations
            "fonctionnalités": ["features", "capacités", "options"],
            "features": ["fonctionnalités", "capacités", "options"],
            "utilisation": ["usage", "use", "utiliser"],
            "usage": ["utilisation", "use", "utiliser"],
        }

        # Intent patterns for better classification
        self.intent_patterns = {
            "team_info": [
                r"qui\s+sont\s+les\s+(collaborateurs|équipe|membres|développeurs)",
                r"(team|équipe|collaborateurs|membres)\s+(sur|de|du|dans)",
                r"(composition|responsabilités)\s+(équipe|team)",
                r"(vincent|nicolas|emin|fraillon|lambropoulos|calyaka)",
                r"(équipe|team|collaborateurs|membres|développeurs)\s+(développement|project|isschat)",
                r"(développement|project)\s+(équipe|team)",
                r"(développeurs|developers)\s+(du|of|on)\s+(système|project|isschat)",
            ],
            "project_info": [
                r"(qu est-ce que|what is|c est quoi)\s+(isschat|le projet)",
                r"(description|présentation|overview)\s+(du projet|d isschat)",
                r"(objectif|but|goal)\s+(du projet|d isschat)",
            ],
            "technical_info": [
                r"(comment|how)\s+(utiliser|use|configurer|configure)",
                r"(installation|setup|configuration)",
                r"(problème|error|erreur|bug)",
            ],
            "feature_info": [
                r"(fonctionnalités|features|capacités)",
                r"(peut|can)\s+(faire|do)",
                r"(options|paramètres|settings)",
            ],
        }

    @property
    def embedding_service(self):
        """Lazy loading of embedding service"""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def process_query(self, query: str) -> QueryProcessingResult:
        """
        Process a query with semantic understanding and expansion.

        Args:
            query: Original user query

        Returns:
            QueryProcessingResult with expanded queries and intent
        """
        try:
            # Clean and normalize query
            normalized_query = self._normalize_query(query)

            # Extract keywords
            keywords = self._extract_keywords(normalized_query)

            # Detect intent
            intent = self._classify_intent(normalized_query)

            # Generate semantic variations
            semantic_variations = self._generate_semantic_variations(normalized_query, keywords)

            # Create expanded queries
            expanded_queries = self._create_expanded_queries(normalized_query, semantic_variations)

            # Calculate confidence based on intent detection and semantic coverage
            confidence = self._calculate_confidence(intent, keywords, semantic_variations)

            result = QueryProcessingResult(
                original_query=query,
                expanded_queries=expanded_queries,
                intent=intent,
                keywords=keywords,
                semantic_variations=semantic_variations,
                confidence=confidence,
            )

            self.logger.debug(f"Query processed: '{query}' -> {len(expanded_queries)} variations, intent: {intent}")
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

    def _classify_intent(self, query: str) -> str:
        """Classify query intent based on patterns with priority"""
        # Score each intent based on pattern matches
        intent_scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            intent_scores[intent] = score

        # Special handling for conflicting keywords
        # If query contains both team and technical terms, analyze context
        if intent_scores.get("team_info", 0) > 0 and intent_scores.get("technical_info", 0) > 0:
            # Check if team terms are more prominent
            team_terms = ["équipe", "team", "collaborateurs", "membres", "développeurs"]
            tech_terms = ["configuration", "installation", "problème", "erreur"]

            team_count = sum(1 for term in team_terms if term in query)
            tech_count = sum(1 for term in tech_terms if term in query)

            # If team terms are more prominent or equal, prefer team_info
            if team_count >= tech_count:
                return "team_info"

        # Special case: if only technical_info matched but we have team terms, check context
        elif intent_scores.get("technical_info", 0) > 0 and intent_scores.get("team_info", 0) == 0:
            team_terms = ["équipe", "team", "collaborateurs", "membres", "développeurs"]
            if any(term in query for term in team_terms):
                # Check if it's about team configuration rather than system configuration
                if "configuration" in query and any(term in query for term in team_terms):
                    return "team_info"

        # Return intent with highest score
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            if intent_scores[best_intent] > 0:
                return best_intent

        return "general"

    def _generate_semantic_variations(self, query: str, keywords: List[str]) -> List[str]:
        """Generate semantic variations of the query"""
        variations = []

        # Generate variations based on synonym mappings
        for keyword in keywords:
            if keyword in self.semantic_mappings:
                synonyms = self.semantic_mappings[keyword]
                for synonym in synonyms:
                    # Replace keyword with synonym in query
                    variation = query.replace(keyword, synonym)
                    if variation != query and variation not in variations:
                        variations.append(variation)

        # Add context-specific variations based on intent
        intent_variations = self._get_intent_variations(query)
        variations.extend(intent_variations)

        return variations

    def _get_intent_variations(self, query: str) -> List[str]:
        """Generate intent-specific query variations"""
        variations = []

        # Team info variations
        if any(word in query for word in ["collaborateurs", "équipe", "team", "membres"]):
            variations.extend(
                [
                    "équipe composition responsabilités",
                    "membres développeurs isschat",
                    "vincent nicolas emin fraillon lambropoulos calyaka",
                    "team composition isschat project",
                    "collaborateurs projet isschat",
                ]
            )

        # Project info variations
        if any(word in query for word in ["projet", "isschat", "application"]):
            variations.extend(
                [
                    "isschat description présentation",
                    "projet objectif but",
                    "application fonctionnalités",
                    "système isschat overview",
                ]
            )

        return variations

    def _create_expanded_queries(self, original_query: str, semantic_variations: List[str]) -> List[str]:
        """Create final list of expanded queries"""
        expanded_queries = [original_query]

        # Add semantic variations
        for variation in semantic_variations:
            if variation not in expanded_queries:
                expanded_queries.append(variation)

        # Limit to reasonable number of queries
        return expanded_queries[:5]

    def _calculate_confidence(self, intent: str, keywords: List[str], variations: List[str]) -> float:
        """Calculate confidence score for query processing"""
        confidence = 0.5  # Base confidence

        # Boost confidence for recognized intent
        if intent != "general":
            confidence += 0.2

        # Boost confidence for meaningful keywords
        if len(keywords) > 0:
            confidence += min(0.2, len(keywords) * 0.05)

        # Boost confidence for semantic variations found
        if len(variations) > 0:
            confidence += min(0.1, len(variations) * 0.02)

        return min(1.0, confidence)

    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            embedding1 = self.embedding_service.encode_single(text1)
            embedding2 = self.embedding_service.encode_single(text2)
            return self.embedding_service.similarity(embedding1, embedding2)
        except Exception as e:
            self.logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.0

    def expand_query_with_embeddings(self, query: str, candidate_texts: List[str], threshold: float = 0.7) -> List[str]:
        """
        Expand query based on semantic similarity with candidate texts.
        Useful for finding semantically similar content even with different keywords.
        """
        try:
            query_embedding = self.embedding_service.encode_single(query)
            expanded_terms = []

            for text in candidate_texts:
                text_embedding = self.embedding_service.encode_single(text)
                similarity = self.embedding_service.similarity(query_embedding, text_embedding)

                if similarity >= threshold:
                    expanded_terms.append(text)

            return expanded_terms

        except Exception as e:
            self.logger.error(f"Embedding-based query expansion failed: {e}")
            return []
