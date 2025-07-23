"""
Conversation context tracking and entity extraction for context-aware RAG retrieval.
Maintains conversation state and enriches queries with relevant context from previous turns.
"""

import logging
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

from ..embeddings import get_embedding_service


@dataclass
class EntityMention:
    """Represents an entity mentioned in conversation"""

    entity: str
    entity_type: str
    mention_count: int = 1
    first_mentioned: datetime = field(default_factory=datetime.now)
    last_mentioned: datetime = field(default_factory=datetime.now)
    context_phrases: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ConversationTurn:
    """Represents a single conversation turn"""

    query: str
    response: str
    timestamp: datetime
    entities: List[EntityMention]
    topics: List[str]
    intent: str


@dataclass
class ConversationContext:
    """Tracks context across conversation turns"""

    conversation_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    active_entities: Dict[str, EntityMention] = field(default_factory=dict)
    topic_evolution: List[str] = field(default_factory=list)
    context_window: int = 5  # Number of recent turns to consider for context


class ConversationContextTracker:
    """
    Tracks conversation context and extracts key entities for context-aware retrieval.
    Maintains state across conversation turns and provides context enrichment for queries.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._embedding_service = None

        # Conversation contexts indexed by conversation_id
        self.contexts: Dict[str, ConversationContext] = {}

        # Entity patterns for recognition (French and English)
        self.entity_patterns = {
            "project": [
                r"\b(teora|téora)\b",
                r"\b(isschat)\b",
                r"\b(project|projet)\s+([A-Z][a-zA-Z0-9-_]*)",
                r"\b(application|app)\s+([A-Z][a-zA-Z0-9-_]*)",
                r"\b(système|system)\s+([A-Z][a-zA-Z0-9-_]*)",
            ],
            "person": [
                r"\b(vincent|nicolas|emin|fraillon|lambropoulos|calyaka)\b",
                r"\b([A-Z][a-z]+)\s+(fraillon|lambropoulos|calyaka)\b",
                r"\b(mr|mme|m\.)\s+([A-Z][a-z]+)\b",
            ],
            "team": [
                r"\b(équipe|team)(?:\s+(?:de\s+)?([A-Z][a-zA-Z0-9-_]*))?",
                r"\b(collaborateurs|développeurs|members)\s+(?:sur|de|du)\s+([A-Z][a-zA-Z0-9-_]*)",
                r"\b(group|groupe)\s+([A-Z][a-zA-Z0-9-_]*)",
            ],
            "feature": [
                r"\b(fonctionnalité|feature)(?:\s+([a-zA-Z0-9-_]+))?",
                r"\b(module|component|composant)\s+([a-zA-Z0-9-_]+)",
                r"\b(service|api)\s+([a-zA-Z0-9-_]+)",
                r"\b(configuration)\b",
            ],
            "location": [
                r"\b(dans|in)\s+(?:le|la|les)?\s*([A-Z][a-zA-Z0-9-_]*)",
                r"\b(sur|on)\s+(?:le|la|les)?\s*([A-Z][a-zA-Z0-9-_]*)",
                r"\b(répertoire|directory|folder|dossier)\s+([a-zA-Z0-9-_/]+)",
            ],
            "technology": [
                r"\b(azure|aws|gcp|kubernetes|docker|streamlit|python|javascript|react|vue)\b",
                r"\b(database|db|mongodb|postgresql|mysql|redis)\b",
                r"\b(api|rest|graphql|grpc|http|https)\b",
            ],
        }

    @property
    def embedding_service(self):
        """Lazy loading of embedding service"""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def track_conversation_turn(
        self, conversation_id: str, query: str, response: str, intent: str = "general"
    ) -> ConversationTurn:
        """
        Track a new conversation turn and extract entities/topics.

        Args:
            conversation_id: Unique conversation identifier
            query: User query
            response: System response
            intent: Detected intent from query processing

        Returns:
            ConversationTurn object with extracted entities
        """
        try:
            # Initialize context if not exists
            if conversation_id not in self.contexts:
                self.contexts[conversation_id] = ConversationContext(conversation_id=conversation_id)

            context = self.contexts[conversation_id]

            # Extract entities from query and response
            query_entities = self._extract_entities(query)
            response_entities = self._extract_entities(response)

            # Combine and deduplicate entities
            all_entities = self._merge_entities(query_entities + response_entities)

            # Extract topics from query
            topics = self._extract_topics(query, response)

            # Create conversation turn
            turn = ConversationTurn(
                query=query,
                response=response,
                timestamp=datetime.now(),
                entities=all_entities,
                topics=topics,
                intent=intent,
            )

            # Update context
            context.turns.append(turn)
            self._update_active_entities(context, all_entities)
            self._update_topic_evolution(context, topics)

            # Keep only recent turns within context window
            if len(context.turns) > context.context_window:
                context.turns = context.turns[-context.context_window :]

            self.logger.debug(f"Tracked turn for {conversation_id}: {len(all_entities)} entities, topics: {topics}")
            return turn

        except Exception as e:
            self.logger.error(f"Error tracking conversation turn: {e}")
            # Return minimal turn on error
            return ConversationTurn(
                query=query, response=response, timestamp=datetime.now(), entities=[], topics=[], intent=intent
            )

    def enrich_query_with_context(
        self, conversation_id: str, current_query: str, max_context_entities: int = 5
    ) -> Tuple[str, Dict[str, any]]:
        """
        Enrich current query with relevant context from previous conversation turns.

        Args:
            conversation_id: Conversation identifier
            current_query: Current user query
            max_context_entities: Maximum number of context entities to include

        Returns:
            Tuple of (enriched_query, context_metadata)
        """
        try:
            if conversation_id not in self.contexts:
                return current_query, {"context_applied": False}

            context = self.contexts[conversation_id]

            # Get relevant context entities
            relevant_entities = self._get_relevant_context_entities(context, current_query, max_context_entities)

            if not relevant_entities:
                return current_query, {"context_applied": False}

            # Build enriched query
            enriched_query = self._build_enriched_query(current_query, relevant_entities)

            # Build context metadata
            context_metadata = {
                "context_applied": True,
                "original_query": current_query,
                "enriched_query": enriched_query,
                "context_entities": [
                    {
                        "entity": entity.entity,
                        "type": entity.entity_type,
                        "confidence": entity.confidence,
                        "mention_count": entity.mention_count,
                    }
                    for entity in relevant_entities
                ],
                "context_turns_used": len(context.turns),
                "active_topics": context.topic_evolution[-3:] if context.topic_evolution else [],
            }

            self.logger.info(f"Enriched query for {conversation_id}: '{current_query}' -> '{enriched_query}'")
            return enriched_query, context_metadata

        except Exception as e:
            self.logger.error(f"Error enriching query with context: {e}")
            return current_query, {"context_applied": False, "error": str(e)}

    def _extract_entities(self, text: str) -> List[EntityMention]:
        """Extract entities from text using pattern matching"""
        entities = []
        text_lower = text.lower()

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    # Extract the main entity based on groups available
                    groups = match.groups()
                    if groups:
                        # Find the most specific non-None group (usually the last meaningful one)
                        entity_text = None
                        for group in reversed(groups):
                            if group and group.strip():
                                entity_text = group.strip()
                                break

                        # If no meaningful group found, use the first group
                        if not entity_text:
                            entity_text = groups[0].strip() if groups[0] else match.group(0).strip()
                    else:
                        # No groups, use full match
                        entity_text = match.group(0).strip()

                    if len(entity_text) > 1:  # Filter out very short matches
                        # Get surrounding context
                        start = max(0, match.start() - 20)
                        end = min(len(text), match.end() + 20)
                        context_phrase = text[start:end].strip()

                        entities.append(
                            EntityMention(
                                entity=entity_text,
                                entity_type=entity_type,
                                context_phrases=[context_phrase],
                                confidence=self._calculate_entity_confidence(entity_text, entity_type),
                            )
                        )

        return entities

    def _extract_topics(self, query: str, response: str) -> List[str]:
        """Extract main topics from query and response"""
        topics = []
        combined_text = f"{query} {response}".lower()

        # Domain-specific topic keywords
        topic_keywords = {
            "team_management": ["équipe", "team", "collaborateurs", "membres", "développeurs", "responsabilités"],
            "project_info": ["projet", "project", "application", "système", "isschat", "teora"],
            "configuration": ["configuration", "config", "paramètres", "settings", "setup"],
            "features": ["fonctionnalités", "features", "capacités", "options", "modules"],
            "documentation": ["documentation", "docs", "guide", "manuel", "aide", "help"],
            "technical_support": ["problème", "error", "bug", "erreur", "issue", "support"],
            "deployment": ["déploiement", "deployment", "installation", "setup", "production"],
            "integration": ["intégration", "integration", "api", "connexion", "connection"],
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                topics.append(topic)

        return topics

    def _merge_entities(self, entities: List[EntityMention]) -> List[EntityMention]:
        """Merge duplicate entities and update mention counts"""
        entity_map = {}

        for entity in entities:
            key = f"{entity.entity.lower()}_{entity.entity_type}"
            if key in entity_map:
                existing = entity_map[key]
                existing.mention_count += 1
                existing.last_mentioned = entity.first_mentioned
                existing.context_phrases.extend(entity.context_phrases)
            else:
                entity_map[key] = entity

        return list(entity_map.values())

    def _update_active_entities(self, context: ConversationContext, entities: List[EntityMention]):
        """Update active entities in conversation context"""
        for entity in entities:
            key = f"{entity.entity.lower()}_{entity.entity_type}"
            if key in context.active_entities:
                existing = context.active_entities[key]
                existing.mention_count += entity.mention_count
                existing.last_mentioned = entity.last_mentioned
                existing.context_phrases.extend(entity.context_phrases)
                # Keep only recent context phrases
                existing.context_phrases = existing.context_phrases[-5:]
            else:
                context.active_entities[key] = entity

    def _update_topic_evolution(self, context: ConversationContext, topics: List[str]):
        """Update topic evolution in conversation context"""
        for topic in topics:
            if not context.topic_evolution or context.topic_evolution[-1] != topic:
                context.topic_evolution.append(topic)

        # Keep only recent topics
        if len(context.topic_evolution) > 10:
            context.topic_evolution = context.topic_evolution[-10:]

    def _get_relevant_context_entities(
        self, context: ConversationContext, current_query: str, max_entities: int
    ) -> List[EntityMention]:
        """Get most relevant context entities for current query"""
        if not context.active_entities:
            return []

        # Calculate relevance scores for each entity
        entity_scores = []
        current_query_lower = current_query.lower()

        for entity in context.active_entities.values():
            score = 0

            # Recency bonus (more recent = higher score)
            minutes_since_last = (datetime.now() - entity.last_mentioned).total_seconds() / 60
            recency_bonus = max(0, 1 - (minutes_since_last / 60))  # Decay over 1 hour
            score += recency_bonus * 0.3

            # Mention frequency bonus
            frequency_bonus = min(1.0, entity.mention_count / 5)
            score += frequency_bonus * 0.2

            # Semantic relevance to current query
            semantic_bonus = self._calculate_semantic_relevance(entity, current_query_lower)
            score += semantic_bonus * 0.4

            # Entity confidence bonus
            score += entity.confidence * 0.1

            entity_scores.append((entity, score))

        # Sort by score and return top entities
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        return [entity for entity, score in entity_scores[:max_entities] if score > 0.2]

    def _calculate_semantic_relevance(self, entity: EntityMention, query: str) -> float:
        """Calculate semantic relevance between entity and query"""
        # Simple keyword-based relevance (could be enhanced with embeddings)
        entity_lower = entity.entity.lower()

        # Direct mention
        if entity_lower in query:
            return 1.0

        # Partial match
        if any(word in query for word in entity_lower.split()):
            return 0.7

        # Context phrase similarity
        for context_phrase in entity.context_phrases:
            context_lower = context_phrase.lower()
            if any(word in query for word in context_lower.split() if len(word) > 3):
                return 0.5

        # Type-based relevance
        type_relevance = {
            "project": 0.3 if any(word in query for word in ["projet", "project", "application"]) else 0,
            "person": 0.3 if any(word in query for word in ["qui", "who", "équipe", "team"]) else 0,
            "team": 0.3 if any(word in query for word in ["équipe", "team", "collaborateurs"]) else 0,
            "feature": 0.2 if any(word in query for word in ["fonctionnalité", "feature", "module"]) else 0,
        }

        return type_relevance.get(entity.entity_type, 0)

    def _build_enriched_query(self, original_query: str, entities: List[EntityMention]) -> str:
        """Build enriched query with context entities"""
        # Start with original query
        enriched_parts = [original_query]

        # Group entities by type for better organization
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity.entity_type].append(entity.entity)

        # Add context in a structured way
        context_parts = []

        for entity_type, entity_list in entities_by_type.items():
            if entity_type == "project":
                context_parts.append(f"projet: {', '.join(entity_list)}")
            elif entity_type == "person":
                context_parts.append(f"personne: {', '.join(entity_list)}")
            elif entity_type == "team":
                context_parts.append(f"équipe: {', '.join(entity_list)}")
            elif entity_type == "feature":
                context_parts.append(f"fonctionnalité: {', '.join(entity_list)}")
            else:
                context_parts.append(f"{entity_type}: {', '.join(entity_list)}")

        if context_parts:
            enriched_parts.append(f"[contexte: {'; '.join(context_parts)}]")

        return " ".join(enriched_parts)

    def _calculate_entity_confidence(self, entity: str, entity_type: str) -> float:
        """Calculate confidence score for extracted entity"""
        confidence = 0.5

        # Known entities get higher confidence
        known_entities = {
            "project": ["teora", "téora", "isschat"],
            "person": ["vincent", "nicolas", "emin", "fraillon", "lambropoulos", "calyaka"],
            "technology": ["azure", "streamlit", "python", "javascript", "react"],
        }

        if entity_type in known_entities:
            if entity.lower() in known_entities[entity_type]:
                confidence = 0.9
            elif any(known in entity.lower() for known in known_entities[entity_type]):
                confidence = 0.7

        # Length-based confidence
        if len(entity) > 10:
            confidence += 0.1
        elif len(entity) < 3:
            confidence -= 0.2

        # Capitalization patterns
        if entity[0].isupper() and entity_type in ["person", "project"]:
            confidence += 0.1

        return max(0.1, min(1.0, confidence))

    def get_conversation_summary(self, conversation_id: str) -> Dict[str, any]:
        """Get summary of conversation context"""
        if conversation_id not in self.contexts:
            return {"exists": False}

        context = self.contexts[conversation_id]

        return {
            "exists": True,
            "conversation_id": conversation_id,
            "turn_count": len(context.turns),
            "active_entities": len(context.active_entities),
            "topics": context.topic_evolution,
            "entities_by_type": {
                entity_type: [
                    entity.entity for entity in context.active_entities.values() if entity.entity_type == entity_type
                ]
                for entity_type in set(e.entity_type for e in context.active_entities.values())
            },
            "last_turn_timestamp": context.turns[-1].timestamp.isoformat() if context.turns else None,
        }

    def clear_conversation_context(self, conversation_id: str):
        """Clear context for a specific conversation"""
        if conversation_id in self.contexts:
            del self.contexts[conversation_id]
            self.logger.info(f"Cleared context for conversation {conversation_id}")

    def clear_old_contexts(self, max_age_hours: int = 24):
        """Clear contexts older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []

        for conv_id, context in self.contexts.items():
            if context.turns and context.turns[-1].timestamp < cutoff_time:
                to_remove.append(conv_id)

        for conv_id in to_remove:
            del self.contexts[conv_id]

        if to_remove:
            self.logger.info(f"Cleared {len(to_remove)} old conversation contexts")
