"""
Query reformulation service for resolving coreferences and clarifying implicit references.
Uses LLM to reformulate user queries with context from recent conversation exchanges.
"""

import logging
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..config import get_config


@dataclass
class ConversationExchange:
    """Represents a single exchange in the conversation"""

    user_message: str
    assistant_message: str


class ReformulationService:
    """
    Service for reformulating user queries to resolve coreferences and clarify implicit references.
    Takes the user query and recent conversation context to produce a clear, autonomous query.
    """

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY required for reformulation service")

    def reformulate_query(
        self, user_query: str, recent_exchanges: List[ConversationExchange], max_exchanges: int = 3
    ) -> str:
        """
        Reformulate a user query using recent conversation context.

        Args:
            user_query: The original user query
            recent_exchanges: List of recent conversation exchanges
            max_exchanges: Maximum number of recent exchanges to consider

        Returns:
            Reformulated query that is autonomous and clear
        """
        try:
            self.logger.info(f"🔄 REFORMULATION START: '{user_query}'")
            self.logger.info(f"📝 Recent exchanges count: {len(recent_exchanges)}")

            # Log the exchanges for debugging
            for i, exchange in enumerate(recent_exchanges[-3:], 1):
                assistant_preview = exchange.assistant_message[:50]
                self.logger.info(
                    f"  Exchange {i}: User='{exchange.user_message}' -> Assistant='{assistant_preview}...'"
                )

            # Check autonomy
            is_autonomous = self._is_query_autonomous(user_query)
            self.logger.info(f"🔍 Query autonomy check: {is_autonomous}")

            # If no context or query is already clear, return original
            if not recent_exchanges:
                self.logger.info("❌ No recent exchanges - skipping reformulation")
                return user_query

            if is_autonomous:
                self.logger.info("✅ Query is autonomous - skipping reformulation")
                return user_query

            # Limit to recent exchanges
            context_exchanges = recent_exchanges[-max_exchanges:] if recent_exchanges else []
            self.logger.info(f"📚 Using {len(context_exchanges)} exchanges for context")

            # Build reformulation prompt
            prompt = self._build_reformulation_prompt(user_query, context_exchanges)
            self.logger.info(f"📝 Reformulation prompt built (length: {len(prompt)} chars)")

            # Call LLM for reformulation
            self.logger.info("🤖 Calling LLM for reformulation...")
            reformulated = self._call_llm_for_reformulation(prompt)

            # Validate and return reformulated query
            if reformulated and reformulated.strip():
                self.logger.info(f"✅ REFORMULATION SUCCESS: '{user_query}' -> '{reformulated}'")
                return reformulated.strip()
            else:
                self.logger.warning("⚠️ Reformulation returned empty result, using original query")
                return user_query

        except Exception as e:
            self.logger.error(f"❌ Reformulation failed: {e}")
            import traceback

            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            # Fallback to original query on any error
            return user_query

    def _is_query_autonomous(self, query: str) -> bool:
        """
        Check if a query is already autonomous and doesn't need reformulation.

        Args:
            query: The user query to check

        Returns:
            True if query is autonomous, False if it needs reformulation
        """
        query_lower = query.lower().strip()
        import re

        query_normalized = re.sub(r"[^\w\s]", " ", query_lower).strip()
        words = query_normalized.split()

        self.logger.debug(f"🔍 Autonomy check for: '{query}'")
        self.logger.debug(f"  Normalized: '{query_normalized}'")
        self.logger.debug(f"  Words: {words}")

        # Check various autonomy indicators
        if self._has_coreference_indicators(words):
            return False

        if self._has_implicit_reference_patterns(query_lower):
            return False

        if self._has_ambiguous_references(words):
            return False

        if self._has_french_question_patterns(query_lower):
            return False

        if self._has_demonstrative_references(query_lower):
            return False

        if self._should_force_reformulation():
            return False

        self.logger.debug("  ✅ Query appears autonomous")
        return True

    def _has_coreference_indicators(self, words: list) -> bool:
        """Check for coreference indicators at the beginning of the query"""
        coreference_indicators = [
            "il",
            "elle",
            "ils",
            "elles",  # French pronouns
            "he",
            "she",
            "they",
            "it",  # English pronouns
            "ça",
            "cela",
            "that",
            "this",  # Demonstratives
            "le",
            "la",
            "les",  # French definite articles (when ambiguous)
        ]

        if words and words[0] in coreference_indicators:
            self.logger.debug(f"  ❌ Found coreference indicator at start: '{words[0]}'")
            return True
        return False

    def _has_implicit_reference_patterns(self, query_lower: str) -> bool:
        """Check for implicit reference patterns"""
        import re

        implicit_indicators = [
            "comment faire",
            "how to do",
            "how to",
            "comment utiliser",
            "how to use",
            "comment installer",
            "how to install",
            "où est",
            "where is",
            "where are",
            "qu'est-ce que c'est",
            "what is it",
            "comment ça marche",
            "how does it work",
            "pourquoi",
            "why",
        ]

        for indicator in implicit_indicators:
            if indicator in query_lower:
                remaining_text = query_lower.split(indicator, 1)[-1] if len(query_lower.split(indicator, 1)) > 1 else ""
                remaining_normalized = re.sub(r"[^\w\s]", " ", remaining_text).strip()
                remaining_words = remaining_normalized.split()
                if any(word in remaining_words for word in ["it", "that", "ça", "cela"]):
                    self.logger.debug(f"  ❌ Found implicit reference pattern: '{indicator}' with ambiguous reference")
                    return True
        return False

    def _has_ambiguous_references(self, words: list) -> bool:
        """Check for standalone ambiguous references"""
        ambiguous_words = ["it", "that", "this", "they", "them", "ça", "cela"]
        for word in ambiguous_words:
            if word in words:
                self.logger.debug(f"  ❌ Found ambiguous word: '{word}'")
                return True
        return False

    def _has_french_question_patterns(self, query_lower: str) -> bool:
        """Check for French questioning patterns that need context"""
        french_question_patterns = [
            "et qui",
            "et quoi",
            "et où",
            "et quand",
            "et comment",
            "qui l'utilise",
            "qui utilise",
            "qui fait",
            "qui peut",
            "qui en sont",
            "qui en est",
            "qu'en est",
            "en sont",
            "en est",
        ]

        for pattern in french_question_patterns:
            if pattern in query_lower:
                self.logger.debug(f"  ❌ Found French question pattern needing context: '{pattern}'")
                return True
        return False

    def _has_demonstrative_references(self, query_lower: str) -> bool:
        """Check for demonstrative pronouns that need context resolution"""
        demonstrative_references = [
            "ce projet",
            "cette application",
            "ce système",
            "cette solution",
            "cet outil",
            "ces outils",
            "ce document",
            "cette documentation",
            "ce processus",
            "cette méthode",
            "ce code",
            "cette fonction",
        ]

        for ref in demonstrative_references:
            if ref in query_lower:
                self.logger.debug(f"  ❌ Found demonstrative reference needing context: '{ref}'")
                return True
        return False

    def _should_force_reformulation(self) -> bool:
        """Check if configuration forces reformulation for all queries"""
        if self.config.force_reformulate_all_queries:
            self.logger.debug("  🔄 Configuration forces reformulation for all queries")
            return True
        return False

    def _build_reformulation_prompt(self, user_query: str, context_exchanges: List[ConversationExchange]) -> str:
        """
        Build the prompt for LLM reformulation.

        Args:
            user_query: The original user query
            context_exchanges: Recent conversation exchanges for context

        Returns:
            Formatted prompt for the LLM
        """
        # Build conversation context
        context_text = ""
        if context_exchanges:
            context_text = "Recent conversation context:\n"
            for i, exchange in enumerate(context_exchanges, 1):
                context_text += f"Exchange {i}:\n"
                context_text += f"User: {exchange.user_message}\n"
                context_text += f"Assistant: {exchange.assistant_message}\n\n"

        prompt = f"""You are a query reformulation assistant. Your task is to reformulate user queries to make them
autonomous and clear by resolving coreferences and clarifying implicit references.

Here is the recent conversation context:

{context_text}

Current user query: "{user_query}"

Instructions:
1. Analyze the current query for pronouns (he, she, they, it, il, elle, ils, elles, ça, cela) and implicit references
2. Use the conversation context to resolve what these pronouns and references refer to
3. Reformulate the query to be completely autonomous - someone reading only the reformulated query
should understand exactly what is being asked
4. Keep the reformulated query concise but specific
5. Maintain the original intent and question type
6. If the query is already autonomous, return it unchanged

Examples:
- "How do I install it?" → "How do I install Docker?" (if Docker was mentioned in context)
- "What can they do?" → "What can Vincent and Nicolas do?" (if Vincent and Nicolas were mentioned in context)
- "Tell me more about that" → "Tell me more about the authentication system" (if auth system was mentioned in context)

Reformulated query:"""

        return prompt

    def _call_llm_for_reformulation(self, prompt: str) -> Optional[str]:
        """
        Call the LLM to reformulate the query.

        Args:
            prompt: The reformulation prompt

        Returns:
            Reformulated query or None if failed
        """
        headers = {"Authorization": f"Bearer {self.config.openrouter_api_key}", "Content-Type": "application/json"}

        payload = {
            "model": self.config.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,  # Low temperature for consistent reformulation
            "max_tokens": self.config.reformulation_max_tokens,
        }

        try:
            response = requests.post(
                f"{self.config.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.reformulation_timeout,
            )
            response.raise_for_status()

            data = response.json()
            reformulated = data["choices"][0]["message"]["content"].strip()

            # Clean up the response (remove quotes, extra formatting)
            reformulated = reformulated.strip("\"'")

            return reformulated

        except requests.RequestException as e:
            self.logger.error(f"LLM API error during reformulation: {e}")
            return None
        except (KeyError, IndexError) as e:
            self.logger.error(f"Invalid LLM response format during reformulation: {e}")
            return None

    def is_ready(self) -> bool:
        """Check if the reformulation service is ready to use"""
        return bool(self.config.openrouter_api_key)

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "type": "reformulation_service",
            "ready": self.is_ready(),
            "config": {
                "llm_model": self.config.llm_model,
                "api_key_configured": bool(self.config.openrouter_api_key),
                "base_url": self.config.openrouter_base_url,
                "timeout": self.config.reformulation_timeout,
                "max_tokens": self.config.reformulation_max_tokens,
            },
        }
