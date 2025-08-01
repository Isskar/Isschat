from typing import Dict, Any, List
import logging
import requests
import re

from ...config import get_config
from src.rag.tools.prompt_templates import PromptTemplates
from ...core.documents import RetrievalDocument


class GenerationTool:
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY required for generation")

    def generate(self, query: str, documents: List[RetrievalDocument], numerical_context: Any = None) -> Dict[str, Any]:
        """
        Generate response from query and retrieved documents

        Args:
            query: User query
            documents: Retrieved documents with scores
            numerical_context: Optional numerical query processing result

        Returns:
            Dict with answer, sources, etc.
        """
        try:
            # Filter documents based on relevance
            print(f"🔍 GENERATION: Received {len(documents)} documents for query '{query}'")
            relevant_documents = self._filter_relevant_documents(query, documents)
            print(f"📄 GENERATION: After filtering, {len(relevant_documents)} documents remain")

            # Prepare context from relevant documents
            context = self._prepare_context(relevant_documents)

            avg_score = sum(doc.score for doc in documents) / len(documents) if documents else 0.0
            prompt = self._build_prompt(query, context, avg_score, numerical_context)

            llm_response = self._call_openrouter(prompt)

            # Only show sources if documents are relevant
            sources = self._format_sources(relevant_documents) if relevant_documents else ""

            return {
                "answer": llm_response["answer"],
                "sources": sources,
                "token_count": llm_response.get("token_count", 0),
                "generation_time": llm_response.get("generation_time", 0.0),
                "success": True,
            }

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return {
                "answer": f"Sorry, an error occurred during generation: {str(e)}",
                "sources": "",
                "token_count": 0,
                "generation_time": 0.0,
                "success": False,
                "error": str(e),
            }

    def _prepare_context(self, documents: List[RetrievalDocument]) -> str:
        if not documents:
            return "No relevant documents found."

        # Adjust max content length based on number of documents to fit in context window
        # Increased limits to preserve arborescence content
        max_content_per_doc = max(800, 3000 // len(documents)) if documents else 1200
        context_parts = [doc.to_context_section(max_content_per_doc) for doc in documents]
        return "\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str, avg_score: float = 0.0, numerical_context: Any = None) -> str:
        """Build prompt based on context quality"""
        # Add numerical context if available
        if numerical_context:
            numerical_info = self._format_numerical_context(numerical_context)
            context = f"{context}\n\n{numerical_info}"

        return PromptTemplates.get_default_template().format(context=context, query=query)

    def _format_numerical_context(self, numerical_context: Any) -> str:
        """Format numerical context for inclusion in prompt"""
        if not numerical_context:
            return ""

        # Handle case where numerical_context is a string (error case)
        if isinstance(numerical_context, str):
            return f"## Numerical Analysis\n**Note:** {numerical_context}\n"

        formatted_context = "## Numerical Analysis\n"

        if hasattr(numerical_context, "aggregated_value") and numerical_context.aggregated_value is not None:
            formatted_context += f"**Aggregated Result:** {numerical_context.aggregated_value}\n"

        if hasattr(numerical_context, "explanation") and numerical_context.explanation:
            formatted_context += f"**Explanation:** {numerical_context.explanation}\n"

        if hasattr(numerical_context, "confidence") and numerical_context.confidence > 0:
            formatted_context += f"**Confidence:** {numerical_context.confidence:.2f}\n"

        return formatted_context

    def _call_openrouter(self, prompt: str) -> Dict[str, Any]:
        """Call OpenRouter API"""
        import time

        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Helicone-Auth": f"Bearer {self.config.helicone_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.llm_temperature,
            "max_tokens": self.config.llm_max_tokens,
        }

        try:
            response = requests.post(
                "https://openrouter.helicone.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.openrouter_timeout,
            )
            response.raise_for_status()

            data = response.json()
            end_time = time.time()

            answer = data["choices"][0]["message"]["content"]
            token_count = data.get("usage", {}).get("total_tokens", 0)

            return {"answer": answer.strip(), "token_count": token_count, "generation_time": end_time - start_time}

        except requests.RequestException as e:
            raise RuntimeError(f"OpenRouter API error: {e}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Invalid OpenRouter response format: {e}")

    def _format_sources(self, documents: List[RetrievalDocument]) -> str:
        """Format sources for display"""
        if not documents:
            return ""
        # Deduplicate sources based on title and URL
        seen_sources = set()
        unique_sources = []

        for doc in documents:
            source_key = (doc.title, doc.url)
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_sources.append(doc.to_source_link())

        return " • ".join(unique_sources)

    def is_ready(self) -> bool:
        return bool(self.config.openrouter_api_key)

    def get_stats(self) -> Dict[str, Any]:
        """Generation tool statistics"""
        return {
            "type": "generation_tool",
            "ready": self.is_ready(),
            "config": {
                "llm_model": self.config.llm_model,
                "llm_temperature": self.config.llm_temperature,
                "llm_max_tokens": self.config.llm_max_tokens,
                "api_key_configured": bool(self.config.openrouter_api_key),
            },
        }

    def _filter_relevant_documents(self, query: str, documents: List[RetrievalDocument]) -> List[RetrievalDocument]:
        """
        Filter documents based on relevance to the query.
        Uses a more flexible approach to avoid missing important documents.
        """
        if not documents:
            return []

        # If source filtering is disabled, return all documents
        if not self.config.source_filtering_enabled:
            return documents

        # Normalize query for analysis
        query_lower = query.lower().strip()

        # Define patterns for different query types
        greeting_patterns = [
            r"^(hello|hi|hey|bonjour|salut|coucou)$",
            r"^(hello|hi|hey|bonjour|salut|coucou)\s*[!.]*$",
            r"^(comment\s+ça\s+va|how\s+are\s+you|ça\s+va).*$",
            r"^(merci|thank\s+you|thanks).*$",
            r"^(au\s+revoir|bye|goodbye|à\s+bientôt).*$",
        ]

        # Define generic terms that shouldn't show sources
        generic_terms = {
            "test",
            "tests",
            "testing",
            "exemple",
            "example",
            "demo",
            "sample",
            "ok",
            "okay",
            "oui",
            "yes",
            "non",
            "no",
            "bien",
            "good",
            "bad",
            "help",
            "aide",
            "info",
            "information",
        }

        # Check if query is a simple greeting/social interaction
        is_greeting = any(re.match(pattern, query_lower) for pattern in greeting_patterns)

        # Check if query is a single generic term
        is_generic_term = query_lower.strip() in generic_terms

        if is_greeting or is_generic_term:
            # For greetings and generic terms, don't show any sources
            return []

        # Extract meaningful keywords from query (excluding stop words)
        stop_words = {
            "le",
            "la",
            "les",
            "un",
            "une",
            "des",
            "du",
            "de",
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
            "can",
            "could",
            "should",
        }

        query_keywords = [word for word in query_lower.split() if word not in stop_words and len(word) > 2]

        # If no meaningful keywords, likely a general query
        if not query_keywords:
            return []

        # More flexible filtering approach
        relevant_documents = []
        min_score_threshold = self.config.min_source_score_threshold
        min_relevance_threshold = self.config.min_source_relevance_threshold

        print(f"🔍 FILTERING: Keywords extracted: {query_keywords}")

        for i, doc in enumerate(documents):
            relevance_score = self._calculate_document_relevance(query_keywords, doc)

            print(f"📄 DOC {i + 1}: score={doc.score:.3f}, relevance={relevance_score:.3f}")

            # Choose filtering approach based on configuration
            if self.config.use_flexible_filtering:
                # Flexible multi-criteria approach: document is relevant if it meets EITHER:
                # 1. High vector similarity (even if keyword matching is poor) - semantic relevance
                # 2. Good vector similarity AND decent keyword relevance - traditional approach
                # 3. Exceptional keyword relevance (even if vector similarity is lower) - exact matches

                high_vector_threshold = min_score_threshold + 0.2  # e.g., 0.5 if min is 0.3
                exceptional_relevance_threshold = min_relevance_threshold + 0.3  # e.g., 0.5 if min is 0.2

                is_highly_similar = doc.score >= high_vector_threshold
                is_traditionally_relevant = (
                    doc.score >= min_score_threshold and relevance_score >= min_relevance_threshold
                )
                is_exceptionally_relevant = relevance_score >= exceptional_relevance_threshold

                print(
                    f"  Flexible thresholds: high_vector={high_vector_threshold:.3f}, "
                    f"traditional=({min_score_threshold:.3f},{min_relevance_threshold:.3f}), "
                    f"exceptional={exceptional_relevance_threshold:.3f}"
                )

                if is_highly_similar:
                    relevant_documents.append(doc)
                    print(f"  ✅ DOC {i + 1}: ACCEPTED (high vector similarity)")
                elif is_traditionally_relevant:
                    relevant_documents.append(doc)
                    print(f"  ✅ DOC {i + 1}: ACCEPTED (traditional criteria)")
                elif is_exceptionally_relevant:
                    relevant_documents.append(doc)
                    print(f"  ✅ DOC {i + 1}: ACCEPTED (exceptional keyword relevance)")
                else:
                    print(f"  ❌ DOC {i + 1}: REJECTED")
            else:
                # Traditional strict approach: both criteria must be met
                print(
                    f"  Traditional thresholds: vector>={min_score_threshold:.3f}, "
                    f"relevance>={min_relevance_threshold:.3f}"
                )

                if doc.score >= min_score_threshold and relevance_score >= min_relevance_threshold:
                    relevant_documents.append(doc)
                    print(f"  ✅ DOC {i + 1}: ACCEPTED (traditional criteria)")
                else:
                    print(f"  ❌ DOC {i + 1}: REJECTED")

        # Sort by combined score (similarity + relevance)
        relevant_documents.sort(key=lambda doc: doc.score, reverse=True)

        # Ensure we don't filter out all documents if we have good candidates
        if not relevant_documents and documents:
            # If no documents pass the flexible criteria, take the top scoring document
            # This prevents scenarios where important information is completely filtered out
            top_doc = max(documents, key=lambda d: d.score)
            if top_doc.score >= (min_score_threshold - 0.1):  # Slightly more lenient
                relevant_documents = [top_doc]
                print(f"🔄 FALLBACK: Taking top document with score {top_doc.score:.3f}")

        # Log filtering results
        self.logger.debug(f"Filtered {len(documents)} documents to {len(relevant_documents)} relevant ones")

        return relevant_documents

    def _calculate_document_relevance(self, query_keywords: List[str], document: RetrievalDocument) -> float:
        """
        Calculate how relevant a document is to the query keywords.
        Enhanced with French language support and synonym matching.
        Returns a score between 0 and 1.
        """
        if not query_keywords:
            return 0.0

        # Get document content and metadata
        content = document.content.lower()
        title = document.metadata.get("title", "").lower()

        # Enhanced keyword matching with French variations and common synonyms
        content_matches = 0
        title_matches = 0

        for keyword in query_keywords:
            # Direct match
            if keyword in content:
                content_matches += 1
            if keyword in title:
                title_matches += 1

            # French variations and common synonyms
            keyword_variations = self._get_keyword_variations(keyword)
            for variation in keyword_variations:
                if variation in content:
                    content_matches += 0.8  # Slightly lower weight for variations
                if variation in title:
                    title_matches += 0.8

        # Calculate base relevance score
        # Title matches are weighted higher than content matches
        max_possible_matches = len(query_keywords) * 2  # 2 for title weight
        relevance_score = (title_matches * 2 + content_matches) / max_possible_matches

        # Additional boost for exact phrase matches
        query_phrase = " ".join(query_keywords)
        if query_phrase in content:
            relevance_score += 0.2

        if query_phrase in title:
            relevance_score += 0.3

        # Boost for partial phrase matches (useful for reformulated queries)
        if len(query_keywords) > 1:
            for i in range(len(query_keywords) - 1):
                partial_phrase = " ".join(query_keywords[i : i + 2])
                if partial_phrase in content:
                    relevance_score += 0.1
                if partial_phrase in title:
                    relevance_score += 0.15

        return min(1.0, relevance_score)

    def _get_keyword_variations(self, keyword: str) -> List[str]:
        """
        Get variations of a keyword for better matching.
        Handles French conjugations, plurals, and common synonyms.
        """
        variations = []

        # Common French and English synonyms/variations
        synonym_map = {
            # Team/people related
            "équipe": ["team", "groupe", "collaborateurs", "membres"],
            "team": ["équipe", "groupe", "collaborateurs", "membres"],
            "collaborateurs": ["équipe", "team", "membres", "personnes"],
            "membres": ["équipe", "team", "collaborateurs", "personnes"],
            # Project related
            "projet": ["project", "application", "app", "système"],
            "project": ["projet", "application", "app", "système"],
            "application": ["app", "projet", "project", "système"],
            # Technical terms
            "configuration": ["config", "paramètres", "settings", "setup"],
            "installation": ["install", "setup", "configuration"],
            "utilisation": ["usage", "use", "emploi"],
            "fonctionnalités": ["features", "capacités", "options"],
            "features": ["fonctionnalités", "capacités", "options"],
            # Common verbs and their variations
            "utiliser": ["use", "employer", "utilise", "utilisent"],
            "installer": ["install", "installe", "installent", "setup"],
            "configurer": ["configure", "config", "configure", "setup"],
        }

        # Add direct synonyms
        if keyword in synonym_map:
            variations.extend(synonym_map[keyword])

        # Add common French plural/singular variations
        if keyword.endswith("s") and len(keyword) > 3:
            variations.append(keyword[:-1])  # Remove 's' for singular
        elif not keyword.endswith("s"):
            variations.append(keyword + "s")  # Add 's' for plural

        # Add common French verb conjugations
        verb_endings = {
            "er": ["e", "es", "ent", "ez", "ons"],  # aimer -> aime, aimes, etc.
            "ir": ["is", "it", "issent", "issez", "issons"],  # finir -> finis, finit, etc.
            "re": ["", "s", "ent", "ez", "ons"],  # prendre -> prend, prends, etc.
        }

        for ending, conjugations in verb_endings.items():
            if keyword.endswith(ending):
                root = keyword[: -len(ending)]
                for conj in conjugations:
                    variations.append(root + conj)

        # Remove duplicates and the original keyword
        variations = list(set(v for v in variations if v != keyword and len(v) > 2))

        return variations
