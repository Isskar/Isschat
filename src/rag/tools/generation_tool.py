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

    def generate(self, query: str, documents: List[RetrievalDocument], history: str = "") -> Dict[str, Any]:
        """
        Generate response from query and retrieved documents

        Args:
            query: User query
            documents: Retrieved documents with scores
            history: Conversation history

        Returns:
            Dict with answer, sources, etc.
        """
        try:
            # Filter documents based on relevance
            relevant_documents = self._filter_relevant_documents(query, documents)
            
            context = self._prepare_context(relevant_documents)

            avg_score = sum(doc.score for doc in relevant_documents) / len(relevant_documents) if relevant_documents else 0.0
            prompt = self._build_prompt(query, context, history, avg_score)

            llm_response = self._call_openrouter(prompt)

            # Only show sources if documents are relevant
            sources = self._format_sources(relevant_documents) if relevant_documents else "No sources"

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
                "sources": "Error",
                "token_count": 0,
                "generation_time": 0.0,
                "success": False,
                "error": str(e),
            }

    def _prepare_context(self, documents: List[RetrievalDocument]) -> str:
        if not documents:
            return "No relevant documents found."

        # Adjust max content length based on number of documents to fit in context window
        # Reduced for smaller Confluence pages to avoid truncation
        max_content_per_doc = max(300, 1500 // len(documents)) if documents else 600
        context_parts = [doc.to_context_section(max_content_per_doc) for doc in documents]
        return "\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str, history: str = "", avg_score: float = 0.0) -> str:
        """Build prompt based on context quality"""
        history_section = f"{history}\n" if history.strip() else ""

        return PromptTemplates.get_default_template().format(context=context, history=history_section, query=query)

    def _call_openrouter(self, prompt: str) -> Dict[str, Any]:
        """Call OpenRouter API"""
        import time

        start_time = time.time()

        headers = {"Authorization": f"Bearer {self.config.openrouter_api_key}", "Content-Type": "application/json"}

        payload = {
            "model": self.config.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.llm_temperature,
            "max_tokens": self.config.llm_max_tokens,
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30
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
            return "No sources"
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
        Returns only documents that are truly relevant to avoid showing irrelevant sources.
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
            r'^(hello|hi|hey|bonjour|salut|coucou)$',
            r'^(hello|hi|hey|bonjour|salut|coucou)\s*[!.]*$',
            r'^(comment\s+ça\s+va|how\s+are\s+you|ça\s+va).*$',
            r'^(merci|thank\s+you|thanks).*$',
            r'^(au\s+revoir|bye|goodbye|à\s+bientôt).*$',
        ]
        
        # Define generic terms that shouldn't show sources
        generic_terms = {
            'test', 'tests', 'testing', 'exemple', 'example', 'demo', 'sample',
            'ok', 'okay', 'oui', 'yes', 'non', 'no', 'bien', 'good', 'bad',
            'help', 'aide', 'info', 'information'
        }
        
        # Check if query is a simple greeting/social interaction
        is_greeting = any(re.match(pattern, query_lower) for pattern in greeting_patterns)
        
        # Check if query is a single generic term
        is_generic_term = query_lower.strip() in generic_terms
        
        if is_greeting or is_generic_term:
            # For greetings and generic terms, don't show any sources
            return []
        
        # Calculate relevance scores
        relevant_documents = []
        
        # Extract meaningful keywords from query (excluding stop words)
        stop_words = {
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'est', 'sont', 'avec',
            'sur', 'dans', 'pour', 'par', 'qui', 'que', 'quoi', 'comment', 'où', 'quand', 'pourquoi',
            'the', 'a', 'an', 'and', 'or', 'is', 'are', 'with', 'on', 'in', 'for', 'by', 'who', 'what',
            'where', 'when', 'why', 'how', 'this', 'that', 'can', 'could', 'should'
        }
        
        query_keywords = [
            word for word in query_lower.split() 
            if word not in stop_words and len(word) > 2
        ]
        
        # If no meaningful keywords, likely a general query
        if not query_keywords:
            return []
        
        # Calculate relevance for each document
        for doc in documents:
            relevance_score = self._calculate_document_relevance(query_keywords, doc)
            
            # Apply thresholds from configuration
            min_score_threshold = self.config.min_source_score_threshold
            min_relevance_threshold = self.config.min_source_relevance_threshold
            
            # Document is relevant if it meets both criteria
            if doc.score >= min_score_threshold and relevance_score >= min_relevance_threshold:
                relevant_documents.append(doc)
        
        # Sort by combined score (similarity + relevance)
        relevant_documents.sort(key=lambda doc: doc.score, reverse=True)
        
        # Log filtering results
        self.logger.debug(f"Filtered {len(documents)} documents to {len(relevant_documents)} relevant ones")
        
        return relevant_documents
    
    def _calculate_document_relevance(self, query_keywords: List[str], document: RetrievalDocument) -> float:
        """
        Calculate how relevant a document is to the query keywords.
        Returns a score between 0 and 1.
        """
        if not query_keywords:
            return 0.0
        
        # Get document content and metadata
        content = document.content.lower()
        title = document.metadata.get('title', '').lower()
        
        # Count keyword matches
        content_matches = sum(1 for keyword in query_keywords if keyword in content)
        title_matches = sum(1 for keyword in query_keywords if keyword in title)
        
        # Calculate relevance score
        # Title matches are weighted higher than content matches
        relevance_score = (title_matches * 2 + content_matches) / (len(query_keywords) * 2)
        
        # Additional boost for exact phrase matches
        query_phrase = ' '.join(query_keywords)
        if query_phrase in content:
            relevance_score += 0.2
        
        if query_phrase in title:
            relevance_score += 0.3
        
        return min(1.0, relevance_score)
