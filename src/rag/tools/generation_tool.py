from typing import Dict, Any, List
import logging
import requests

from ...config import get_config
from src.rag.tools.prompt_templates import PromptTemplates
from ...core.documents import RetrievalDocument


class GenerationTool:
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY required for generation")

    def generate(
        self, query: str, documents: List[RetrievalDocument], history: str = "", numerical_context: Any = None
    ) -> Dict[str, Any]:
        """
        Generate response from query and retrieved documents

        Args:
            query: User query
            documents: Retrieved documents with scores
            history: Conversation history
            numerical_context: Optional numerical query processing result

        Returns:
            Dict with answer, sources, etc.
        """
        try:
            context = self._prepare_context(documents)

            avg_score = sum(doc.score for doc in documents) / len(documents) if documents else 0.0
            prompt = self._build_prompt(query, context, history, avg_score, numerical_context)

            llm_response = self._call_openrouter(prompt)

            sources = self._format_sources(documents)

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
        # Increased limits to preserve arborescence content
        max_content_per_doc = max(800, 3000 // len(documents)) if documents else 1200
        context_parts = [doc.to_context_section(max_content_per_doc) for doc in documents]
        return "\n\n".join(context_parts)

    def _build_prompt(
        self, query: str, context: str, history: str = "", avg_score: float = 0.0, numerical_context: Any = None
    ) -> str:
        """Build prompt based on context quality"""
        history_section = f"{history}\n" if history.strip() else ""

        # Add numerical context if available
        if numerical_context:
            numerical_info = self._format_numerical_context(numerical_context)
            context = f"{context}\n\n{numerical_info}"

        return PromptTemplates.get_default_template().format(context=context, history=history_section, query=query)

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
