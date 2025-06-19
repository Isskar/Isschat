"""
OpenRouter-based generator implementation.
"""

from typing import Dict, Any
import time
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.utils import convert_to_secret_str

from src.core.interfaces import GenerationResult, RetrievalResult
from src.core.exceptions import GenerationError, ConfigurationError
from src.generation.base_generator import BaseGenerator
from src.generation.prompt_templates import PromptTemplates
from src.core.config import get_config


class OpenRouterGenerator(BaseGenerator):
    """OpenRouter-based generator implementation."""

    def __init__(self, model_name: str, temperature: float, max_tokens: int):
        """
        Initialize OpenRouter generator.

        Args:
            model_name: Model to use for generation
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None
        self._prompt = None
        self._chain = None

    def _initialize_llm(self):
        """Initialize the language model."""
        if ChatOpenAI is None or get_config is None:
            raise ConfigurationError("Required dependencies not available")

        try:
            config = get_config()
            api_key = convert_to_secret_str(config.openrouter_api_key)
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in configuration")

            self._llm = ChatOpenAI(
                model_name="anthropic/claude-sonnet-4",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                request_timeout=60,  # Increase timeout to 60 seconds
            )

            # Create prompt template
            prompt = PromptTemplates.get_default_template()
            self._prompt = PromptTemplate(template=prompt, input_variables=["context", "question"])

            # Create the chain
            self._chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self._prompt
                | self._llm
                | StrOutputParser()
            )

        except Exception as e:
            raise GenerationError(f"Failed to initialize OpenRouter generator: {str(e)}")

    async def _generate_async(self, context: str, query: str) -> str:
        """Generate answer asynchronously."""
        try:
            # Ensure we have an event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Generate answer with async call
            answer = await self._chain.ainvoke({"context": context, "question": query})
            return answer
        except Exception as e:
            raise GenerationError(f"Failed to generate answer: {str(e)}")

    def generate(self, query: str, retrieval_result: RetrievalResult) -> GenerationResult:
        """
        Generate answer based on query and retrieved documents.

        Args:
            query: User query
            retrieval_result: Result from retrieval step

        Returns:
            GenerationResult with answer and metadata
        """
        if self._chain is None:
            self._initialize_llm()

        try:
            start_time = time.time()

            # Prepare context from retrieved documents
            context = "\n\n".join(
                [
                    f"Document {i + 1}:\nTitle: {doc.metadata.get('title', 'N/A')}\nContent: {doc.page_content}"
                    for i, doc in enumerate(retrieval_result.documents[:3])  # Use top 3 documents
                ]
            )

            # Generate answer using direct synchronous call
            answer = self._chain.invoke({"context": context, "question": query})

            # Format sources
            sources = self._format_sources(retrieval_result.documents)

            generation_time = time.time() - start_time

            return GenerationResult(
                answer=answer,
                sources=sources,
                generation_time=generation_time,
                token_count=len(answer.split()),  # Simple token count approximation
            )

        except Exception as e:
            raise GenerationError(f"Failed to generate answer for query '{query}': {str(e)}")

    def _format_sources(self, documents, k: int = 2) -> str:
        """Format source documents into a readable string with clickable Confluence links."""
        if not documents:
            return "Désolé, je n'ai pas trouvé de ressources utiles pour répondre à votre question"

        sources = []
        for doc in documents:
            title = doc.metadata.get("title", "Document")
            source = doc.metadata.get("source", "")
            url = doc.metadata.get("url", "#")

            # Format pour Streamlit avec liens cliquables
            if url and url != "#":
                # Create a clickable HTML link for Streamlit
                source_link = f"[{title}]({url})"
            else:
                source_link = title

            # Add source information if available
            if source:
                source_link += f" ({source})"

            sources.append(source_link)

        if sources:
            k = min(k, len(sources))
            # Get unique sources
            unique_sources = list(dict.fromkeys(sources))[:k]
            sources_str = "  \n- ".join(unique_sources)

            if len(unique_sources) == 1:
                return f"Voici la source que j'ai utilisée pour répondre à votre question:  \n- {sources_str}"
            elif len(unique_sources) > 1:
                return f"Voici les {len(unique_sources)} sources que j'ai utilisées pour répondre à votre question:  \n- {sources_str}"  # noqa

        return "Désolé, je n'ai pas trouvé de ressources utiles pour répondre à votre question"

    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "generator_type": "OpenRouter",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "status": "initialized" if self._llm else "not_initialized",
        }
