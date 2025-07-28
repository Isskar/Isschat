"""
Fast pipeline optimized for Streamlit performance.
Uses direct components to minimize overhead and achieve <2s response times.
"""

import logging
import time
from typing import Tuple, Optional

from ..config import get_config
from ..storage.data_manager import get_data_manager
from .tools.semantic_retrieval_tool import SemanticRetrievalTool
from .tools.generation_tool import GenerationTool


class FastRAGPipeline:
    """
    Optimized RAG pipeline for maximum performance.
    Uses direct components without LlamaIndex overhead.
    Target: <2s response time after warm-up.
    """

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_manager = get_data_manager()

        # Initialize tools (lazy loading for performance)
        self._retrieval_tool = None
        self._generation_tool = None
        self._warmed_up = False  # Track if warm-up is done

        # Simple conversation memory (dict-based, no LlamaIndex overhead)
        self.conversations = {}  # conversation_id -> messages list

        self.logger.info("✅ Fast RAG pipeline initialized")

    def _get_retrieval_tool(self):
        """Lazy initialization of retrieval tool"""
        if self._retrieval_tool is None:
            self._retrieval_tool = SemanticRetrievalTool()
        return self._retrieval_tool

    def _get_generation_tool(self):
        """Lazy initialization of generation tool"""
        if self._generation_tool is None:
            self._generation_tool = GenerationTool()
        return self._generation_tool

    def _perform_full_warmup(self):
        """Perform full warm-up of embedding and tools (deferred to first query)"""
        try:
            warmup_start = time.time()

            # Warm up embedding service through retrieval tool
            retrieval_tool = self._get_retrieval_tool()
            if hasattr(retrieval_tool, "_embedding_service"):
                embedding_service = retrieval_tool._embedding_service
                if hasattr(embedding_service, "encode_single"):
                    # Warm up with a simple query
                    embedding_service.encode_single("test query for warmup")
                    self.logger.debug("✅ Embedding service warmed up")

            # Warm up generation tool
            generation_tool = self._get_generation_tool()
            if hasattr(generation_tool, "is_ready"):
                generation_tool.is_ready()
                self.logger.debug("✅ Generation tool warmed up")

            self._warmed_up = True
            warmup_time = (time.time() - warmup_start) * 1000
            self.logger.info(f"✅ Full warm-up completed in {warmup_time:.0f}ms")

        except Exception as e:
            self.logger.warning(f"Warm-up failed: {e}")
            self._warmed_up = True  # Mark as warmed to avoid retry loops

    def process_query(
        self,
        query: str,
        user_id: str = "anonymous",
        conversation_id: Optional[str] = None,
        verbose: bool = False,
    ) -> Tuple[str, str]:
        """
        Process query with maximum performance.

        Args:
            query: User question
            user_id: User ID for logging
            conversation_id: Conversation ID for context
            verbose: Enable verbose logging

        Returns:
            Tuple (answer, sources)
        """
        start_time = time.time()

        try:
            # Deferred warm-up: full warm-up on first query only
            if not self._warmed_up:
                if verbose:
                    self.logger.info("⚡ Performing deferred warm-up (first query)...")
                self._perform_full_warmup()

            if verbose:
                self.logger.info(f"🚀 Fast processing: '{query[:50]}...'")

            # Setup conversation context
            if not conversation_id:
                conversation_id = f"conv_{int(time.time())}_{user_id}"

            # Get conversation history (lightweight)
            history = self._get_conversation_history(conversation_id)

            # Add current query to memory
            self._add_to_conversation(conversation_id, "user", query)

            # 1. Fast retrieval (no semantic expansion, keep reranking)
            retrieval_tool = self._get_retrieval_tool()
            docs = retrieval_tool.retrieve(
                query=query,
                k=self.config.search_k,
                use_semantic_expansion=False,  # Disabled for speed
                use_semantic_reranking=True,  # Keep for quality
            )

            # 2. Fast generation with minimal history
            generation_tool = self._get_generation_tool()
            generation_result = generation_tool.generate(
                query=query,
                documents=docs,
                history=history,  # Last 3 exchanges only
            )

            answer = generation_result.get("answer", "No answer generated")
            sources = self._format_sources(docs)

            # Add response to memory
            self._add_to_conversation(conversation_id, "assistant", answer)

            response_time = (time.time() - start_time) * 1000

            if verbose:
                self.logger.info(f"✅ Fast response: {response_time:.0f}ms")

            # Save to persistent storage (async-like, don't wait)
            try:
                self.data_manager.save_conversation(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    question=query,
                    answer=answer,
                    response_time_ms=response_time,
                    sources=sources,
                    metadata={
                        "pipeline_type": "fast_rag",
                        "response_time_ms": response_time,
                    },
                )
            except Exception as e:
                self.logger.warning(f"Failed to save conversation: {e}")

            return answer, sources

        except Exception as e:
            self.logger.error(f"Fast pipeline failed: {e}")
            response_time = (time.time() - start_time) * 1000
            error_answer = f"Sorry, an error occurred: {str(e)}"
            return error_answer, ""

    def _get_conversation_history(self, conversation_id: str, max_exchanges: int = 3) -> str:
        """Get lightweight conversation history"""
        if conversation_id not in self.conversations:
            return ""

        messages = self.conversations[conversation_id]

        # Take only last max_exchanges*2 messages (user + assistant pairs)
        recent_messages = messages[-(max_exchanges * 2) :]

        history = []
        for msg in recent_messages:
            if msg["role"] == "user":
                history.append(f"User: {msg['content']}")
            else:
                history.append(f"Assistant: {msg['content']}")

        return "\n".join(history)

    def _add_to_conversation(self, conversation_id: str, role: str, content: str):
        """Add message to conversation memory"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        self.conversations[conversation_id].append({"role": role, "content": content, "timestamp": time.time()})

        # Keep only last 10 messages per conversation (memory management)
        if len(self.conversations[conversation_id]) > 10:
            self.conversations[conversation_id] = self.conversations[conversation_id][-10:]

    def _format_sources(self, docs) -> str:
        """Format sources from documents"""
        if not docs:
            return ""

        sources = []
        seen = set()

        for doc in docs:
            metadata = doc.metadata or {}
            title = metadata.get("title", "Document")
            url = metadata.get("url", "")

            source_link = f"[{title}]({url})" if url else title

            if source_link not in seen:
                seen.add(source_link)
                sources.append(source_link)

        return " • ".join(sources)

    def clear_conversation(self, conversation_id: str):
        """Clear conversation history"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

    def is_ready(self) -> bool:
        """Check if pipeline is ready"""
        try:
            retrieval_tool = self._get_retrieval_tool()
            return retrieval_tool.is_ready()
        except Exception:
            return False

    def get_stats(self):
        """Get pipeline statistics"""
        return {
            "type": "fast_rag_pipeline",
            "ready": self.is_ready(),
            "active_conversations": len(self.conversations),
            "total_messages": sum(len(conv) for conv in self.conversations.values()),
            "optimizations": ["No semantic expansion", "Lightweight memory", "Direct components", "Minimal overhead"],
        }


class FastRAGPipelineFactory:
    """Factory for fast RAG pipeline"""

    @staticmethod
    def create_fast_pipeline() -> FastRAGPipeline:
        """Create optimized fast pipeline"""
        return FastRAGPipeline()
