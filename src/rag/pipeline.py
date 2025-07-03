import logging
import time
from typing import Tuple, Dict, Any, Optional

from ..config import get_config
from ..storage.data_manager import get_data_manager
from src.rag.tools import RetrievalTool, GenerationTool


class RAGPipeline:
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_manager = get_data_manager()

        self.retrieval_tool = RetrievalTool()
        self.generation_tool = GenerationTool()

        self.logger.info("✅ RAG pipeline initialized")

    def process_query(
        self,
        query: str,
        history: str = "",
        user_id: str = "anonymous",
        conversation_id: Optional[str] = None,
        verbose: bool = False,
    ) -> Tuple[str, str]:
        """
        Process a complete query (retrieval + generation)

        Args:
            query: User question
            history: Conversation history
            user_id: User ID for logs
            conversation_id: Conversation ID for logs
            verbose: Detailed display

        Returns:
            Tuple (answer, sources)
        """
        start_time = time.time()

        try:
            if verbose:
                self.logger.info(f"🔍 Processing query: '{query[:100]}...'")

            if verbose:
                self.logger.info("📥 Step 1: Searching for relevant documents")

            search_results = self.retrieval_tool.retrieve(query)

            if verbose:
                self.logger.info(f"📄 {len(search_results)} documents found")

            if verbose:
                self.logger.info("🤖 Step 2: Generating response")

            generation_result = self.generation_tool.generate(query=query, documents=search_results, history=history)

            answer = generation_result["answer"]
            sources = generation_result["sources"]

            response_time = (time.time() - start_time) * 1000  # in ms

            if verbose:
                self.logger.info(f"✅ Response generated in {response_time:.0f}ms")

            try:
                conv_id = conversation_id or f"conv_{int(time.time())}"
                self.data_manager.save_conversation(
                    user_id=user_id,
                    conversation_id=conv_id,
                    question=query,
                    answer=answer,
                    response_time_ms=response_time,
                    sources=self._format_sources_for_storage(search_results),
                    metadata={
                        "pipeline_type": "rag",
                        "num_retrieved_docs": len(search_results),
                        "generation_success": generation_result["success"],
                    },
                )
            except Exception as e:
                self.logger.warning(f"Failed to save conversation: {e}")

            return answer, sources

        except Exception as e:
            error_msg = f"RAG pipeline error: {str(e)}"
            self.logger.error(error_msg)

            try:
                response_time = (time.time() - start_time) * 1000
                conv_id = conversation_id or f"conv_{int(time.time())}"
                self.data_manager.save_conversation(
                    user_id=user_id,
                    conversation_id=conv_id,
                    question=query,
                    answer=f"Error: {error_msg}",
                    response_time_ms=response_time,
                    metadata={"pipeline_type": "rag", "error": str(e)},
                )
            except Exception:
                pass

            return f"Sorry, an error occurred: {str(e)}", "System error"

    def _format_sources_for_storage(self, formatted_docs) -> list[dict]:
        """Format sources for storage"""
        sources = []
        for doc in formatted_docs:
            sources.append(
                {
                    "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "score": doc.score,
                    "metadata": doc.metadata,
                }
            )
        return sources

    def is_ready(self) -> bool:
        try:
            retrieval_ready = self.retrieval_tool.is_ready()
            generation_ready = self.generation_tool.is_ready()

            return retrieval_ready and generation_ready
        except Exception as e:
            self.logger.error(f"Readiness check failed: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        try:
            retrieval_stats = self.retrieval_tool.get_stats()
            generation_stats = self.generation_tool.get_stats()

            return {
                "pipeline_type": "rag_pipeline",
                "ready": self.is_ready(),
                "config": {
                    "embeddings_model": self.config.embeddings_model,
                    "llm_model": self.config.llm_model,
                    "vectordb_collection": self.config.vectordb_collection,
                    "search_k": self.config.search_k,
                },
                "retrieval_tool": retrieval_stats,
                "generation_tool": generation_stats,
                "data_manager": self.data_manager.get_info(),
            }
        except Exception as e:
            return {"pipeline_type": "rag_pipeline", "ready": False, "error": str(e)}

    def check_pipeline(self, test_query: str = "What does this documentation describe?") -> Dict[str, Any]:
        try:
            if not self.is_ready():
                return {"success": False, "error": "Pipeline not ready", "details": self.get_status()}

            start_time = time.time()
            answer, sources = self.process_query(test_query, verbose=True)
            end_time = time.time()

            return {
                "success": True,
                "test_query": test_query,
                "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                "sources": sources,
                "response_time_ms": (end_time - start_time) * 1000,
                "pipeline_status": self.get_status(),
            }

        except Exception as e:
            return {"success": False, "error": str(e), "test_query": test_query}

    def get_conversation_history(self, user_id: Optional[str] = None, limit: int = 50) -> list[dict]:
        try:
            return self.data_manager.get_conversation_history(user_id=user_id, limit=limit)
        except Exception as e:
            self.logger.error(f"Failed to retrieve history: {e}")
            return []


class RAGPipelineFactory:
    @staticmethod
    def create_default_pipeline() -> RAGPipeline:
        pipeline = RAGPipeline()

        if not pipeline.is_ready():
            logging.warning("⚠️ RAG pipeline created but not ready - check that the vector database is built")

        return pipeline
