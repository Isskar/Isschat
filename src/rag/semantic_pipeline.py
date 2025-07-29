"""
Enhanced RAG pipeline with semantic understanding capabilities.
Integrates semantic query processing and retrieval for better accuracy.
"""

import logging
import time
from typing import Tuple, Dict, Any, Optional, List

from ..config import get_config
from ..storage.data_manager import get_data_manager
from .tools.semantic_retrieval_tool import SemanticRetrievalTool
from .tools.generation_tool import GenerationTool
from .query_processor import QueryProcessor
from .reformulation_service import ReformulationService, ConversationExchange


class SemanticRAGPipeline:
    """
    Enhanced RAG pipeline with semantic understanding capabilities.
    Handles misleading keywords through semantic query processing.
    """

    def __init__(self, use_semantic_features: bool = True):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_manager = get_data_manager()
        self.use_semantic_features = use_semantic_features

        # Initialize tools
        self.semantic_retrieval_tool = SemanticRetrievalTool()
        self.generation_tool = GenerationTool()
        self.query_processor = QueryProcessor()
        self.reformulation_service = ReformulationService()

        self.logger.info("✅ Semantic RAG pipeline initialized")

    def process_query(
        self,
        query: str,
        history: str = "",
        user_id: str = "anonymous",
        conversation_id: Optional[str] = None,
        verbose: bool = False,
        use_semantic_expansion: bool = True,
        use_semantic_reranking: bool = True,
    ) -> Tuple[str, str]:
        """
        Process a complete query with semantic understanding.

        Args:
            query: User question
            history: Conversation history
            user_id: User ID for logs
            conversation_id: Conversation ID for logs
            verbose: Detailed display
            use_semantic_expansion: Enable semantic query expansion
            use_semantic_reranking: Enable semantic re-ranking

        Returns:
            Tuple (answer, sources)
        """
        start_time = time.time()

        try:
            if verbose:
                self.logger.info(f"🔍 Processing query with semantic understanding: '{query[:100]}...'")

            # Step 1: Query reformulation (new step)
            reformulated_query = query
            self.logger.info(f"🚀 PIPELINE START: Processing query: '{query}'")
            self.logger.info(
                f"📜 History provided: {bool(history and history.strip())} (length: {len(history) if history else 0})"
            )

            # Debug output that should be visible in Streamlit logs
            print(f"🚀 SEMANTIC PIPELINE: Processing '{query}'")
            print(f"📜 History length: {len(history) if history else 0}")

            if history and history.strip():
                if verbose:
                    self.logger.info("🔄 Step 1: Query reformulation")

                self.logger.info("📥 Extracting recent exchanges from history...")
                # Extract recent exchanges from history
                recent_exchanges = self._extract_exchanges_from_history(history)
                self.logger.info(f"📚 Extracted {len(recent_exchanges)} exchanges from history")

                # Log the history format for debugging
                self.logger.debug(f"📜 Raw history format: {repr(history[:200])}...")

                # Reformulate query to resolve coreferences
                self.logger.info("🔄 Calling ReformulationService...")
                print(f"🔄 CALLING REFORMULATION for: '{query}'")
                reformulated_query = self.reformulation_service.reformulate_query(query, recent_exchanges)
                print(f"🔄 REFORMULATION RESULT: '{reformulated_query}'")

                if reformulated_query != query:
                    self.logger.info(f"✅ QUERY REFORMULATED: '{query}' -> '{reformulated_query}'")
                    if verbose:
                        self.logger.info(f"📝 Query reformulated: '{query}' -> '{reformulated_query}'")
                    # For debugging - this should be visible in logs
                    print(f"🔄 REFORMULATION: '{query}' -> '{reformulated_query}'")
                else:
                    self.logger.info(f"⚪ Query unchanged after reformulation: '{query}'")
            else:
                self.logger.info("⚪ No history provided - skipping reformulation step")
                print("⚪ NO HISTORY - skipping reformulation")

            # Step 2: Process query for semantic understanding (optional legacy processing)
            query_result = None
            if self.use_semantic_features and use_semantic_expansion:
                if verbose:
                    self.logger.info("🧠 Step 2: Semantic query processing")

                # Use reformulated query for semantic processing
                query_result = self.query_processor.process_query(reformulated_query)

                if verbose:
                    self.logger.info(
                        f"📝 Intent: {query_result.intent}, "
                        f"Variations: {len(query_result.expanded_queries)}, "
                        f"Confidence: {query_result.confidence:.2f}"
                    )

            # Step 3: Semantic retrieval
            if verbose:
                self.logger.info("📥 Step 3: Semantic document retrieval")

            # Use reformulated query for retrieval
            print(f"🔍 VECTOR SEARCH: Using query '{reformulated_query}' for retrieval")
            search_results = self.semantic_retrieval_tool.retrieve(
                query=reformulated_query,
                use_semantic_expansion=use_semantic_expansion and self.use_semantic_features,
                use_semantic_reranking=use_semantic_reranking and self.use_semantic_features,
            )
            print(f"📄 VECTOR SEARCH: Found {len(search_results)} results")

            if verbose:
                self.logger.info(f"📄 {len(search_results)} documents retrieved")
                if search_results:
                    top_score = search_results[0].score
                    avg_score = sum(doc.score for doc in search_results) / len(search_results)
                    self.logger.info(f"📊 Top score: {top_score:.3f}, Average score: {avg_score:.3f}")

            # Step 4: Generate response
            if verbose:
                self.logger.info("🤖 Step 4: Generating response")

            # Use reformulated query for generation to ensure consistent filtering
            generation_result = self.generation_tool.generate(
                query=reformulated_query, documents=search_results, history=""
            )

            answer = generation_result["answer"]
            sources = generation_result["sources"]

            response_time = (time.time() - start_time) * 1000  # in ms

            if verbose:
                self.logger.info(f"✅ Response generated in {response_time:.0f}ms")

            # Save conversation with enhanced metadata
            try:
                conv_id = conversation_id or f"conv_{int(time.time())}"
                metadata = {
                    "pipeline_type": "semantic_rag",
                    "num_retrieved_docs": len(search_results),
                    "generation_success": generation_result["success"],
                    "semantic_features_enabled": self.use_semantic_features,
                    "semantic_expansion_used": use_semantic_expansion,
                    "semantic_reranking_used": use_semantic_reranking,
                    "query_reformulated": reformulated_query != query,
                    "reformulated_query": reformulated_query if reformulated_query != query else None,
                }

                # Add query processing metadata if available
                if query_result:
                    metadata.update(
                        {
                            "query_intent": query_result.intent,
                            "query_confidence": query_result.confidence,
                            "num_query_variations": len(query_result.expanded_queries),
                            "semantic_keywords": query_result.keywords,
                        }
                    )

                self.data_manager.save_conversation(
                    user_id=user_id,
                    conversation_id=conv_id,
                    question=query,
                    answer=answer,
                    response_time_ms=response_time,
                    sources=self._format_sources_for_storage(search_results),
                    metadata=metadata,
                )
                print(f"💾 SAVED CONVERSATION: '{query}' to conversation_id={conv_id}")
            except Exception as e:
                self.logger.warning(f"Failed to save conversation: {e}")

            return answer, sources

        except Exception as e:
            error_msg = f"Semantic RAG pipeline error: {str(e)}"
            self.logger.error(error_msg)

            # Save error conversation
            try:
                response_time = (time.time() - start_time) * 1000
                conv_id = conversation_id or f"conv_{int(time.time())}"
                self.data_manager.save_conversation(
                    user_id=user_id,
                    conversation_id=conv_id,
                    question=query,
                    answer=f"Error: {error_msg}",
                    response_time_ms=response_time,
                    metadata={"pipeline_type": "semantic_rag", "error": str(e)},
                )
            except Exception:
                pass

            return f"Sorry, an error occurred: {str(e)}", "System error"

    def compare_with_basic_retrieval(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Compare semantic retrieval with basic retrieval for evaluation.

        Args:
            query: Test query
            k: Number of results to compare

        Returns:
            Comparison results
        """
        try:
            # Semantic retrieval
            semantic_results = self.semantic_retrieval_tool.retrieve(
                query=query, k=k, use_semantic_expansion=True, use_semantic_reranking=True
            )

            # Basic retrieval (no semantic features)
            basic_results = self.semantic_retrieval_tool.retrieve(
                query=query, k=k, use_semantic_expansion=False, use_semantic_reranking=False
            )

            # Query processing info
            query_result = self.query_processor.process_query(query)

            return {
                "query": query,
                "query_processing": {
                    "intent": query_result.intent,
                    "confidence": query_result.confidence,
                    "keywords": query_result.keywords,
                    "expanded_queries": query_result.expanded_queries,
                },
                "semantic_retrieval": {
                    "count": len(semantic_results),
                    "scores": [r.score for r in semantic_results],
                    "top_content": semantic_results[0].content[:200] + "..." if semantic_results else None,
                    "avg_score": (
                        sum(r.score for r in semantic_results) / len(semantic_results) if semantic_results else 0
                    ),
                    "metadata_sample": semantic_results[0].metadata if semantic_results else None,
                },
                "basic_retrieval": {
                    "count": len(basic_results),
                    "scores": [r.score for r in basic_results],
                    "top_content": basic_results[0].content[:200] + "..." if basic_results else None,
                    "avg_score": sum(r.score for r in basic_results) / len(basic_results) if basic_results else 0,
                },
                "improvement_metrics": {
                    "score_improvement": (
                        (semantic_results[0].score - basic_results[0].score)
                        if semantic_results and basic_results
                        else 0
                    ),
                    "avg_score_improvement": (
                        (sum(r.score for r in semantic_results) / len(semantic_results))
                        - (sum(r.score for r in basic_results) / len(basic_results))
                        if semantic_results and basic_results
                        else 0
                    ),
                    "semantic_advantage": (
                        semantic_results[0].score > basic_results[0].score
                        if semantic_results and basic_results
                        else False
                    ),
                },
            }

        except Exception as e:
            return {"error": str(e), "query": query}

    def _format_sources_for_storage(self, formatted_docs) -> list[dict]:
        """Format sources for storage with enhanced metadata"""
        sources = []
        for doc in formatted_docs:
            source_info = {
                "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "score": doc.score,
                "metadata": doc.metadata,
            }

            # Add semantic-specific metadata if available
            if "semantic_score" in doc.metadata:
                source_info["semantic_metadata"] = {
                    "semantic_score": doc.metadata.get("semantic_score"),
                    "intent_bonus": doc.metadata.get("intent_bonus"),
                    "keyword_bonus": doc.metadata.get("keyword_bonus"),
                    "matched_query": doc.metadata.get("matched_query"),
                    "query_weight": doc.metadata.get("query_weight"),
                }

            sources.append(source_info)

        return sources

    def _extract_exchanges_from_history(self, history: str) -> List[ConversationExchange]:
        """
        Extract conversation exchanges from history string.

        Args:
            history: Formatted history string from format_chat_history

        Returns:
            List of ConversationExchange objects
        """
        exchanges = []

        if not history or not history.strip():
            self.logger.debug("📭 Empty history provided")
            return exchanges

        # Split history into lines and process
        lines = [line.strip() for line in history.split("\n") if line.strip()]
        self.logger.debug(f"📄 Processing {len(lines)} lines from history")

        current_user_msg = None

        for i, line in enumerate(lines):
            self.logger.debug(f"  Line {i + 1}: '{line[:100]}...'")

            if line.startswith("User: "):
                current_user_msg = line[6:]  # Remove 'User: '
                self.logger.debug(f"    -> Found user message: '{current_user_msg}'")
            elif line.startswith("Assistant: ") and current_user_msg:
                assistant_msg = line[11:]  # Remove 'Assistant: '
                exchange = ConversationExchange(user_message=current_user_msg, assistant_message=assistant_msg)
                exchanges.append(exchange)
                self.logger.debug(
                    f"    -> Created exchange: User='{current_user_msg}' Assistant='{assistant_msg[:50]}...'"
                )
                current_user_msg = None

        self.logger.info(f"📑 Extracted {len(exchanges)} exchanges total")

        # Return most recent exchanges (limit to avoid too much context)
        final_exchanges = exchanges[-5:] if len(exchanges) > 5 else exchanges
        self.logger.info(f"📋 Using {len(final_exchanges)} most recent exchanges")

        return final_exchanges

    def is_ready(self) -> bool:
        """Check if the pipeline is ready"""
        try:
            retrieval_ready = self.semantic_retrieval_tool.is_ready()
            generation_ready = self.generation_tool.is_ready()

            return retrieval_ready and generation_ready
        except Exception as e:
            self.logger.error(f"Readiness check failed: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        try:
            retrieval_stats = self.semantic_retrieval_tool.get_stats()
            generation_stats = self.generation_tool.get_stats()

            return {
                "pipeline_type": "semantic_rag_pipeline",
                "ready": self.is_ready(),
                "semantic_features_enabled": self.use_semantic_features,
                "config": {
                    "embeddings_model": self.config.embeddings_model,
                    "llm_model": self.config.llm_model,
                    "vectordb_collection": self.config.vectordb_collection,
                    "search_k": self.config.search_k,
                    "search_fetch_k": self.config.search_fetch_k,
                },
                "semantic_retrieval_tool": retrieval_stats,
                "generation_tool": generation_stats,
                "data_manager": self.data_manager.get_info(),
                "capabilities": {
                    "semantic_query_expansion": True,
                    "semantic_reranking": True,
                    "intent_classification": True,
                    "multilingual_support": True,
                    "synonym_handling": True,
                    "misleading_keyword_resolution": True,
                },
            }
        except Exception as e:
            return {"pipeline_type": "semantic_rag_pipeline", "ready": False, "error": str(e)}


class SemanticRAGPipelineFactory:
    """Factory for creating semantic RAG pipelines"""

    @staticmethod
    def create_semantic_pipeline(use_semantic_features: bool = True) -> SemanticRAGPipeline:
        """Create a semantic RAG pipeline"""
        pipeline = SemanticRAGPipeline(use_semantic_features=use_semantic_features)

        if not pipeline.is_ready():
            logging.warning("⚠️ Semantic RAG pipeline created but not ready - check that the vector database is built")

        return pipeline

    @staticmethod
    def create_comparison_pipeline() -> Tuple[SemanticRAGPipeline, SemanticRAGPipeline]:
        """Create both semantic and basic pipelines for comparison"""
        semantic_pipeline = SemanticRAGPipelineFactory.create_semantic_pipeline(use_semantic_features=True)
        basic_pipeline = SemanticRAGPipelineFactory.create_semantic_pipeline(use_semantic_features=False)

        return semantic_pipeline, basic_pipeline
