"""
Enhanced RAG pipeline with semantic understanding capabilities.
Now powered by LlamaIndex for optimized query processing and memory management.
"""

import logging
import time
from typing import Tuple, Dict, Any, Optional

from ..config import get_config
from ..storage.data_manager import get_data_manager
from .llama_index_pipeline import LlamaIndexRAGPipelineFactory


class SemanticRAGPipeline:
    """
    Enhanced RAG pipeline powered by LlamaIndex.
    Provides advanced query transformation and unified memory management.
    """

    def __init__(self, use_semantic_features: bool = True):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_manager = get_data_manager()
        self.use_semantic_features = use_semantic_features

        # Initialize LlamaIndex pipeline based on configuration
        pipeline_type = getattr(self.config, "llamaindex_pipeline_type", "hyde")

        if pipeline_type == "decompose":
            self.llama_pipeline = LlamaIndexRAGPipelineFactory.create_decompose_pipeline()
        elif pipeline_type == "hybrid":
            self.llama_pipeline = LlamaIndexRAGPipelineFactory.create_hybrid_pipeline()
        elif pipeline_type == "simple":
            self.llama_pipeline = LlamaIndexRAGPipelineFactory.create_simple_pipeline()
        else:  # default to hyde
            self.llama_pipeline = LlamaIndexRAGPipelineFactory.create_hyde_pipeline()

        self.logger.info(f"✅ LlamaIndex-powered RAG pipeline initialized (type: {pipeline_type})")

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
        Process a complete query using LlamaIndex pipeline.

        Args:
            query: User question
            history: Conversation history (now managed by ChatMemoryBuffer)
            user_id: User ID for logs
            conversation_id: Conversation ID for logs
            verbose: Detailed display
            use_semantic_expansion: Deprecated - managed by LlamaIndex
            use_semantic_reranking: Deprecated - managed by LlamaIndex

        Returns:
            Tuple (answer, sources)
        """
        try:
            if verbose:
                self.logger.info(f"🔍 Processing query with LlamaIndex: '{query[:100]}...'")

            # Delegate to LlamaIndex pipeline which handles:
            # - HyDE query transformation (replaces costly semantic expansion)
            # - Memory management (ChatMemoryBuffer)
            # - Optimized retrieval (single enhanced query vs 5 basic queries)
            # - Response generation
            answer, sources = self.llama_pipeline.process_query(
                query=query,
                user_id=user_id,
                conversation_id=conversation_id,
                verbose=verbose,
            )

            if verbose:
                self.logger.info("✅ Query processed successfully with LlamaIndex")

            return answer, sources

        except Exception as e:
            self.logger.error(f"LlamaIndex pipeline failed: {e}")
            error_answer = f"Sorry, an error occurred while processing your question: {str(e)}"
            return error_answer, ""

    def start_new_conversation(self, conversation_id: str) -> bool:
        """Start a new conversation"""
        if hasattr(self, "llama_pipeline"):
            return self.llama_pipeline.start_new_conversation(conversation_id)
        return False

    def continue_conversation(self, conversation_id: str, user_id: str = "anonymous") -> bool:
        """Continue an existing conversation by loading its history"""
        if hasattr(self, "llama_pipeline"):
            return self.llama_pipeline.continue_conversation(conversation_id, user_id)
        return False

    def load_conversation_history(self, conversation_id: str, user_id: str = "anonymous") -> bool:
        """Load conversation history into memory"""
        if hasattr(self, "llama_pipeline"):
            return self.llama_pipeline.load_conversation_history(conversation_id, user_id)
        return False

    def clear_memory(self):
        """Clear conversation memory"""
        if hasattr(self, "llama_pipeline"):
            self.llama_pipeline.clear_memory()
            self.logger.debug("LlamaIndex memory cleared")

    def get_memory_summary(self) -> str:
        """Get current memory content summary"""
        if hasattr(self, "llama_pipeline"):
            return self.llama_pipeline.get_memory_summary()
        return "Memory not available"

    def is_ready(self) -> bool:
        """Check if pipeline is ready"""
        try:
            return hasattr(self, "llama_pipeline") and self.llama_pipeline.is_ready()
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        if hasattr(self, "llama_pipeline"):
            stats = self.llama_pipeline.get_stats()
            stats.update(
                {
                    "wrapper_type": "semantic_rag_pipeline",
                    "llama_index_enabled": True,
                }
            )
            return stats

        return {"wrapper_type": "semantic_rag_pipeline", "llama_index_enabled": False, "ready": False}

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

    def test_problematic_query(self, query: str = "qui sont les collaborateurs sur Isschat") -> Dict[str, Any]:
        """
        Test the pipeline with the specific problematic query about collaborators.

        Args:
            query: The problematic query to test

        Returns:
            Detailed test results
        """
        try:
            # Test with full semantic pipeline
            start_time = time.time()
            answer, sources = self.process_query(query, verbose=True)
            response_time = (time.time() - start_time) * 1000

            # Get comparison data
            comparison = self.compare_with_basic_retrieval(query)

            # Analyze if the answer contains team information
            team_keywords = ["vincent", "nicolas", "emin", "fraillon", "lambropoulos", "calyaka", "équipe", "team"]
            answer_lower = answer.lower()
            team_mentions = [keyword for keyword in team_keywords if keyword in answer_lower]

            return {
                "test_query": query,
                "semantic_pipeline_result": {
                    "answer": answer,
                    "sources": sources,
                    "response_time_ms": response_time,
                    "team_keywords_found": team_mentions,
                    "contains_team_info": len(team_mentions) > 2,
                },
                "comparison": comparison,
                "success_criteria": {
                    "finds_team_info": len(team_mentions) > 2,
                    "mentions_specific_names": any(name in answer_lower for name in ["vincent", "nicolas", "emin"]),
                    "better_than_basic": comparison.get("improvement_metrics", {}).get("semantic_advantage", False),
                },
                "pipeline_status": self.get_status(),
            }

        except Exception as e:
            return {"error": str(e), "test_query": query}

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

    def check_pipeline(self, test_query: str = "qui sont les collaborateurs sur Isschat") -> Dict[str, Any]:
        """Check pipeline with default problematic query"""
        try:
            if not self.is_ready():
                return {"success": False, "error": "Pipeline not ready", "details": self.get_status()}

            # Run the problematic query test
            test_result = self.test_problematic_query(test_query)

            return {
                "success": test_result.get("success_criteria", {}).get("finds_team_info", False),
                "test_result": test_result,
                "pipeline_status": self.get_status(),
            }

        except Exception as e:
            return {"success": False, "error": str(e), "test_query": test_query}


class SemanticRAGPipelineFactory:
    """Factory for creating semantic RAG pipelines"""

    @staticmethod
    def create_semantic_pipeline(use_semantic_features: bool = True) -> SemanticRAGPipeline:
        """Create a LlamaIndex-powered semantic RAG pipeline"""
        pipeline = SemanticRAGPipeline(use_semantic_features=use_semantic_features)

        if not pipeline.is_ready():
            logging.warning("⚠️ LlamaIndex RAG pipeline created but not ready - check that the vector database is built")

        return pipeline

    @staticmethod
    def create_hyde_pipeline() -> SemanticRAGPipeline:
        """Create pipeline optimized with HyDE query transformation"""
        pipeline = SemanticRAGPipeline()
        # Override with HyDE-specific configuration
        pipeline.llama_pipeline = LlamaIndexRAGPipelineFactory.create_hyde_pipeline()
        return pipeline

    @staticmethod
    def create_decompose_pipeline() -> SemanticRAGPipeline:
        """Create pipeline optimized with query decomposition"""
        pipeline = SemanticRAGPipeline()
        # Override with decompose-specific configuration
        pipeline.llama_pipeline = LlamaIndexRAGPipelineFactory.create_decompose_pipeline()
        return pipeline

    @staticmethod
    def create_hybrid_pipeline() -> SemanticRAGPipeline:
        """Create pipeline with both HyDE and decomposition"""
        pipeline = SemanticRAGPipeline()
        # Override with hybrid configuration
        pipeline.llama_pipeline = LlamaIndexRAGPipelineFactory.create_hybrid_pipeline()
        return pipeline

    @staticmethod
    def create_comparison_pipeline() -> Tuple[SemanticRAGPipeline, SemanticRAGPipeline]:
        """Create both LlamaIndex and simple pipelines for comparison"""
        llamaindex_pipeline = SemanticRAGPipelineFactory.create_hyde_pipeline()
        simple_pipeline = SemanticRAGPipeline()
        simple_pipeline.llama_pipeline = LlamaIndexRAGPipelineFactory.create_simple_pipeline()

        return llamaindex_pipeline, simple_pipeline
