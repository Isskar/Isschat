"""
Enhanced RAG pipeline with semantic understanding capabilities.
Integrates semantic query processing, context-aware retrieval, and conversation tracking for better accuracy.
"""

import logging
import time
from typing import Tuple, Dict, Any, Optional

from ..config import get_config
from ..storage.data_manager import get_data_manager
from .tools.semantic_retrieval_tool import SemanticRetrievalTool
from .tools.generation_tool import GenerationTool
from .query_processor import QueryProcessor
from .conversation_context import ConversationContextTracker


class SemanticRAGPipeline:
    """
    Enhanced RAG pipeline with semantic understanding and context-aware capabilities.
    Handles misleading keywords through semantic query processing and maintains conversation context.
    """

    def __init__(self, use_semantic_features: bool = True, use_context_awareness: bool = True):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_manager = get_data_manager()
        self.use_semantic_features = use_semantic_features
        self.use_context_awareness = use_context_awareness

        # Initialize tools
        self.semantic_retrieval_tool = SemanticRetrievalTool()
        self.generation_tool = GenerationTool()
        self.query_processor = QueryProcessor()

        # Initialize context tracker
        if self.use_context_awareness:
            self.context_tracker = ConversationContextTracker()
        else:
            self.context_tracker = None

        self.logger.info(f"✅ Semantic RAG pipeline initialized (context-aware: {use_context_awareness})")

    def process_query(
        self,
        query: str,
        history: str = "",
        user_id: str = "anonymous",
        conversation_id: Optional[str] = None,
        verbose: bool = False,
        use_semantic_expansion: bool = True,
        use_semantic_reranking: bool = True,
        use_context_enrichment: bool = True,
    ) -> Tuple[str, str]:
        """
        Process a complete query with semantic understanding and context awareness.

        Args:
            query: User question
            history: Conversation history
            user_id: User ID for logs
            conversation_id: Conversation ID for logs
            verbose: Detailed display
            use_semantic_expansion: Enable semantic query expansion
            use_semantic_reranking: Enable semantic re-ranking
            use_context_enrichment: Enable conversational context enrichment

        Returns:
            Tuple (answer, sources)
        """
        start_time = time.time()
        original_query = query
        context_metadata = {"context_applied": False}

        try:
            if verbose:
                self.logger.info(f"🔍 Processing query with semantic understanding: '{query[:100]}...'")

            # Step 0: Context enrichment (if enabled and conversation_id provided)
            if self.use_context_awareness and use_context_enrichment and conversation_id and self.context_tracker:
                if verbose:
                    self.logger.info("🧩 Step 0: Context-aware query enrichment")

                enriched_query, context_metadata = self.context_tracker.enrich_query_with_context(
                    conversation_id, query
                )

                if context_metadata.get("context_applied", False):
                    query = enriched_query
                    if verbose:
                        self.logger.info(f"📝 Query enriched with context: '{original_query}' -> '{query[:150]}...'")
                        self.logger.info(
                            f"🏷️ Context entities: {[e['entity'] for e in context_metadata.get('context_entities', [])]}"
                        )

            # Step 1: Process query for semantic understanding
            query_result = None
            if self.use_semantic_features and use_semantic_expansion:
                if verbose:
                    self.logger.info("🧠 Step 1: Semantic query processing")

                query_result = self.query_processor.process_query(query)

                if verbose:
                    self.logger.info(
                        f"📝 Intent: {query_result.intent}, "
                        f"Variations: {len(query_result.expanded_queries)}, "
                        f"Confidence: {query_result.confidence:.2f}"
                    )

            # Step 2: Context-aware semantic retrieval
            if verbose:
                self.logger.info("📥 Step 2: Context-aware semantic document retrieval")

            search_results = self.semantic_retrieval_tool.retrieve(
                query=query,  # Use potentially enriched query
                use_semantic_expansion=use_semantic_expansion and self.use_semantic_features,
                use_semantic_reranking=use_semantic_reranking and self.use_semantic_features,
            )

            if verbose:
                self.logger.info(f"📄 {len(search_results)} documents retrieved")
                if search_results:
                    top_score = search_results[0].score
                    avg_score = sum(doc.score for doc in search_results) / len(search_results)
                    self.logger.info(f"📊 Top score: {top_score:.3f}, Average score: {avg_score:.3f}")

            # Step 3: Generate response with context awareness
            if verbose:
                self.logger.info("🤖 Step 3: Generating response")

            # Provide original query to generation tool for natural response
            generation_result = self.generation_tool.generate(
                query=original_query,  # Use original query for natural response generation
                documents=search_results,
                history=history,
            )

            answer = generation_result["answer"]
            sources = generation_result["sources"]

            response_time = (time.time() - start_time) * 1000  # in ms

            if verbose:
                self.logger.info(f"✅ Response generated in {response_time:.0f}ms")

            # Step 4: Track conversation turn for context (if enabled)
            if self.use_context_awareness and conversation_id and self.context_tracker:
                intent = query_result.intent if query_result else "general"
                self.context_tracker.track_conversation_turn(
                    conversation_id=conversation_id, query=original_query, response=answer, intent=intent
                )

                if verbose:
                    self.logger.info("🧩 Conversation turn tracked for future context")

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
                    "context_awareness_enabled": self.use_context_awareness,
                    "context_enrichment_used": use_context_enrichment,
                    "original_query": original_query,
                }

                # Add context metadata if available
                if context_metadata.get("context_applied", False):
                    metadata.update(
                        {
                            "context_applied": True,
                            "enriched_query": context_metadata.get("enriched_query"),
                            "context_entities": context_metadata.get("context_entities", []),
                            "context_turns_used": context_metadata.get("context_turns_used", 0),
                            "active_topics": context_metadata.get("active_topics", []),
                        }
                    )
                else:
                    metadata["context_applied"] = False

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

            status = {
                "pipeline_type": "semantic_rag_pipeline",
                "ready": self.is_ready(),
                "semantic_features_enabled": self.use_semantic_features,
                "context_awareness_enabled": self.use_context_awareness,
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
                    "conversation_context_tracking": self.use_context_awareness,
                    "context_aware_retrieval": self.use_context_awareness,
                    "entity_extraction": self.use_context_awareness,
                },
            }

            # Add context tracker status if available
            if self.context_tracker:
                active_conversations = len(self.context_tracker.contexts)
                total_turns = sum(len(ctx.turns) for ctx in self.context_tracker.contexts.values())
                status["context_tracker"] = {
                    "active_conversations": active_conversations,
                    "total_tracked_turns": total_turns,
                    "conversations": list(self.context_tracker.contexts.keys()),
                }

            return status
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

    def test_context_aware_conversation(self) -> Dict[str, Any]:
        """Test context-aware conversation with implicit references"""
        if not self.use_context_awareness or not self.context_tracker:
            return {"success": False, "error": "Context awareness not enabled"}

        try:
            test_conversation_id = f"test_context_{int(time.time())}"

            # Simulate a multi-turn conversation
            test_turns = [
                {
                    "query": "Qui travaille sur le projet Teora ?",
                    "expected_entities": ["teora"],
                    "expected_types": ["project"],
                },
                {
                    "query": "Quelles sont leurs responsabilités ?",  # Implicit reference to Teora team
                    "expected_context": True,
                    "expected_enrichment": ["teora"],
                },
                {
                    "query": "Comment puis-je les contacter ?",  # Further implicit reference
                    "expected_context": True,
                    "expected_enrichment": ["teora"],
                },
            ]

            test_results = []

            for i, turn in enumerate(test_turns):
                self.logger.info(f"Testing turn {i + 1}: {turn['query']}")

                # Process query with context
                answer, sources = self.process_query(
                    query=turn["query"], conversation_id=test_conversation_id, verbose=True, use_context_enrichment=True
                )

                # Get context summary
                context_summary = self.context_tracker.get_conversation_summary(test_conversation_id)

                turn_result = {
                    "turn": i + 1,
                    "query": turn["query"],
                    "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                    "sources_found": bool(sources and sources != "System error"),
                    "context_summary": context_summary,
                    "success": True,
                }

                # Check expectations
                if "expected_entities" in turn:
                    found_entities = [
                        e
                        for e in context_summary.get("entities_by_type", {}).get("project", [])
                        if any(exp in e.lower() for exp in turn["expected_entities"])
                    ]
                    turn_result["entities_extracted"] = found_entities
                    turn_result["entity_extraction_success"] = len(found_entities) > 0

                if turn.get("expected_context", False):
                    # Check if context was applied (this would be in metadata, simulated here)
                    turn_result["context_expected"] = True
                    turn_result["context_likely_applied"] = context_summary.get("active_entities", 0) > 0

                test_results.append(turn_result)

            # Overall assessment
            entity_extraction_success = any(r.get("entity_extraction_success", False) for r in test_results)
            context_progression = len(test_results) > 1 and test_results[-1]["context_summary"]["turn_count"] == len(
                test_turns
            )

            return {
                "success": True,
                "test_type": "context_aware_conversation",
                "conversation_id": test_conversation_id,
                "turns_tested": len(test_turns),
                "results": test_results,
                "overall_assessment": {
                    "entity_extraction_working": entity_extraction_success,
                    "context_tracking_working": context_progression,
                    "conversation_continuity": context_progression,
                },
                "final_context_state": self.context_tracker.get_conversation_summary(test_conversation_id),
            }

        except Exception as e:
            return {"success": False, "error": str(e), "test_type": "context_aware_conversation"}

    def get_context_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get context summary for a conversation"""
        if not self.context_tracker:
            return {"error": "Context tracking not enabled"}

        return self.context_tracker.get_conversation_summary(conversation_id)

    def clear_conversation_context(self, conversation_id: str):
        """Clear context for a specific conversation"""
        if self.context_tracker:
            self.context_tracker.clear_conversation_context(conversation_id)


class SemanticRAGPipelineFactory:
    """Factory for creating semantic RAG pipelines"""

    @staticmethod
    def create_semantic_pipeline(
        use_semantic_features: bool = True, use_context_awareness: bool = True
    ) -> SemanticRAGPipeline:
        """Create a semantic RAG pipeline with optional context awareness"""
        pipeline = SemanticRAGPipeline(
            use_semantic_features=use_semantic_features, use_context_awareness=use_context_awareness
        )

        if not pipeline.is_ready():
            logging.warning("⚠️ Semantic RAG pipeline created but not ready - check that the vector database is built")

        return pipeline

    @staticmethod
    def create_context_aware_pipeline() -> SemanticRAGPipeline:
        """Create a fully featured context-aware semantic RAG pipeline"""
        return SemanticRAGPipelineFactory.create_semantic_pipeline(
            use_semantic_features=True, use_context_awareness=True
        )

    @staticmethod
    def create_basic_pipeline() -> SemanticRAGPipeline:
        """Create a basic pipeline without semantic features or context awareness"""
        return SemanticRAGPipelineFactory.create_semantic_pipeline(
            use_semantic_features=False, use_context_awareness=False
        )

    @staticmethod
    def create_comparison_pipeline() -> Tuple[SemanticRAGPipeline, SemanticRAGPipeline, SemanticRAGPipeline]:
        """Create context-aware, semantic, and basic pipelines for comparison"""
        context_aware_pipeline = SemanticRAGPipelineFactory.create_context_aware_pipeline()
        semantic_pipeline = SemanticRAGPipelineFactory.create_semantic_pipeline(
            use_semantic_features=True, use_context_awareness=False
        )
        basic_pipeline = SemanticRAGPipelineFactory.create_basic_pipeline()

        return context_aware_pipeline, semantic_pipeline, basic_pipeline
