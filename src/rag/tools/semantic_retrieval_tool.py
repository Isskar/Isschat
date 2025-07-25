"""
Semantic retrieval tool with enhanced query processing and re-ranking capabilities.
Improves upon the basic retrieval tool by adding semantic understanding.
"""

import logging
from typing import List, Dict, Any, Optional

from ...config import get_config
from ...embeddings import get_embedding_service
from ...vectordb import VectorDBFactory
from ...core.documents import RetrievalDocument
from ..query_processor import QueryProcessor, QueryProcessingResult


class SemanticRetrievalTool:
    """
    Enhanced retrieval tool with semantic understanding capabilities.
    Handles misleading keywords through query expansion and semantic re-ranking.
    """

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Centralized services - lazy loading
        self._embedding_service = None
        self._vector_db = None
        self._query_processor = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of services"""
        if self._initialized:
            return

        try:
            # Initialize services
            self._embedding_service = get_embedding_service()
            self._vector_db = VectorDBFactory.create_from_config()
            self._query_processor = QueryProcessor()

            self._initialized = True
            self.logger.debug("Semantic retrieval tool initialized")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize semantic retrieval tool: {e}")

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        use_semantic_expansion: bool = True,
        use_semantic_reranking: bool = True,
    ) -> List[RetrievalDocument]:
        """
        Retrieve relevant documents with semantic understanding.

        Args:
            query: User query
            k: Number of results to return
            filter_conditions: Filter conditions
            use_semantic_expansion: Whether to use query expansion
            use_semantic_reranking: Whether to use semantic re-ranking

        Returns:
            List of semantically relevant documents
        """
        self._initialize()

        try:
            # Process query for semantic understanding
            if use_semantic_expansion:
                query_result = self._query_processor.process_query(query)
                self.logger.debug(
                    f"Query processed: intent={query_result.intent}, variations={len(query_result.expanded_queries)}"
                )
            else:
                query_result = QueryProcessingResult(
                    original_query=query,
                    expanded_queries=[query],
                    intent="general",
                    keywords=query.split(),
                    semantic_variations=[],
                    confidence=0.5,
                )

            # Retrieve documents using multi-query approach
            all_results = self._multi_query_retrieval(query_result, filter_conditions)

            # Apply semantic re-ranking if enabled
            if use_semantic_reranking and len(all_results) > 1:
                all_results = self._semantic_rerank(query_result, all_results)

            # Determine number of results to return
            return_k = k if k is not None else self.config.search_k
            final_results = all_results[:return_k]

            self.logger.debug(f"Semantic retrieval: {len(final_results)} results for '{query[:50]}...'")
            return final_results

        except Exception as e:
            self.logger.error(f"Semantic retrieval failed: {e}")
            # Fallback to basic retrieval
            return self._basic_retrieval(query, k, filter_conditions)

    def _multi_query_retrieval(
        self, query_result: QueryProcessingResult, filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalDocument]:
        """
        Perform retrieval using multiple query variations and merge results.
        """
        all_candidates = []
        seen_content = set()

        # Weight queries based on their origin
        query_weights = {
            query_result.original_query: 1.0,  # Original query has highest weight
        }

        # Lower weights for expanded queries
        for i, expanded_query in enumerate(query_result.expanded_queries[1:], 1):
            query_weights[expanded_query] = max(0.3, 1.0 - (i * 0.1))

        # Retrieve for each query variation
        for query_text in query_result.expanded_queries:
            try:
                # Generate embedding for this query variation
                query_embedding = self._embedding_service.encode_query(query_text)

                # Search in vector database
                search_results = self._vector_db.search(
                    query_embedding=query_embedding, k=self.config.search_fetch_k, filter_conditions=filter_conditions
                )

                # Convert to retrieval documents with weighted scores
                query_weight = query_weights.get(query_text, 0.5)
                for result in search_results:
                    # Avoid duplicates based on content
                    content_key = result.document.content[:200]  # First 200 chars as key
                    if content_key not in seen_content:
                        seen_content.add(content_key)

                        # Adjust score based on query weight
                        weighted_score = result.score * query_weight

                        retrieval_doc = RetrievalDocument(
                            content=result.document.content,
                            metadata=result.document.metadata or {},
                            score=weighted_score,
                        )

                        # Add query information to metadata
                        retrieval_doc.metadata["matched_query"] = query_text
                        retrieval_doc.metadata["query_weight"] = query_weight
                        retrieval_doc.metadata["original_score"] = result.score

                        all_candidates.append(retrieval_doc)

            except Exception as e:
                self.logger.warning(f"Failed to retrieve for query '{query_text}': {e}")
                continue

        # Sort by weighted score
        all_candidates.sort(key=lambda x: x.score, reverse=True)

        # Return top candidates (more than final k to allow for re-ranking)
        max_candidates = min(len(all_candidates), self.config.search_fetch_k * 2)
        return all_candidates[:max_candidates]

    def _semantic_rerank(
        self, query_result: QueryProcessingResult, candidates: List[RetrievalDocument]
    ) -> List[RetrievalDocument]:
        """
        Re-rank candidates using intent and keyword matching only.
        Removed redundant semantic similarity re-computation.
        """
        if not candidates:
            return candidates

        try:
            # Calculate intent and keyword bonuses without re-embedding
            for candidate in candidates:
                # Intent matching bonus
                intent_bonus = self._calculate_intent_bonus(query_result.intent, candidate)

                # Keyword matching bonus
                keyword_bonus = self._calculate_keyword_bonus(query_result.keywords, candidate)

                # Combine scores: keep original vector score + small bonuses
                # 90% original vector score, 7% intent, 3% keywords
                combined_score = 0.90 * candidate.score + 0.07 * intent_bonus + 0.03 * keyword_bonus

                # Update candidate score and add debugging info
                candidate.score = combined_score
                candidate.metadata["intent_bonus"] = intent_bonus
                candidate.metadata["keyword_bonus"] = keyword_bonus
                candidate.metadata["combined_score"] = combined_score

            # Sort by combined score
            candidates.sort(key=lambda x: x.score, reverse=True)

            self.logger.debug(f"Lightweight re-ranking applied to {len(candidates)} candidates")
            return candidates

        except Exception as e:
            self.logger.error(f"Re-ranking failed: {e}")
            return candidates

    def _calculate_intent_bonus(self, intent: str, candidate: RetrievalDocument) -> float:
        """Calculate bonus score based on intent matching"""
        content_lower = candidate.content.lower()

        intent_keywords = {
            "team_info": [
                "équipe",
                "team",
                "collaborateurs",
                "membres",
                "vincent",
                "nicolas",
                "emin",
                "fraillon",
                "lambropoulos",
                "calyaka",
                "composition",
                "responsabilités",
            ],
            "project_info": ["projet", "isschat", "application", "description", "objectif", "but"],
            "technical_info": ["configuration", "installation", "utilisation", "problème", "erreur"],
            "feature_info": ["fonctionnalités", "features", "capacités", "options", "paramètres"],
        }

        if intent in intent_keywords:
            keywords = intent_keywords[intent]
            matches = sum(1 for keyword in keywords if keyword in content_lower)
            return min(1.0, matches / len(keywords))

        return 0.0

    def _calculate_keyword_bonus(self, keywords: List[str], candidate: RetrievalDocument) -> float:
        """Calculate bonus score based on keyword matching"""
        if not keywords:
            return 0.0

        content_lower = candidate.content.lower()
        matches = sum(1 for keyword in keywords if keyword in content_lower)
        return min(1.0, matches / len(keywords))

    def _basic_retrieval(
        self, query: str, k: Optional[int] = None, filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalDocument]:
        """Fallback to basic retrieval if semantic retrieval fails"""
        try:
            return_k = k if k is not None else self.config.search_k

            query_embedding = self._embedding_service.encode_query(query)
            search_results = self._vector_db.search(
                query_embedding=query_embedding, k=return_k, filter_conditions=filter_conditions
            )

            return [
                RetrievalDocument(
                    content=result.document.content, metadata=result.document.metadata or {}, score=result.score
                )
                for result in search_results
            ]

        except Exception as e:
            self.logger.error(f"Basic retrieval fallback failed: {e}")
            return []

    def is_ready(self) -> bool:
        """Check if the retrieval tool is ready"""
        try:
            self._initialize()
            return self._vector_db.exists() and self._vector_db.count() > 0
        except Exception as e:
            self.logger.error(f"Readiness check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval tool"""
        try:
            self._initialize()

            # Basic stats
            db_info = self._vector_db.get_info()
            embedding_info = self._embedding_service.get_info()

            return {
                "type": "semantic_retrieval_tool",
                "ready": self.is_ready(),
                "config": {
                    "search_k": self.config.search_k,
                    "search_fetch_k": self.config.search_fetch_k,
                    "vectordb_collection": self.config.vectordb_collection,
                    "embeddings_model": self.config.embeddings_model,
                },
                "vector_db": db_info,
                "embedding_service": embedding_info,
                "features": {
                    "semantic_expansion": True,
                    "semantic_reranking": True,
                    "intent_classification": True,
                    "multi_query_retrieval": True,
                },
            }

        except Exception as e:
            return {"type": "semantic_retrieval_tool", "ready": False, "error": str(e)}

    def test_semantic_retrieval(self, test_query: str = "qui sont les collaborateurs sur Isschat") -> Dict[str, Any]:
        """Test semantic retrieval with the specific problematic query"""
        try:
            self._initialize()

            if not self.is_ready():
                return {"success": False, "error": "Vector DB empty or not accessible"}

            # Test with semantic features enabled
            semantic_results = self.retrieve(test_query, k=5, use_semantic_expansion=True, use_semantic_reranking=True)

            # Test with semantic features disabled (basic retrieval)
            basic_results = self.retrieve(test_query, k=5, use_semantic_expansion=False, use_semantic_reranking=False)

            return {
                "success": True,
                "query": test_query,
                "semantic_results": {
                    "count": len(semantic_results),
                    "scores": [r.score for r in semantic_results],
                    "sample_content": semantic_results[0].content[:200] + "..." if semantic_results else None,
                    "matched_queries": [r.metadata.get("matched_query") for r in semantic_results[:3]],
                },
                "basic_results": {
                    "count": len(basic_results),
                    "scores": [r.score for r in basic_results],
                    "sample_content": basic_results[0].content[:200] + "..." if basic_results else None,
                },
                "improvement": {
                    "score_improvement": (
                        (semantic_results[0].score - basic_results[0].score)
                        if semantic_results and basic_results
                        else 0
                    ),
                    "semantic_features_help": (
                        len(semantic_results) > 0
                        and (not basic_results or semantic_results[0].score > basic_results[0].score)
                    ),
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
