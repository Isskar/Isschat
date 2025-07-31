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


class SemanticRetrievalTool:
    """
    Modernized vector retrieval tool that works with reformulated queries.
    Provides optional semantic re-ranking for improved relevance.
    """

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Centralized services - lazy loading
        self._embedding_service = None
        self._vector_db = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of services"""
        if self._initialized:
            return

        try:
            # Initialize services
            self._embedding_service = get_embedding_service()
            self._vector_db = VectorDBFactory.create_from_config()

            self._initialized = True
            self.logger.debug("Vector retrieval tool initialized")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize semantic retrieval tool: {e}")

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        use_semantic_reranking: bool = True,
    ) -> List[RetrievalDocument]:
        """
        Retrieve relevant documents using vector similarity.

        Args:
            query: User query
            k: Number of results to return
            filter_conditions: Filter conditions
            use_semantic_reranking: Whether to use semantic re-ranking

        Returns:
            List of relevant documents
        """
        self._initialize()

        try:
            # Debug output to verify what query is received
            print(f"🔍 RETRIEVAL TOOL: Received query '{query}'")

            all_results = self._direct_vector_retrieval(query, filter_conditions)

            if use_semantic_reranking and len(all_results) > 1:
                all_results = self._semantic_rerank_simple(query, all_results)

            return_k = k if k is not None else self.config.search_k
            final_results = all_results[:return_k]

            self.logger.debug(f"Vector retrieval: {len(final_results)} results for '{query[:50]}...'")
            return final_results

        except Exception as e:
            self.logger.error(f"Vector retrieval failed: {e}")
            # Fallback to basic retrieval
            return self._basic_retrieval(query, k, filter_conditions)

    def _direct_vector_retrieval(
        self, query: str, filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalDocument]:
        """
        Perform direct vector retrieval.
        """
        try:
            query_embedding = self._embedding_service.encode_query(query)

            search_results = self._vector_db.search(
                query_embedding=query_embedding, k=self.config.search_fetch_k, filter_conditions=filter_conditions
            )

            retrieval_docs = []
            for result in search_results:
                retrieval_doc = RetrievalDocument(
                    content=result.document.content,
                    metadata=result.document.metadata or {},
                    score=result.score,
                )
                retrieval_docs.append(retrieval_doc)

            retrieval_docs.sort(key=lambda x: x.score, reverse=True)

            self.logger.debug(f"Vector retrieval: {len(retrieval_docs)} documents found")
            return retrieval_docs

        except Exception as e:
            self.logger.error(f"Vector retrieval failed: {e}")
            return []

    def _semantic_rerank_simple(self, query: str, candidates: List[RetrievalDocument]) -> List[RetrievalDocument]:
        """
        Semantic re-ranking based on content similarity with the query.
        """
        if not candidates:
            return candidates

        try:
            query_embedding = self._embedding_service.encode_single(query)

            for candidate in candidates:
                content_embedding = self._embedding_service.encode_single(candidate.content)
                semantic_score = self._embedding_service.similarity(query_embedding, content_embedding)

                query_keywords = [word.lower() for word in query.split() if len(word) > 2]
                content_lower = candidate.content.lower()
                keyword_matches = sum(1 for keyword in query_keywords if keyword in content_lower)
                keyword_bonus = min(0.3, keyword_matches / len(query_keywords)) if query_keywords else 0

                combined_score = 0.7 * candidate.score + 0.2 * semantic_score + 0.1 * keyword_bonus

                candidate.score = combined_score
                candidate.metadata["semantic_score"] = semantic_score
                candidate.metadata["keyword_bonus"] = keyword_bonus
                candidate.metadata["combined_score"] = combined_score

            candidates.sort(key=lambda x: x.score, reverse=True)

            self.logger.debug(f"Semantic re-ranking applied to {len(candidates)} candidates")
            return candidates

        except Exception as e:
            self.logger.error(f"Semantic re-ranking failed: {e}")
            return candidates

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
                "type": "vector_retrieval_tool",
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
                    "direct_vector_retrieval": True,
                    "semantic_reranking": True,
                    "query_reformulation_compatible": True,
                    "french_language_support": True,
                },
            }

        except Exception as e:
            return {"type": "vector_retrieval_tool", "ready": False, "error": str(e)}
