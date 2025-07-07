from typing import List, Dict, Any, Optional
import logging

from ...config import get_config
from ...embeddings import get_embedding_service
from ...vectordb import VectorDBFactory
from ...core.documents import RetrievalDocument, SearchResult


class RetrievalTool:
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
            # Centralized embedding service
            self._embedding_service = get_embedding_service()

            # Vector DB depuis factory
            self._vector_db = VectorDBFactory.create_from_config()

            self._initialized = True
            self.logger.debug("Retrieval tool initialized with centralized services")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize retrieval tool: {e}")

    def retrieve(
        self, query: str, k: Optional[int] = None, filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalDocument]:
        """
        Retrieve relevant documents for a query

        Args:
            query: User query
            k: Number of results to return for generation (otherwise uses config.search_k)
            filter_conditions: Filter conditions

        Returns:
            List of retrieval documents (top k documents from search_fetch_k candidates)
        """
        self._initialize()

        try:
            # Number of candidates to fetch from vector DB
            fetch_k = self.config.search_fetch_k

            # Number of results to return for generation
            return_k = k if k is not None else self.config.search_k

            # Generate query embedding via centralized service
            query_embedding = self._embedding_service.encode_query(query)

            # Search in vector DB - fetch more candidates
            search_results = self._vector_db.search(
                query_embedding=query_embedding, k=fetch_k, filter_conditions=filter_conditions
            )

            # Convert to retrieval documents
            retrieval_docs = self._format_documents(search_results)

            # Keep only the k best documents for generation
            top_k_docs = retrieval_docs[:return_k]

            self.logger.debug(
                f"Retrieval: fetched {len(retrieval_docs)} candidates, "
                f"returning top {len(top_k_docs)} for '{query[:50]}...'"
            )
            return top_k_docs

        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            raise RuntimeError(f"Retrieval failed: {e}")

    def _format_documents(self, search_results: List[SearchResult]) -> List[RetrievalDocument]:
        """Convert SearchResult objects to RetrievalDocument objects"""
        retrieval_docs = []

        for result in search_results:
            # Extract content and metadata from SearchResult
            content = result.document.content
            metadata = result.document.metadata or {}
            score = result.score

            retrieval_doc = RetrievalDocument(content=content, metadata=metadata, score=score)
            retrieval_docs.append(retrieval_doc)

        return retrieval_docs

    def is_ready(self) -> bool:
        try:
            self._initialize()
            return self._vector_db.exists() and self._vector_db.count() > 0
        except Exception as e:
            self.logger.error(f"Readiness check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        try:
            self._initialize()

            # Stats vector DB
            db_info = self._vector_db.get_info()

            # Stats embedding service
            embedding_info = self._embedding_service.get_info()

            return {
                "type": "retrieval_tool",
                "ready": self.is_ready(),
                "config": {
                    "search_k": self.config.search_k,
                    "search_fetch_k": self.config.search_fetch_k,
                    "vectordb_collection": self.config.vectordb_collection,
                    "embeddings_model": self.config.embeddings_model,
                },
                "vector_db": db_info,
                "embedding_service": embedding_info,
            }

        except Exception as e:
            return {"type": "retrieval_tool", "ready": False, "error": str(e)}

    def test_retrieval(self, test_query: str = "test query") -> Dict[str, Any]:
        try:
            self._initialize()

            if not self.is_ready():
                return {"success": False, "error": "Vector DB empty or not accessible"}

            results = self.retrieve(test_query, k=3)

            return {
                "success": True,
                "query": test_query,
                "results_count": len(results),
                "scores": [r.score for r in results] if results else [],
                "sample_content": results[0].content[:100] + "..." if results else None,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
