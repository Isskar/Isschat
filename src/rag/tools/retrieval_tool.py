"""
Unified retrieval tool using centralized services.
Replaces previous retrieval_tool.py with centralized services.
"""

from typing import List, Dict, Any, Optional
import logging

from ...config import get_config
from ...embeddings import get_embedding_service
from ...vectordb import VectorDBFactory, SearchResult


class RetrievalTool:
    """Retrieval tool using centralized services"""

    def __init__(self):
        """Initialize with centralized services"""
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
    ) -> List[SearchResult]:
        """
        Retrieve relevant documents for a query

        Args:
            query: User query
            k: Number of results (otherwise uses config)
            filter_conditions: Filter conditions

        Returns:
            List of search results
        """
        self._initialize()

        try:
            # Number of results
            search_k = k if k is not None else self.config.search_k

            # Generate query embedding via centralized service
            query_embedding = self._embedding_service.encode_query(query)

            # Search in vector DB
            results = self._vector_db.search(
                query_embedding=query_embedding, k=search_k, filter_conditions=filter_conditions
            )

            self.logger.debug(f"Retrieval: {len(results)} documents found for '{query[:50]}...'")
            return results

        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            raise RuntimeError(f"Retrieval failed: {e}")

    def is_ready(self) -> bool:
        """Check if the tool is ready"""
        try:
            self._initialize()
            return self._vector_db.exists() and self._vector_db.count() > 0
        except Exception as e:
            self.logger.error(f"Readiness check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Retrieval tool statistics"""
        try:
            self._initialize()

            # Stats vector DB
            db_info = self._vector_db.get_info()

            # Stats embedding service
            embedding_info = self._embedding_service.get_info()

            return {
                "type": "unified_retrieval_tool",
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
            return {"type": "unified_retrieval_tool", "ready": False, "error": str(e)}

    def test_retrieval(self, test_query: str = "test query") -> Dict[str, Any]:
        """Test the retrieval system"""
        try:
            self._initialize()

            if not self.is_ready():
                return {"success": False, "error": "Vector DB empty or not accessible"}

            # Retrieval test
            results = self.retrieve(test_query, k=3)

            return {
                "success": True,
                "query": test_query,
                "results_count": len(results),
                "scores": [r.score for r in results] if results else [],
                "sample_content": results[0].document.content[:100] + "..." if results else None,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
