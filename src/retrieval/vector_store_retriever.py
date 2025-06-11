"""
Vector store retriever implementation.
Clean retriever that works with pre-loaded vector stores.
"""

from typing import Dict, Any, Optional
import time

from src.core.interfaces import RetrievalResult, Document
from src.core.exceptions import RetrievalError
from src.retrieval.base_retriever import BaseRetriever


class VectorStoreRetriever(BaseRetriever):
    """
    Simple retriever that works with pre-loaded vector stores.
    No database management - assumes vector store is already loaded and ready.
    """

    def __init__(self, vector_store, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize retriever with a pre-loaded vector store.

        Args:
            vector_store: Pre-loaded vector store instance
            search_kwargs: Search configuration
        """
        if not vector_store:
            raise ValueError("vector_store is required and must be pre-loaded")

        self.vector_store = vector_store
        self.search_kwargs = search_kwargs or {"k": 3, "fetch_k": 5}
        self.retriever = vector_store.as_retriever(search_kwargs=self.search_kwargs)

        print("✅ VectorStoreRetriever initialized with pre-loaded vector store")

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            RetrievalResult with documents and metadata
        """
        try:
            start_time = time.time()

            # Use custom top_k if different from default
            if top_k != self.search_kwargs.get("k", 3):
                search_kwargs = self.search_kwargs.copy()
                search_kwargs["k"] = top_k
                search_kwargs["fetch_k"] = max(top_k + 2, search_kwargs.get("fetch_k", 5))
                retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
            else:
                retriever = self.retriever

            # Retrieve documents
            docs = retriever.invoke(query)

            # Convert to our format
            documents = []
            scores = []
            for doc in docs:
                documents.append(Document(page_content=doc.page_content, metadata=doc.metadata))
                scores.append(getattr(doc, "score", 0.0))

            retrieval_time = time.time() - start_time

            return RetrievalResult(documents=documents, scores=scores, query=query, retrieval_time=retrieval_time)

        except Exception as e:
            raise RetrievalError(f"Failed to retrieve documents for query '{query}': {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        try:
            return {
                "status": "initialized",
                "retriever_type": "VectorStore Retriever",
                "search_kwargs": self.search_kwargs,
                "vector_store_type": type(self.vector_store).__name__,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
