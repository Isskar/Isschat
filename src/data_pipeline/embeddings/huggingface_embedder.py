"""
HuggingFace embedding implementation.
Refactored to use centralized embeddings manager.
"""

from typing import List, Dict, Any
from .base_embedder import BaseEmbedder

from src.core.interfaces import Document
from src.core.embeddings_manager import EmbeddingsManager


class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace-based embedding generator using centralized configuration."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace embedder using centralized embeddings manager.

        Args:
            config: Configuration containing model settings (now uses centralized config)
        """
        self.config = config
        # Use centralized embeddings manager instead of creating our own model
        self.embeddings = EmbeddingsManager.get_embeddings()
        self._dimension = None

        print("✅ HuggingFaceEmbedder initialized with centralized embeddings")

    def _get_dimension(self):
        """Get embedding dimension from the centralized embeddings."""
        if self._dimension is None:
            # Get dimension from a test embedding
            test_embedding = self.embeddings.embed_query("test")
            self._dimension = len(test_embedding)
        return self._dimension

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for documents using centralized embeddings with progress monitoring.

        Args:
            documents: List of documents to embed

        Returns:
            List[List[float]]: Document embeddings
        """
        # Extract text content from documents
        texts = [doc.page_content for doc in documents]
        total_docs = len(texts)

        print(f"🔄 Starting embedding of {total_docs} documents...")
        print(f"🔄 Using model: {self.embeddings.model_name}")

        # Get batch size from config or use a safe default
        from src.core.config import get_config

        config = get_config()
        batch_size = min(config.embeddings_batch_size, 8)  # Cap at 8 for safety

        print(f"🔄 Processing in batches of {batch_size}")

        all_embeddings = []

        # Process in smaller chunks with progress reporting
        for i in range(0, total_docs, batch_size):
            batch_end = min(i + batch_size, total_docs)
            batch_texts = texts[i:batch_end]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs - 1) // batch_size + 1

            print(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch_texts)} documents)")

            try:
                # Generate embeddings for this batch
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)

                print(f"✅ Batch {batch_num} completed ({len(all_embeddings)}/{total_docs} total)")

            except Exception as e:
                print(f"❌ Error in batch {batch_num}: {str(e)}")
                print(f"🔍 Batch size: {len(batch_texts)}")
                print(f"🔍 First text preview: {batch_texts[0][:100] if batch_texts else 'Empty'}...")
                raise

        print(f"🎉 Successfully generated {len(all_embeddings)} embeddings!")
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query using centralized embeddings.

        Args:
            query: Query text to embed

        Returns:
            List[float]: Query embedding
        """
        return self.embeddings.embed_query(query)

    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension.

        Returns:
            int: Embedding dimension
        """
        return self._get_dimension()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model from centralized manager.

        Returns:
            Dict: Model information
        """
        embeddings_info = EmbeddingsManager.get_model_info()
        return {
            "status": "using_centralized_embeddings",
            "embeddings_info": embeddings_info,
            "dimension": self.get_embedding_dimension(),
        }
