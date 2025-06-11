"""
FAISS vector store implementation.
Refactored to work with centralized configuration.
"""

from typing import List, Dict, Any
from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.vector_store.base_store import BaseVectorStore
from src.core.interfaces import Document
from src.core.exceptions import VectorStoreError
from src.core.embeddings_manager import EmbeddingsManager


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store implementation with centralized configuration."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FAISS vector store.

        Args:
            config: Configuration dictionary containing persist_directory and other settings
        """
        self.config = config
        self.persist_directory = Path(config.get("persist_directory", "./data/vector_db"))
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Use centralized embeddings manager
        self.embeddings = EmbeddingsManager.get_embeddings()
        self._store = None

        print(f"✅ FAISSVectorStore initialized with persist_directory: {self.persist_directory}")

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        try:
            # Convert our Document format to LangChain format
            langchain_docs = []
            for doc in documents:
                from langchain_core.documents import Document as LangChainDocument

                langchain_docs.append(LangChainDocument(page_content=doc.page_content, metadata=doc.metadata))

            if self._store is None:
                # Create new store
                self._store = FAISS.from_documents(langchain_docs, self.embeddings)
            else:
                # Add to existing store
                self._store.add_documents(langchain_docs)

        except Exception as e:
            raise VectorStoreError(f"Failed to add documents: {str(e)}")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search."""
        if self._store is None:
            raise VectorStoreError("Vector store not initialized")

        try:
            results = self._store.similarity_search(query, k=k)

            # Convert back to our Document format
            documents = []
            for result in results:
                documents.append(Document(page_content=result.page_content, metadata=result.metadata))

            return documents

        except Exception as e:
            raise VectorStoreError(f"Similarity search failed: {str(e)}")

    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the vector store."""
        # FAISS doesn't support deletion directly
        # This would require rebuilding the index
        raise NotImplementedError("FAISS doesn't support document deletion")

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if self._store is None:
            return {"status": "not_initialized", "document_count": 0}

        try:
            # Get basic stats
            return {
                "status": "initialized",
                "store_type": "FAISS",
                "embeddings_model": self.config.embeddings_model,
                "persist_directory": str(self.persist_directory),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def save_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """Save documents with their embeddings to the vector store."""
        try:
            # DEBUG: Log input parameters
            print(f"🔍 save_documents called with {len(documents)} documents and {len(embeddings)} embeddings")

            if not documents:
                print("⚠️ No documents to save, skipping...")
                return

            if not embeddings:
                print("⚠️ No embeddings provided, skipping...")
                return

            # DEBUG: Check first document structure
            print(f"🔍 First document keys: {list(documents[0].keys()) if documents else 'No documents'}")

            # Convert dict documents back to our Document format
            doc_objects = []
            for i, doc_dict in enumerate(documents):
                try:
                    # Check if document has content or page_content
                    content = doc_dict.get("page_content") or doc_dict.get("content", "")
                    if not content:
                        print(f"⚠️ Document {i} has no content, skipping...")
                        continue

                    doc_objects.append(Document(page_content=content, metadata=doc_dict.get("metadata", {})))
                except Exception as e:
                    print(f"❌ Error processing document {i}: {str(e)}")
                    continue

            print(f"🔍 Converted {len(doc_objects)} documents to Document objects")

            if not doc_objects:
                print("⚠️ No valid documents after conversion, skipping...")
                return

            # Add documents to the store
            self.add_documents(doc_objects)

            # Persist to disk
            self.persist()

        except Exception as e:
            print(f"❌ save_documents error: {str(e)}")
            print(f"🔍 Error type: {type(e)}")
            import traceback

            print(f"🔍 Traceback: {traceback.format_exc()}")
            raise VectorStoreError(f"Failed to save documents: {str(e)}")

    def persist(self) -> None:
        """Persist the vector store to disk."""
        if self._store is None:
            raise VectorStoreError("No store to persist")

        try:
            self._store.save_local(str(self.persist_directory))
        except Exception as e:
            raise VectorStoreError(f"Failed to persist store: {str(e)}")

    def load(self):
        """Load the vector store from disk and return self for chaining."""
        try:
            index_file = self.persist_directory / "index.faiss"
            if index_file.exists():
                self._store = FAISS.load_local(
                    str(self.persist_directory), self.embeddings, allow_dangerous_deserialization=True
                )
                return self
            else:
                raise VectorStoreError("No persisted store found")

        except Exception as e:
            raise VectorStoreError(f"Failed to load store: {str(e)}")

    def as_retriever(self, **kwargs):
        """Get a retriever interface for this store."""
        if self._store is None:
            raise VectorStoreError("Vector store not initialized")

        return self._store.as_retriever(**kwargs)
