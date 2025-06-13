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
        from src.core.config import get_config

        self.config = config
        self.app_config = get_config()
        self.persist_directory = Path(self.app_config.persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Use centralized embeddings manager
        self.embeddings = EmbeddingsManager.get_embeddings()
        self._store = None
        self._is_loaded = False  # Flag to prevent multiple loads

        print(f"✅ FAISSVectorStore initialized with persist_directory: {self.persist_directory}")

    def ensure_loaded(self) -> bool:
        """Ensure the vector store is loaded only once and handle rebuilds"""
        if self._is_loaded:
            return True

        try:
            # Check if rebuild is required
            if self.app_config.should_rebuild_vector_db():
                print("🔄 Vector DB rebuild required due to configuration changes")
                return self._trigger_rebuild()

            # Try to load existing DB
            if self._try_load_existing():
                self._is_loaded = True
                print("✅ Vector DB loaded from existing storage")
                return True

            # No existing DB found, trigger rebuild
            print("🆕 No existing vector DB found, rebuild required")
            return self._trigger_rebuild()

        except Exception as e:
            print(f"❌ Error ensuring vector DB is loaded: {e}")
            return False

    def _try_load_existing(self) -> bool:
        """Try to load existing vector database"""
        try:
            if self.app_config.is_using_azure_storage():
                return self._load_from_azure()
            else:
                return self._load_from_local()
        except Exception as e:
            print(f"⚠️ Could not load existing DB: {e}")
            return False

    def _load_from_local(self) -> bool:
        """Load vector DB from local storage"""
        index_file = self.persist_directory / "index.faiss"
        if index_file.exists():
            self._store = FAISS.load_local(
                str(self.persist_directory), self.embeddings, allow_dangerous_deserialization=True
            )
            return True
        return False

    def _load_from_azure(self) -> bool:
        """Load vector DB from Azure storage"""
        vector_db_path = self.app_config.get_vector_db_path()
        index_file_path = f"{vector_db_path}/index.faiss"

        if self.app_config.file_exists(index_file_path):
            # Download necessary files
            files_to_download = ["index.faiss", "index.pkl"]

            for filename in files_to_download:
                azure_file_path = f"{vector_db_path}/{filename}"
                local_file_path = self.persist_directory / filename

                if self.app_config.file_exists(azure_file_path):
                    file_data = self.app_config.read_file(azure_file_path)
                    with open(local_file_path, "wb") as f:
                        f.write(file_data)

            # Load from downloaded files
            return self._load_from_local()

        return False

    def _trigger_rebuild(self) -> bool:
        """Trigger vector database rebuild"""
        print("🚧 Triggering vector database rebuild...")

        # Clean up old DB
        self._cleanup_old_db()

        # Reset state
        self._store = None
        self._is_loaded = False

        # Try to trigger automatic rebuild if pipeline manager is available
        try:
            from src.data_pipeline.pipeline_manager import PipelineManager

            pipeline_manager = PipelineManager()

            print("🔄 Starting automatic rebuild...")
            success = pipeline_manager.run_full_pipeline()

            if success:
                # Mark DB as built with current version
                self.app_config.save_db_version()
                # Try to reload
                if self._try_load_existing():
                    self._is_loaded = True
                    print("✅ Rebuild completed and DB reloaded")
                    return True

        except ImportError:
            print("⚠️ PipelineManager not available, manual rebuild required")
        except Exception as e:
            print(f"❌ Error during automatic rebuild: {e}")

        return False

    def _cleanup_old_db(self):
        """Clean up old vector database files"""
        try:
            if self.app_config.is_using_azure_storage():
                # For Azure, delete files via abstraction
                vector_db_path = self.app_config.get_vector_db_path()
                files_to_delete = ["index.faiss", "index.pkl", "db_version.txt"]

                for filename in files_to_delete:
                    file_path = f"{vector_db_path}/{filename}"
                    if self.app_config.file_exists(file_path):
                        self.app_config.delete_file(file_path)
            else:
                # For local, delete files from directory
                for file_path in self.persist_directory.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()

            print("🗑️ Old vector database cleaned up")

        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")

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
        """Perform similarity search with automatic loading"""
        # Ensure the vector store is loaded
        if not self.ensure_loaded():
            raise VectorStoreError("Could not load vector database")

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

    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """Alias for similarity_search to match test expectations."""
        return self.similarity_search(query, k)

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
            # Get document count from FAISS index
            document_count = 0
            if hasattr(self._store, "index") and hasattr(self._store.index, "ntotal"):
                document_count = self._store.index.ntotal

            # Get embeddings model info
            embeddings_info = EmbeddingsManager.get_model_info()

            return {
                "status": "initialized",
                "store_type": "FAISS",
                "document_count": document_count,
                "embeddings_model": embeddings_info.get("model_name", "unknown"),
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
        """Persist the vector store to disk and Azure if configured"""
        if self._store is None:
            raise VectorStoreError("No store to persist")

        try:
            # Always save locally first
            self._store.save_local(str(self.persist_directory))

            # If using Azure storage, upload to Azure
            if self.app_config.is_using_azure_storage():
                self._upload_to_azure()

            # Save DB version to mark as current
            self.app_config.save_db_version()

        except Exception as e:
            raise VectorStoreError(f"Failed to persist store: {str(e)}")

    def _upload_to_azure(self):
        """Upload vector database files to Azure storage"""
        try:
            vector_db_path = self.app_config.get_vector_db_path()

            # Upload all files in the persist directory
            for file_path in self.persist_directory.glob("*"):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        file_data = f.read()

                    azure_file_path = f"{vector_db_path}/{file_path.name}"
                    self.app_config.write_file(azure_file_path, file_data)

            print(f"📤 Vector database uploaded to Azure: {vector_db_path}")

        except Exception as e:
            print(f"❌ Error uploading to Azure: {e}")
            raise

    def load(self):
        """Load the vector store from disk and return self for chaining."""
        try:
            if self.ensure_loaded():
                return self
            else:
                raise VectorStoreError("Could not load vector store")
        except Exception as e:
            raise VectorStoreError(f"Failed to load store: {str(e)}")

    def as_retriever(self, **kwargs):
        """Get a retriever interface for this store."""
        if self._store is None:
            raise VectorStoreError("Vector store not initialized")

        return self._store.as_retriever(**kwargs)
