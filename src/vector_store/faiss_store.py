"""
FAISS vector store implementation.
Refactored to work with centralized configuration.
"""

from typing import List, Dict, Any
from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.vector_store.base_store import BaseVectorStore
from src.core.interfaces import Document
from src.core.exceptions import VectorStoreError, StorageAccessError
from src.core.embeddings_manager import EmbeddingsManager


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store implementation with centralized configuration."""

    def __init__(self, config: Dict[str, Any], storage_service=None):
        """
        Initialize FAISS vector store with dependency injection.

        Args:
            config: Configuration dictionary containing persist_directory and other settings
            storage_service: Optional storage service for file operations
        """
        from src.core.config import get_config, _ensure_config_initialized

        self.config = config
        self.app_config = get_config()

        # Get storage service from config if not provided
        if storage_service is None:
            config_manager = _ensure_config_initialized()
            self.storage_service = config_manager.get_storage_service()
        else:
            self.storage_service = storage_service

        # Use relative path for storage service
        self.persist_directory_path = "vector_db"

        # Determine storage type
        self.storage_type = type(self.storage_service._storage).__name__

        # Create directory using storage service
        if hasattr(self.storage_service._storage, "create_directory"):
            self.storage_service._storage.create_directory(self.persist_directory_path)

        # Only create local directory if using LocalStorage
        if self.storage_type == "LocalStorage":
            self.persist_directory = Path(self.app_config.persist_directory)
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ FAISSVectorStore initialized with LOCAL storage: {self.persist_directory}")
        else:
            # For Azure storage, we don't need local directory
            self.persist_directory = None
            print(f"✅ FAISSVectorStore initialized with AZURE storage: {self.persist_directory_path}")

        # Use centralized embeddings manager
        self.embeddings = EmbeddingsManager.get_embeddings()
        self._store = None
        self._is_loaded = False  # Flag to prevent multiple loads

    def ensure_loaded(self) -> bool:
        """Ensure the vector store is loaded only once"""
        if self._is_loaded:
            return True

        try:
            # Try to load existing DB
            if self._try_load_existing():
                self._is_loaded = True
                print("✅ Vector DB loaded from existing storage")
                return True

            # No existing DB found
            print("🆕 No existing vector DB found")
            return False

        except Exception as e:
            print(f"❌ Error ensuring vector DB is loaded: {e}")
            return False

    def _try_load_existing(self) -> bool:
        """Try to load existing vector database based on storage type"""
        from src.core.exceptions import StorageAccessError

        try:
            # Check storage type to determine loading strategy
            storage_type = type(self.storage_service._storage).__name__

            if storage_type == "LocalStorage":
                return self._load_from_local()
            elif storage_type == "AzureStorage":
                return self._load_from_azure()
            else:
                print(f"⚠️ Unknown storage type: {storage_type}, falling back to local")
                return self._load_from_local()

        except StorageAccessError:
            # Re-raise storage access errors instead of returning False
            raise
        except Exception as e:
            print(f"⚠️ Could not load existing DB: {e}")
            return False

    def _load_from_local(self) -> bool:
        """Load vector database from local storage"""
        if self.persist_directory is None:
            raise VectorStoreError("Local storage not configured - using Azure storage instead")

        try:
            index_file = self.persist_directory / "index.faiss"
            if index_file.exists():
                self._store = FAISS.load_local(
                    str(self.persist_directory), self.embeddings, allow_dangerous_deserialization=True
                )
                print("✅ Loaded vector DB from local storage")
                return True
            return False
        except Exception as e:
            print(f"⚠️ Error loading from local storage: {e}")
            return False

    def _load_from_azure(self) -> bool:
        """Load vector database from Azure blob storage"""
        from src.core.exceptions import StorageAccessError

        try:
            # Check if index file exists in blob storage
            index_file_path = f"{self.persist_directory_path}/index.faiss"

            # Test Azure access first
            try:
                file_exists = self.storage_service.file_exists(index_file_path)
            except Exception as e:
                # If we can't even check if files exist, it's an access problem
                raise StorageAccessError(
                    f"Cannot access Azure storage to check for vector database: {str(e)}",
                    storage_type="Azure",
                    original_error=e,
                )

            if file_exists:
                # Download FAISS files to local temp directory
                import tempfile
                import os

                with tempfile.TemporaryDirectory() as temp_dir:
                    # Download FAISS files to temp directory
                    faiss_files = ["index.faiss", "index.pkl"]

                    for file_name in faiss_files:
                        remote_path = f"{self.persist_directory_path}/{file_name}"
                        if self.storage_service.file_exists(remote_path):
                            try:
                                file_data = self.storage_service.read_binary_file(remote_path)
                                if file_data:
                                    local_file_path = os.path.join(temp_dir, file_name)
                                    with open(local_file_path, "wb") as f:
                                        f.write(file_data)
                            except Exception as e:
                                # If we can't read files, it's an access problem
                                raise StorageAccessError(
                                    f"Cannot read vector database files from Azure storage: {str(e)}",
                                    storage_type="Azure",
                                    original_error=e,
                                )

                    # Load FAISS from temp directory
                    self._store = FAISS.load_local(temp_dir, self.embeddings, allow_dangerous_deserialization=True)
                    print("✅ Loaded vector DB from Azure blob storage")
                    return True
            return False
        except StorageAccessError:
            # Re-raise storage access errors
            raise
        except Exception as e:
            print(f"⚠️ Error loading from Azure storage: {e}")
            return False

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
        """Persist the vector store based on storage type"""
        if self._store is None:
            raise VectorStoreError("No store to persist")

        try:
            # Check storage type to determine saving strategy
            storage_type = type(self.storage_service._storage).__name__

            if storage_type == "LocalStorage":
                self._persist_to_local()
            elif storage_type == "AzureStorage":
                # Validate Azure access before persisting
                self._validate_azure_access()
                self._persist_to_azure()
            else:
                print(f"⚠️ Unknown storage type: {storage_type}, falling back to local")
                self._persist_to_local()

        except StorageAccessError:
            # Re-raise storage access errors with clear context
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to persist store: {str(e)}")

    def _persist_to_local(self) -> None:
        """Persist vector store to local storage"""
        if self.persist_directory is None:
            raise VectorStoreError("Local storage not configured - using Azure storage instead")

        self._store.save_local(str(self.persist_directory))
        print(f"✅ Vector database persisted to local storage: {self.persist_directory}")

    def _validate_azure_access(self) -> None:
        """Validate Azure storage access before operations"""
        try:
            # Test write access by creating a small test file
            test_path = f"{self.persist_directory_path}/.access_test"
            test_data = b"access_test"

            if not self.storage_service.write_binary_file(test_path, test_data):
                raise StorageAccessError(
                    "❌ IMPOSSIBLE D'ÉCRIRE SUR AZURE STORAGE: Test d'écriture échoué.\n"
                    "Vous avez demandé d'utiliser Azure Storage mais l'écriture est impossible.\n"
                    "Vérifiez vos permissions Azure.",
                    storage_type="azure",
                )

            # Clean up test file
            self.storage_service.delete_file(test_path)
            print("✅ Azure Storage access validation successful")

        except StorageAccessError:
            raise
        except Exception as e:
            raise StorageAccessError(
                f"❌ IMPOSSIBLE D'ACCÉDER AU STORAGE AZURE: {str(e)}\n"
                "Vous avez demandé d'utiliser Azure Storage mais l'accès est impossible.\n"
                "Vérifiez vos permissions et la configuration Azure.",
                storage_type="azure",
                original_error=e,
            )

    def _persist_to_azure(self) -> None:
        """Persist vector store to Azure blob storage"""
        import tempfile
        import os

        try:
            # Save to temporary directory first
            with tempfile.TemporaryDirectory() as temp_dir:
                self._store.save_local(temp_dir)

                # Upload FAISS files to blob storage
                faiss_files = ["index.faiss", "index.pkl"]

                for file_name in faiss_files:
                    local_file_path = os.path.join(temp_dir, file_name)
                    if os.path.exists(local_file_path):
                        with open(local_file_path, "rb") as f:
                            file_data = f.read()

                        remote_path = f"{self.persist_directory_path}/{file_name}"
                        success = self.storage_service.write_binary_file(remote_path, file_data)

                        if not success:
                            raise StorageAccessError(
                                f"❌ ÉCHEC D'ÉCRITURE SUR AZURE: Impossible d'écrire {file_name}\n"
                                "Vous avez demandé d'utiliser Azure Storage mais l'écriture a échoué.\n"
                                "Vérifiez vos permissions Azure.",
                                storage_type="azure",
                            )

                print(f"✅ Vector database persisted to Azure blob storage: {self.persist_directory_path}")

        except StorageAccessError:
            raise
        except Exception as e:
            raise StorageAccessError(
                f"❌ ERREUR LORS DE LA PERSISTANCE AZURE: {str(e)}\n"
                "Vous avez demandé d'utiliser Azure Storage mais la persistance a échoué.\n"
                "Vérifiez vos permissions et la configuration Azure.",
                storage_type="azure",
                original_error=e,
            )

    def load(self):
        """Load the vector store from disk and return self for chaining."""
        try:
            if self.ensure_loaded():
                return self
            else:
                raise VectorStoreError("Could not load vector store")
        except StorageAccessError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to load store: {str(e)}")

    def rebuild_database(self, force_azure: bool = False) -> bool:
        """
        Rebuild the vector database with proper storage validation

        Args:
            force_azure: Force using Azure storage even if not configured

        Returns:
            True if rebuild successful, False otherwise

        Raises:
            StorageAccessError: If requested storage is not accessible
            RebuildError: If rebuild fails due to storage issues
        """
        from ..core.exceptions import RebuildError

        try:
            # Determine target storage type
            storage_type = type(self.storage_service._storage).__name__

            # If force_azure is requested but we're not using Azure, raise error
            if force_azure and storage_type != "AzureStorage":
                raise RebuildError(
                    "❌ REBUILD AZURE DEMANDÉ MAIS STORAGE LOCAL CONFIGURÉ:\n"
                    "Vous avez demandé un rebuild avec Azure Storage mais le système est configuré "
                    "pour le stockage local.\n"
                    "Configurez USE_AZURE_STORAGE=true et AZURE_STORAGE_ACCOUNT pour utiliser Azure.",
                    storage_type="local",
                    requested_storage="azure",
                )

            # If using Azure, validate access before rebuild
            if storage_type == "AzureStorage":
                print("🔍 Validation de l'accès Azure Storage avant rebuild...")
                self._validate_azure_access()
                print("✅ Accès Azure Storage validé")

                # Clear any existing local data to prevent mixed storage
                self._clear_local_storage()
                print("🧹 Stockage local nettoyé pour éviter les conflits")

            # Proceed with rebuild
            print(f"🔄 Début du rebuild de la base vectorielle ({storage_type})...")

            # Reset the store
            self._store = None
            self._is_loaded = False

            # Trigger rebuild through the database manager
            from ..data_pipeline.integrated_data_ingestion_pipeline import DataIngestionPipeline

            pipeline = DataIngestionPipeline()

            success = pipeline.build_database()

            if success:
                print(f"✅ Rebuild terminé avec succès ({storage_type})")
                return True
            else:
                raise RebuildError(
                    f"❌ ÉCHEC DU REBUILD: La construction de la base de données a échoué.\n"
                    f"Storage utilisé: {storage_type}",
                    storage_type=storage_type.lower(),
                )

        except StorageAccessError:
            # Re-raise storage access errors
            raise
        except RebuildError:
            # Re-raise rebuild errors
            raise
        except Exception as e:
            raise RebuildError(
                f"❌ ERREUR INATTENDUE LORS DU REBUILD: {str(e)}\nStorage type: {storage_type}",
                storage_type=storage_type.lower() if "storage_type" in locals() else "unknown",
            )

    def _clear_local_storage(self) -> None:
        """Clear local storage to prevent conflicts when using Azure"""
        try:
            import shutil

            if self.persist_directory and self.persist_directory.exists():
                shutil.rmtree(self.persist_directory)
                print(f"🧹 Cleared local storage: {self.persist_directory}")
            else:
                print("🧹 No local storage to clear (using Azure storage)")
        except Exception as e:
            print(f"⚠️ Warning: Could not clear local storage: {e}")

    def as_retriever(self, **kwargs):
        """Get a retriever interface for this store."""
        if self._store is None:
            raise VectorStoreError("Vector store not initialized")

        return self._store.as_retriever(**kwargs)
