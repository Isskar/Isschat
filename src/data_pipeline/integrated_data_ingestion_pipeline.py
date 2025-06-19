"""
Centralized vector database manager.
Handles database existence, building, and loading.
"""

from pathlib import Path
from src.core.config import get_config
from src.core.exceptions import StorageAccessError


class DataIngestionPipeline:
    """Data ingestion pipeline for vector database building and management"""

    def __init__(self, config=None, storage_service=None):
        """
        Initialize database manager.

        Args:
            config: Optional configuration (uses get_config() by default)
            storage_service: Optional storage service for file operations
        """
        self.config = config or get_config()
        self.persist_path = Path(self.config.persist_directory)
        self.index_file = self.persist_path / "index.faiss"
        # Get storage service from config if not provided
        if storage_service is None:
            from src.core.config import _ensure_config_initialized

            config_manager = _ensure_config_initialized()
            self.storage_service = config_manager.get_storage_service()
        else:
            self.storage_service = storage_service

    def database_exists(self) -> bool:
        """
        Check if the database exists in the configured storage.

        Returns:
            True if database exists, False otherwise

        Raises:
            StorageAccessError: If storage access fails
        """
        try:
            # Check storage type to determine where to look for the database
            storage_type = type(self.storage_service._storage).__name__

            if storage_type == "AzureStorage":
                # For Azure, check if the index file exists in blob storage
                try:
                    return self.storage_service.file_exists("vector_db/index.faiss")
                except Exception as e:
                    raise StorageAccessError(
                        f"Cannot access Azure storage to check database existence: {str(e)}",
                        storage_type="Azure",
                        original_error=e,
                    )
            else:
                # For local storage, check local files
                return self.persist_path.exists() and self.index_file.exists()

        except StorageAccessError:
            # Re-raise storage access errors
            raise
        except Exception as e:
            # Any other error during existence check
            raise StorageAccessError(
                f"Failed to check database existence: {str(e)}",
                storage_type=type(self.storage_service._storage).__name__,
                original_error=e,
            )

    def build_database(self) -> bool:
        """
        Build the database using the new architecture.

        Returns:
            True if build succeeded, False otherwise
        """
        try:
            # Import new architecture components
            from src.data_pipeline.pipeline_manager import PipelineManager
            from src.data_pipeline.extractors.confluence_extractor import ConfluenceExtractor
            from src.data_pipeline.processors.document_filter import DocumentFilter
            from src.data_pipeline.processors.chunker import DocumentChunker
            from src.data_pipeline.processors.post_processor import PostProcessor
            from src.data_pipeline.embeddings.huggingface_embedder import HuggingFaceEmbedder
            from src.vector_store.faiss_store import FAISSVectorStore

            print("🚀 Building vector database using centralized configuration...")

            # Unified configuration from centralized config
            config_dict = {
                "persist_directory": self.config.persist_directory,
                "confluence_url": self._get_confluence_url(),
                "confluence_email": self.config.confluence_email_address,
                "confluence_api_key": self.config.confluence_private_api_key,
                "model_name": self.config.embeddings_model,
                "device": self.config.embeddings_device,
                "batch_size": self.config.embeddings_batch_size,
            }

            confluence_config = {
                "confluence_url": self._get_confluence_url(),
                "confluence_email_address": self.config.confluence_email_address,
                "confluence_private_api_key": self.config.confluence_private_api_key,
                "confluence_space_key": getattr(self.config, "confluence_space_key", ""),
            }

            # Initialize pipeline with centralized configuration
            pipeline = PipelineManager(config_dict)
            pipeline.set_extractor(ConfluenceExtractor(confluence_config))
            pipeline.set_filter(DocumentFilter(config_dict))
            pipeline.set_chunker(DocumentChunker(config_dict))
            pipeline.set_post_processor(PostProcessor(config_dict))
            pipeline.set_embedder(HuggingFaceEmbedder(config_dict))
            pipeline.set_vector_store(FAISSVectorStore(config_dict, storage_service=self.storage_service))

            # Run the pipeline
            stats = pipeline.run_pipeline()
            print(f"✅ Database built successfully: {stats['storage']['stored_documents']} documents stored")
            return True

        except Exception as e:
            print(f"❌ Failed to build database: {str(e)}")
            return False

    def _get_confluence_url(self) -> str:
        """
        Helper to build Confluence URL.

        Returns:
            Complete Confluence URL
        """
        if self.config.confluence_space_name.startswith("http"):
            return self.config.confluence_space_name
        return f"https://{self.config.confluence_space_name}.atlassian.net"

    def ensure_database(self, force_rebuild: bool = False) -> bool:
        """
        Ensure the database exists, build it if necessary.

        Args:
            force_rebuild: Force rebuild even if DB exists

        Returns:
            True if database is ready, False otherwise

        Raises:
            StorageAccessError: If storage access fails
        """
        try:
            if not force_rebuild and self.database_exists():
                print("✅ Using existing vector database")
                return True
        except StorageAccessError as e:
            # If we can't access storage, we can't build or load the database
            print(f"❌ Storage access failed: {e}")
            raise e

        print("🔄 Database not found or rebuild requested")
        return self.build_database()

    def load_vector_store(self):
        """
        Load existing vector store.

        Returns:
            Loaded vector store instance or None if failed

        Raises:
            StorageAccessError: If storage access fails
        """
        try:
            from src.vector_store.faiss_store import FAISSVectorStore

            config_dict = {
                "persist_directory": self.config.persist_directory,
                "confluence_url": self._get_confluence_url(),
                "confluence_email": self.config.confluence_email_address,
                "confluence_api_key": self.config.confluence_private_api_key,
            }

            vector_store = FAISSVectorStore(config_dict, storage_service=self.storage_service)
            loaded_store = vector_store.load()

            if loaded_store is None:
                print("❌ Failed to load vector store: loaded_store is None")
                return None

            print("✅ Vector store loaded successfully")
            return loaded_store

        except StorageAccessError:
            raise
        except Exception as e:
            print(f"❌ Failed to load vector store: {str(e)}")
            return None

    def get_database_info(self) -> dict:
        """
        Get information about the database state.

        Returns:
            Dictionary with database information
        """
        return {
            "exists": self.database_exists(),
            "persist_directory": str(self.persist_path),
            "index_file": str(self.index_file),
            "index_file_exists": self.index_file.exists() if self.persist_path.exists() else False,
            "embeddings_model": self.config.embeddings_model,
            "search_config": {"k": self.config.search_k, "fetch_k": self.config.search_fetch_k},
        }
