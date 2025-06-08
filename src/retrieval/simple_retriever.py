"""
Simple retriever implementation - current FAISS-based approach.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import time

from langchain_huggingface import HuggingFaceEmbeddings

from core.interfaces import RetrievalResult, Document
from core.exceptions import RetrievalError, ConfigurationError
from retrieval.base_retriever import BaseRetriever


class SimpleRetriever(BaseRetriever):
    """
    Simple FAISS-based retriever implementation.
    This is the current retrieval approach used in HelpDesk.
    """

    def __init__(self, embeddings: Optional[HuggingFaceEmbeddings] = None, **kwargs):
        """
        Initialize simple retriever.

        Args:
            embeddings: HuggingFace embeddings model
            **kwargs: Additional configuration parameters
        """
        self.embeddings = embeddings or self._get_default_embeddings()
        self.search_kwargs = kwargs.get(
            "search_kwargs",
            {
                "k": 3,
                "fetch_k": 5,
            },
        )
        self._db = None
        self._retriever = None

    def _get_default_embeddings(self) -> HuggingFaceEmbeddings:
        """Get default embeddings configuration."""
        if HuggingFaceEmbeddings is None:
            raise ConfigurationError("langchain_huggingface not available")

        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={
                "device": "cpu",
                "trust_remote_code": False,
            },
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
        )

    def _initialize_db(self, force_rebuild: bool = False):
        """Initialize the vector database using new architecture only."""
        try:
            success = self._initialize_db_new_architecture(force_rebuild)
            if not success:
                raise RetrievalError("Failed to initialize new architecture")
        except Exception as e:
            raise RetrievalError(f"Failed to initialize FAISS database: {str(e)}")

    def _initialize_db_new_architecture(self, force_rebuild: bool = False) -> bool:
        """Initialize using new data pipeline architecture."""
        try:
            # Import new architecture components
            from src.data_pipeline.pipeline_manager import PipelineManager
            from src.data_pipeline.extractors.confluence_extractor import ConfluenceExtractor
            from src.data_pipeline.processors.document_filter import DocumentFilter
            from src.data_pipeline.processors.chunker import DocumentChunker
            from src.data_pipeline.processors.post_processor import PostProcessor
            from src.data_pipeline.embeddings.huggingface_embedder import HuggingFaceEmbedder
            from src.vector_store.faiss_store import FAISSVectorStore
            from src.core.config import get_config

            config = get_config()

            # Check if DB already exists and we don't need to rebuild
            persist_path = Path(config.persist_directory)
            index_file = persist_path / "index.faiss"

            if not force_rebuild and persist_path.exists() and index_file.exists():
                print("✅ Using existing vector database")
                # Load existing vector store
                # Convert config object to dict for FAISSVectorStore
                config_dict = {
                    "persist_directory": config.persist_directory,
                    "confluence_url": config.confluence_space_name
                    if config.confluence_space_name.startswith("http")
                    else f"https://{config.confluence_space_name}.atlassian.net",
                    "confluence_email": config.confluence_email_address,
                    "confluence_api_key": config.confluence_private_api_key,
                }

                vector_store = FAISSVectorStore(config_dict)

                try:
                    loaded_store = vector_store.load()  # This loads the store internally and returns self
                except Exception as load_error:
                    print(f"❌ ERROR: Failed to load vector store: {load_error}")
                    return False

                if loaded_store is None:
                    print("❌ ERROR: loaded_store is None, cannot create retriever")
                    return False

                try:
                    self._retriever = loaded_store.as_retriever(search_kwargs=self.search_kwargs)
                    # IMPORTANT: Set self._db for compatibility with dynamic search_kwargs
                    self._db = loaded_store._store
                except Exception as retriever_error:
                    print(f"❌ ERROR: Failed to create retriever: {retriever_error}")
                    return False

                return True

            print("🚀 Building vector database using new architecture...")

            # Create pipeline components
            confluence_config = {
                "confluence_url": config.confluence_space_name
                if config.confluence_space_name.startswith("http")
                else f"https://{config.confluence_space_name}.atlassian.net",
                "confluence_email_address": config.confluence_email_address,
                "confluence_private_api_key": config.confluence_private_api_key,
                "confluence_space_key": getattr(config, "confluence_space_key", ""),
            }

            # Convert config object to dict for pipeline components
            config_dict = {
                "persist_directory": config.persist_directory,
                "confluence_url": config.confluence_space_name
                if config.confluence_space_name.startswith("http")
                else f"https://{config.confluence_space_name}.atlassian.net",
                "confluence_email": config.confluence_email_address,
                "confluence_api_key": config.confluence_private_api_key,
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu",
                "batch_size": 32,
            }

            # Initialize pipeline
            pipeline = PipelineManager(config_dict)
            pipeline.set_extractor(ConfluenceExtractor(confluence_config))
            pipeline.set_filter(DocumentFilter(config_dict))
            pipeline.set_chunker(DocumentChunker(config_dict))
            pipeline.set_post_processor(PostProcessor(config_dict))
            pipeline.set_embedder(HuggingFaceEmbedder(config_dict))
            pipeline.set_vector_store(FAISSVectorStore(config_dict))

            # Run the pipeline
            stats = pipeline.run_pipeline()
            print(f"✅ Pipeline completed successfully: {stats}")

            # Load the created vector store
            vector_store = FAISSVectorStore(config_dict)

            try:
                loaded_store = vector_store.load()  # This loads the store internally and returns self
            except Exception as load_error:
                print(f"❌ ERROR: Failed to load new vector store: {load_error}")
                return False

            if loaded_store is None:
                print("❌ ERROR: loaded_store is None after pipeline, cannot create retriever")
                return False

            try:
                self._retriever = loaded_store.as_retriever(search_kwargs=self.search_kwargs)
                # IMPORTANT: Set self._db for compatibility with dynamic search_kwargs
                self._db = loaded_store._store
            except Exception as retriever_error:
                print(f"❌ ERROR: Failed to create new retriever: {retriever_error}")
                return False

            return True

        except Exception as e:
            print(f"❌ New architecture failed: {str(e)}")
            return False

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            RetrievalResult with documents and metadata
        """
        if self._retriever is None:
            self._initialize_db()

        try:
            start_time = time.time()

            # Update search kwargs if top_k is different
            if top_k != self.search_kwargs.get("k", 3):
                search_kwargs = self.search_kwargs.copy()
                search_kwargs["k"] = top_k
                search_kwargs["fetch_k"] = max(top_k + 2, search_kwargs.get("fetch_k", 5))
                # FIXED: Use self._db if available, otherwise use self._retriever
                if hasattr(self, "_db") and self._db is not None:
                    retriever = self._db.as_retriever(search_kwargs=search_kwargs)
                else:
                    # Fallback to default retriever if _db is not available
                    retriever = self._retriever
            else:
                retriever = self._retriever

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
        if self._db is None:
            return {"status": "not_initialized"}

        try:
            # Try to get some basic stats
            return {
                "status": "initialized",
                "retriever_type": "Simple FAISS",
                "search_kwargs": self.search_kwargs,
                "embeddings_model": "all-MiniLM-L6-v2",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
