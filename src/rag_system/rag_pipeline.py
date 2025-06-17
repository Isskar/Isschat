"""
Main RAG pipeline orchestrating retrieval and generation.
Refactored to use centralized configuration and database management.
"""

from typing import Dict, Any, Tuple, Optional

from src.core.config import get_config
from src.data_pipeline.offline_db_manager import OfflineDatabaseManager
from src.core.embeddings_manager import EmbeddingsManager
from src.core.exceptions import ConfigurationError, StorageAccessError
from src.rag_system.query_processor import QueryProcessor


class RAGPipeline:
    """
    Main RAG pipeline that orchestrates retrieval and generation.
    Now handles database verification and initialization automatically.
    """

    def __init__(self, config=None, force_rebuild: bool = False):
        """
        Initialize RAG pipeline with automatic database verification.

        Args:
            config: Optional configuration (uses get_config() by default)
            force_rebuild: Force database rebuild
        """
        from src.generation.openrouter_generator import OpenRouterGenerator

        self.config = config or get_config()
        self.query_processor = QueryProcessor()

        # 1. Ensure database exists
        print("🔍 Checking vector database...")
        self.db_manager = OfflineDatabaseManager(self.config)
        try:
            # Check if database exists - this will raise StorageAccessError if storage is inaccessible
            if not self.db_manager.database_exists() and not force_rebuild:
                print("⚠️  Vector database not found. Building database...")
                force_rebuild = True

            if not self.db_manager.ensure_database(force_rebuild):
                raise ConfigurationError("Failed to initialize vector database")

        except StorageAccessError as e:
            # If we can't access storage, we can't proceed
            raise ConfigurationError(
                f"Cannot access storage for vector database: {e}. "
                f"Please check your storage configuration and credentials."
            ) from e

        # 2. Load vector store
        print("📚 Loading vector store...")
        try:
            self.vector_store = self.db_manager.load_vector_store()
            if not self.vector_store:
                # If loading fails, try to rebuild database
                print("⚠️  Failed to load vector store. Attempting to rebuild...")
                if self.db_manager.build_database():
                    self.vector_store = self.db_manager.load_vector_store()
                    if not self.vector_store:
                        raise ConfigurationError("Failed to load vector store after rebuild")
                else:
                    raise ConfigurationError("Failed to build and load vector store")
        except StorageAccessError as e:
            # If we can't access storage, we can't proceed
            raise ConfigurationError(
                f"Cannot access storage for vector database: {e}. "
                f"Please check your storage configuration and credentials."
            ) from e

        # 3. Create retriever with centralized configuration
        search_kwargs = {"k": self.config.search_k, "fetch_k": self.config.search_fetch_k}
        self.retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)

        # 4. Create generator
        self.generator = OpenRouterGenerator(
            model_name=self.config.generator_model_name,
            temperature=self.config.generator_temperature,
            max_tokens=self.config.generator_max_tokens,
        )

        print("✅ RAG Pipeline initialized successfully")

    def process_query(self, query: str, top_k: Optional[int] = None, verbose: bool = True) -> Tuple[str, str]:
        """
        Process a query through the RAG pipeline.

        Args:
            query: User query
            top_k: Number of documents to retrieve (uses config default if None)
            verbose: Whether to print verbose output

        Returns:
            Tuple of (answer, sources)
        """
        try:
            if verbose:
                print(f"\n🔍 Processing query: '{query}'")
                print("=" * 50)

            # Step 0: Process and analyze the query
            query_analysis = self.query_processor.process_query(query)

            # Step 1: Determine retriever to use based on top_k
            search_query = query_analysis.processed_query if query_analysis.processed_query.strip() else query

            # Use custom top_k if provided, otherwise use default retriever
            if top_k and top_k != self.config.search_k:
                search_kwargs = {"k": top_k, "fetch_k": max(top_k + 2, self.config.search_fetch_k)}
                retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
                docs = retriever.invoke(search_query)
            else:
                docs = self.retriever.invoke(search_query)

            # Convert to our format for compatibility
            from src.core.interfaces import RetrievalResult, Document

            documents = []
            scores = []
            for doc in docs:
                documents.append(Document(page_content=doc.page_content, metadata=doc.metadata))
                scores.append(getattr(doc, "score", 0.0))

            retrieval_result = RetrievalResult(
                documents=documents,
                scores=scores,
                query=search_query,
                retrieval_time=0.0,  # We don't track time here anymore
            )

            if verbose:
                print(f"\n📚 Retrieved {len(retrieval_result.documents)} documents:")
                for i, doc in enumerate(retrieval_result.documents):
                    print(f"  Document {i + 1}:")
                    print(f"    Title: {doc.metadata.get('title', 'N/A')}")
                    print(f"    Source: {doc.metadata.get('source', 'N/A')}")
                    print(f"    Content: {doc.page_content[:150]}...")
                    if i < len(retrieval_result.scores):
                        print(f"    Score: {retrieval_result.scores[i]:.4f}")
                    print()

            # Step 2: Generate answer using original query for context
            generation_result = self.generator.generate(query, retrieval_result)

            if verbose:
                print(f"⚡ Generation completed in {generation_result.generation_time:.2f}s")
                print(f"📝 Token count: {generation_result.token_count}")
                print(f"\n📖 Sources:\n{generation_result.sources}")
                print("=" * 50)

            return generation_result.answer, generation_result.sources

        except Exception as e:
            error_msg = f"RAG pipeline failed: {str(e)}"
            if verbose:
                print(f"❌ Error: {error_msg}")
            return (
                f"Désolé, une erreur s'est produite lors du traitement de votre question: {error_msg}",
                "Aucune source disponible",
            )

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return {
            "pipeline_type": "Modern RAG with Centralized Configuration",
            "database_info": self.db_manager.get_database_info(),
            "embeddings_info": EmbeddingsManager.get_model_info(),
            "generator_stats": self.generator.get_stats(),
            "query_processor": "Enabled",
            "config": {
                "search_k": self.config.search_k,
                "search_fetch_k": self.config.search_fetch_k,
                "embeddings_model": self.config.embeddings_model,
                "embeddings_device": self.config.embeddings_device,
            },
        }


class RAGPipelineFactory:
    """Factory for creating RAG pipelines with centralized configuration."""

    @staticmethod
    def create_default_pipeline(force_rebuild: bool = False) -> RAGPipeline:
        """
        Create a default RAG pipeline with automatic database management.

        Args:
            force_rebuild: Force database rebuild

        Returns:
            Configured RAGPipeline
        """
        return RAGPipeline(force_rebuild=force_rebuild)

    @staticmethod
    def create_pipeline(config=None, force_rebuild: bool = False, **kwargs) -> RAGPipeline:
        """
        Create a RAG pipeline with custom configuration.

        Args:
            config: Custom configuration (optional)
            force_rebuild: Force database rebuild
            **kwargs: Additional configuration (deprecated, use config instead)

        Returns:
            Configured RAGPipeline
        """
        if kwargs:
            print("⚠️  Warning: **kwargs is deprecated, use config parameter instead")

        return RAGPipeline(config=config, force_rebuild=force_rebuild)

    @staticmethod
    def create_pipeline_with_custom_embeddings(embeddings_model: str, force_rebuild: bool = False) -> RAGPipeline:
        """
        Create a RAG pipeline with custom embeddings model.

        Args:
            embeddings_model: Custom embeddings model name
            force_rebuild: Force database rebuild

        Returns:
            Configured RAGPipeline
        """
        from src.core.config import get_config
        from dataclasses import replace

        # Get default config and override embeddings model
        base_config = get_config()
        custom_config = replace(base_config, embeddings_model=embeddings_model)

        # Reset embeddings manager to use new model
        EmbeddingsManager.reset()

        return RAGPipeline(config=custom_config, force_rebuild=force_rebuild)
