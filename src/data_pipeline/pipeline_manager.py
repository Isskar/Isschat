"""
Pipeline manager for orchestrating the offline data processing pipeline.
"""

from typing import Dict, Any, Optional
from .extractors.base_extractor import BaseExtractor
from .processors.document_filter import DocumentFilter
from .processors.chunker import DocumentChunker
from .processors.post_processor import PostProcessor
from .embeddings.base_embedder import BaseEmbedder

from src.vector_store.base_store import BaseVectorStore


class PipelineManager:
    """Orchestrates the offline data processing pipeline."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline manager.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.extractor: Optional[BaseExtractor] = None
        self.filter: Optional[DocumentFilter] = None
        self.chunker: Optional[DocumentChunker] = None
        self.post_processor: Optional[PostProcessor] = None
        self.embedder: Optional[BaseEmbedder] = None
        self.vector_store: Optional[BaseVectorStore] = None

        self.stats = {}

    def set_extractor(self, extractor: BaseExtractor) -> "PipelineManager":
        """Set the document extractor."""
        self.extractor = extractor
        return self

    def set_filter(self, document_filter: DocumentFilter) -> "PipelineManager":
        """Set the document filter."""
        self.filter = document_filter
        return self

    def set_chunker(self, chunker: DocumentChunker) -> "PipelineManager":
        """Set the document chunker."""
        self.chunker = chunker
        return self

    def set_post_processor(self, post_processor: PostProcessor) -> "PipelineManager":
        """Set the post processor."""
        self.post_processor = post_processor
        return self

    def set_embedder(self, embedder: BaseEmbedder) -> "PipelineManager":
        """Set the embedder."""
        self.embedder = embedder
        return self

    def set_vector_store(self, vector_store: BaseVectorStore) -> "PipelineManager":
        """Set the vector store."""
        self.vector_store = vector_store
        return self

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete offline pipeline.

        Returns:
            Dict: Pipeline execution statistics
        """
        if not self._validate_components():
            raise ValueError("Pipeline components not properly configured")

        # Step 1: Extract documents
        print("Step 1: Extracting documents...")
        documents = self.extractor.extract()
        self.stats["extraction"] = {"document_count": len(documents)}

        # Step 2: Filter documents
        if self.filter:
            print("Step 2: Filtering documents...")
            filtered_docs = self.filter.filter_documents(documents)
            self.stats["filtering"] = self.filter.get_filter_stats(documents, filtered_docs)
            documents = filtered_docs

        # Step 3: Chunk documents
        if self.chunker:
            print("Step 3: Chunking documents...")
            original_docs = documents.copy()
            documents = self.chunker.chunk_documents(documents)
            self.stats["chunking"] = self.chunker.get_chunking_stats(original_docs, documents)

        # Step 4: Post-process documents
        if self.post_processor:
            print("Step 4: Post-processing documents...")
            original_docs = documents.copy()
            documents = self.post_processor.process_documents(documents)
            self.stats["post_processing"] = self.post_processor.get_processing_stats(original_docs, documents)

        # Step 5: Generate embeddings
        print("Step 5: Generating embeddings...")
        print(f"🔍 About to embed {len(documents)} documents")

        try:
            import time

            start_time = time.time()

            embeddings = self.embedder.embed_documents(documents)

            end_time = time.time()
            duration = end_time - start_time

            print(f"🎉 Successfully generated {len(embeddings)} embeddings in {duration:.1f} seconds")
            print(f"🔍 Average time per document: {duration / len(documents):.3f}s")

            if embeddings:
                print(f"🔍 First embedding shape: {len(embeddings[0]) if embeddings[0] else 'None'}")

            self.stats["embedding"] = {
                "document_count": len(documents),
                "embedding_count": len(embeddings),
                "embedding_dimension": self.embedder.get_embedding_dimension(),
                "duration_seconds": duration,
                "avg_time_per_doc": duration / len(documents) if documents else 0,
            }
        except MemoryError as e:
            print(f"❌ Memory error during embedding generation: {str(e)}")
            print("💡 Try reducing batch_size in config or using a lighter model")
            raise
        except Exception as e:
            print(f"❌ Embedding generation failed: {str(e)}")
            print(f"🔍 Error type: {type(e)}")
            import traceback

            print(f"🔍 Traceback: {traceback.format_exc()}")

            raise

        # Step 6: Store in vector database
        print("Step 6: Storing in vector database...")

        doc_dicts = [doc.to_dict() for doc in documents]

        self.vector_store.save_documents(doc_dicts, embeddings)
        self.stats["storage"] = {"stored_documents": len(doc_dicts), "stored_embeddings": len(embeddings)}

        print("Pipeline completed successfully!")
        return self.stats

    def _validate_components(self) -> bool:
        """
        Validate that required components are set.

        Returns:
            bool: True if all required components are set
        """
        print("🔍 Validating pipeline components...")

        required_components = [
            ("extractor", self.extractor),
            ("embedder", self.embedder),
            ("vector_store", self.vector_store),
        ]

        for name, component in required_components:
            if component is None:
                print(f"❌ Error: {name} is required but not set")
                return False
            else:
                print(f"✅ {name}: {type(component).__name__}")

        # Additional validation for optional components
        optional_components = [
            ("filter", self.filter),
            ("chunker", self.chunker),
            ("post_processor", self.post_processor),
        ]

        for name, component in optional_components:
            if component is not None:
                print(f"✅ {name}: {type(component).__name__}")
            else:
                print(f"ℹ️ {name}: Not configured (optional)")

        print("✅ All required components are properly configured")
        return True

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get pipeline execution statistics.

        Returns:
            Dict: Pipeline statistics
        """
        return self.stats.copy()

    def reset_stats(self):
        """Reset pipeline statistics."""
        self.stats = {}
