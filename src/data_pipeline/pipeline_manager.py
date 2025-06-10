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
        embeddings = self.embedder.embed_documents(documents)
        self.stats["embedding"] = {
            "document_count": len(documents),
            "embedding_count": len(embeddings),
            "embedding_dimension": self.embedder.get_embedding_dimension(),
        }

        # Step 6: Store in vector database
        print("Step 6: Storing in vector database...")

        # DEBUG: Check document structure
        if documents:
            print(f"🔍 First document type: {type(documents[0])}")
            print(f"🔍 First document attributes: {dir(documents[0])}")
            if hasattr(documents[0], "content"):
                print("🔍 Document has 'content' attribute")
            if hasattr(documents[0], "page_content"):
                print("🔍 Document has 'page_content' attribute")
            print(f"🔍 First document dict: {documents[0].to_dict()}")

        doc_dicts = [doc.to_dict() for doc in documents]
        print(f"🔍 Number of documents to store: {len(doc_dicts)}")
        print(f"🔍 Number of embeddings: {len(embeddings)}")

        if doc_dicts:
            print(f"🔍 First doc_dict keys: {list(doc_dicts[0].keys())}")

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
        required_components = [
            ("extractor", self.extractor),
            ("embedder", self.embedder),
            ("vector_store", self.vector_store),
        ]

        for name, component in required_components:
            if component is None:
                print(f"Error: {name} is required but not set")
                return False

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
