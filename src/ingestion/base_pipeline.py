"""
Base ingestion pipeline for all data sources.
"""

import logging
import time
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from ..config import get_config
from ..embeddings import get_embedding_service
from ..vectordb import VectorDBFactory, Document as VectorDocument
from ..core.interfaces import Document
from .processors.chunker import DocumentChunker


class BaseIngestionPipeline(ABC):
    """
    Abstract base class for all ingestion pipelines.
    Handles the common flow: Extract -> Chunk -> Embed -> Store
    """

    def __init__(self):
        """Initialize pipeline with services."""
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Services communs
        self.embedding_service = get_embedding_service()
        self.vector_db = VectorDBFactory.create_from_config()

        # Chunker par défaut (peut être remplacé dans les sous-classes)
        self.chunker = DocumentChunker(
            {"chunk_size": self.config.chunk_size, "chunk_overlap": self.config.chunk_overlap, "separator": "\n\n"}
        )

        # Statistiques
        self.stats = self._init_stats()

    def _init_stats(self) -> Dict[str, Any]:
        """Initialize statistics dictionary."""
        return {
            "start_time": None,
            "end_time": None,
            "documents_extracted": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "documents_stored": 0,
            "errors": [],
        }

    @abstractmethod
    def extract_documents(self) -> List[Document]:
        """
        Extract documents from the source.
        Must be implemented by subclasses.

        Returns:
            List of extracted documents
        """
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """Get the name of the data source (e.g., 'confluence', 'github')."""
        pass

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        Can be overridden for custom chunking logic.
        """
        chunks = self.chunker.chunk_documents(documents)

        # Log statistics
        stats = self.chunker.get_chunking_stats(documents, chunks)
        self.logger.info(
            f"✂️ Chunking: {stats['original_count']} docs → "
            f"{stats['chunk_count']} chunks "
            f"(avg. {stats['avg_chunks_per_doc']:.1f} chunks/doc)"
        )

        return chunks

    def generate_embeddings(self, chunks: List[Document]) -> List[List[float]]:
        """Generate embeddings for chunks."""
        texts = [chunk.page_content for chunk in chunks]

        self.logger.info(f"🔢 Generating embeddings for {len(texts)} chunks")
        embeddings_array = self.embedding_service.encode_texts(texts, show_progress=True)

        embeddings = embeddings_array.tolist()
        self.logger.info(f"✅ {len(embeddings)} embeddings generated (dim: {self.embedding_service.dimension})")

        return embeddings

    def store_documents(self, chunks: List[Document], embeddings: List[List[float]]) -> None:
        """Store chunks and embeddings in vector database."""
        vector_docs = []

        for idx, chunk in enumerate(chunks):
            # Enrichir les métadonnées
            metadata = chunk.metadata.copy()
            metadata.update(
                {
                    "source": self.get_source_name(),
                    "chunk_length": len(chunk.page_content),
                    "embedding_model": self.embedding_service.model_name,
                    "ingestion_timestamp": time.time(),
                }
            )

            # Générer un ID déterministe
            doc_id = self._generate_document_id(chunk.page_content, metadata, idx)

            vector_doc = VectorDocument(id=doc_id, content=chunk.page_content, metadata=metadata)
            vector_docs.append(vector_doc)

        self.logger.info(f"💾 Storing {len(vector_docs)} documents in vector database")
        self.vector_db.add_documents(vector_docs, embeddings)

        count = self.vector_db.count()
        self.logger.info(f"✅ Documents stored. Total in database: {count}")

    def _generate_document_id(self, content: str, metadata: Dict[str, Any], index: int) -> str:
        """Generate deterministic document ID."""
        source_id = metadata.get("page_id", metadata.get("doc_id", "unknown"))
        chunk_index = metadata.get("chunk_index", index)

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return f"{self.get_source_name()}_{source_id}_{chunk_index}_{content_hash}"

    def run(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Execute the complete ingestion pipeline.

        Args:
            force_rebuild: If True, delete existing data before ingestion

        Returns:
            Dictionary with ingestion results
        """
        self.logger.info(f"🚀 Starting {self.get_source_name()} ingestion")
        self.stats = self._init_stats()
        self.stats["start_time"] = time.time()

        try:
            # Handle rebuild if requested
            if force_rebuild:
                self.logger.info("🔄 Force rebuild: deleting existing collection")
                try:
                    self.vector_db.delete_collection()
                except Exception as e:
                    self.logger.warning(f"Unable to delete collection: {e}")

            # Step 1: Extract documents
            self.logger.info("📥 Step 1: Extracting documents")
            documents = self.extract_documents()
            self.stats["documents_extracted"] = len(documents)

            if not documents:
                raise ValueError(f"No documents extracted from {self.get_source_name()}")

            # Step 2: Chunk documents
            self.logger.info("✂️ Step 2: Chunking documents")
            chunks = self.chunk_documents(documents)
            self.stats["chunks_created"] = len(chunks)

            # Step 3: Generate embeddings
            self.logger.info("🔢 Step 3: Generating embeddings")
            embeddings = self.generate_embeddings(chunks)
            self.stats["embeddings_generated"] = len(embeddings)

            # Step 4: Store in vector database
            self.logger.info("💾 Step 4: Storing in vector database")
            self.store_documents(chunks, embeddings)
            self.stats["documents_stored"] = len(chunks)

            # Finalize
            self.stats["end_time"] = time.time()
            duration = self.stats["end_time"] - self.stats["start_time"]

            # Optimize collection if possible
            self._optimize_vector_db()

            # Return results
            results = {
                "success": True,
                "duration_seconds": duration,
                "statistics": self.stats,
                "vector_db_info": self.vector_db.get_info(),
                "embedding_info": self.embedding_service.get_info(),
            }

            self.logger.info(f"✅ Ingestion completed successfully in {duration:.1f}s")
            self.logger.info(
                f"📊 Documents: {self.stats['documents_extracted']} → "
                f"Chunks: {self.stats['chunks_created']} → "
                f"VectorDB: {self.stats['documents_stored']}"
            )

            return results

        except Exception as e:
            self.stats["end_time"] = time.time()
            self.stats["errors"].append(str(e))

            self.logger.error(f"❌ Ingestion failed: {e}")

            return {"success": False, "error": str(e), "statistics": self.stats}

    def _optimize_vector_db(self) -> None:
        """Optimize vector database if supported."""
        try:
            if hasattr(self.vector_db, "optimize_collection"):
                self.logger.info("⚡ Optimizing vector database collection")
                self.vector_db.optimize_collection()
        except Exception as e:
            self.logger.warning(f"Optimization failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        try:
            return {
                "pipeline_ready": True,
                "source": self.get_source_name(),
                "vector_db": {
                    "exists": self.vector_db.exists(),
                    "count": self.vector_db.count(),
                    "info": self.vector_db.get_info(),
                },
                "embedding_service": self.embedding_service.get_info(),
                "config": {
                    "embeddings_model": self.config.embeddings_model,
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                },
            }
        except Exception as e:
            return {"pipeline_ready": False, "error": str(e)}
