"""
Unified ingestion pipeline for Isschat.
Uses unified config, centralized embedding service and Qdrant with HNSW.
"""

import logging
import time
import uuid
from typing import List, Dict, Any

from ..config import get_config, get_path_manager
from ..embeddings import get_embedding_service
from ..vectordb import VectorDBFactory, Document as VectorDocument
from ..core.interfaces import Document as CoreDocument
from .extractors.confluence_extractor import ConfluenceExtractor
from .processors.chunker import DocumentChunker


class IngestionPipeline:
    """Centralized ingestion pipeline"""

    def __init__(self):
        """Initialize pipeline with unified config"""
        self.config = get_config()
        self.path_manager = get_path_manager()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Centralized services
        self.embedding_service = get_embedding_service()
        self.vector_db = VectorDBFactory.create_from_config()

        # Processors
        self.chunker = DocumentChunker(
            {"chunk_size": self.config.chunk_size, "chunk_overlap": self.config.chunk_overlap, "separator": "\n\n"}
        )

        # Metrics
        self.stats = {
            "start_time": None,
            "end_time": None,
            "documents_extracted": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "documents_stored": 0,
            "errors": [],
        }

    def run_confluence_ingestion(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Execute complete Confluence ingestion

        Args:
            force_rebuild: Completely rebuild the vector database

        Returns:
            Dict: Ingestion results
        """
        self.logger.info("🚀 Starting Confluence ingestion")
        self.stats["start_time"] = time.time()

        try:
            # Check Confluence configuration
            if not self._validate_confluence_config():
                raise ValueError("Invalid Confluence configuration")

            # Handle rebuild
            if force_rebuild:
                self.logger.info("🔄 Force rebuild: deleting existing collection")
                try:
                    self.vector_db.delete_collection()
                except Exception as e:
                    self.logger.warning(f"Unable to delete collection: {e}")

            # Step 1: Extraction
            self.logger.info("📥 Step 1: Extracting Confluence documents")
            documents = self._extract_confluence_documents()
            self.stats["documents_extracted"] = len(documents)

            if not documents:
                raise ValueError("No documents extracted from Confluence")

            # Step 2: Chunking
            self.logger.info("✂️ Step 2: Chunking documents")
            chunks = self._chunk_documents(documents)
            self.stats["chunks_created"] = len(chunks)

            # Step 3: Generate embeddings
            self.logger.info("🔢 Step 3: Generating embeddings")
            embeddings = self._generate_embeddings(chunks)
            self.stats["embeddings_generated"] = len(embeddings)

            # Step 4: Vector storage
            self.logger.info("💾 Step 4: Storing in Qdrant")
            self._store_in_vector_db(chunks, embeddings)
            self.stats["documents_stored"] = len(chunks)

            # Finalize
            self.stats["end_time"] = time.time()
            duration = self.stats["end_time"] - self.stats["start_time"]

            # Optimize Qdrant collection
            self.logger.info("⚡ Optimizing Qdrant collection")
            try:
                if hasattr(self.vector_db, "optimize_collection"):
                    self.vector_db.optimize_collection()
            except Exception as e:
                self.logger.warning(f"Optimization failed: {e}")

            # Final results
            results = {
                "success": True,
                "duration_seconds": duration,
                "statistics": self.stats,
                "vector_db_info": self.vector_db.get_info(),
                "embedding_info": self.embedding_service.get_info(),
            }

            self.logger.info(f"✅ Ingestion completed successfully in {duration:.1f}s")
            self.logger.info(
                f"📊 Documents: {self.stats['documents_extracted']} → Chunks: {self.stats['chunks_created']} → VectorDB: {self.stats['documents_stored']}"
            )

            return results

        except Exception as e:
            self.stats["end_time"] = time.time()
            self.stats["errors"].append(str(e))

            error_results = {"success": False, "error": str(e), "statistics": self.stats}

            self.logger.error(f"❌ Ingestion failed: {e}")
            return error_results

    def _validate_confluence_config(self) -> bool:
        """Validate Confluence configuration"""
        required_fields = [
            self.config.confluence_api_key,
            self.config.confluence_space_key,
            self.config.confluence_space_name,
            self.config.confluence_email,
        ]

        if not all(required_fields):
            missing = []
            if not self.config.confluence_api_key:
                missing.append("CONFLUENCE_PRIVATE_API_KEY")
            if not self.config.confluence_space_key:
                missing.append("CONFLUENCE_SPACE_KEY")
            if not self.config.confluence_space_name:
                missing.append("CONFLUENCE_SPACE_NAME")
            if not self.config.confluence_email:
                missing.append("CONFLUENCE_EMAIL_ADDRESS")

            self.logger.error(f"Missing configuration: {missing}")
            return False

        return True

    def _extract_confluence_documents(self) -> List[CoreDocument]:
        """Extract documents from Confluence"""
        extractor_config = {
            "confluence_private_api_key": self.config.confluence_api_key,
            "confluence_space_key": self.config.confluence_space_key,
            "confluence_space_name": self.config.confluence_space_name,
            "confluence_email_address": self.config.confluence_email,
            "confluence_url": self.config.confluence_url,
        }

        extractor = ConfluenceExtractor(extractor_config)

        # Validate connection
        if not extractor.validate_connection():
            raise ValueError("Unable to connect to Confluence")

        # Extract documents
        documents = extractor.extract()

        if not documents:
            raise ValueError("No documents extracted from Confluence")

        self.logger.info(f"📄 {len(documents)} documents extracted from Confluence")
        return documents

    def _chunk_documents(self, documents: List[CoreDocument]) -> List[CoreDocument]:
        """Split documents into chunks"""
        chunks = self.chunker.chunk_documents(documents)

        # Chunking statistics
        stats = self.chunker.get_chunking_stats(documents, chunks)
        self.logger.info(
            f"✂️ Chunking: {stats['original_count']} docs → {stats['chunk_count']} chunks (avg. {stats['avg_chunks_per_doc']:.1f} chunks/doc)"
        )

        return chunks

    def _generate_embeddings(self, chunks: List[CoreDocument]) -> List[List[float]]:
        """Generate embeddings for chunks"""
        # Extract texts
        texts = [chunk.page_content for chunk in chunks]

        # Generate embeddings via centralized service
        self.logger.info(f"🔢 Generating embeddings for {len(texts)} chunks")
        embeddings_array = self.embedding_service.encode_texts(texts, show_progress=True)

        # Convert to list of lists
        embeddings = embeddings_array.tolist()

        self.logger.info(f"✅ {len(embeddings)} embeddings generated (dim: {self.embedding_service.dimension})")
        return embeddings

    def _store_in_vector_db(self, chunks: List[CoreDocument], embeddings: List[List[float]]) -> None:
        """Store chunks and embeddings in Qdrant"""
        # Convert chunks to VectorDocument format
        vector_docs = []
        for i, chunk in enumerate(chunks):
            # Create enriched metadata
            metadata = chunk.metadata.copy()
            metadata.update(
                {
                    "chunk_length": len(chunk.page_content),
                    "embedding_model": self.embedding_service.model_name,
                    "ingestion_timestamp": time.time(),
                }
            )

            vector_doc = VectorDocument(id=str(uuid.uuid4()), content=chunk.page_content, metadata=metadata)
            vector_docs.append(vector_doc)

        # Store in Qdrant
        self.logger.info(f"💾 Storing {len(vector_docs)} documents in Qdrant")
        self.vector_db.add_documents(vector_docs, embeddings)

        # Verify storage
        count = self.vector_db.count()
        self.logger.info(f"✅ Documents stored. Total in database: {count}")

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        try:
            vector_info = self.vector_db.get_info()
            embedding_info = self.embedding_service.get_info()

            return {
                "pipeline_ready": True,
                "vector_db": {"exists": self.vector_db.exists(), "count": self.vector_db.count(), "info": vector_info},
                "embedding_service": embedding_info,
                "config": {
                    "embeddings_model": self.config.embeddings_model,
                    "chunk_size": self.config.chunk_size,
                    "vectordb_collection": self.config.vectordb_collection,
                    "vectordb_index_type": self.config.vectordb_index_type,
                },
            }
        except Exception as e:
            return {"pipeline_ready": False, "error": str(e)}

    def check_pipeline(self) -> Dict[str, Any]:
        results = {"embedding_service": False, "vector_db": False, "confluence_config": False, "errors": []}

        try:
            test_embedding = self.embedding_service.encode_single("test")
            results["embedding_service"] = len(test_embedding) == self.embedding_service.dimension
        except Exception as e:
            results["errors"].append(f"Embedding service: {e}")

        try:
            info = self.vector_db.get_info()
            results["vector_db"] = "error" not in info
        except Exception as e:
            results["errors"].append(f"Vector DB: {e}")

        try:
            results["confluence_config"] = self._validate_confluence_config()
        except Exception as e:
            results["errors"].append(f"Confluence config: {e}")

        results["overall_success"] = all(
            [results["embedding_service"], results["vector_db"], results["confluence_config"]]
        )

        return results


def create_ingestion_pipeline() -> IngestionPipeline:
    """Factory to create ingestion pipeline"""
    return IngestionPipeline()
