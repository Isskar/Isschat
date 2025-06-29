"""
Optimized Qdrant client with HNSW index.
"""

import uuid
import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    HnswConfigDiff,
    OptimizersConfigDiff,
)

from .interface import VectorDatabase, Document, SearchResult
from ..config import get_config


class QdrantVectorDB(VectorDatabase):
    """Optimized Qdrant client with HNSW and advanced configuration"""

    def __init__(self, collection_name: Optional[str] = None, embedding_dim: Optional[int] = None):
        """
        Initialize Qdrant client with unified config

        Args:
            collection_name: Collection name (otherwise uses config)
            embedding_dim: Embedding dimension (otherwise deduced from model)
        """
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configuration from unified config
        self.collection_name = collection_name or self.config.vectordb_collection

        # Embedding dimension (deduced from embedding model)
        if embedding_dim:
            self.embedding_dim = embedding_dim
        else:
            from ..embeddings import get_embedding_service

            self.embedding_dim = get_embedding_service().dimension

        # Advanced HNSW configuration
        self.distance_metric = Distance.COSINE

        # Initialize Qdrant client - always use localhost server
        self.client = QdrantClient(host="localhost", port=self.config.vectordb_port)
        self.logger.info(f"Client Qdrant: localhost:{self.config.vectordb_port}")

        # Create collection with HNSW
        self._ensure_collection_with_hnsw()

    def _ensure_collection_with_hnsw(self) -> None:
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self.logger.info(f"Creating collection '{self.collection_name}' with HNSW")

                # Optimized HNSW configuration
                hnsw_config = HnswConfigDiff(
                    m=48,  # Number of bidirectional connections (more = better precision)
                    ef_construct=512,  # Dynamic neighborhood size (more = better quality)
                    full_scan_threshold=20000,  # Threshold for full scan vs HNSW
                    max_indexing_threads=8,  # Indexing parallelization
                    on_disk=True,  # Store index on disk for large collections
                )

                # Optimizer configuration
                optimizers_config = OptimizersConfigDiff(
                    default_segment_number=2,  # Default number of segments
                    max_segment_size=200000,  # Max segment size
                    memmap_threshold=200000,  # Threshold for memory mapping
                    indexing_threshold=50000,  # Threshold to start indexing
                )

                # Create collection with advanced config
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim, distance=self.distance_metric, hnsw_config=hnsw_config
                    ),
                    optimizers_config=optimizers_config,
                )

                self.logger.info(f"Collection '{self.collection_name}' created with HNSW (dim={self.embedding_dim})")
            else:
                self.logger.debug(f"Collection '{self.collection_name}' already exists")

        except Exception as e:
            raise ConnectionError(f"Failed to create Qdrant collection: {e}")

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the collection by searching for original_doc_id in payload"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Search by original_doc_id in payload instead of point ID
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=[FieldCondition(key="original_doc_id", match=MatchValue(value=doc_id))]),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(result[0]) > 0
        except Exception:
            return False

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """Add documents with embeddings optimized by batch"""
        if len(documents) != len(embeddings):
            raise ValueError("Nombre documents != nombre embeddings")

        if not documents:
            return

        # Filter out existing documents to avoid duplicates
        new_documents = []
        new_embeddings = []
        existing_count = 0

        for doc, embedding in zip(documents, embeddings):
            if not self.document_exists(doc.id):
                new_documents.append(doc)
                new_embeddings.append(embedding)
            else:
                existing_count += 1

        if existing_count > 0:
            self.logger.info(f"Skipped {existing_count} existing documents")

        if not new_documents:
            self.logger.info("No new documents to add")
            return

        self.logger.info(f"Adding {len(new_documents)} new documents to '{self.collection_name}'")

        # Prepare points for Qdrant
        points = []
        for doc, embedding in zip(new_documents, new_embeddings):
            # Generate valid UUID for Qdrant point ID
            point_id = str(uuid.uuid4())

            # Payload (content + metadata + original doc ID for deduplication)
            payload = {"content": doc.content, "original_doc_id": doc.id}
            if doc.metadata:
                payload.update(doc.metadata)

            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

        try:
            # Batch upsert with optimization
            batch_size = 100  # Optimal batch size for Qdrant
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True,  # Wait for confirmation
                )
                self.logger.debug(f"Batch {i // batch_size + 1}/{(len(points) - 1) // batch_size + 1} added")

            self.logger.info(f"{len(new_documents)} documents added successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to add Qdrant documents: {e}")

    def search(
        self, query_embedding: List[float], k: int = 3, filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Optimized search with HNSW"""
        try:
            # Build filter if provided
            qdrant_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(FieldCondition(key=key, match={"value": value}))
                if conditions:
                    qdrant_filter = Filter(must=conditions)

            # Search with optimized HNSW parameters
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                query_filter=qdrant_filter,
                search_params={
                    "hnsw_ef": max(k * 2, 128),  # ef parameter for HNSW search
                    "exact": False,  # Use HNSW (faster than exact)
                },
            )

            # Convert results
            results = []
            for result in search_results:
                payload = result.payload
                content = payload.pop("content", "")
                metadata = payload

                document = Document(id=str(result.id), content=content, metadata=metadata)

                results.append(SearchResult(document=document, score=result.score))

            self.logger.debug(f"Search: {len(results)} results found")
            return results

        except Exception as e:
            raise RuntimeError(f"Qdrant search failed: {e}")

    def exists(self) -> bool:
        """Check collection existence"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            return self.collection_name in collection_names
        except Exception:
            return False

    def count(self) -> int:
        """Total number of documents"""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception:
            return 0

    def delete_collection(self) -> None:
        """Delete collection completely"""
        try:
            self.client.delete_collection(self.collection_name)
            self.logger.info(f"Collection '{self.collection_name}' deleted")
        except Exception as e:
            raise RuntimeError(f"Failed to delete collection: {e}")

    def get_info(self) -> Dict[str, Any]:
        """Detailed information about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)

            return {
                "type": "qdrant",
                "collection_name": self.collection_name,
                "path": str(self.path),
                "host": self.config.vectordb_host,
                "port": self.config.vectordb_port,
                "points_count": collection_info.points_count,
                "embedding_dim": self.embedding_dim,
                "distance_metric": self.distance_metric.value,
                "status": collection_info.status.value,
                "config": collection_info.config.model_dump() if collection_info.config else None,
                "index_type": "hnsw",
            }
        except Exception as e:
            return {"type": "qdrant", "collection_name": self.collection_name, "error": str(e)}

    def optimize_collection(self) -> None:
        """Optimize collection (HNSW indexing)"""
        try:
            # Trigger manual optimization
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=0  # Force immediate indexing
                ),
            )
            self.logger.info(f"Collection '{self.collection_name}' optimization started")
        except Exception as e:
            self.logger.warning(f"Optimization failed: {e}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Detailed collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)

            return {
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_per_segment": info.points_count // max(info.segments_count, 1),
                "segments_count": info.segments_count,
                "disk_data_size": getattr(info, "disk_data_size", "N/A"),
                "ram_data_size": getattr(info, "ram_data_size", "N/A"),
            }
        except Exception as e:
            return {"error": str(e)}
