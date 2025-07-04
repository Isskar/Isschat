"""
Weaviate client implementation with optimized configuration.
"""

import logging
from typing import List, Dict, Any, Optional

import weaviate
from weaviate.classes.config import Configure, Property, VectorDistances, DataType
from weaviate.classes.query import Filter

from .interface import VectorDatabase
from ..core.documents import VectorDocument, SearchResult
from ..config import get_config
from ..config.secrets import get_weaviate_api_key, get_weaviate_url


class WeaviateVectorDB(VectorDatabase):
    """Weaviate client with optimized configuration"""

    def __init__(self, collection_name: Optional[str] = None, embedding_dim: Optional[int] = None):
        """
        Initialize Weaviate client

        Args:
            collection_name: Collection name (otherwise uses config)
            embedding_dim: Embedding dimension (otherwise deduced from model)
        """
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.collection_name = collection_name or self.config.vectordb_collection.replace("-", "_").title()

        if embedding_dim:
            self.embedding_dim = embedding_dim
        else:
            from ..embeddings import get_embedding_service

            self.embedding_dim = get_embedding_service().dimension

        # Initialize Weaviate client
        weaviate_api_key = get_weaviate_api_key()
        weaviate_url = get_weaviate_url()

        if not weaviate_api_key or not weaviate_url:
            raise ValueError("WEAVIATE_API_KEY and WEAVIATE_URL must be configured")

        auth_credentials = weaviate.auth.AuthApiKey(api_key=weaviate_api_key)
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url, auth_credentials=auth_credentials, skip_init_checks=True
        )
        self.logger.info(f"Weaviate client connected: localhost:{self.config.vectordb_port or 8080}")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist"""
        try:
            if not self.client.collections.exists(self.collection_name):
                self.logger.info(f"Creating collection '{self.collection_name}'")

                self.client.collections.create(
                    name=self.collection_name,
                    vectorizer_config=Configure.Vectorizer.none(),
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE, ef=256, ef_construction=256, max_connections=32
                    ),
                    inverted_index_config=Configure.inverted_index(
                        bm25_b=0.75,
                        bm25_k1=1.2,
                    ),
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="original_doc_id", data_type=DataType.TEXT),
                    ],
                )

                self.logger.info(f"Collection '{self.collection_name}' created with HNSW (dim={self.embedding_dim})")
            else:
                self.logger.debug(f"Collection '{self.collection_name}' already exists")

        except Exception as e:
            raise ConnectionError(f"Failed to create Weaviate collection: {e}")

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the collection by searching for original_doc_id"""
        try:
            collection = self.client.collections.get(self.collection_name)

            response = collection.query.fetch_objects(
                where=Filter.by_property("original_doc_id").equal(doc_id), limit=1
            )

            return len(response.objects) > 0
        except Exception:
            return False

    def add_documents(self, documents: List[VectorDocument], embeddings: List[List[float]]) -> None:
        """Add documents with embeddings optimized by batch"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents != number of embeddings")

        if not documents:
            return

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

        try:
            collection = self.client.collections.get(self.collection_name)

            # Prepare data objects
            data_objects = []
            for doc, embedding in zip(new_documents, new_embeddings):
                properties = {"content": doc.content, "original_doc_id": doc.id}

                # Add metadata properties
                if doc.metadata:
                    for key, value in doc.metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            properties[key] = value
                        else:
                            properties[key] = str(value)

                data_objects.append({"properties": properties, "vector": embedding})

            # Batch insert with progress tracking
            batch_size = 100
            for i in range(0, len(data_objects), batch_size):
                batch = data_objects[i : i + batch_size]

                with collection.batch.dynamic() as batch_context:
                    for obj in batch:
                        batch_context.add_object(properties=obj["properties"], vector=obj["vector"])

                self.logger.debug(f"Batch {i // batch_size + 1}/{(len(data_objects) - 1) // batch_size + 1} added")

            self.logger.info(f"{len(new_documents)} documents added successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to add Weaviate documents: {e}")

    def search(
        self, query_embedding: List[float], k: int = 3, filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search with Weaviate vector similarity"""
        try:
            collection = self.client.collections.get(self.collection_name)

            # Build filter if conditions provided
            where_filter = None
            if filter_conditions:
                filters = []
                for key, value in filter_conditions.items():
                    filters.append(Filter.by_property(key).equal(value))

                if len(filters) == 1:
                    where_filter = filters[0]
                else:
                    where_filter = Filter.all_of(filters)

            if where_filter:
                response = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=k,
                    distance=0.8,  # Equivalent to score_threshold=0.2 in cosine similarity
                    return_metadata=["distance"],
                ).where(where_filter)
            else:
                response = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=k,
                    distance=0.8,  # Equivalent to score_threshold=0.2 in cosine similarity
                    return_metadata=["distance"],
                )

            results = []
            for obj in response.objects:
                properties = obj.properties
                content = properties.pop("content", "")
                original_doc_id = properties.pop("original_doc_id", str(obj.uuid))

                # Remaining properties become metadata
                metadata = properties

                document = VectorDocument(id=original_doc_id, content=content, metadata=metadata)

                # Convert distance to similarity score (1 - distance for cosine)
                score = 1.0 - obj.metadata.distance if obj.metadata.distance else 1.0
                results.append(SearchResult(document=document, score=score))

            self.logger.debug(f"Search: {len(results)} results found")
            return results

        except Exception as e:
            raise RuntimeError(f"Weaviate search failed: {e}")

    def exists(self) -> bool:
        """Check collection existence"""
        try:
            return self.client.collections.exists(self.collection_name)
        except Exception:
            return False

    def count(self) -> int:
        """Total number of documents"""
        try:
            collection = self.client.collections.get(self.collection_name)
            response = collection.aggregate.over_all(total_count=True)
            return response.total_count
        except Exception:
            return 0

    def delete_collection(self) -> None:
        """Delete collection completely"""
        try:
            self.client.collections.delete(self.collection_name)
            self.logger.info(f"Collection '{self.collection_name}' deleted")
            # Recreate the collection immediately after deletion
            self._ensure_collection()
        except Exception as e:
            raise RuntimeError(f"Failed to delete collection: {e}")

    def get_info(self) -> Dict[str, Any]:
        """Detailed information about the collection"""
        try:
            collection = self.client.collections.get(self.collection_name)
            config = collection.config.get()

            return {
                "type": "weaviate",
                "collection_name": self.collection_name,
                "host_port": f"{self.config.vectordb_host}:{self.config.vectordb_port or 8080}",
                "host": self.config.vectordb_host,
                "port": self.config.vectordb_port or 8080,
                "points_count": self.count(),
                "embedding_dim": self.embedding_dim,
                "distance_metric": "cosine",
                "vectorizer": config.vectorizer,
                "index_type": "hnsw",
                "vector_index_config": str(config.vector_index_config) if config.vector_index_config else None,
            }
        except Exception as e:
            return {"type": "weaviate", "collection_name": self.collection_name, "error": str(e)}

    def optimize_collection(self) -> None:
        """Optimize collection (Weaviate handles this automatically)"""
        try:
            # Weaviate automatically optimizes indices, but we can log this action
            self.logger.info(f"Collection '{self.collection_name}' optimization (handled automatically by Weaviate)")
        except Exception as e:
            self.logger.warning(f"Optimization logging failed: {e}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Detailed collection statistics"""
        try:
            count = self.count()

            return {
                "points_count": count,
                "vectors_count": count,
                "indexed_vectors_count": count,  # Weaviate keeps all vectors indexed
                "collection_name": self.collection_name,
                "vectorizer": "none",  # We provide our own embeddings
                "distance_metric": "cosine",
            }
        except Exception as e:
            return {"error": str(e)}

    def __del__(self):
        """Cleanup connection"""
        try:
            if hasattr(self, "client"):
                self.client.close()
        except Exception:
            pass
