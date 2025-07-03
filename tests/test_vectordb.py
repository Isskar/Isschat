import pytest
from unittest.mock import Mock, patch
from src.vectordb.interface import VectorDatabase
from src.core.documents import VectorDocument, SearchResult
from src.vectordb.weaviate_client import WeaviateVectorDB


class TestVectorDocument:
    """Tests for VectorDocument dataclass."""

    def test_document_creation(self):
        """Test document creation."""
        doc = VectorDocument(id="test-1", content="Test content", metadata={"source": "test"})

        assert doc.id == "test-1"
        assert doc.content == "Test content"
        assert doc.metadata == {"source": "test"}

    def test_document_defaults(self):
        """Test document with default values."""
        doc = VectorDocument(content="Test", metadata={})

        assert doc.id is None
        assert doc.content == "Test"
        assert doc.metadata == {}


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test search result creation."""
        doc = VectorDocument(id="test-1", content="Test content", metadata={})
        result = SearchResult(document=doc, score=0.85)

        assert result.document == doc
        assert result.score == 0.85


class MockVectorDatabase(VectorDatabase):
    """Mock implementation of VectorDatabase for testing."""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_documents(self, documents, embeddings):
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding, k=3, filter_conditions=None):
        return [SearchResult(doc, 0.9) for doc in self.documents[:k]]

    def exists(self):
        return True

    def count(self):
        return len(self.documents)

    def delete_collection(self):
        self.documents.clear()
        self.embeddings.clear()

    def get_info(self):
        return {"type": "mock", "count": len(self.documents)}


class TestVectorDatabase:
    """Tests for VectorDatabase interface."""

    def test_mock_implementation(self):
        """Test mock implementation works correctly."""
        db = MockVectorDatabase()

        docs = [VectorDocument(id="1", content="test", metadata={})]
        embeddings = [[0.1, 0.2, 0.3]]

        db.add_documents(docs, embeddings)

        assert db.count() == 1
        assert db.exists() is True

        results = db.search([0.1, 0.2, 0.3])
        assert len(results) == 1
        assert results[0].document.id == "1"


class TestWeaviateVectorDB:
    """Tests for WeaviateVectorDB."""

    @patch("src.vectordb.weaviate_client.get_weaviate_api_key")
    @patch("src.vectordb.weaviate_client.get_weaviate_url")
    @patch("src.vectordb.weaviate_client.get_config")
    @patch("src.vectordb.weaviate_client.weaviate.connect_to_weaviate_cloud")
    def test_init_success(self, mock_connect, mock_config, mock_url, mock_key):
        """Test successful initialization."""
        mock_key.return_value = "test-key"
        mock_url.return_value = "https://test.weaviate.network"
        mock_config.return_value = Mock(vectordb_collection="test-collection", vectordb_port=8080)
        mock_client = Mock()
        mock_client.collections.exists.return_value = True
        mock_connect.return_value = mock_client

        with patch("src.embeddings.get_embedding_service") as mock_embed:
            mock_embed.return_value.dimension = 384

            db = WeaviateVectorDB()

            assert db.collection_name == "Test_Collection"
            assert db.embedding_dim == 384

    @patch("src.vectordb.weaviate_client.get_weaviate_api_key")
    @patch("src.vectordb.weaviate_client.get_weaviate_url")
    def test_init_missing_credentials(self, mock_url, mock_key):
        """Test initialization with missing credentials."""
        mock_key.return_value = None
        mock_url.return_value = None

        with pytest.raises(ValueError, match="WEAVIATE_API_KEY and WEAVIATE_URL must be configured"):
            WeaviateVectorDB()

    def test_document_exists_true(self):
        """Test document existence check returns True."""
        db = WeaviateVectorDB.__new__(WeaviateVectorDB)  # Skip __init__
        db.client = Mock()
        db.collection_name = "test"

        mock_collection = Mock()
        mock_response = Mock()
        mock_response.objects = [Mock()]  # Non-empty list
        mock_collection.query.fetch_objects.return_value = mock_response
        db.client.collections.get.return_value = mock_collection

        result = db.document_exists("test-doc")

        assert result is True

    def test_document_exists_false(self):
        """Test document existence check returns False."""
        db = WeaviateVectorDB.__new__(WeaviateVectorDB)  # Skip __init__
        db.client = Mock()
        db.collection_name = "test"

        mock_collection = Mock()
        mock_response = Mock()
        mock_response.objects = []  # Empty list
        mock_collection.query.fetch_objects.return_value = mock_response
        db.client.collections.get.return_value = mock_collection

        result = db.document_exists("test-doc")

        assert result is False

    def test_document_exists_exception(self):
        """Test document existence check handles exceptions."""
        db = WeaviateVectorDB.__new__(WeaviateVectorDB)  # Skip __init__
        db.client = Mock()
        db.collection_name = "test"

        db.client.collections.get.side_effect = Exception("Connection error")

        result = db.document_exists("test-doc")

        assert result is False

    def test_add_documents_empty_list(self):
        """Test adding empty document list."""
        db = WeaviateVectorDB.__new__(WeaviateVectorDB)  # Skip __init__
        db.logger = Mock()

        db.add_documents([], [])

        # Should not raise any exception and should return early

    def test_add_documents_mismatch_length(self):
        """Test adding documents with mismatched embeddings length."""
        db = WeaviateVectorDB.__new__(WeaviateVectorDB)  # Skip __init__

        docs = [VectorDocument(id="1", content="test", metadata={})]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]  # Different length

        with pytest.raises(ValueError, match="Number of documents != number of embeddings"):
            db.add_documents(docs, embeddings)

    def test_exists_true(self):
        """Test collection exists returns True."""
        db = WeaviateVectorDB.__new__(WeaviateVectorDB)  # Skip __init__
        db.client = Mock()
        db.collection_name = "test"

        db.client.collections.exists.return_value = True

        result = db.exists()

        assert result is True

    def test_exists_exception(self):
        """Test collection exists handles exceptions."""
        db = WeaviateVectorDB.__new__(WeaviateVectorDB)  # Skip __init__
        db.client = Mock()
        db.collection_name = "test"

        db.client.collections.exists.side_effect = Exception("Connection error")

        result = db.exists()

        assert result is False

    def test_count_success(self):
        """Test document count returns correct number."""
        db = WeaviateVectorDB.__new__(WeaviateVectorDB)  # Skip __init__
        db.client = Mock()
        db.collection_name = "test"

        mock_collection = Mock()
        mock_response = Mock()
        mock_response.total_count = 42
        mock_collection.aggregate.over_all.return_value = mock_response
        db.client.collections.get.return_value = mock_collection

        result = db.count()

        assert result == 42

    def test_count_exception(self):
        """Test document count handles exceptions."""
        db = WeaviateVectorDB.__new__(WeaviateVectorDB)  # Skip __init__
        db.client = Mock()
        db.collection_name = "test"

        db.client.collections.get.side_effect = Exception("Connection error")

        result = db.count()

        assert result == 0

    def test_get_info_success(self):
        """Test get_info returns correct information."""
        db = WeaviateVectorDB.__new__(WeaviateVectorDB)  # Skip __init__
        db.client = Mock()
        db.collection_name = "test"
        db.embedding_dim = 384
        db.config = Mock(vectordb_host="localhost", vectordb_port=8080)

        mock_collection = Mock()
        mock_config = Mock()
        mock_config.vectorizer = "none"
        mock_config.vector_index_config = "hnsw"
        mock_collection.config.get.return_value = mock_config
        db.client.collections.get.return_value = mock_collection

        with patch.object(db, "count", return_value=10):
            result = db.get_info()

        assert result["type"] == "weaviate"
        assert result["collection_name"] == "test"
        assert result["points_count"] == 10
        assert result["embedding_dim"] == 384

    def test_get_info_exception(self):
        """Test get_info handles exceptions."""
        db = WeaviateVectorDB.__new__(WeaviateVectorDB)  # Skip __init__
        db.client = Mock()
        db.collection_name = "test"

        db.client.collections.get.side_effect = Exception("Connection error")

        result = db.get_info()

        assert result["type"] == "weaviate"
        assert result["collection_name"] == "test"
        assert "error" in result
