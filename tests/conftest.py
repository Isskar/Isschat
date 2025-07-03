"""
Pytest configuration and fixtures for Isschat tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
from dataclasses import dataclass


@dataclass
class TestConfig:
    """Test configuration object."""

    # Embedding configuration
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_device: str = "cpu"
    embeddings_batch_size: int = 32
    embeddings_normalize: bool = True
    embeddings_trust_remote_code: bool = False
    embeddings_cache_dir: str = "./cache"
    embeddings_dimension: int = 384

    # Vector database configuration
    vector_store_path: str = "./test_vector_store"
    weaviate_host: str = "localhost"
    weaviate_port: int = 8080
    weaviate_collection: str = "test_collection"

    # Chunking configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Storage configuration
    storage_type: str = "local"
    azure_account_name: str = "test_account"
    azure_container_name: str = "test-container"

    # Ingestion configuration
    confluence_base_url: str = "https://test.atlassian.net"
    confluence_username: str = "test@example.com"
    confluence_token: str = "test_token"


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TestConfig()


@pytest.fixture
def mock_embeddings():
    """Provide mock embeddings service."""
    mock = Mock()
    # Mock for 384-dimensional embeddings (all-MiniLM-L6-v2)
    mock_embedding = [0.1] * 384
    mock.embed_documents.return_value = [mock_embedding, mock_embedding]
    mock.embed_query.return_value = mock_embedding
    mock.get_dimension.return_value = 384
    mock.calculate_similarity.return_value = 0.85
    return mock


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    from src.core.documents import BaseDocument

    return [
        BaseDocument(
            content="Ceci est le premier document de test avec du contenu substantiel pour tester le chunking.",  # noqa
            metadata={
                "title": "Document 1",
                "source": "confluence",
                "original_doc_id": "doc1",
                "page_id": "12345",
                "space_key": "TEST",
                "url": "https://test.atlassian.net/doc1",
            },
        ),
        BaseDocument(
            content="Ceci est le deuxième document de test avec des informations importantes sur l'architecture.",
            metadata={
                "title": "Document 2",
                "source": "confluence",
                "original_doc_id": "doc2",
                "page_id": "67890",
                "space_key": "ARCH",
                "url": "https://test.atlassian.net/doc2",
            },
        ),
    ]


@pytest.fixture
def temp_dir():
    """Provide temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_weaviate_client():
    """Provide mock Weaviate client."""
    mock = Mock()
    mock.collections.exists.return_value = True
    mock.collections.create.return_value = True
    mock.collections.get.return_value.batch.dynamic.return_value.__enter__.return_value.add_object.return_value = True
    mock.collections.get.return_value.query.near_vector.return_value.objects = [
        Mock(properties={"content": "test", "original_doc_id": "doc1"}, metadata=Mock(distance=0.1), uuid="1"),
        Mock(properties={"content": "test2", "original_doc_id": "doc2"}, metadata=Mock(distance=0.2), uuid="2"),
    ]
    mock.collections.get.return_value.aggregate.over_all.return_value.total_count = 100
    return mock


@pytest.fixture
def mock_storage():
    """Provide mock storage adapter."""
    mock = Mock()
    mock.save_data.return_value = True
    mock.load_data.return_value = {"test": "data"}
    mock.list_files.return_value = ["file1.json", "file2.json"]
    mock.file_exists.return_value = True
    return mock


@pytest.fixture
def mock_confluence_connector():
    """Provide mock Confluence connector."""
    mock = Mock()
    mock.check_connection.return_value = True
    mock.get_pages.return_value = [
        {
            "id": "12345",
            "title": "Test Page 1",
            "body": "<p>Test content 1</p>",
            "space": {"key": "TEST"},
            "_links": {"webui": "/pages/12345"},
        },
        {
            "id": "67890",
            "title": "Test Page 2",
            "body": "<p>Test content 2</p>",
            "space": {"key": "TEST"},
            "_links": {"webui": "/pages/67890"},
        },
    ]
    return mock


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances before each test."""
    # Reset all singleton instances
    singletons_to_reset = ["src.embeddings.service", "src.storage.storage_factory"]

    for module_name in singletons_to_reset:
        try:
            module = __import__(module_name, fromlist=[""])
            # Reset singleton instances if they have a reset method
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, "reset") and callable(attr.reset):
                    attr.reset()
                # Clear singleton caches
                if hasattr(attr, "_instance"):
                    attr._instance = None
        except (ImportError, AttributeError):
            pass

    yield

    # Clean up after test - same reset logic
    for module_name in singletons_to_reset:
        try:
            module = __import__(module_name, fromlist=[""])
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, "reset") and callable(attr.reset):
                    attr.reset()
                if hasattr(attr, "_instance"):
                    attr._instance = None
        except (ImportError, AttributeError):
            pass
