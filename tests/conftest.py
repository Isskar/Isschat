"""
Pytest configuration and fixtures for Isschat tests.
"""

import pytest
from unittest.mock import Mock
from dataclasses import dataclass


@dataclass
class TestConfig:
    """Test configuration object."""

    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_device: str = "cpu"
    embeddings_batch_size: int = 32
    embeddings_normalize: bool = True
    embeddings_trust_remote_code: bool = False
    embeddings_cache_dir: str = "./cache"
    vector_store_path: str = "./test_vector_store"
    chunk_size: int = 1000
    chunk_overlap: int = 200


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TestConfig()


@pytest.fixture
def mock_embeddings():
    """Provide mock embeddings object."""
    mock = Mock()
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    from src.core.interfaces import Document

    return [
        Document(
            page_content="Ceci est le premier document de test.", metadata={"title": "Document 1", "source": "doc1.md"}
        ),
        Document(
            page_content="Ceci est le deuxième document de test.", metadata={"title": "Document 2", "source": "doc2.md"}
        ),
    ]


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances before each test."""
    # Reset EmbeddingsManager singleton
    try:
        from src.core.embeddings_manager import EmbeddingsManager

        EmbeddingsManager.reset()
    except ImportError:
        pass

    yield

    # Clean up after test
    try:
        from src.core.embeddings_manager import EmbeddingsManager

        EmbeddingsManager.reset()
    except ImportError:
        pass
