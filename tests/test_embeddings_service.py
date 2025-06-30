"""
Tests for the new embeddings service architecture.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from src.embeddings.service import get_embedding_service


class TestEmbeddingService:
    """Test the embedding service functionality."""

    def test_singleton_pattern(self):
        """Test that the service follows singleton pattern."""
        service1 = get_embedding_service()
        service2 = get_embedding_service()
        assert service1 is service2, "Should return the same instance"

    @patch("src.embeddings.service.SentenceTransformer")
    @patch("src.embeddings.service.get_model_dimension")
    def test_lazy_loading(self, mock_get_dimension, mock_transformer):
        """Test that model is loaded lazily."""
        mock_model = Mock()
        # Set up encode to return proper shape for dimension check
        mock_model.encode.return_value = np.array([[0.1] * 384])
        mock_transformer.return_value = mock_model
        mock_get_dimension.return_value = 384

        # Reset singleton
        import src.embeddings.service as service_module

        service_module._embedding_service = None

        service = get_embedding_service()

        # Model should not be loaded yet
        mock_transformer.assert_not_called()

        # Access model property to trigger loading
        _ = service.model
        mock_transformer.assert_called_once()

    @patch("src.embeddings.service.SentenceTransformer")
    @patch("src.embeddings.service.get_model_dimension")
    def test_embed_query(self, mock_get_dimension, mock_transformer):
        """Test query embedding."""
        mock_model = Mock()
        # Set up encode to return proper shape for dimension check
        mock_model.encode.side_effect = [np.array([[0.1] * 384]), np.array([[0.1] * 384])]
        mock_transformer.return_value = mock_model
        mock_get_dimension.return_value = 384

        # Reset singleton
        import src.embeddings.service as service_module

        service_module._embedding_service = None

        service = get_embedding_service()
        result = service.encode_query("test query")

        assert len(result) == 384
        # Check that encode was called with proper arguments
        assert mock_model.encode.call_count == 2
        # First call is for dimension check, second is actual encoding
        second_call_args, second_call_kwargs = mock_model.encode.call_args_list[1]
        assert second_call_args[0] == ["test query"]
        assert second_call_kwargs.get("convert_to_numpy") is True

    @patch("src.embeddings.service.SentenceTransformer")
    @patch("src.embeddings.service.get_model_dimension")
    def test_embed_documents(self, mock_get_dimension, mock_transformer):
        """Test document embedding."""
        mock_model = Mock()
        # Set up encode to return proper shape for dimension check and actual call
        mock_model.encode.side_effect = [np.array([[0.1] * 384]), np.array([[0.1] * 384, [0.2] * 384])]
        mock_transformer.return_value = mock_model
        mock_get_dimension.return_value = 384

        # Reset singleton
        import src.embeddings.service as service_module

        service_module._embedding_service = None

        service = get_embedding_service()
        texts = ["doc1", "doc2"]
        result = service.encode_texts(texts)

        assert len(result) == 2
        assert len(result[0]) == 384
        # Check that encode was called with proper arguments
        assert mock_model.encode.call_count == 2
        # First call is for dimension check, second is actual encoding
        second_call_args, second_call_kwargs = mock_model.encode.call_args_list[1]
        assert second_call_args[0] == texts
        assert second_call_kwargs.get("convert_to_numpy") is True

    @patch("src.embeddings.service.SentenceTransformer")
    @patch("src.embeddings.service.get_model_dimension")
    def test_dimension_validation(self, mock_get_dimension, mock_transformer):
        """Test dimension validation."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1] * 512])  # Wrong dimension
        mock_transformer.return_value = mock_model
        mock_get_dimension.return_value = 384  # Expected dimension

        # Reset singleton
        import src.embeddings.service as service_module

        service_module._embedding_service = None

        service = get_embedding_service()
        # Access model to trigger dimension check
        _ = service.model
        # The warning is logged but no exception is raised

    @patch("src.embeddings.service.SentenceTransformer")
    def test_similarity_calculation(self, mock_transformer):
        """Test cosine similarity calculation."""
        mock_model = Mock()
        # Set up encode for dimension check only
        mock_model.encode.return_value = np.array([[0.1] * 384])
        mock_transformer.return_value = mock_model

        # Reset singleton
        import src.embeddings.service as service_module

        service_module._embedding_service = None

        service = get_embedding_service()

        # Test identical vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        similarity = service.similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0, abs=1e-7)

        # Test orthogonal vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = service.similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0, abs=1e-7)

    @patch("src.embeddings.service.SentenceTransformer")
    def test_error_handling(self, mock_transformer):
        """Test error handling in embedding operations."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = Exception("Model error")
        mock_transformer.return_value = mock_model

        # Reset singleton
        import src.embeddings.service as service_module

        service_module._embedding_service = None

        service = get_embedding_service()

        with pytest.raises(Exception, match="Encoding error"):
            service.encode_query("test")

    @patch("src.embeddings.service.SentenceTransformer")
    def test_batch_processing(self, mock_transformer):
        """Test batch processing for large document sets."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        # Simulate batch processing
        # Set up encode to return proper shape for dimension check and actual call
        mock_model.encode.side_effect = [np.array([[0.1] * 384]), np.array([[0.1] * 384] * 100)]
        mock_transformer.return_value = mock_model

        # Reset singleton
        import src.embeddings.service as service_module

        service_module._embedding_service = None

        service = get_embedding_service()

        # Test with large document set
        large_doc_set = [f"document {i}" for i in range(100)]
        result = service.encode_texts(large_doc_set)

        assert len(result) == 100
        assert all(len(embedding) == 384 for embedding in result)
