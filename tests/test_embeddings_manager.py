from unittest.mock import patch
from src.core.embeddings_manager import EmbeddingsManager


def test_embeddings_manager_is_singleton(test_config, mock_embeddings):
    with patch("src.core.embeddings_manager.HuggingFaceEmbeddings", return_value=mock_embeddings):
        with patch("src.core.config.get_config", return_value=test_config):
            embeddings1 = EmbeddingsManager.get_embeddings()

            embeddings2 = EmbeddingsManager.get_embeddings()

            assert embeddings1 is embeddings2


def test_embeddings_manager_reset():
    EmbeddingsManager.reset()

    assert EmbeddingsManager._instance is None
    assert EmbeddingsManager._config_hash is None


def test_get_model_info_not_initialized():
    EmbeddingsManager.reset()

    info = EmbeddingsManager.get_model_info()

    assert info["status"] == "not_initialized"


def test_get_model_info_initialized(test_config, mock_embeddings):
    with patch("src.core.embeddings_manager.HuggingFaceEmbeddings", return_value=mock_embeddings):
        with patch("src.core.config.get_config", return_value=test_config):
            EmbeddingsManager.get_embeddings(test_config)

            info = EmbeddingsManager.get_model_info()

            assert info["status"] == "initialized"
            assert info["model_name"] == test_config.embeddings_model
            assert info["device"] == test_config.embeddings_device


def test_embeddings_manager_config_change(test_config, mock_embeddings):
    with patch("src.core.embeddings_manager.HuggingFaceEmbeddings", return_value=mock_embeddings) as mock_hf:
        with patch("src.core.config.get_config", return_value=test_config):
            EmbeddingsManager.get_embeddings(test_config)

            test_config.embeddings_model = "different-model"

            EmbeddingsManager.get_embeddings(test_config)

            assert mock_hf.call_count == 2
