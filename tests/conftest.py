"""
Configuration pytest simplifiée pour les tests d'ISSCHAT
"""

import pytest
import os
from unittest.mock import Mock, patch

from tests.test_data import TEST_CONFIG
import config


@pytest.fixture
def simple_mock_config():
    """Mock simple de la configuration pour les tests"""
    config_data = config.ConfigurationData(
        confluence_private_api_key=TEST_CONFIG["confluence_private_api_key"],
        confluence_space_key=TEST_CONFIG["confluence_space_key"],
        confluence_space_name=TEST_CONFIG["confluence_space_name"],
        confluence_email_address=TEST_CONFIG["confluence_email_address"],
        openrouter_api_key=TEST_CONFIG["openrouter_api_key"],
        db_path=TEST_CONFIG["db_path"],
        persist_directory=TEST_CONFIG["persist_directory"],
    )

    with patch("config.get_config") as mock_get_config:
        mock_get_config.return_value = config_data
        yield config_data


@pytest.fixture
def simple_mock_dependencies(simple_mock_config):
    """Mock simple de toutes les dépendances de HelpDesk"""
    with (
        patch("src.help_desk.HuggingFaceEmbeddings") as mock_hf_embeddings,
        patch("src.help_desk.ChatOpenAI") as mock_chat_openai,
        patch("src.load_db.DataLoader") as mock_data_loader,
    ):
        # Configuration des mocks simples
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 384
        mock_hf_embeddings.return_value = mock_embeddings

        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        mock_db = Mock()
        mock_retriever = Mock()
        mock_db.as_retriever.return_value = mock_retriever

        mock_loader_instance = Mock()
        mock_loader_instance.get_db.return_value = mock_db
        mock_loader_instance.set_db.return_value = mock_db
        mock_data_loader.return_value = mock_loader_instance

        yield {"embeddings": mock_embeddings, "llm": mock_llm, "db": mock_db, "config": simple_mock_config}


@pytest.fixture(autouse=True)
def setup_simple_test_environment():
    """Configure l'environnement de test simple"""
    os.environ["ENVIRONMENT"] = "test"
    os.environ["OPENROUTER_API_KEY"] = "fake_test_key"

    yield

    # Nettoyage après les tests
    if "ENVIRONMENT" in os.environ:
        del os.environ["ENVIRONMENT"]
