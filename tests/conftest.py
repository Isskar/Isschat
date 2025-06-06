"""
Configuration pytest simplifiée pour les tests d'ISSCHAT
"""

import pytest
import os

import config


@pytest.fixture(scope="session")
def test_db(tmp_path_factory):
    """Create a test FAISS database with sample documents"""
    from src.load_db import DataLoader
    from src.help_desk import HelpDesk
    from tests.test_data import MOCK_DOCUMENTS

    db_path = tmp_path_factory.mktemp("db") / "test_index"
    loader = DataLoader()

    # Override persist directory for tests
    loader.persist_directory = str(db_path)

    # Process documents through the full pipeline
    splitted_docs = loader.split_docs(MOCK_DOCUMENTS[:2])  # Use first 2 test documents
    db = loader.save_to_db(splitted_docs, embeddings=HelpDesk().embeddings)  # noqa

    return str(db_path)


@pytest.fixture
def ci_config(test_db):
    """Fixture for CI environment configuration"""
    original_env = os.getenv("ENVIRONMENT")

    # Set required environment variables
    os.environ.update(
        {
            "ENVIRONMENT": "ci",
            "OPENROUTER_API_KEY": "fake_test_key",
            "PERSIST_DIRECTORY": test_db,
            # Minimal Confluence config to avoid errors
            "CONFLUENCE_SPACE_NAME": "test_space",
            "CONFLUENCE_SPACE_KEY": "TEST",
            "CONFLUENCE_EMAIL_ADDRESS": "test@example.com",
            "CONFLUENCE_PRIVATE_API_KEY": "fake_api_key",
        }
    )

    # Reset config before test
    config.reset_config()

    yield

    # Cleanup - remove all test environment variables
    for var in [
        "ENVIRONMENT",
        "OPENROUTER_API_KEY",
        "PERSIST_DIRECTORY",
        "CONFLUENCE_SPACE_NAME",
        "CONFLUENCE_SPACE_KEY",
        "CONFLUENCE_EMAIL_ADDRESS",
        "CONFLUENCE_PRIVATE_API_KEY",
    ]:
        if var in os.environ:
            del os.environ[var]
    if original_env:
        os.environ["ENVIRONMENT"] = original_env
