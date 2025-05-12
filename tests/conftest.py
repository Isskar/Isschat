"""
Pytest configuration and fixtures for testing the Isschat application.
"""
import pytest
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture to get the path to the test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def env_vars():
    """Fixture to check if required environment variables are set."""
    required_vars = ["OPENROUTER_API_KEY"]  # Add other required variables here
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        pytest.skip(f"Required environment variables not set: {', '.join(missing_vars)}")
    return {var: os.getenv(var) for var in required_vars}
