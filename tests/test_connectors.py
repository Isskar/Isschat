import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from src.ingestion.connectors.base_connector import BaseConnector, SyncMode, SyncResult
from src.ingestion.connectors.confluence_connector import ConfluenceConnector
from src.core.interfaces import Document


class MockConnector(BaseConnector):
    """Mock implementation of BaseConnector for testing."""

    def __init__(self, config=None):
        super().__init__(config or {})
        self.validate_connection_result = True
        self.extract_result = []

    def validate_connection(self):
        return self.validate_connection_result

    def extract(self):
        return self.extract_result


class TestSyncResult:
    """Tests for SyncResult."""

    def test_init_default_values(self):
        """Test SyncResult initialization with default values."""
        result = SyncResult()

        assert result.success is False
        assert result.documents_retrieved == 0
        assert result.documents_new == 0
        assert result.documents_updated == 0
        assert result.documents_deleted == 0
        assert result.errors == []
        assert result.started_at is None
        assert result.completed_at is None

    def test_duration_seconds_none_when_no_times(self):
        """Test duration_seconds returns None when times not set."""
        result = SyncResult()

        assert result.duration_seconds is None

    def test_duration_seconds_calculation(self):
        """Test duration_seconds calculation."""
        result = SyncResult()
        result.started_at = datetime(2023, 1, 1, 10, 0, 0)
        result.completed_at = datetime(2023, 1, 1, 10, 0, 5)

        assert result.duration_seconds == 5.0


class TestBaseConnector:
    """Tests for BaseConnector."""

    def test_init_with_config(self):
        """Test BaseConnector initialization with config."""
        config = {"test": "value"}
        connector = MockConnector(config)

        assert connector.config == config

    def test_sync_full_mode(self):
        """Test sync with FULL mode."""
        connector = MockConnector()
        test_docs = [Document(page_content="test", metadata={})]
        connector.extract_result = test_docs

        result = connector.sync(SyncMode.FULL)

        assert result.success is True
        assert result.documents_retrieved == 1
        assert result.documents_new == 1
        assert result.started_at is not None
        assert result.completed_at is not None

    def test_sync_incremental_mode_success(self):
        """Test sync with INCREMENTAL mode."""
        connector = MockConnector()
        test_docs = [Document(page_content="test", metadata={})]
        connector.extract_result = test_docs

        since = datetime(2023, 1, 1)
        result = connector.sync(SyncMode.INCREMENTAL, since=since)

        assert result.success is True
        assert result.documents_retrieved == 1
        assert result.documents_updated == 1

    def test_sync_incremental_mode_missing_since(self):
        """Test sync with INCREMENTAL mode missing since parameter."""
        connector = MockConnector()

        result = connector.sync(SyncMode.INCREMENTAL)

        assert result.success is False
        assert "Incremental sync requires 'since' parameter" in result.errors

    def test_sync_specific_mode_success(self):
        """Test sync with SPECIFIC mode."""
        connector = MockConnector()
        test_docs = [Document(page_content="test", metadata={"page_id": "1"})]
        connector.extract_result = test_docs

        result = connector.sync(SyncMode.SPECIFIC, document_ids=["1"])

        assert result.success is True
        assert result.documents_retrieved == 1
        assert result.documents_updated == 1

    def test_sync_specific_mode_missing_ids(self):
        """Test sync with SPECIFIC mode missing document_ids."""
        connector = MockConnector()

        result = connector.sync(SyncMode.SPECIFIC)

        assert result.success is False
        assert "Specific sync requires 'document_ids' parameter" in result.errors

    def test_sync_unsupported_mode(self):
        """Test sync with unsupported mode."""
        connector = MockConnector()

        # Use a value that's not a valid SyncMode (bypass type checking with type: ignore)
        result = connector.sync("invalid_mode")  # type: ignore

        assert result.success is False
        assert "Unsupported sync mode" in result.errors[0]

    def test_get_changed_documents_default(self):
        """Test default get_changed_documents implementation."""
        connector = MockConnector()
        test_docs = [Document(page_content="test", metadata={})]
        connector.extract_result = test_docs

        since = datetime(2023, 1, 1)
        result = connector.get_changed_documents(since)

        assert result == test_docs

    def test_get_documents_by_ids_default(self):
        """Test default get_documents_by_ids implementation."""
        connector = MockConnector()
        test_docs = [
            Document(page_content="test1", metadata={"page_id": "1"}),
            Document(page_content="test2", metadata={"page_id": "2"}),
            Document(page_content="test3", metadata={"page_id": "3"}),
        ]
        connector.extract_result = test_docs

        result = connector.get_documents_by_ids(["1", "3"])

        assert len(result) == 2
        assert result[0].metadata["page_id"] == "1"
        assert result[1].metadata["page_id"] == "3"

    def test_get_sync_capabilities_default(self):
        """Test default sync capabilities."""
        connector = MockConnector()

        capabilities = connector.get_sync_capabilities()

        assert capabilities["full_sync"] is True
        assert capabilities["incremental_sync"] is False
        assert capabilities["specific_sync"] is False
        assert capabilities["change_detection"] is True
        assert capabilities["bulk_retrieval"] is True


class TestConfluenceConnector:
    """Tests for ConfluenceConnector."""

    def test_init_missing_space_name(self):
        """Test initialization with missing space name."""
        config = {}

        with pytest.raises(ValueError, match="Missing confluence_space_key"):
            ConfluenceConnector(config)

    def test_init_missing_space_key(self):
        """Test initialization with missing space key."""
        config = {"confluence_space_name": "test.atlassian.net"}

        with pytest.raises(ValueError, match="Missing confluence_space_key"):
            ConfluenceConnector(config)

    def test_init_missing_auth(self):
        """Test initialization with missing authentication."""
        config = {"confluence_space_name": "test.atlassian.net", "confluence_space_key": "TEST"}

        with pytest.raises(ValueError, match="Missing authentication"):
            ConfluenceConnector(config)

    @patch("src.ingestion.connectors.confluence_connector.ConfluenceReader")
    def test_init_success(self, mock_reader_class):
        """Test successful initialization."""
        mock_reader = Mock()
        mock_reader_class.return_value = mock_reader

        config = {
            "confluence_space_name": "test.atlassian.net",
            "confluence_space_key": "TEST",
            "confluence_email_address": "test@example.com",
            "confluence_private_api_key": "token123",
        }

        connector = ConfluenceConnector(config)

        assert connector.base_url == "test.atlassian.net/wiki"
        assert connector.space_key == "TEST"
        assert connector.username == "test@example.com"
        assert connector.api_token == "token123"

    @patch("src.ingestion.connectors.confluence_connector.ConfluenceReader")
    def test_validate_connection_success(self, mock_reader_class):
        """Test successful connection validation."""
        mock_reader = Mock()
        mock_reader.load_data.return_value = []
        mock_reader_class.return_value = mock_reader

        config = {
            "confluence_space_name": "test.atlassian.net",
            "confluence_space_key": "TEST",
            "confluence_email_address": "test@example.com",
            "confluence_private_api_key": "token123",
        }

        connector = ConfluenceConnector(config)
        result = connector.validate_connection()

        assert result is True
        mock_reader.load_data.assert_called_once()

    @patch("src.ingestion.connectors.confluence_connector.ConfluenceReader")
    def test_validate_connection_failure(self, mock_reader_class):
        """Test connection validation failure."""
        mock_reader = Mock()
        mock_reader.load_data.side_effect = Exception("Connection failed")
        mock_reader_class.return_value = mock_reader

        config = {
            "confluence_space_name": "test.atlassian.net",
            "confluence_space_key": "TEST",
            "confluence_email_address": "test@example.com",
            "confluence_private_api_key": "token123",
        }

        connector = ConfluenceConnector(config)
        result = connector.validate_connection()

        assert result is False

    @patch("src.ingestion.connectors.confluence_connector.ConfluenceReader")
    def test_extract_success(self, mock_reader_class):
        """Test successful document extraction."""
        mock_doc = Mock()
        mock_doc.text = "Test content"
        mock_doc.metadata = {"title": "Test Page", "page_id": "123"}

        mock_reader = Mock()
        mock_reader.load_data.return_value = [mock_doc]
        mock_reader_class.return_value = mock_reader

        config = {
            "confluence_space_name": "test.atlassian.net",
            "confluence_space_key": "TEST",
            "confluence_email_address": "test@example.com",
            "confluence_private_api_key": "token123",
        }

        connector = ConfluenceConnector(config)
        documents = connector.extract()

        assert len(documents) == 1
        assert documents[0].page_content == "Test content"
        assert documents[0].metadata["source"] == "confluence"
        assert documents[0].metadata["space_key"] == "TEST"

    @patch("src.ingestion.connectors.confluence_connector.ConfluenceReader")
    def test_extract_failure(self, mock_reader_class):
        """Test document extraction failure."""
        mock_reader = Mock()
        mock_reader.load_data.side_effect = Exception("API Error")
        mock_reader_class.return_value = mock_reader

        config = {
            "confluence_space_name": "test.atlassian.net",
            "confluence_space_key": "TEST",
            "confluence_email_address": "test@example.com",
            "confluence_private_api_key": "token123",
        }

        connector = ConfluenceConnector(config)
        documents = connector.extract()

        assert documents == []

    @patch("src.ingestion.connectors.confluence_connector.ConfluenceReader")
    def test_get_changed_documents(self, mock_reader_class):
        """Test getting changed documents."""
        mock_doc = Mock()
        mock_doc.text = "Updated content"
        mock_doc.metadata = {"title": "Updated Page", "page_id": "456"}

        mock_reader = Mock()
        mock_reader.load_data.return_value = [mock_doc]
        mock_reader_class.return_value = mock_reader

        config = {
            "confluence_space_name": "test.atlassian.net",
            "confluence_space_key": "TEST",
            "confluence_email_address": "test@example.com",
            "confluence_private_api_key": "token123",
        }

        connector = ConfluenceConnector(config)
        since = datetime(2023, 1, 1)
        documents = connector.get_changed_documents(since)

        assert len(documents) == 1
        assert documents[0].page_content == "Updated content"

    @patch("src.ingestion.connectors.confluence_connector.ConfluenceReader")
    def test_supports_incremental_and_specific(self, mock_reader_class):
        """Test that ConfluenceConnector supports incremental and specific sync."""
        mock_reader_class.return_value = Mock()

        config = {
            "confluence_space_name": "test.atlassian.net",
            "confluence_space_key": "TEST",
            "confluence_email_address": "test@example.com",
            "confluence_private_api_key": "token123",
        }

        connector = ConfluenceConnector(config)

        assert connector._supports_incremental() is True
        assert connector._supports_specific() is True

    @patch("src.ingestion.connectors.confluence_connector.ConfluenceReader")
    def test_convert_llamaindex_documents_error_handling(self, mock_reader_class):
        """Test error handling in document conversion."""
        mock_reader_class.return_value = Mock()

        config = {
            "confluence_space_name": "test.atlassian.net",
            "confluence_space_key": "TEST",
            "confluence_email_address": "test@example.com",
            "confluence_private_api_key": "token123",
        }

        connector = ConfluenceConnector(config)

        # Mock document that will cause an error
        bad_doc = Mock()
        bad_doc.text = "content"
        bad_doc.metadata = Mock()
        bad_doc.metadata.copy.side_effect = Exception("Conversion error")

        result = connector._convert_llamaindex_documents([bad_doc])

        assert result == []
