import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.load_db import DataLoader


class TestDataLoader:
    @pytest.fixture
    def sample_docs(self):
        """Create sample documents for testing"""
        return [
            Document(
                page_content="# Page Title\n\nThis is the first page content.\n## Section 1\nHere is some section content.",  # noqa
                metadata={"source": "https://confluence.example.com/page1", "title": "Page 1"},  # noqa
            ),
            Document(
                page_content="# Another Page\n\nThis is another page with different content.\n## Section A\nContent in section A.",  # noqa
                metadata={"source": "https://confluence.example.com/page2", "title": "Page 2"},  # noqa
            ),
        ]

    def test_split_docs(self, sample_docs):
        """Test document splitting functionality"""
        with (
            patch("src.load_db.MarkdownHeaderTextSplitter") as mock_md_splitter,
            patch("src.load_db.RecursiveCharacterTextSplitter") as mock_text_splitter,
        ):
            # Setup mock markdown splitter
            md_splitter_instance = MagicMock()
            md_splitter_instance.split_text.side_effect = lambda text: [
                Document(page_content=f"Split 1 of {text[:10]}...", metadata={}),
                Document(page_content=f"Split 2 of {text[:10]}...", metadata={}),
            ]
            mock_md_splitter.return_value = md_splitter_instance

            # Setup mock text splitter
            text_splitter_instance = MagicMock()
            text_splitter_instance.split_documents.return_value = [
                Document(page_content="Final split 1", metadata={"title": "Doc 1", "source": "Source 1"}),
                Document(page_content="Final split 2", metadata={"title": "Doc 2", "source": "Source 2"}),
                Document(page_content="Final split 3", metadata={"title": "Doc 3", "source": "Source 3"}),
                Document(page_content="Final split 4", metadata={"title": "Doc 4", "source": "Source 4"}),
            ]
            mock_text_splitter.return_value = text_splitter_instance

            loader = DataLoader()
            result = loader.split_docs(sample_docs)

            # Check that markdown splitter was called twice (once for each document)
            assert md_splitter_instance.split_text.call_count == 2

            # Check that text splitter was called once with all markdown splits
            assert text_splitter_instance.split_documents.call_count == 1

            # Check the final result
            assert len(result) == 4
            assert all(isinstance(doc, Document) for doc in result)
            assert result[0].page_content == "Final split 1"
