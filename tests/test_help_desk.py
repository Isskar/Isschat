import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.help_desk import HelpDesk


class TestHelpDesk:
    @pytest.fixture
    def mock_embedding(self):
        """Mock embedding model"""
        mock = MagicMock()
        mock.embed_query.return_value = [0.1] * 384  # Return dummy embedding vector
        return mock

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM"""
        mock = MagicMock()
        mock.invoke.return_value = "Ceci est une réponse en français."
        return mock

    @pytest.fixture
    def mock_db(self):
        """Mock FAISS vectorstore"""
        mock = MagicMock()
        # Configure mock retriever's get_relevant_documents to return sample documents
        mock.as_retriever.return_value.get_relevant_documents.return_value = [
            Document(
                page_content="Exemple de contenu en français",
                metadata={"title": "Document 1", "source": "https://example.com/doc1"},
            ),
            Document(
                page_content="Un autre exemple de document",
                metadata={"title": "Document 2", "source": "https://example.com/doc2"},
            ),
        ]
        return mock

    def test_init_with_new_db(self, mock_embedding, mock_llm):
        """Test initialization with new_db=True"""
        with (
            patch("src.help_desk.HuggingFaceEmbeddings", return_value=mock_embedding),
            patch("src.help_desk.ChatOpenAI", return_value=mock_llm),
            patch("src.help_desk.load_db.DataLoader") as mock_loader_class,  # noqa
        ):
            # Initialize help desk with new_db=True
            help_desk = HelpDesk(new_db=True)

            # Check that the retriever was configured properly
            assert help_desk.retriever is not None
            assert help_desk.retrieval_qa_chain is not None

    def test_init_with_existing_db(self, mock_embedding, mock_llm):
        """Test initialization with new_db=False"""
        with (
            patch("src.help_desk.HuggingFaceEmbeddings", return_value=mock_embedding),
            patch("src.help_desk.ChatOpenAI", return_value=mock_llm),
            patch("src.help_desk.load_db.DataLoader") as mock_loader_class,  # noqa
        ):
            # Initialize help desk with new_db=False
            help_desk = HelpDesk(new_db=False)

            # Check that the retriever was configured properly
            assert help_desk.retriever is not None
            assert help_desk.retrieval_qa_chain is not None

    def test_template_has_required_components(self):
        """Test that the template has all required components"""
        template = HelpDesk.get_template()

        # Check if it instructs to respond in French
        assert "IMPORTANT: Always respond in French regardless of the language of the question" in template

        # Check if template contains required variables
        assert "{context}" in template
        assert "{question}" in template

    def test_get_prompt(self):
        """Test prompt creation"""
        with (
            patch("src.help_desk.HuggingFaceEmbeddings"),
            patch("src.help_desk.ChatOpenAI"),
            patch("src.help_desk.load_db.DataLoader"),
        ):
            help_desk = HelpDesk()
            prompt = help_desk.prompt

            # Check the prompt is created correctly
            assert prompt.input_variables == ["context", "question"]
            assert isinstance(prompt.template, str)

    def test_retrieval_qa_inference(self, mock_embedding, mock_llm, mock_db):
        """Test retrieval QA inference"""
        with (
            patch("src.help_desk.HuggingFaceEmbeddings", return_value=mock_embedding),
            patch("src.help_desk.ChatOpenAI", return_value=mock_llm),
            patch("src.help_desk.load_db.DataLoader") as mock_loader_class,
        ):
            # Configure the mock DataLoader
            mock_loader = MagicMock()
            mock_loader.get_db.return_value = mock_db
            mock_loader_class.return_value = mock_loader

            help_desk = HelpDesk(new_db=False)

            # Set up a mock for the retrieval chain
            help_desk.retrieval_qa_chain = MagicMock()
            help_desk.retrieval_qa_chain.invoke.return_value = "Voici votre réponse en français."

            # Call the method
            answer, sources = help_desk.retrieval_qa_inference("Comment ça va?", verbose=False)

            # Check the answer
            assert answer == "Voici votre réponse en français."
            assert isinstance(sources, str)
