import pytest
import os
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.rag_evaluation.config_evaluation import EvaluationConfig, DatabaseType
from src.rag_evaluation.models import Evaluation, DocumentEvaluation, ResponseEvaluation
from src.rag_evaluation.mock_db import MockVectorStore
from src.rag_evaluation.evaluator import LLMEvaluator
from src.rag_evaluation.manager import EvaluationManager


class TestRagEvaluation:
    @pytest.fixture
    def mock_config(self):
        """Test configuration for evaluation."""
        config = EvaluationConfig()
        config.database.type = DatabaseType.MOCK
        config.database.mock_data = {
            "How to add a page?": [
                {
                    "content": "To add a page in Confluence, click on the '+' button in the navigation bar...",
                    "metadata": {"title": "Page Creation", "source": "https://example.com/doc1"},
                }
            ]
        }
        config.evaluation_dataset_path = "tests/data/test_evaluation_dataset.tsv"
        config.result_output_path = "tests/data/test_evaluation_results.json"
        config.report_output_path = "tests/data/test_evaluation_report.md"
        return config

    @pytest.fixture
    def test_dataset(self, mock_config):
        """Create a test dataset."""
        path = Path(mock_config.evaluation_dataset_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Example data
        data = [
            [
                "How to add a page?",
                "To add a page in Confluence, click on the '+' button.",
                "Page Creation, Navigation",
            ]
        ]

        # Create DataFrame and save
        df = pd.DataFrame(data, columns=["Question", "Expected Answer", "Expected Context"])
        df.to_csv(path, sep="\t", index=False)

        yield path

        # Clean up after the test
        if path.exists():
            os.unlink(path)

    def test_mock_retriever(self):
        """Test retrieving mock documents."""
        mock_data = {
            "How to add a page?": [
                {"content": "Test content", "metadata": {"title": "Test title", "source": "Test source"}}
            ]
        }

        mock_store = MockVectorStore(mock_data=mock_data)
        retriever = mock_store.as_retriever()

        docs = retriever.get_relevant_documents("How to add a page?")

        assert len(docs) == 1
        assert docs[0].page_content == "Test content"
        assert docs[0].metadata["title"] == "Test title"

    @patch("src.rag_evaluation.evaluator.ComponentFactory")
    @patch("src.rag_evaluation.evaluator.PromptConfig")
    def test_evaluator_documents(self, mock_prompt_config, mock_factory, mock_config):
        """Test evaluating retrieved documents."""
        # Configure LLM mock
        mock_llm = MagicMock()
        # Set up the document agent mock
        mock_document_agent = MagicMock()
        mock_document_agent.invoke.return_value = DocumentEvaluation(
            evaluation=Evaluation.RATHER_GOOD, reason="The documents contain the necessary information."
        )

        # Assign the mock agent to the class directly
        mock_prompt_config.document_evaluation_agent = mock_document_agent

        mock_factory.create_llm.return_value = mock_llm

        evaluator = LLMEvaluator(mock_config)

        docs = [
            {
                "title": "Page Creation",
                "content": "To add a page in Confluence...",
                "source": "https://example.com/doc1",
            }
        ]

        result = evaluator.evaluate_documents("How to add a page?", "Page Creation, Navigation", docs)

        assert isinstance(result, DocumentEvaluation)
        assert result.evaluation == Evaluation.RATHER_GOOD
        assert "necessary" in result.reason

    @patch("src.rag_evaluation.evaluator.ComponentFactory")
    @patch("src.rag_evaluation.evaluator.PromptConfig")
    def test_evaluator_response(self, mock_prompt_config, mock_factory, mock_config):
        """Test evaluating the generated response."""
        # Configure LLM mock
        mock_llm = MagicMock()
        # Set up the response agent mock
        mock_response_agent = MagicMock()
        mock_response_agent.invoke.return_value = ResponseEvaluation(
            evaluation=Evaluation.RATHER_GOOD, reason="The response is correct and complete."
        )

        # Assign the mock agent to the class directly
        mock_prompt_config.response_evaluation_agent = mock_response_agent

        mock_factory.create_llm.return_value = mock_llm

        evaluator = LLMEvaluator(mock_config)

        result = evaluator.evaluate_response(
            "How to add a page?",
            "To add a page in Confluence, click on the '+' button.",
            "To add a page in Confluence, use the '+' button in the navigation.",
        )

        assert isinstance(result, ResponseEvaluation)
        assert result.evaluation == Evaluation.RATHER_GOOD
        assert "correct" in result.reason

    @patch("src.rag_evaluation.manager.ComponentFactory")
    @patch("src.rag_evaluation.manager.LLMEvaluator")
    def test_evaluation_manager(self, mock_evaluator_class, mock_factory, mock_config, test_dataset):
        """Test the complete evaluation execution."""
        # Configure mocks
        mock_help_desk = MagicMock()
        mock_help_desk.retrieval_qa_inference.return_value = (
            "To add a page in Confluence, use the '+' button in the navigation.",
            "Source: https://example.com/doc1",
        )

        mock_doc = MagicMock()
        mock_doc.page_content = "To add a page in Confluence..."
        mock_doc.metadata = {"title": "Page Creation", "source": "https://example.com/doc1"}

        mock_help_desk.retriever.get_relevant_documents.return_value = [mock_doc]

        mock_factory.create_help_desk.return_value = mock_help_desk

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_documents.return_value = DocumentEvaluation(
            evaluation=Evaluation.RATHER_GOOD, reason="The documents contain the necessary information."
        )
        mock_evaluator.evaluate_response.return_value = ResponseEvaluation(
            evaluation=Evaluation.RATHER_GOOD, reason="The response is correct and complete."
        )

        mock_evaluator_class.return_value = mock_evaluator

        # Execute the evaluation manager
        evaluation_manager = EvaluationManager(mock_config)
        results = evaluation_manager.run_evaluation()

        # Verify results
        assert len(results.results) == 1
        assert results.percentage_good_documents == 100.0
        assert results.percentage_good_responses == 100.0
