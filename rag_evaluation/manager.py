"""
Main manager for RAG evaluation.

This module coordinates the entire evaluation process, from reading the dataset
to generating the report.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from config_evaluation import EvaluationConfig, get_config
from models import QuestionEvaluation, EvaluationResults
from evaluator import LLMEvaluator
from factory import ComponentFactory


class EvaluationManager:
    """Main manager for RAG evaluation."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize the manager with given or default configuration.

        Args:
            config: System configuration, if None, uses default configuration
        """
        self.config = config or get_config()
        self.evaluator = LLMEvaluator(self.config)

        # Create RAG model with our factory
        self.rag_model = ComponentFactory.create_help_desk(self.config)

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the evaluation dataset from the configured path.

        Returns:
            pd.DataFrame: Loaded evaluation dataset
        """
        path = Path(self.config.evaluation_dataset_path)
        if not path.exists():
            # Create an example dataset if the file doesn't exist
            self.create_example_dataset(path)

        return pd.read_csv(path, delimiter="\t")

    def create_example_dataset(self, path: Path) -> None:
        """
        Create an example evaluation dataset.

        Args:
            path: Path where to create the example dataset
        """
        # Create parent directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Example data
        data = [
            [
                "How to add a page in Confluence?",
                "To add a page in Confluence, click on the '+' button in the top navigation bar, then select 'Page'.",
                "Confluence Pages, Content Creation, Navigation",
            ],
            [
                "How to manage user permissions?",
                "To manage permissions, access the administration settings, then 'User Management'.",
                "Administration, Security, Permissions",
            ],
        ]

        # Create DataFrame and save
        df = pd.DataFrame(data, columns=["Question", "Expected Answer", "Expected Context"])
        df.to_csv(path, sep="\t", index=False)

        print(f"Example dataset created at {path}")

    def run_evaluation(self) -> EvaluationResults:
        """
        Run complete evaluation on the dataset.

        Returns:
            EvaluationResults: Evaluation results
        """
        dataset = self.load_dataset()
        results = []

        for i, row in dataset.iterrows():
            print(f"Evaluating question {i + 1}/{len(dataset)}: {row['Question']}")

            question = row["Question"]
            expected_answer = row["Expected Answer"]
            expected_context = row["Expected Context"]

            # Run RAG
            generated_response, sources = self.rag_model.retrieval_qa_inference(question, verbose=False)

            # Retrieve documents
            retrieved_documents = self.rag_model.retriever.get_relevant_documents(question)
            docs_info = [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "content": doc.page_content,
                }
                for doc in retrieved_documents
            ]

            # Evaluate documents and response
            eval_docs = self.evaluator.evaluate_documents(question, expected_context, docs_info)
            eval_response = self.evaluator.evaluate_response(question, expected_answer, generated_response)

            # Store results
            question_result = QuestionEvaluation(
                question=question,
                expected_answer=expected_answer,
                expected_context=expected_context,
                generated_response=generated_response,
                retrieved_documents=docs_info,
                document_evaluation=eval_docs,
                response_evaluation=eval_response,
            )

            results.append(question_result)

        return EvaluationResults(results=results)

    def save_results(self, results: EvaluationResults, path: Optional[str] = None) -> None:
        """
        Save evaluation results to JSON format.

        Args:
            results: Evaluation results to save
            path: Path where to save the results
        """
        path = path or self.config.result_output_path
        file_path = Path(path)

        # Create parent directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(results.model_dump_json(indent=2))

        print(f"Results saved to {file_path}")

    def generate_markdown_report(self, results: EvaluationResults, path: Optional[str] = None) -> None:
        """
        Generate an evaluation report in Markdown format.

        Args:
            results: Evaluation results to include in the report
            path: Path where to save the report
        """
        path = path or self.config.report_output_path
        file_path = Path(path)

        # Create parent directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# RAG System Evaluation Report\n\n")

            # Performance summary
            f.write("## Summary\n\n")
            f.write(f"- **Relevant documents**: {results.percentage_good_documents:.2f}%\n")
            f.write(f"- **Correct responses**: {results.percentage_good_responses:.2f}%\n\n")

            # Details for each question
            f.write("## Evaluation Details\n\n")

            for i, res in enumerate(results.results):
                f.write(f"### Question {i + 1}: {res.question}\n\n")

                f.write("#### Document Evaluation\n\n")
                f.write(f"- **Result**: {res.document_evaluation.evaluation.value}\n")
                f.write(f"- **Reason**: {res.document_evaluation.reason}\n\n")

                f.write("#### Response Evaluation\n\n")
                f.write(f"- **Result**: {res.response_evaluation.evaluation.value}\n")
                f.write(f"- **Reason**: {res.response_evaluation.reason}\n\n")

                f.write("#### Details\n\n")
                f.write("**Expected answer**:\n")
                f.write(f"```\n{res.expected_answer}\n```\n\n")

                f.write("**Generated response**:\n")
                f.write(f"```\n{res.generated_response}\n```\n\n")

                f.write("---\n\n")

        print(f"Report generated at {file_path}")


def configure_for_ci() -> EvaluationConfig:
    """
    Configure the system for CI tests with a mock database.

    Returns:
        EvaluationConfig: Configuration for CI tests
    """
    from .config_evaluation import EvaluationConfig, DatabaseType

    # Create base configuration
    config = EvaluationConfig()

    # Configure to use mock database
    config.database.type = DatabaseType.MOCK

    # Example mock data
    config.database.mock_data = {
        "How to add a page?": [
            {
                "content": "To add a page in Confluence, click on the '+' button in the navigation bar...",
                "metadata": {"title": "Page Creation", "source": "https://example.com/doc1"},
            }
        ],
        "How to manage permissions?": [
            {
                "content": "Permission management is done in the administration settings...",
                "metadata": {"title": "Permission Management", "source": "https://example.com/doc2"},
            }
        ],
    }

    # Specific paths for tests
    config.evaluation_dataset_path = "tests/data/evaluation_dataset.tsv"
    config.result_output_path = "tests/data/evaluation_results.json"
    config.report_output_path = "tests/data/evaluation_report.md"

    return config


def run_ci_evaluation() -> EvaluationResults:
    """
    Run evaluation in CI mode.

    Returns:
        EvaluationResults: CI evaluation results
    """
    # Get CI configuration
    config = configure_for_ci()

    # Create evaluation manager
    evaluator = EvaluationManager(config)

    # Run evaluation
    results = evaluator.run_evaluation()

    # Save results
    evaluator.save_results(results)

    # Generate report
    evaluator.generate_markdown_report(results)

    # Return results for verification
    return results


def verify_ci_evaluation(results: EvaluationResults) -> bool:
    """
    Verify if the CI evaluation results are acceptable.

    Args:
        results: CI evaluation results

    Returns:
        bool: True if results are acceptable, False otherwise
    """
    # For example, require at least 70% good responses
    response_threshold = 70.0
    document_threshold = 70.0

    responses_ok = results.percentage_good_responses >= response_threshold
    documents_ok = results.percentage_good_documents >= document_threshold

    if responses_ok and documents_ok:
        print(
            f"✅ Evaluation successful: {results.percentage_good_responses:.2f}% good responses, {results.percentage_good_documents:.2f}% good documents"
        )
        return True
    else:
        print("❌ Evaluation failed:")
        if not responses_ok:
            print(f"   - Responses: {results.percentage_good_responses:.2f}% (threshold: {response_threshold}%)")
        if not documents_ok:
            print(f"   - Documents: {results.percentage_good_documents:.2f}% (threshold: {document_threshold}%)")
        return False
