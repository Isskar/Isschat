"""
Data models for the RAG evaluation system.

This module defines the data structures used to represent evaluations,
results and metrics.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class Evaluation(str, Enum):
    """Binary classification of evaluations."""

    RATHER_GOOD = "rather good"
    RATHER_BAD = "rather bad"


class DocumentEvaluation(BaseModel):
    """Evaluation of the relevance of retrieved documents."""

    model_config = ConfigDict(extra="forbid")

    evaluation: Evaluation = Field(description="Global evaluation of retrieved documents")
    reason: str = Field(description="Explanation of the evaluation")


class ResponseEvaluation(BaseModel):
    """Evaluation of the quality of the generated response."""

    model_config = ConfigDict(extra="forbid")

    evaluation: Evaluation = Field(description="Global evaluation of the generated response")
    reason: str = Field(description="Explanation of the evaluation")


class QuestionEvaluation(BaseModel):
    """Complete evaluation for a given question."""

    question: str
    expected_answer: str
    expected_context: str
    generated_response: str
    retrieved_documents: List[Dict[str, Any]]
    document_evaluation: DocumentEvaluation
    response_evaluation: ResponseEvaluation


class EvaluationResults(BaseModel):
    """Collection of evaluation results with calculated metrics."""

    results: List[QuestionEvaluation]

    @property
    def percentage_good_documents(self) -> float:
        """
        Calculate the percentage of relevant documents.

        Returns:
            float: Percentage of documents evaluated as 'rather good'
        """
        good_docs = sum(1 for r in self.results if r.document_evaluation.evaluation == Evaluation.RATHER_GOOD)
        return (good_docs / len(self.results)) * 100 if self.results else 0

    @property
    def percentage_good_responses(self) -> float:
        """
        Calculate the percentage of correct responses.

        Returns:
            float: Percentage of responses evaluated as 'rather good'
        """
        good_responses = sum(1 for r in self.results if r.response_evaluation.evaluation == Evaluation.RATHER_GOOD)
        return (good_responses / len(self.results)) * 100 if self.results else 0
