"""
Robustness evaluator for testing Isschat's reliability and consistency
"""

from typing import Any
from rag_evaluation.core import BaseEvaluator, LLMJudge, IsschatClient
from rag_evaluation.core.base_evaluator import TestCase, EvaluationResult, EvaluationStatus


class RobustnessEvaluator(BaseEvaluator):
    """Evaluator for testing Isschat's robustness and reliability"""

    def __init__(self, config: Any):
        """Initialize robustness evaluator"""
        super().__init__(config)
        self.isschat_client = IsschatClient()
        self.llm_judge = LLMJudge(config)

    def get_category(self) -> str:
        """Get the category this evaluator handles"""
        return "robustness"

    def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case"""
        try:
            # Get response from Isschat
            response, response_time, sources = self.isschat_client.query(test_case.question)

            # Get perfect answer from test metadata
            perfect_answer = test_case.metadata.get("perfect_answer", {"content": "", "quality_metrics": {}})

            # Compare Isschat response with perfect answer
            comparison_result = self.llm_judge.evaluate_comparison(
                question=test_case.question, isschat_response=response, perfect_answer=perfect_answer.get("content", "")
            )

            # Calculate average score
            scores = [
                comparison_result["relevance"]["score"],
                comparison_result["accuracy"]["score"],
                comparison_result["completeness"]["score"],
                comparison_result["clarity"]["score"],
            ]
            average_score = sum(scores) / len(scores)

            # Determine test status
            status = EvaluationStatus.PASSED if average_score >= 0.7 else EvaluationStatus.FAILED

            return EvaluationResult(
                test_id=test_case.test_id,
                category=test_case.category,
                test_name=test_case.test_name,
                question=test_case.question,
                response=response,
                status=status,
                score=average_score,
                evaluation_details={
                    "comparison_details": comparison_result,
                    "reasoning": comparison_result.get("overall_comparison", "No overall comparison provided"),
                },
                response_time=response_time,
                sources=sources,
                metadata={
                    "quality_scores": {
                        "relevance": comparison_result["relevance"]["score"],
                        "accuracy": comparison_result["accuracy"]["score"],
                        "completeness": comparison_result["completeness"]["score"],
                        "clarity": comparison_result["clarity"]["score"],
                    }
                },
            )

        except Exception as e:
            return EvaluationResult(
                test_id=test_case.test_id,
                category=test_case.category,
                test_name=test_case.test_name,
                question=test_case.question,
                response="",
                status=EvaluationStatus.ERROR,
                score=0.0,
                error_message=str(e),
                metadata={"error": str(e)},
            )
