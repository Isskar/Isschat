"""
Performance comparison evaluator for Isschat vs Human efficiency
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from ..core.base_evaluator import BaseEvaluator, EvaluationResult, TestCase, EvaluationStatus
from ..core.isschat_client import IsschatClient
from ..core.llm_judge import LLMJudge


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single test"""

    response_time: float
    efficiency_ratio: float  # human_time / isschat_time
    relevance_score: float


class PerformanceComparisonEvaluator(BaseEvaluator):
    """Evaluator for comparing Isschat performance against human benchmarks"""

    def __init__(self, config: Any):
        """Initialize performance comparison evaluator"""
        super().__init__(config)
        self.isschat_client = IsschatClient()
        self.llm_judge = LLMJudge(config)

        # Load test dataset
        self.test_dataset = self._load_test_dataset()

        # Performance thresholds (seconds)
        self.easy_threshold = 2.0
        self.intermediate_threshold = 5.0
        self.hard_threshold = 15.0

    def get_category(self) -> str:
        """Get the category this evaluator handles"""
        return "performance_comparison"

    def _load_test_dataset(self) -> List[Dict[str, Any]]:
        """Load performance comparison test dataset"""
        dataset_path = Path(__file__).parent.parent / "config" / "test_datasets" / "performance_comparison_tests.json"

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Performance comparison test dataset not found at {dataset_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in performance comparison test dataset: {e}")

    def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case"""
        try:
            # Run Isschat query and measure time
            start_time = time.time()
            response, response_time, sources = self.isschat_client.query(test_case.question)
            actual_response_time = time.time() - start_time

            # Get human estimate from test metadata
            human_estimate_str = test_case.metadata.get("human_estimate", "30s")
            human_estimate = self._parse_time_estimate(human_estimate_str)

            # Calculate efficiency ratio
            efficiency_ratio = human_estimate / actual_response_time if actual_response_time > 0 else 0

            # Evaluate quality using LLM judge
            quality_evaluation = self.llm_judge.evaluate_performance(
                question=test_case.question,
                response=response,
                expected=test_case.expected_behavior,
                response_time=actual_response_time,
                complexity=test_case.metadata.get("complexity", "medium"),
            )

            relevance_score = quality_evaluation.get("score", 0.5)

            # Check if test passed based on response time threshold
            complexity = test_case.metadata.get("complexity", "medium")
            passed = self._check_performance_threshold(actual_response_time, complexity)

            # Create evaluation result
            evaluation_result = EvaluationResult(
                test_id=test_case.test_id,
                category=test_case.category,
                test_name=test_case.test_name,
                question=test_case.question,
                response=response,
                expected_behavior=test_case.expected_behavior,
                status=EvaluationStatus.PASSED if passed else EvaluationStatus.FAILED,
                score=relevance_score,
                evaluation_details={
                    "response_time": actual_response_time,
                    "human_estimate": human_estimate,
                    "efficiency_ratio": efficiency_ratio,
                    "reasoning": f"Response time: {actual_response_time:.2f}s, Efficiency: {efficiency_ratio:.1f}x",
                },
                response_time=actual_response_time,
                sources=sources,
                metadata={
                    "response_time": actual_response_time,
                    "human_estimate": human_estimate,
                    "efficiency_ratio": efficiency_ratio,
                    "complexity": complexity,
                },
            )

            return evaluation_result

        except Exception as e:
            return EvaluationResult(
                test_id=test_case.test_id,
                category=test_case.category,
                test_name=test_case.test_name,
                question=test_case.question,
                response="",
                expected_behavior=test_case.expected_behavior,
                status=EvaluationStatus.ERROR,
                score=0.0,
                error_message=str(e),
                metadata={"error": str(e)},
            )

    def evaluate_performance_comparison(self, complexity_filter: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate performance comparison across all complexity levels"""
        print("🚀 Starting Performance Comparison Evaluation")
        print("=" * 60)

        results = []

        # Filter tests by complexity if specified
        tests_to_run = self.test_dataset
        if complexity_filter:
            tests_to_run = [test for test in self.test_dataset if test["complexity"] == complexity_filter]
            print(f"📊 Running {complexity_filter} complexity tests only")

        # Group tests by complexity
        tests_by_complexity = {}
        for test in tests_to_run:
            complexity = test["complexity"]
            if complexity not in tests_by_complexity:
                tests_by_complexity[complexity] = []
            tests_by_complexity[complexity].append(test)

        # Evaluate each complexity level
        for complexity, tests in tests_by_complexity.items():
            print(f"\n🔍 Evaluating {complexity} complexity tests...")
            complexity_results = self._evaluate_complexity_level(complexity, tests)
            results.extend(complexity_results)

            # Print complexity summary
            self._print_complexity_summary(complexity, complexity_results)

        # Calculate overall summary
        overall_summary = self._calculate_overall_summary(results)

        return {
            "category": "performance_comparison",
            "results": results,
            "summary": overall_summary,
            "timestamp": time.time(),
        }

    def _evaluate_complexity_level(self, complexity: str, tests: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """Evaluate tests for a specific complexity level"""
        results = []

        for test in tests:
            print(f"  📝 Testing: {test['test_name']}")

            try:
                # Convert test dict to TestCase
                test_case = TestCase.from_dict(test)

                # Evaluate the test case
                result = self.evaluate_single(test_case)
                results.append(result)

                # Print test result
                status = "✅" if result.passed else "❌"
                response_time = result.metadata.get("response_time", 0)
                human_estimate = result.metadata.get("human_estimate", 0)
                efficiency_ratio = result.metadata.get("efficiency_ratio", 0)

                print(
                    f"    {status} {response_time:.2f}s (vs {human_estimate:.1f}s human) - "
                    f"Efficiency: {efficiency_ratio:.1f}x"
                )

                # Print the question and response
                print(f"\n    Question: {test_case.question}")
                print(f"    Response: {result.response}\n")

            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
                error_result = EvaluationResult(
                    test_id=test["test_id"],
                    category=test["category"],
                    test_name=test["test_name"],
                    question=test["question"],
                    response="",
                    expected_behavior=test["expected_behavior"],
                    status=EvaluationStatus.ERROR,
                    score=0.0,
                    error_message=str(e),
                    metadata={"error": str(e), "complexity": complexity},
                )
                results.append(error_result)

        return results

    def _parse_time_estimate(self, time_str: str) -> float:
        """Parse human time estimate string to seconds"""
        time_str = time_str.lower().strip()

        if "min" in time_str:
            minutes = float(time_str.replace("min", "").strip())
            return minutes * 60
        elif "s" in time_str:
            seconds = float(time_str.replace("s", "").strip())
            return seconds
        else:
            return float(time_str)

    def _check_performance_threshold(self, response_time: float, complexity: str) -> bool:
        """Check if response time meets threshold for complexity level"""
        if complexity == "easy" and response_time <= self.easy_threshold:
            return True
        elif complexity == "intermediate" and response_time <= self.intermediate_threshold:
            return True
        elif complexity == "hard" and response_time <= self.hard_threshold:
            return True
        return False

    def _print_complexity_summary(self, complexity: str, results: List[EvaluationResult]):
        """Print summary for a complexity level"""
        if not results:
            return

        passed_count = sum(1 for r in results if r.passed)
        total_tests = len(results)

        response_times = [r.metadata.get("response_time", 0) for r in results if "response_time" in r.metadata]
        efficiency_ratios = [r.metadata.get("efficiency_ratio", 0) for r in results if "efficiency_ratio" in r.metadata]
        relevance_scores = [r.score for r in results]

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        avg_efficiency_ratio = sum(efficiency_ratios) / len(efficiency_ratios) if efficiency_ratios else 0.0
        avg_relevance_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

        print(f"\n📊 {complexity.upper()} COMPLEXITY SUMMARY:")
        print("-" * 40)
        print(f"Tests: {passed_count}/{total_tests} passed ({passed_count / total_tests:.1%})")
        print(f"Avg Response Time: {avg_response_time:.2f}s")
        print(f"Avg Efficiency Ratio: {avg_efficiency_ratio:.1f}x")
        print(f"Avg Relevance Score: {avg_relevance_score:.2f}")

    def _calculate_overall_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate overall summary statistics"""
        total_tests = len(results)
        total_passed = sum(1 for r in results if r.passed)

        response_times = [r.metadata.get("response_time", 0) for r in results if "response_time" in r.metadata]
        efficiency_ratios = [r.metadata.get("efficiency_ratio", 0) for r in results if "efficiency_ratio" in r.metadata]
        relevance_scores = [r.score for r in results]

        return {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_tests - total_passed,
            "overall_pass_rate": total_passed / total_tests if total_tests > 0 else 0.0,
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0.0,
            "avg_efficiency_ratio": sum(efficiency_ratios) / len(efficiency_ratios) if efficiency_ratios else 0.0,
            "avg_relevance_score": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0,
        }
