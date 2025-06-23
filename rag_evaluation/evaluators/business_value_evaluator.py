"""
Business Value Evaluator for measuring Isschat's business impact and efficiency
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from rag_evaluation.core import BaseEvaluator, LLMJudge, IsschatClient
from rag_evaluation.core.base_evaluator import TestCase, EvaluationResult, EvaluationStatus


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single test"""

    response_time: float
    efficiency_ratio: float  # human_time / isschat_time
    relevance_score: float
    quality_comparison: Dict[str, float]  # Comparaison des métriques de qualité


class BusinessValueEvaluator(BaseEvaluator):
    """Business Value Evaluator for measuring Isschat's business impact and efficiency"""

    def __init__(self, config: Any):
        """Initialize business value evaluator"""
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
        return "business_value"

    def _load_test_dataset(self) -> List[Dict[str, Any]]:
        """Load business value test dataset"""
        dataset_path = Path(__file__).parent.parent / "config" / "test_datasets" / "bva_tests.json"

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Business value test dataset not found at {dataset_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in business value test dataset: {e}")

    def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case"""
        try:
            # Run Isschat query and measure time
            start_time = time.time()
            response, response_time, sources = self.isschat_client.query(test_case.question)
            actual_response_time = time.time() - start_time

            # Get human estimate and perfect answer from test metadata
            human_estimate = test_case.metadata.get("human_estimate", 30)
            perfect_answer = test_case.metadata.get("perfect_answer", {"content": "", "quality_metrics": {}})

            # Calculate efficiency ratio
            efficiency_ratio = human_estimate / actual_response_time if actual_response_time > 0 else 0

            # Compare Isschat response with perfect answer using LLM judge
            comparison_result = self.llm_judge.evaluate_comparison(
                question=test_case.question, isschat_response=response, perfect_answer=perfect_answer.get("content", "")
            )

            # Calculate average score from comparison metrics
            scores = [
                comparison_result["relevance"]["score"],
                comparison_result["accuracy"]["score"],
                comparison_result["completeness"]["score"],
                comparison_result["clarity"]["score"],
            ]
            average_score = sum(scores) / len(scores)

            # Check if test passed based on response time and quality comparison
            complexity = test_case.metadata.get("complexity", "medium")
            time_passed = self._check_performance_threshold(actual_response_time, complexity)
            quality_passed = average_score >= 0.7  # Seuil de qualité à 70%
            passed = time_passed and quality_passed

            # Create evaluation result
            evaluation_result = EvaluationResult(
                test_id=test_case.test_id,
                category=test_case.category,
                test_name=test_case.test_name,
                question=test_case.question,
                response=response,
                status=EvaluationStatus.PASSED if passed else EvaluationStatus.FAILED,
                score=average_score,
                evaluation_details={
                    "response_time": actual_response_time,
                    "human_estimate": human_estimate,
                    "efficiency_ratio": efficiency_ratio,
                    "comparison_details": comparison_result,
                    "time_passed": time_passed,
                    "quality_passed": quality_passed,
                    "reasoning": comparison_result.get("overall_comparison", "No overall comparison provided"),
                },
                response_time=actual_response_time,
                sources=sources,
                metadata={
                    "response_time": actual_response_time,
                    "human_estimate": human_estimate,
                    "efficiency_ratio": efficiency_ratio,
                    "complexity": complexity,
                    "quality_scores": {
                        "relevance": comparison_result["relevance"]["score"],
                        "accuracy": comparison_result["accuracy"]["score"],
                        "completeness": comparison_result["completeness"]["score"],
                        "clarity": comparison_result["clarity"]["score"],
                    },
                },
            )

            # Display detailed results
            self._display_evaluation_results(evaluation_result, response, comparison_result)

            return evaluation_result

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

    def _compare_quality_metrics(
        self, isschat_metrics: Dict[str, float], human_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compare quality metrics between Isschat and human responses"""
        bva = {}

        # Compare each metric
        for metric in ["relevance", "accuracy", "completeness", "clarity"]:
            isschat_score = isschat_metrics.get(metric, 0.0)
            human_score = human_metrics.get(metric, 0.0)

            if human_score > 0:  # Only compare if human score is available
                difference = isschat_score - human_score
                bva[metric] = {
                    "isschat_score": isschat_score,
                    "human_score": human_score,
                    "difference": difference,
                    "relative_performance": (isschat_score / human_score) if human_score > 0 else 0,
                }
            else:
                bva[metric] = {
                    "isschat_score": isschat_score,
                    "human_score": None,
                    "difference": None,
                    "relative_performance": None,
                }

        # Calculate overall bva
        bva["overall"] = {
            "average_difference": sum(c["difference"] for c in bva.values() if c["difference"] is not None)
            / len([c for c in bva.values() if c["difference"] is not None])
            if any(c["difference"] is not None for c in bva.values())
            else None,
            "better_than_human": sum(1 for c in bva.values() if c["difference"] is not None and c["difference"] > 0),
            "equal_to_human": sum(1 for c in bva.values() if c["difference"] is not None and c["difference"] == 0),
            "worse_than_human": sum(1 for c in bva.values() if c["difference"] is not None and c["difference"] < 0),
        }

        return bva

    def _check_quality_threshold(self, quality_bva: Dict[str, Any]) -> bool:
        """Check if quality metrics meet thresholds"""
        if not quality_bva.get("overall"):
            return True  # Si pas de comparaison possible (pas de référence humaine), on considère que c'est passé

        # Le test passe si Isschat est meilleur ou égal à l'humain dans au moins 50% des métriques
        metrics_with_bva = quality_bva["overall"]["better_than_human"] + quality_bva["overall"]["equal_to_human"]
        total_metrics_compared = metrics_with_bva + quality_bva["overall"]["worse_than_human"]

        if total_metrics_compared == 0:
            return True

        return (metrics_with_bva / total_metrics_compared) >= 0.5

    def evaluate_business_value(self, complexity_filter: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate business value across all complexity levels"""
        print("🚀 Starting Business Value Evaluation")
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
            "category": "business_value",
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
                    status=EvaluationStatus.ERROR,
                    score=0.0,
                    error_message=str(e),
                    metadata={"error": str(e), "complexity": complexity},
                )
                results.append(error_result)

        return results

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

        # Collect quality bvas
        quality_stats = {"better_than_human": 0, "equal_to_human": 0, "worse_than_human": 0}

        for r in results:
            if "quality_bva" in r.metadata:
                bva = r.metadata["quality_bva"].get("overall", {})
                quality_stats["better_than_human"] += bva.get("better_than_human", 0)
                quality_stats["equal_to_human"] += bva.get("equal_to_human", 0)
                quality_stats["worse_than_human"] += bva.get("worse_than_human", 0)

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        avg_efficiency_ratio = sum(efficiency_ratios) / len(efficiency_ratios) if efficiency_ratios else 0.0

        print(f"\n📊 {complexity.upper()} COMPLEXITY SUMMARY:")
        print("-" * 40)
        print(f"Tests: {passed_count}/{total_tests} passed ({passed_count / total_tests:.1%})")
        print(f"Avg Response Time: {avg_response_time:.2f}s")
        print(f"Avg Efficiency Ratio: {avg_efficiency_ratio:.1f}x")
        print("\nQuality bva with Human:")
        print(f"Better than human: {quality_stats['better_than_human']}")
        print(f"Equal to human: {quality_stats['equal_to_human']}")
        print(f"Worse than human: {quality_stats['worse_than_human']}")

    def _calculate_overall_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate overall summary statistics"""
        total_tests = len(results)
        total_passed = sum(1 for r in results if r.passed)

        response_times = [r.metadata.get("response_time", 0) for r in results if "response_time" in r.metadata]
        efficiency_ratios = [r.metadata.get("efficiency_ratio", 0) for r in results if "efficiency_ratio" in r.metadata]

        # Aggregate quality bvas
        quality_stats = {"better_than_human": 0, "equal_to_human": 0, "worse_than_human": 0}

        for r in results:
            if "quality_bva" in r.metadata:
                bva = r.metadata["quality_bva"].get("overall", {})
                quality_stats["better_than_human"] += bva.get("better_than_human", 0)
                quality_stats["equal_to_human"] += bva.get("equal_to_human", 0)
                quality_stats["worse_than_human"] += bva.get("worse_than_human", 0)

        return {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_tests - total_passed,
            "overall_pass_rate": total_passed / total_tests if total_tests > 0 else 0.0,
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0.0,
            "avg_efficiency_ratio": sum(efficiency_ratios) / len(efficiency_ratios) if efficiency_ratios else 0.0,
            "quality_bva": {
                "better_than_human": quality_stats["better_than_human"],
                "equal_to_human": quality_stats["equal_to_human"],
                "worse_than_human": quality_stats["worse_than_human"],
                "total_bvas": sum(quality_stats.values()),
            },
        }
