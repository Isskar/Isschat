"""
Business Value Evaluator for measuring Isschat's business impact and efficiency
"""

import time
import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
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
    quality_bva: Dict[str, float]  # Comparaison des métriques de qualité


logger = logging.getLogger(__name__)


class BusinessValueEvaluator(BaseEvaluator):
    BVA_PROMPT = (
        "Tu es un évaluateur expert de la valeur métier des réponses d'un chatbot IA. "
        "Ton objectif est d'évaluer si la réponse d'Isschat est utile, précise et "
        "pertinente pour un utilisateur métier.\n"
        'Tu dois comparer la "Réponse d\'Isschat" à la "Réponse parfaite" en te basant '
        'sur la "Question" de l\'utilisateur.\n\n'
        "Évalue la réponse sur les quatre critères suivants :\n"
        "1.  **Relevance** : La réponse d'Isschat est-elle pertinente par rapport à la "
        "question de l'utilisateur ?\n"
        "2.  **Accuracy** : La réponse d'Isschat est-elle factuellement exacte et "
        "cohérente avec la réponse parfaite ?\n"
        "3.  **Completeness** : La réponse d'Isschat couvre-t-elle tous les aspects "
        "clés de la réponse parfaite ?\n"
        "4.  **Clarity** : La réponse d'Isschat est-elle claire, concise et facile à "
        "comprendre pour un utilisateur métier ?\n\n"
        "Question : {question}\n"
        "Réponse d'Isschat : {isschat_response}\n"
        "Réponse parfaite : {perfect_answer}\n\n"
        "Fournis ton évaluation dans le format suivant, sans texte d'introduction ni "
        "formatage supplémentaire. Chaque score DOIT être un float entre 0.0 et 1.0. "
        "Ne réponds qu'avec le format ci-dessous :\n"
        "Relevance Score: <float_between_0.0_and_1.0>\n"
        "Relevance Reasoning: <one_sentence_reasoning>\n"
        "Accuracy Score: <float_between_0.0_and_1.0>\n"
        "Accuracy Reasoning: <one_sentence_reasoning>\n"
        "Completeness Score: <float_between_0.0_and_1.0>\n"
        "Completeness Reasoning: <one_sentence_reasoning>\n"
        "Clarity Score: <float_between_0.0_and_1.0>\n"
        "Clarity Reasoning: <one_sentence_reasoning>\n"
        "Overall BVA: <overall_assessment_of_business_value>"
    )
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

    def _query_system(self, test_case: TestCase) -> Tuple[str, float, List[str]]:
        """Query the system and return response, response time, and sources."""
        start_time = time.time()
        response, _, sources = self.isschat_client.query(test_case.question)
        response_time = time.time() - start_time
        return response, response_time, sources

    def _evaluate_semantically(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Evaluate the response semantically using a structured text prompt."""
        perfect_answer = test_case.metadata.get("perfect_answer", {}).get("content", "")
        prompt = self.BVA_PROMPT.format(
            question=test_case.question,
            isschat_response=response,
            perfect_answer=perfect_answer,
        )

        llm_response = self.llm_judge.llm.invoke(prompt).content.strip()
        logger.info(f"LLM BVA raw response for {test_case.test_id}:\n{llm_response}")

        bva_details = self._parse_bva_response(llm_response)

        scores = [
            bva_details["relevance"]["score"],
            bva_details["accuracy"]["score"],
            bva_details["completeness"]["score"],
            bva_details["clarity"]["score"],
        ]
        average_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "score": average_score,
            "reasoning": bva_details.get("overall_bva", "No overall BVA provided"),
            "bva_details": bva_details,
            "passes_criteria": average_score >= 0.7,
        }

    def _parse_bva_response(self, response: str) -> Dict[str, Any]:
        """Parse the structured text response from the LLM for BVA evaluation."""
        parsed = {
            "relevance": {"score": 0.0, "reasoning": "Parsing failed"},
            "accuracy": {"score": 0.0, "reasoning": "Parsing failed"},
            "completeness": {"score": 0.0, "reasoning": "Parsing failed"},
            "clarity": {"score": 0.0, "reasoning": "Parsing failed"},
            "overall_bva": "Parsing failed",
        }

        # Handle overall BVA first to capture multi-line reasoning
        overall_bva_match = re.search(r"Overall BVA:(.*)", response, re.DOTALL)
        if overall_bva_match:
            parsed["overall_bva"] = overall_bva_match.group(1).strip()

        lines = response.splitlines()
        for line in lines:
            for metric in ["Relevance", "Accuracy", "Completeness", "Clarity"]:
                if line.startswith(f"{metric} Score:"):
                    score_str = line.split(f"{metric} Score:")[1].strip()
                    try:
                        score = float(score_str)
                        parsed[metric.lower()]["score"] = max(0.0, min(1.0, score))
                    except (ValueError, IndexError):
                        parsed[metric.lower()]["score"] = 0.0
                elif line.startswith(f"{metric} Reasoning:"):
                    reasoning = line.split(f"{metric} Reasoning:")[1].strip()
                    parsed[metric.lower()]["reasoning"] = reasoning

        return parsed

    def _create_success_result(
        self,
        test_case: TestCase,
        response: str,
        evaluation: Dict[str, Any],
        response_time: float,
        sources: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Create a successful evaluation result with business value metrics."""
        human_estimate = test_case.metadata.get("human_estimate", 30)
        complexity = test_case.metadata.get("complexity", "medium")

        efficiency_ratio = human_estimate / response_time if response_time > 0 else 0
        time_passed = self._check_performance_threshold(response_time, complexity)
        quality_passed = evaluation["passes_criteria"]

        status = EvaluationStatus.PASSED if quality_passed else EvaluationStatus.FAILED

        # Add business-specific details to the evaluation and metadata
        evaluation["response_time"] = response_time
        evaluation["human_estimate"] = human_estimate
        evaluation["efficiency_ratio"] = efficiency_ratio
        evaluation["time_passed"] = time_passed
        evaluation["quality_passed"] = quality_passed

        metadata = {
            "response_time": response_time,
            "human_estimate": human_estimate,
            "efficiency_ratio": efficiency_ratio,
            "complexity": complexity,
            "quality_scores": {
                k: v["score"] for k, v in evaluation["bva_details"].items() if isinstance(v, dict) and "score" in v
            },
        }

        perfect_answer = test_case.metadata.get("perfect_answer", "")
        return EvaluationResult(
            test_id=test_case.test_id,
            category=self.get_category(),
            test_name=test_case.test_name,
            question=test_case.question,
            response=response,
            expected_behavior=perfect_answer,
            status=status,
            score=evaluation["score"],
            evaluation_details=evaluation,
            response_time=response_time,
            sources=sources or [],
            metadata=metadata,
        )

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

        for i, test_data in enumerate(tests):
            print(f"  📝 Testing: {test_data['test_name']}")

            # Convert test dict to TestCase
            test_case = TestCase(
                test_id=f"bva_{complexity}_{i}",
                category=self.get_category(),
                test_name=test_data["test_name"],
                question=test_data["question"],
                expected_behavior=test_data["perfect_answer"],
                metadata={
                    "complexity": complexity,
                    "perfect_answer": test_data["perfect_answer"],
                    "human_estimate": test_data["human_estimate"],
                },
            )

            try:
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
                    test_id=test_case.test_id,
                    category=test_case.category,
                    test_name=test_case.test_name,
                    question=test_case.question,
                    response="",
                    expected_behavior=test_case.expected_behavior,
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
