"""
Base evaluator classes and data structures for Isschat evaluation system
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EvaluationStatus(Enum):
    """Evaluation status enumeration"""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    MEASURED = "measured"  # For metric collection without pass/fail judgment


@dataclass
class EvaluationResult:
    """Data structure for evaluation results"""

    # Test identification
    test_id: str
    category: str
    test_name: str

    # Test data
    question: str
    response: str
    expected_behavior: str

    # Evaluation results
    status: EvaluationStatus
    score: float  # 0.0 to 1.0
    evaluation_details: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    response_time: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if test passed"""
        return self.status == EvaluationStatus.PASSED

    @property
    def failed(self) -> bool:
        """Check if test failed"""
        return self.status == EvaluationStatus.FAILED

    @property
    def has_error(self) -> bool:
        """Check if test had an error"""
        return self.status == EvaluationStatus.ERROR

    @property
    def is_measured(self) -> bool:
        """Check if test was measured (no pass/fail judgment)"""
        return self.status == EvaluationStatus.MEASURED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "test_id": self.test_id,
            "category": self.category,
            "test_name": self.test_name,
            "question": self.question,
            "response": self.response,
            "expected_behavior": self.expected_behavior,
            "status": self.status.value,
            "score": self.score,
            "evaluation_details": self.evaluation_details,
            "response_time": self.response_time,
            "timestamp": self.timestamp.isoformat(),
            "error_message": self.error_message,
            "sources": self.sources,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create from dictionary"""
        return cls(
            test_id=data["test_id"],
            category=data["category"],
            test_name=data["test_name"],
            question=data["question"],
            response=data["response"],
            expected_behavior=data["expected_behavior"],
            status=EvaluationStatus(data["status"]),
            score=data["score"],
            evaluation_details=data.get("evaluation_details", {}),
            response_time=data.get("response_time", 0.0),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            error_message=data.get("error_message"),
            sources=data.get("sources", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TestCase:
    """Data structure for test cases"""

    test_id: str
    category: str
    test_name: str
    question: str
    expected_behavior: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional context for conversational tests
    conversation_context: List[Dict[str, str]] = field(default_factory=list)

    # Optional complexity level
    complexity_level: str = "medium"  # simple, medium, complex

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_id": self.test_id,
            "category": self.category,
            "test_name": self.test_name,
            "question": self.question,
            "expected_behavior": self.expected_behavior,
            "metadata": self.metadata,
            "conversation_context": self.conversation_context,
            "complexity_level": self.complexity_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """Create from dictionary"""

        return cls(
            test_id=data["test_id"],
            category=data["category"],
            test_name=data["test_name"],
            question=data["question"],
            expected_behavior=data["expected_behavior"],
            metadata=data.get("metadata", {}),
            conversation_context=data.get("conversation_context", []),
            complexity_level=data.get("complexity_level", "medium"),
        )


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators"""

    def __init__(self, config: Any):
        """Initialize evaluator with configuration"""
        self.config = config
        self.results: List[EvaluationResult] = []

    @abstractmethod
    def get_category(self) -> str:
        """Get the category this evaluator handles"""
        pass

    def requires_test_cases(self) -> bool:
        """Check if this evaluator requires test cases to be loaded"""
        return True

    def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """Template method for evaluation - implements common flow"""
        try:
            # Step 1: Query the system
            response, response_time, sources = self._query_system(test_case)

            # Step 2: Check for errors
            if response.startswith("ERROR:"):
                return self._create_error_result(test_case, response, response_time, sources)

            # Step 3: Evaluate semantically (abstract method)
            evaluation = self._evaluate_semantically(test_case, response)

            # Step 4: Log results
            self._log_evaluation_result(test_case, evaluation)

            # Step 5: Create success result
            return self._create_success_result(test_case, response, evaluation, response_time, sources)

        except Exception as e:
            return self._create_exception_result(test_case, e)

    @abstractmethod
    def _query_system(self, test_case: TestCase) -> Tuple[str, float, List[str]]:
        """Query the system - implemented by subclasses"""
        pass

    @abstractmethod
    def _evaluate_semantically(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Evaluate response semantically - implemented by subclasses"""
        pass

    def _create_error_result(
        self, test_case: TestCase, response: str, response_time: float, sources: List[str] = Optional[None]
    ) -> EvaluationResult:
        """Create an error result for failed responses"""
        return EvaluationResult(
            test_id=test_case.test_id,
            category=test_case.category,
            test_name=test_case.test_name,
            question=test_case.question,
            response=response,
            expected_behavior=test_case.expected_behavior,
            status=EvaluationStatus.ERROR,
            score=0.0,
            response_time=response_time,
            error_message=response,
            sources=sources or [],
        )

    def _create_success_result(
        self,
        test_case: TestCase,
        response: str,
        evaluation: Dict[str, Any],
        response_time: float,
        sources: List[str] = Optional[None],
    ) -> EvaluationResult:
        """Create a successful evaluation result"""
        status = EvaluationStatus.PASSED if evaluation["passes_criteria"] else EvaluationStatus.FAILED
        return EvaluationResult(
            test_id=test_case.test_id,
            category=test_case.category,
            test_name=test_case.test_name,
            question=test_case.question,
            response=response,
            expected_behavior=test_case.expected_behavior,
            status=status,
            score=evaluation["score"],
            evaluation_details=evaluation,
            response_time=response_time,
            sources=sources or [],
            metadata=test_case.metadata,
        )

    def _create_exception_result(self, test_case: TestCase, exception: Exception) -> EvaluationResult:
        """Create an error result for exceptions"""
        logger.error(f"Error evaluating test {test_case.test_id}: {str(exception)}")
        return EvaluationResult(
            test_id=test_case.test_id,
            category=test_case.category,
            test_name=test_case.test_name,
            question=test_case.question,
            response="",
            expected_behavior=test_case.expected_behavior,
            status=EvaluationStatus.ERROR,
            score=0.0,
            error_message=str(exception),
            metadata=test_case.metadata,
        )

    def _log_evaluation_result(self, test_case: TestCase, evaluation: Dict[str, Any]):
        """Log evaluation results consistently"""
        test_type = test_case.metadata.get("test_type", "generic")
        reasoning = evaluation.get("reasoning", "No reasoning provided")

        logger.info(
            f"Test {test_case.test_id} ({test_type}): LLM evaluation score={evaluation['score']}, "
            f"passes={evaluation['passes_criteria']}"
        )
        logger.info(f"LLM Judge Comment: {reasoning}")

    def evaluate_batch(self, test_cases: List[TestCase]) -> List[EvaluationResult]:
        """Evaluate a batch of test cases"""
        results = []

        for i, test_case in enumerate(test_cases, 1):
            try:
                print(f"Evaluating {i}/{len(test_cases)}: {test_case.test_name}")
                result = self.evaluate_single(test_case)
                results.append(result)
                self.results.append(result)

                # Rate limiting
                if hasattr(self.config, "request_delay") and i < len(test_cases):
                    import time

                    time.sleep(self.config.request_delay)

            except Exception as e:
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
                )
                results.append(error_result)
                self.results.append(error_result)
                print(f"Error evaluating {test_case.test_name}: {e}")

        return results

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for this evaluator"""
        if not self.results:
            return {}

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if r.failed)
        errors = sum(1 for r in self.results if r.has_error)
        measured = sum(1 for r in self.results if r.is_measured)

        avg_score = sum(r.score for r in self.results) / total
        avg_response_time = sum(r.response_time for r in self.results) / total

        return {
            "category": self.get_category(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "measured": measured,
            "pass_rate": passed / total if total > 0 else 0.0,
            "average_score": avg_score,
            "average_response_time": avg_response_time,
        }

    def clear_results(self):
        """Clear stored results"""
        self.results.clear()

    def format_detailed_summary(self) -> str:
        """Format detailed summary for this evaluator (optional override)"""
        return ""
