"""
Base evaluator classes and data structures for Isschat evaluation system
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class TestCategory(Enum):
    """Test category enumeration"""

    ROBUSTNESS = "robustness"
    CONVERSATIONAL = "conversational"


class EvaluationStatus(Enum):
    """Evaluation status enumeration"""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class EvaluationResult:
    """Data structure for evaluation results"""

    # Test identification
    test_id: str
    category: TestCategory
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "test_id": self.test_id,
            "category": self.category.value,
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
            category=TestCategory(data["category"]),
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
    category: TestCategory
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
            "category": self.category.value,
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
            category=TestCategory(data["category"]),
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
    def get_category(self) -> TestCategory:
        """Get the category this evaluator handles"""
        pass

    @abstractmethod
    def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case"""
        pass

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

        avg_score = sum(r.score for r in self.results) / total
        avg_response_time = sum(r.response_time for r in self.results) / total

        return {
            "category": self.get_category().value,
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": passed / total if total > 0 else 0.0,
            "average_score": avg_score,
            "average_response_time": avg_response_time,
        }

    def clear_results(self):
        """Clear stored results"""
        self.results.clear()
