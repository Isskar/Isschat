"""
Data models for the RAG evaluation system
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class TestType(str, Enum):
    """Types of evaluation tests"""

    ROBUSTNESS = "robustness"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    CONSISTENCY = "consistency"
    RETRIEVAL = "retrieval"
    END_TO_END = "end_to_end"


class RobustnessTestType(str, Enum):
    """Specific robustness test types"""

    KNOWLEDGE_INTERNAL = "knowledge_internal"
    DATA_NONEXISTENT = "data_nonexistent"
    PERSON_FICTIONAL = "person_fictional"
    PERSON_REAL = "person_real"
    OUT_OF_CONTEXT = "out_of_context"
    CONFIDENTIALITY = "confidentiality"
    LANGUAGE_MOLIERE = "language_moliere"
    SYNTHESIS = "synthesis"
    CONVERSATION_HISTORY = "conversation_history"


class Difficulty(str, Enum):
    """Test difficulty levels"""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class TestCase:
    """Individual test case"""

    id: str
    question: str
    expected_answer: Optional[str] = None
    expected_behavior: Optional[str] = None
    test_type: TestType = TestType.QUALITY
    robustness_type: Optional[RobustnessTestType] = None
    difficulty: Difficulty = Difficulty.MEDIUM
    category: str = "general"
    expected_sources: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GenerationScore:
    """Score for generation quality evaluation"""

    relevance: float  # 0-10
    accuracy: float  # 0-10
    completeness: float  # 0-10
    clarity: float  # 0-10
    source_usage: float  # 0-10
    overall_score: float  # Average score
    justification: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevance": self.relevance,
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "source_usage": self.source_usage,
            "overall_score": self.overall_score,
            "justification": self.justification,
        }


@dataclass
class RobustnessScore:
    """Score for robustness test evaluation"""

    score: float  # 0-10
    passed: bool
    justification: str
    test_type: RobustnessTestType

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "justification": self.justification,
            "test_type": self.test_type.value,
        }


@dataclass
class RetrievalScore:
    """Score for retrieval evaluation"""

    precision_at_k: float  # 0-1
    recall_at_k: float  # 0-1
    ndcg_at_k: float  # 0-1
    mrr: float  # 0-1
    k: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "mrr": self.mrr,
            "k": self.k,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for evaluation"""

    response_time: float  # seconds
    tokens_generated: int
    tokens_retrieved: int
    memory_usage: Optional[float] = None  # MB
    cpu_usage: Optional[float] = None  # percentage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response_time": self.response_time,
            "tokens_generated": self.tokens_generated,
            "tokens_retrieved": self.tokens_retrieved,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single test case"""

    test_case_id: str
    question: str
    response: str
    sources: List[str]
    generation_score: Optional[GenerationScore] = None
    robustness_score: Optional[RobustnessScore] = None
    retrieval_score: Optional[RetrievalScore] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    timestamp: datetime = None
    evaluator_version: str = "1.0.0"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "test_case_id": self.test_case_id,
            "question": self.question,
            "response": self.response,
            "sources": self.sources,
            "timestamp": self.timestamp.isoformat(),
            "evaluator_version": self.evaluator_version,
        }

        if self.generation_score:
            result["generation_score"] = self.generation_score.to_dict()
        if self.robustness_score:
            result["robustness_score"] = self.robustness_score.to_dict()
        if self.retrieval_score:
            result["retrieval_score"] = self.retrieval_score.to_dict()
        if self.performance_metrics:
            result["performance_metrics"] = self.performance_metrics.to_dict()

        return result


@dataclass
class BenchmarkResult:
    """Result of a complete benchmark run"""

    benchmark_name: str
    benchmark_version: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_scores: Dict[str, float]
    individual_results: List[EvaluationResult]
    execution_time: float  # seconds
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage"""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "benchmark_version": self.benchmark_version,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "pass_rate": self.pass_rate,
            "average_scores": self.average_scores,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
            "individual_results": [result.to_dict() for result in self.individual_results],
        }


@dataclass
class EvaluationSession:
    """Metadata for an evaluation session"""

    session_id: str
    session_name: str
    description: str
    test_types: List[TestType]
    total_test_cases: int
    completed_test_cases: int = 0
    start_time: datetime = None
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total_test_cases == 0:
            return 0.0
        return (self.completed_test_cases / self.total_test_cases) * 100

    @property
    def duration(self) -> Optional[float]:
        """Calculate session duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "description": self.description,
            "test_types": [t.value for t in self.test_types],
            "total_test_cases": self.total_test_cases,
            "completed_test_cases": self.completed_test_cases,
            "progress_percentage": self.progress_percentage,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "status": self.status,
        }
