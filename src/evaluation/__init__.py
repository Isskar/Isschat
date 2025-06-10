"""
RAG Evaluation System for Isschat

This module provides comprehensive evaluation capabilities for the RAG system including:
- Generation quality evaluation via LLM as Judge
- Retrieval performance evaluation
- End-to-end system evaluation
- Robustness testing
- Benchmark management
"""

from .manager import EvaluationManager
from .generation_evaluator import GenerationEvaluator
from .llm_judge import LLMJudge
from .dataset_manager import DatasetManager
from .results_manager import ResultsManager
from .models import (
    TestCase, EvaluationResult, BenchmarkResult, EvaluationSession,
    GenerationScore, RobustnessScore, RetrievalScore, PerformanceMetrics,
    TestType, RobustnessTestType, Difficulty
)

__all__ = [
    "EvaluationManager",
    "GenerationEvaluator",
    "LLMJudge",
    "DatasetManager",
    "ResultsManager",
    # Models
    "TestCase",
    "EvaluationResult",
    "BenchmarkResult",
    "EvaluationSession",
    "GenerationScore",
    "RobustnessScore",
    "RetrievalScore",
    "PerformanceMetrics",
    # Enums
    "TestType",
    "RobustnessTestType",
    "Difficulty",
]

__version__ = "1.0.0"