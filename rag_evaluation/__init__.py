"""
Isschat Evaluation System
========================

Comprehensive evaluation system for Isschat with support for:
- Model robustness tests
- Conversational history tests
- Performance timing tests
- Feedback system
- HTML and JSON reports
- CI/CD integration
"""

__version__ = "1.0.0"
__author__ = "Isschat Team"

from .core.base_evaluator import BaseEvaluator, EvaluationResult
from .core.isschat_client import IsschatClient
from .core.llm_judge import LLMJudge
from .config.evaluation_config import EvaluationConfig

__all__ = ["BaseEvaluator", "EvaluationResult", "IsschatClient", "LLMJudge", "EvaluationConfig"]
