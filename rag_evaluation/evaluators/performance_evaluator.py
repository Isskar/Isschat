"""
Performance evaluator for testing response time and complexity handling
"""

import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.base_evaluator import BaseEvaluator, TestCategory
from core.isschat_client import IsschatClient
from core.llm_judge import LLMJudge


class PerformanceEvaluator(BaseEvaluator):
    """Evaluator for performance timing tests"""

    def __init__(self, config: Any):
        """Initialize performance evaluator"""
        super().__init__(config)
        self.isschat_client = IsschatClient(conversation_memory=False)
        self.llm_judge = LLMJudge(config)

    def get_category(self) -> TestCategory:
        """Get the category this evaluator handles"""
        return TestCategory.PERFORMANCE
