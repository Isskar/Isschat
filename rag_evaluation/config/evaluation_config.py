"""
Configuration settings for Isschat evaluation system
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any


@dataclass
class EvaluationConfig:
    """Configuration for evaluation system"""

    # CI threshold (only used in CI mode)
    ci_threshold: float = 0.7

    # LLM Judge configuration
    judge_model: str = "google/gemini-2.5-flash-lite-preview-06-17"
    judge_temperature: float = 0.3
    judge_max_tokens: int = 150

    # Report configuration
    output_dir: Path = field(default_factory=lambda: Path("evaluation_results"))

    # CI configuration
    ci_mode: bool = False
    fail_on_threshold: bool = True

    # Evaluators configuration (loaded dynamically)
    _evaluators_config: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Post-initialization validation"""
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load evaluators configuration
        self._load_evaluators_config()

        # Validate thresholds
        if not 0.0 <= self.ci_threshold <= 1.0:
            raise ValueError(f"CI threshold must be between 0.0 and 1.0, got {self.ci_threshold}")

    def _load_evaluators_config(self):
        """Load evaluators configuration from JSON file"""
        config_path = Path(__file__).parent / "evaluators.json"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._evaluators_config = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Evaluators config file not found at {config_path}")
            self._evaluators_config = {}
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in evaluators config: {e}")
            self._evaluators_config = {}

    def get_evaluator_config(self, category: str) -> Dict[str, Any]:
        """Get configuration for a specific evaluator"""
        return self._evaluators_config.get(category, {})

    def get_all_categories(self) -> List[str]:
        """Get all available evaluator categories"""
        return [cat for cat, config in self._evaluators_config.items() if config.get("enabled", True)]

    def get_ci_categories(self) -> List[str]:
        """Get categories that should run in CI mode (all enabled categories)"""
        return self.get_all_categories()

    def get_dataset_path(self, category: str) -> Path:
        """Get full path for dataset file"""
        evaluator_config = self.get_evaluator_config(category)
        if not evaluator_config or "dataset" not in evaluator_config:
            raise ValueError(f"No dataset configured for category: {category}")

        return Path(__file__).parent.parent / evaluator_config["dataset"]

    def get_ci_threshold(self) -> float:
        """Get CI threshold (only used in CI mode)"""
        return self.ci_threshold
