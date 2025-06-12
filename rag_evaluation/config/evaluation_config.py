"""
Configuration settings for Isschat evaluation system
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class EvaluationConfig:
    """Configuration for evaluation system"""

    # Success thresholds
    robustness_threshold: float = 0.7
    robustness_ci_threshold: float = 0.3
    conversational_threshold: float = 0.7
    overall_threshold: float = 0.7

    # LLM Judge configuration
    judge_model: str = "anthropic/claude-sonnet-4"
    judge_temperature: float = 0.3
    judge_max_tokens: int = 150

    # Report configuration
    output_dir: Path = field(default_factory=lambda: Path("evaluation_results"))
    generate_html: bool = True
    generate_json: bool = True
    save_detailed_logs: bool = True

    # CI configuration
    ci_mode: bool = False
    fail_on_threshold: bool = True
    ci_test_categories: List[str] = field(default_factory=lambda: ["robustness"])

    # Rate limiting
    request_delay: float = 1.0
    max_retries: int = 3

    # Dataset paths
    robustness_dataset: str = "config/test_datasets/robustness_tests.json"
    conversational_dataset: str = "config/test_datasets/conversational_tests.json"

    # Evaluation categories
    test_categories: Dict[str, Dict] = field(
        default_factory=lambda: {
            "robustness": {
                "name": "Model Robustness Tests",
                "description": "Tests for model knowledge, data validation, and context handling",
                "weight": 0.5,
            },
            "conversational": {
                "name": "Conversational History Tests",
                "description": "Tests for context continuity and multi-turn conversations",
                "weight": 0.5,
            },
        }
    )

    def __post_init__(self):
        """Post-initialization validation"""
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate thresholds
        for threshold in [
            self.robustness_threshold,
            self.conversational_threshold,
            self.overall_threshold,
        ]:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

    def get_dataset_path(self, category: str) -> Path:
        """Get full path for dataset file"""
        dataset_map = {
            "robustness": self.robustness_dataset,
            "conversational": self.conversational_dataset,
        }

        if category not in dataset_map:
            raise ValueError(f"Unknown category: {category}")

        return Path(__file__).parent.parent / dataset_map[category]

    def get_threshold(self, category: str) -> float:
        """Get threshold for specific category"""
        threshold_map = {
            "robustness": self.robustness_threshold,
            "conversational": self.conversational_threshold,
        }

        return threshold_map.get(category, self.overall_threshold)

    def is_ci_category(self, category: str) -> bool:
        """Check if category should run in CI mode"""
        return category in self.ci_test_categories
