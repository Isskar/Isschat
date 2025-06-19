#!/usr/bin/env python3
"""
Main entry point for Isschat Evaluation System
"""

import json
import sys
import os
import argparse
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from rag_evaluation.config.evaluation_config import EvaluationConfig
from rag_evaluation.core.base_evaluator import TestCase

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EvaluationManager:
    """Main evaluation manager orchestrating all test categories"""

    def __init__(self, config: EvaluationConfig):
        """Initialize evaluation manager"""
        self.config = config
        self.evaluators = {}
        self.results = {}
        self._load_evaluators()

    def _load_evaluators(self):
        """Dynamically load evaluators based on configuration"""
        for category in self.config.get_all_categories():
            evaluator_config = self.config.get_evaluator_config(category)

            try:
                # Import the module
                module = importlib.import_module(evaluator_config["module"])

                # Get the class
                evaluator_class = getattr(module, evaluator_config["class_name"])

                # Create instance
                self.evaluators[category] = evaluator_class(self.config)

            except Exception as e:
                print(f"Warning: Could not load evaluator {category}: {e}")

    def load_test_cases(self, category: str) -> List[TestCase]:
        """Load test cases for a specific category"""
        try:
            dataset_path = self.config.get_dataset_path(category)

            if not dataset_path.exists():
                print(f"⚠️ Dataset file not found: {dataset_path}")
                return []

            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            test_cases = []
            for item in data:
                test_case = TestCase.from_dict(item)
                test_cases.append(test_case)

            print(f"✅ Loaded {len(test_cases)} test cases for {category}")
            return test_cases

        except Exception as e:
            print(f"❌ Error loading test cases for {category}: {e}")
            return []

    def run_category_evaluation(self, category: str) -> Dict[str, Any]:
        """Run evaluation for a specific category"""
        print(f"\nStarting {category} evaluation...")

        # Load test cases
        test_cases = self.load_test_cases(category)
        if not test_cases:
            return {"category": category, "results": [], "summary": {}}

        # Get evaluator
        if category not in self.evaluators:
            print(f"❌ Evaluator for category '{category}' not found. Available: {list(self.evaluators.keys())}")
            return {"category": category, "results": [], "summary": {"error": f"Evaluator not found for {category}"}}

        evaluator = self.evaluators[category]

        # Run evaluation
        results = evaluator.evaluate_batch(test_cases)

        # Get summary
        summary = evaluator.get_summary_stats()

        print(
            f"✅ Completed {category} evaluation: {summary.get('passed', 0)}/{summary.get('total_tests', 0)} passed"  # noqa: E501
        )

        return {"category": category, "results": [r.to_dict() for r in results], "summary": summary}

    def run_full_evaluation(self, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run full evaluation across all or specified categories"""
        print("Starting Isschat Evaluation System")
        print("=" * 60)

        # Determine which categories to run
        available_categories = self.config.get_all_categories()

        if categories:
            selected_categories = [cat for cat in categories if cat in available_categories]
        else:
            selected_categories = available_categories

        # Filter for CI mode if enabled
        if self.config.ci_mode:
            ci_categories = self.config.get_ci_categories()
            selected_categories = [cat for cat in selected_categories if cat in ci_categories]
            print(f"🔧 CI Mode: Running only {selected_categories}")

        evaluation_results = {}
        overall_stats = {
            "total_tests": 0,
            "total_passed": 0,
            "total_failed": 0,
            "total_errors": 0,
            "total_measured": 0,
            "category_results": {},
        }

        # Run evaluations
        for category in selected_categories:
            try:
                category_result = self.run_category_evaluation(category)
                evaluation_results[category] = category_result

                # Update overall stats
                summary = category_result["summary"]
                overall_stats["total_tests"] += summary.get("total_tests", 0)
                overall_stats["total_passed"] += summary.get("passed", 0)
                overall_stats["total_failed"] += summary.get("failed", 0)
                overall_stats["total_errors"] += summary.get("errors", 0)
                overall_stats["total_measured"] += summary.get("measured", 0)
                overall_stats["category_results"][category] = summary

            except Exception as e:
                print(f"❌ Error in {category} evaluation: {e}")
                evaluation_results[category] = {
                    "category": category,
                    "results": [],
                    "summary": {"error": str(e)},
                }

        # Calculate overall metrics
        if overall_stats["total_tests"] > 0:
            overall_stats["overall_pass_rate"] = overall_stats["total_passed"] / overall_stats["total_tests"]
        else:
            overall_stats["overall_pass_rate"] = 0.0

        # Store results
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "ci_mode": self.config.ci_mode,
                "categories_run": selected_categories,
                "ci_threshold": self.config.get_ci_threshold() if self.config.ci_mode else None,
            },
            "overall_stats": overall_stats,
            "category_results": evaluation_results,
        }

        return self.results

    def check_thresholds(self) -> bool:
        """Check if evaluation results meet CI threshold"""
        if not self.results:
            return False

        # Only check thresholds in CI mode
        if not self.config.ci_mode:
            return True

        overall_stats = self.results.get("overall_stats", {})
        overall_pass_rate = overall_stats.get("overall_pass_rate", 0.0)

        # Check CI threshold
        return overall_pass_rate >= self.config.get_ci_threshold()

    def save_results(self, output_path: Optional[Path] = None) -> None:
        """Save evaluation results to file"""
        if not self.results:
            print("⚠️ No results to save")
            return

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
            output_path = self.config.output_dir / filename

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)

            print(f"💾 Results saved to: {output_path}")

        except Exception as e:
            print(f"❌ Error saving results: {e}")

    def print_summary(self) -> None:
        """Print evaluation summary"""
        if not self.results:
            print("⚠️ No results to summarize")
            return

        overall_stats = self.results.get("overall_stats", {})

        print(f"\n{'=' * 60}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total Tests: {overall_stats.get('total_tests', 0)}")

        if self.config.ci_mode:
            print(f"Passed: {overall_stats.get('total_passed', 0)} ({overall_stats.get('overall_pass_rate', 0):.1%})")
            print(f"Failed: {overall_stats.get('total_failed', 0)}")
        else:
            print(f"Measured: {overall_stats.get('total_measured', 0)}")

        print(f"Errors: {overall_stats.get('total_errors', 0)}")

        # Category breakdown
        print("\nCATEGORY SCORES:")
        print(f"{'-' * 60}")

        category_results = overall_stats.get("category_results", {})
        for category, summary in category_results.items():
            if "error" in summary:
                print(f"{category.upper()}: ERROR - {summary['error']}")
                continue

            total = summary.get("total_tests", 0)

            if self.config.ci_mode:
                passed = summary.get("passed", 0)
                pass_rate = summary.get("pass_rate", 0.0)
                print(f"{category.upper()}: {passed}/{total} ({pass_rate:.1%})")
            else:
                measured = summary.get("measured", 0)
                avg_score = summary.get("average_score", 0.0)
                print(f"{category.upper()}: {measured}/{total} tests (avg score: {avg_score:.3f})")

            # Show detailed metrics if evaluator supports it
            if category in self.evaluators:
                detailed_summary = self.evaluators[category].format_detailed_summary()
                if detailed_summary:
                    print(detailed_summary)

        # CI threshold check (only in CI mode)
        if self.config.ci_mode:
            meets_thresholds = self.check_thresholds()
            threshold_status = "✅ PASSED" if meets_thresholds else "❌ FAILED"
            ci_threshold = self.config.get_ci_threshold()
            print(f"\n🎯 CI THRESHOLD CHECK ({ci_threshold:.1%}): {threshold_status}")

        print(f"{'=' * 60}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Isschat Evaluation System")
    # Get available categories dynamically
    temp_config = EvaluationConfig()
    available_categories = temp_config.get_all_categories()

    parser.add_argument(
        "--categories",
        nargs="+",
        choices=available_categories,
        help="Specific categories to evaluate",
    )
    parser.add_argument("--ci", action="store_true", help="Run in CI mode")
    parser.add_argument("--output", type=str, help="Output file path")

    args = parser.parse_args()

    try:
        # Load configuration
        config = EvaluationConfig()
        if args.ci:
            config.ci_mode = True

        # Create evaluation manager
        manager = EvaluationManager(config)

        # Run evaluation
        results = manager.run_full_evaluation(args.categories)  # noqa

        # Save results
        output_path = Path(args.output) if args.output else None
        manager.save_results(output_path)

        # Print summary
        manager.print_summary()

        # Exit with appropriate code for CI
        if config.ci_mode and config.fail_on_threshold:
            meets_thresholds = manager.check_thresholds()
            sys.exit(0 if meets_thresholds else 1)

    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
