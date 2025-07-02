#!/usr/bin/env python3
"""
Main entry point for Isschat Evaluation System
"""

import json
import os
import sys
import argparse
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from rag_evaluation.config import EvaluationConfig
from rag_evaluation.core.base_evaluator import TestCase

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EvaluationManager:
    """Main evaluation manager orchestrating all test categories"""

    def __init__(self, config: EvaluationConfig):
        """Initialize evaluation manager"""
        self.config = config
        self.evaluator_classes = {}  # Store classes, not instances
        self.evaluators = {}  # Cache for instantiated evaluators
        self.results = {}
        self._load_evaluator_classes()

    def _load_evaluator_classes(self):
        """Dynamically load evaluator classes (but don't instantiate them yet)"""
        for category in self.config.get_all_categories():
            evaluator_config = self.config.get_evaluator_config(category)

            try:
                module = importlib.import_module(evaluator_config["module"])

                evaluator_class = getattr(module, evaluator_config["class_name"])

                self.evaluator_classes[category] = evaluator_class

            except Exception as e:
                print(f"Warning: Could not load evaluator class {category}: {e}")

    def _get_evaluator(self, category: str):
        """Get evaluator instance, creating it lazily if needed"""
        if category not in self.evaluators:
            if category not in self.evaluator_classes:
                raise ValueError(f"Evaluator class for category '{category}' not found")

            # Lazy instantiation - only create when needed
            print(f"🔧 Initializing {category} evaluator...")
            self.evaluators[category] = self.evaluator_classes[category](self.config)

        return self.evaluators[category]

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

        try:
            evaluator = self._get_evaluator(category)
        except ValueError as e:
            print(f"❌ {e}. Available: {list(self.evaluator_classes.keys())}")
            return {"category": category, "results": [], "summary": {"error": str(e)}}

        # Check if this evaluator requires test cases
        if evaluator.requires_test_cases():
            test_cases = self.load_test_cases(category)
            if not test_cases:
                return {"category": category, "results": [], "summary": {}}

            # Run evaluation with test cases
            results = evaluator.evaluate_batch(test_cases)
        else:
            # For evaluators that don't need test cases
            print(f"Running {category} evaluation, no test cases needed...")
            if hasattr(evaluator, "evaluate_without_test_cases"):
                results = evaluator.evaluate_without_test_cases()
            else:
                # Fallback: create a dummy test case for compatibility
                dummy_test_case = TestCase(
                    test_id=f"{category}_analysis_001",
                    category=category,
                    test_name=f"{category.title()} Analysis",
                    question=f"Perform {category} analysis",
                    expected_behavior=f"Complete {category} analysis",
                )
                results = [evaluator.evaluate_single(dummy_test_case)]

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

    def save_results(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Save evaluation results to file

        Returns:
            Path to the saved file, or None if saving failed
        """
        if not self.results:
            print("⚠️ No results to save")
            return None

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
            output_path = self.config.output_dir / filename

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)

            print(f"💾 Results saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return None

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
            passed = summary.get("passed", 0)
            measured = summary.get("measured", 0)

            if self.config.ci_mode:
                pass_rate = summary.get("pass_rate", 0.0)
                print(f"{category.upper()}: {passed}/{total} tests (pass rate: {pass_rate:.1%})")
            else:
                # Only show pass/fail ratio for categories with LLM as a judge
                if measured == total and passed == 0:
                    # This category only has measured tests (no LLM judge)
                    avg_score = summary.get("average_score", 0.0)
                    print(f"{category.upper()}: {total} tests measured (avg score: {avg_score:.3f})")
                else:
                    # This category has LLM judge evaluation
                    avg_score = summary.get("average_score", 0.0)
                    print(f"{category.upper()}: {passed}/{total} tests (avg score: {avg_score:.3f})")

            # Show detailed metrics if evaluator supports it
            if category in self.evaluator_classes:
                try:
                    evaluator = self._get_evaluator(category)
                    detailed_summary = evaluator.format_detailed_summary()
                    if detailed_summary:
                        print(detailed_summary)
                except Exception:
                    # Skip detailed summary if evaluator can't be instantiated
                    pass

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
    parser.add_argument("--output", type=str, help="Output file path for JSON results")
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate an HTML report after evaluation",
    )
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
        saved_results_path = manager.save_results(output_path)

        # Print summary
        manager.print_summary()

        # Generate HTML report if requested
        if args.html_report:
            if saved_results_path:
                print("\nGenerating HTML report...")
                try:
                    # Lazy import to avoid circular dependency issues
                    from rag_evaluation.report_generator import generate_html_report

                    report_path = generate_html_report(saved_results_path)
                    print(f"✅ HTML report generated: {report_path}")
                except Exception as e:
                    print(f"❌ Error generating HTML report: {e}")
            else:
                print("⚠️ Cannot generate HTML report, results were not saved.")

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
