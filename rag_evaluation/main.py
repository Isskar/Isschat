#!/usr/bin/env python3
"""
Main entry point for Isschat Evaluation System
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rag_evaluation.config.evaluation_config import EvaluationConfig
from rag_evaluation.core.base_evaluator import TestCase, TestCategory
from rag_evaluation.evaluators.robustness_evaluator import RobustnessEvaluator
from rag_evaluation.evaluators.conversational_evaluator import ConversationalEvaluator


class EvaluationManager:
    """Main evaluation manager orchestrating all test categories"""

    def __init__(self, config: EvaluationConfig):
        """Initialize evaluation manager"""
        self.config = config
        self.evaluators = {
            TestCategory.ROBUSTNESS: RobustnessEvaluator(config),
            TestCategory.CONVERSATIONAL: ConversationalEvaluator(config),
        }
        self.results = {}

    def load_test_cases(self, category: TestCategory) -> List[TestCase]:
        """Load test cases for a specific category"""
        try:
            dataset_path = self.config.get_dataset_path(category.value)

            if not dataset_path.exists():
                print(f"⚠️  Dataset file not found: {dataset_path}")
                return []

            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            test_cases = []
            for item in data:
                test_case = TestCase.from_dict(item)
                test_cases.append(test_case)

            print(f"✅ Loaded {len(test_cases)} test cases for {category.value}")
            return test_cases

        except Exception as e:
            print(f"❌ Error loading test cases for {category.value}: {e}")
            return []

    def run_category_evaluation(self, category: TestCategory) -> Dict[str, Any]:
        """Run evaluation for a specific category"""
        print(f"\n🔍 Starting {category.value} evaluation...")

        # Load test cases
        test_cases = self.load_test_cases(category)
        if not test_cases:
            return {"category": category.value, "results": [], "summary": {}}

        # Get evaluator
        evaluator = self.evaluators[category]

        # Run evaluation
        results = evaluator.evaluate_batch(test_cases)

        # Get summary
        summary = evaluator.get_summary_stats()

        print(
            f"✅ Completed {category.value} evaluation: {summary.get('passed', 0)}/{summary.get('total_tests', 0)} passed"  # noqa: E501
        )

        return {"category": category.value, "results": [r.to_dict() for r in results], "summary": summary}

    def run_full_evaluation(self, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run full evaluation across all or specified categories"""
        print("🚀 Starting Isschat Evaluation System")
        print("=" * 60)

        # Determine which categories to run
        if categories:
            selected_categories = [TestCategory(cat) for cat in categories if cat in [c.value for c in TestCategory]]
        else:
            selected_categories = list(TestCategory)

        # Filter for CI mode if enabled
        if self.config.ci_mode:
            selected_categories = [cat for cat in selected_categories if self.config.is_ci_category(cat.value)]
            print(f"🔧 CI Mode: Running only {[cat.value for cat in selected_categories]}")

        evaluation_results = {}
        overall_stats = {
            "total_tests": 0,
            "total_passed": 0,
            "total_failed": 0,
            "total_errors": 0,
            "category_results": {},
        }

        # Run evaluations
        for category in selected_categories:
            try:
                category_result = self.run_category_evaluation(category)
                evaluation_results[category.value] = category_result

                # Update overall stats
                summary = category_result["summary"]
                overall_stats["total_tests"] += summary.get("total_tests", 0)
                overall_stats["total_passed"] += summary.get("passed", 0)
                overall_stats["total_failed"] += summary.get("failed", 0)
                overall_stats["total_errors"] += summary.get("errors", 0)
                overall_stats["category_results"][category.value] = summary

            except Exception as e:
                print(f"❌ Error in {category.value} evaluation: {e}")
                evaluation_results[category.value] = {
                    "category": category.value,
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
                "categories_run": [cat.value for cat in selected_categories],
                "thresholds": {
                    "robustness": self.config.robustness_threshold
                    if not self.config.ci_mode
                    else self.config.robustness_ci_threshold,
                    "conversational": self.config.conversational_threshold,
                    "overall": self.config.overall_threshold,
                },
            },
            "overall_stats": overall_stats,
            "category_results": evaluation_results,
        }

        return self.results

    def check_thresholds(self) -> bool:
        """Check if evaluation results meet configured thresholds"""
        if not self.results:
            return False

        overall_stats = self.results.get("overall_stats", {})
        overall_pass_rate = overall_stats.get("overall_pass_rate", 0.0)

        # Check overall threshold
        meets_overall = overall_pass_rate >= self.config.overall_threshold

        # Check category-specific thresholds
        category_results = overall_stats.get("category_results", {})
        category_checks = {}

        for category, summary in category_results.items():
            if "error" in summary:
                category_checks[category] = False
                continue

            pass_rate = summary.get("pass_rate", 0.0)
            threshold = self.config.get_threshold(category)
            category_checks[category] = pass_rate >= threshold

        all_categories_pass = all(category_checks.values()) if category_checks else True

        return meets_overall and all_categories_pass

    def save_results(self, output_path: Optional[Path] = None) -> None:
        """Save evaluation results to file"""
        if not self.results:
            print("⚠️  No results to save")
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
            print("⚠️  No results to summarize")
            return

        overall_stats = self.results.get("overall_stats", {})

        print(f"\n{'=' * 60}")
        print("📊 EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total Tests: {overall_stats.get('total_tests', 0)}")
        print(f"Passed: {overall_stats.get('total_passed', 0)} ({overall_stats.get('overall_pass_rate', 0):.1%})")
        print(f"Failed: {overall_stats.get('total_failed', 0)}")
        print(f"Errors: {overall_stats.get('total_errors', 0)}")

        # Category breakdown
        print("\n📋 CATEGORY BREAKDOWN:")
        print(f"{'-' * 60}")

        category_results = overall_stats.get("category_results", {})
        for category, summary in category_results.items():
            if "error" in summary:
                print(f"{category.upper()}: ERROR - {summary['error']}")
                continue

            total = summary.get("total_tests", 0)
            passed = summary.get("passed", 0)
            pass_rate = summary.get("pass_rate", 0.0)
            threshold = self.config.get_threshold(category)
            status = "✅" if pass_rate >= threshold else "❌"

            print(f"{category.upper()}: {passed}/{total} ({pass_rate:.1%}) {status} (threshold: {threshold:.1%})")

        # Threshold check
        meets_thresholds = self.check_thresholds()
        threshold_status = "✅ PASSED" if meets_thresholds else "❌ FAILED"
        print(f"\n🎯 THRESHOLD CHECK: {threshold_status}")

        # Add document relevance test results to the summary
        print("\n📑 DOCUMENT RELEVANCE TEST RESULTS:")
        print(f"{'-' * 60}")

        # Iterate through all test results to find document relevance evaluations
        for category, category_data in self.results.get("category_results", {}).items():
            for result in category_data.get("results", []):
                if "evaluation_details" in result and "document_relevance" in result["evaluation_details"]:
                    doc_relevance = result["evaluation_details"]["document_relevance"]
                    test_id = result["test_id"]
                    test_name = result["test_name"]
                    expected_count = doc_relevance.get("expected_count", 0)
                    matched_count = doc_relevance.get("matched_count", 0)
                    retrieved_count = doc_relevance.get("retrieved_count", 0)

                    # Display document relevance results for this test
                    status = "✅" if doc_relevance.get("passes_criteria", False) else "❌"
                    print(
                        f"{test_id} - {test_name}: {matched_count}/{expected_count} expected documents found {status}"
                    )

                    # Display retrieved sources if available
                    if "retrieved_sources" in doc_relevance and doc_relevance["retrieved_sources"]:
                        print(f"  Retrieved sources ({retrieved_count}):")
                        for source in doc_relevance["retrieved_sources"]:
                            print(f"  - {source}")
                    print()

        print(f"{'=' * 60}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Isschat Evaluation System")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["robustness", "conversational"],
        help="Specific categories to evaluate",
    )
    parser.add_argument("--ci", action="store_true", help="Run in CI mode")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--config", type=str, help="Configuration file path")

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
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
