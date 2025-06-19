#!/usr/bin/env python3
"""
Performance Comparison Runner for Isschat

This script runs performance comparison tests to measure Isschat's efficiency
against human benchmarks across different complexity levels.
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_evaluation.config.evaluation_config import EvaluationConfig
from rag_evaluation.evaluators.performance_comparison_evaluator import PerformanceComparisonEvaluator


def main():
    """Main entry point for performance comparison"""
    parser = argparse.ArgumentParser(description="Run Isschat Performance Comparison Tests")
    parser.add_argument(
        "--complexity", choices=["easy", "intermediate", "hard"], help="Run tests for specific complexity level only"
    )
    parser.add_argument("--output", type=str, help="Output file path for results (JSON format)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    try:
        # Initialize configuration
        config = EvaluationConfig()

        # Initialize evaluator
        evaluator = PerformanceComparisonEvaluator(config)

        print("🚀 Isschat Performance Comparison System")
        print("=" * 60)
        print("📊 Comparing Isschat vs Human performance")
        if args.complexity:
            print(f"🎯 Running {args.complexity} complexity tests only")
        print()

        # Run evaluation
        results = evaluator.evaluate_performance_comparison(complexity_filter=args.complexity)

        # Print final summary
        print("\n" + "=" * 60)
        print("📈 FINAL PERFORMANCE COMPARISON SUMMARY")
        print("=" * 60)

        summary = results["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['total_passed']} ({summary['overall_pass_rate']:.1%})")
        print(f"Failed: {summary['total_failed']}")

        print("\n⏱️  PERFORMANCE METRICS:")
        print("-" * 40)
        print(f"Average Response Time: {summary['avg_response_time']:.2f}s")
        print(f"Average Efficiency Ratio: {summary['avg_efficiency_ratio']:.1f}x")

        # Print quality comparison summary
        if "quality_comparison" in summary:
            quality = summary["quality_comparison"]
            total_comparisons = quality["total_comparisons"]

            print("\n🎯 QUALITY COMPARISON WITH HUMAN:")
            print("-" * 40)
            if total_comparisons > 0:
                print(
                    "Better than human: "
                    f"{quality['better_than_human']} "
                    f"({quality['better_than_human'] / total_comparisons:.1%})"
                )
                print(
                    f"Equal to human: {quality['equal_to_human']} ({quality['equal_to_human'] / total_comparisons:.1%})"
                )
                print(
                    "Worse than human: "
                    f"{quality['worse_than_human']} "
                    f"({quality['worse_than_human'] / total_comparisons:.1%})"
                )
                print(f"Total comparisons: {total_comparisons}")
            else:
                print("No quality comparisons available (no human responses provided)")

        # Print complexity breakdown
        if "complexity_breakdown" in summary:
            print("\n📋 COMPLEXITY BREAKDOWN:")
            print("-" * 40)
            for complexity, stats in summary["complexity_breakdown"].items():
                print(f"\n{complexity.upper()}:")
                print(f"  Tests: {stats['passed']}/{stats['total_tests']} passed ({stats['pass_rate']:.1%})")
                print("  Performance:")
                print(f"    Avg Response Time: {stats['avg_response_time']:.2f}s")
                print(f"    Avg Efficiency: {stats['avg_efficiency_ratio']:.1f}x")

                if "quality_stats" in stats:
                    quality = stats["quality_stats"]
                    total = sum(quality.values())
                    if total > 0:
                        print("  Quality vs Human:")
                        print(
                            f"    Better: {quality['better_than_human']} ({quality['better_than_human'] / total:.1%})"
                        )
                        print(f"    Equal: {quality['equal_to_human']} ({quality['equal_to_human'] / total:.1%})")
                        print(f"    Worse: {quality['worse_than_human']} ({quality['worse_than_human'] / total:.1%})")

        # Save results if output specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert results to serializable format
            results_dict = {
                "timestamp": results["timestamp"],
                "category": results["category"],
                "summary": results["summary"],
                "results": [
                    {
                        "test_id": r.test_id,
                        "test_name": r.test_name,
                        "question": r.question,
                        "isschat_response": r.response,  # Clear label for Isschat's answer
                        "passed": r.passed,
                        "score": r.score,
                        "evaluation_details": r.evaluation_details,
                        "metadata": r.metadata,
                    }
                    for r in results["results"]
                ],
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=2)

            print(f"\n💾 Results saved to: {output_path}")

        # Exit with appropriate code
        quality_threshold = 0.5  # 50% des métriques doivent être meilleures ou égales à l'humain
        if "quality_comparison" in summary:
            quality = summary["quality_comparison"]
            total = quality["total_comparisons"]
            if total > 0:
                quality_ratio = (quality["better_than_human"] + quality["equal_to_human"]) / total
            else:
                quality_ratio = 1.0  # Si pas de comparaison possible, on considère que c'est passé
        else:
            quality_ratio = 1.0

        if summary["overall_pass_rate"] >= 0.8 and quality_ratio >= quality_threshold:
            print("\n✅ Performance comparison completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  Performance comparison completed with some failures")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error during performance comparison: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
