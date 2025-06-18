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
        print(f"Average Response Time: {summary['avg_response_time']:.2f}s")
        print(f"Average Efficiency Ratio: {summary['avg_efficiency_ratio']:.1f}x")
        print(f"Average Relevance Score: {summary['avg_relevance_score']:.2f}")

        # Print complexity breakdown
        if "complexity_breakdown" in summary:
            print("\n📋 COMPLEXITY BREAKDOWN:")
            print("-" * 40)
            for complexity, stats in summary["complexity_breakdown"].items():
                print(f"{complexity.upper()}:")
                print(f"  Tests: {stats['passed']}/{stats['total_tests']} passed ({stats['pass_rate']:.1%})")
                print(f"  Avg Response Time: {stats['avg_response_time']:.2f}s")
                print(f"  Avg Efficiency: {stats['avg_efficiency_ratio']:.1f}x")

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
        if summary["overall_pass_rate"] >= 0.8:
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
