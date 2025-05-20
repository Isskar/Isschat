#!/usr/bin/env python3
"""
Command-line interface for the RAG evaluation system.

This module provides a command-line entry point to run
the RAG system evaluation.
"""

import argparse
import sys
from datetime import datetime

from config_evaluation import get_config, DatabaseType
from manager import EvaluationManager, run_ci_evaluation, verify_ci_evaluation


def main():
    """
    Main entry point for RAG evaluation.
    """
    parser = argparse.ArgumentParser(description="RAG Evaluation System")
    parser.add_argument("--ci", action="store_true", help="Run in CI mode with mock database")
    parser.add_argument("--dataset", help="Path to the evaluation dataset")
    parser.add_argument("--output", help="Output path for results")
    parser.add_argument(
        "--report",
        help="Output path for report",
        default=f"Isschat/data/evaluation_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.md",
    )
    parser.add_argument("--db-mock", action="store_true", help="Use a mock database")
    parser.add_argument(
        "--response-threshold", type=float, default=70.0, help="Minimum threshold for correct responses"
    )
    parser.add_argument(
        "--document-threshold", type=float, default=70.0, help="Minimum threshold for relevant documents"
    )
    args = parser.parse_args()

    if args.ci:
        print("Running in CI mode with mock database")
        results = run_ci_evaluation()
        success = verify_ci_evaluation(results)
        # For CI, return an error code if evaluation fails
        if not success:
            sys.exit(1)
    else:
        # Normal mode with default configuration
        config = get_config()

        # Apply command line arguments
        if args.dataset:
            config.evaluation_dataset_path = args.dataset
        if args.output:
            config.result_output_path = args.output
        if args.report:
            config.report_output_path = args.report
        if args.db_mock:
            config.database.type = DatabaseType.MOCK

        evaluator = EvaluationManager(config)

        print("Starting evaluation...")
        results = evaluator.run_evaluation()

        # Save results
        evaluator.save_results(results)

        # Generate report
        evaluator.generate_markdown_report(results)

        # Display summary
        print("\n=== Evaluation Summary ===")
        print(f"Relevant documents: {results.percentage_good_documents:.2f}%")
        print(f"Correct responses: {results.percentage_good_responses:.2f}%")

        # Check if results meet thresholds
        response_threshold = args.response_threshold
        document_threshold = args.document_threshold

        if (
            results.percentage_good_responses >= response_threshold
            and results.percentage_good_documents >= document_threshold
        ):
            print("✅ Evaluation successful!")
        else:
            print("❌ Evaluation below thresholds:")
            if results.percentage_good_responses < response_threshold:
                print(f"   - Responses: {results.percentage_good_responses:.2f}% (threshold: {response_threshold}%)")
            if results.percentage_good_documents < document_threshold:
                print(f"   - Documents: {results.percentage_good_documents:.2f}% (threshold: {document_threshold}%)")


if __name__ == "__main__":
    main()
