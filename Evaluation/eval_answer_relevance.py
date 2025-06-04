#!/usr/bin/env python3
"""
Automated Evaluation System for Isschat
Evaluates chatbot responses against expected behaviors using LLM as judge
"""

import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add the src directory to Python path to import Isschat modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from help_desk import HelpDesk
from config import get_config
from langchain_openai import ChatOpenAI
from langchain_core.utils.utils import convert_to_secret_str

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Data class to hold evaluation results"""

    question: str
    isschat_response: str
    expected_answer: str
    evaluation: str
    response_time: float
    error: str = ""


class IsschatEvaluator:
    """Main evaluator class for Isschat responses"""

    def __init__(self, dataset_path: str = "question_dataset.json"):
        """Initialize the evaluator with dataset path"""
        self.dataset_path = Path(__file__).parent / dataset_path
        self.results: List[EvaluationResult] = []

        # Initialize Isschat
        logger.info("Initializing Isschat...")
        try:
            self.isschat = HelpDesk(new_db=False)  # Use existing DB for faster startup
            logger.info("Isschat initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Isschat: {e}")
            raise

        # Initialize LLM judge
        logger.info("Initializing LLM judge...")
        try:
            self.judge_llm = self._initialize_judge_llm()
            logger.info("LLM judge initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM judge: {e}")
            raise

    def _initialize_judge_llm(self) -> ChatOpenAI:
        """Initialize the LLM that will act as judge"""
        config = get_config()
        api_key = convert_to_secret_str(config.openrouter_api_key)

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in configuration")

        # Use a more capable model for evaluation
        judge_llm = ChatOpenAI(
            model_name="anthropic/claude-3.5-sonnet",  # More capable model for evaluation
            temperature=0.0,  # Deterministic evaluation
            max_tokens=100,  # Short responses for binary evaluation
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )
        return judge_llm

    def load_dataset(self) -> List[Dict]:
        """Load the question dataset from JSON file"""
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            logger.info(f"Loaded {len(dataset)} questions from {self.dataset_path}")
            return dataset
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {self.dataset_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in dataset file: {e}")
            raise

    def query_isschat(self, question: str) -> Tuple[str, float]:
        """Query Isschat and measure response time"""
        start_time = time.time()
        try:
            # Use verbose=False to reduce log noise during evaluation
            response, sources = self.isschat.retrieval_qa_inference(question, verbose=False)
            response_time = time.time() - start_time

            # Combine response and sources for complete answer
            full_response = f"{response}\n\nSources: {sources}" if sources else response

            logger.info(f"Isschat responded in {response_time:.2f}s")
            return full_response, response_time

        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Error querying Isschat: {e}")
            return f"ERROR: {str(e)}", response_time

    def evaluate_response(self, question: str, isschat_response: str, expected_answer: str) -> str:
        """Use LLM judge to evaluate the response"""

        # Create evaluation prompt
        evaluation_prompt = f"""Tu es un évaluateur expert pour un chatbot d'entreprise nommé Isschat.

QUESTION POSÉE: {question}

RÉPONSE D'ISSCHAT: {isschat_response}

COMPORTEMENT ATTENDU: {expected_answer}

INSTRUCTIONS:
- Évalue si la réponse d'Isschat correspond au comportement attendu
- Considère le contexte d'un chatbot d'entreprise basé sur Confluence
- Réponds UNIQUEMENT par "rather good" ou "rather bad"
- "rather good" = la réponse respecte globalement le comportement attendu
- "rather bad" = la réponse ne respecte pas le comportement attendu

ÉVALUATION:"""

        try:
            # Get evaluation from judge LLM
            evaluation = self.judge_llm.invoke(evaluation_prompt).content.strip().lower()

            # Normalize the response
            if "rather good" in evaluation:
                return "rather good"
            elif "rather bad" in evaluation:
                return "rather bad"
            else:
                logger.warning(f"Unexpected evaluation response: {evaluation}")
                return "rather bad"  # Default to bad if unclear

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return "rather bad"  # Default to bad on error

    def run_evaluation(self) -> List[EvaluationResult]:
        """Run the complete evaluation process"""
        logger.info("Starting evaluation process...")

        # Load dataset
        dataset = self.load_dataset()

        # Process each question
        for i, item in enumerate(dataset, 1):
            question = item["question"]
            expected_answer = item["expected_answer"]

            logger.info(f"Processing question {i}/{len(dataset)}: {question[:50]}...")

            # Query Isschat
            isschat_response, response_time = self.query_isschat(question)

            # Skip evaluation if there was an error
            if isschat_response.startswith("ERROR:"):
                evaluation = "rather bad"
                error = isschat_response
            else:
                # Evaluate response
                evaluation = self.evaluate_response(question, isschat_response, expected_answer)
                error = ""

            # Store result
            result = EvaluationResult(
                question=question,
                isschat_response=isschat_response,
                expected_answer=expected_answer,
                evaluation=evaluation,
                response_time=response_time,
                error=error,
            )
            self.results.append(result)

            logger.info(f"Question {i} evaluated as: {evaluation}")

            # Add small delay to avoid overwhelming the system
            time.sleep(1)

        logger.info("Evaluation process completed")
        return self.results

    def save_results(self, output_path: Optional[str] = None) -> None:
        """Save evaluation results back to JSON file"""
        if output_path is None:
            output_path = self.dataset_path

        # Convert results to the expected JSON format
        output_data = []
        for result in self.results:
            output_data.append(
                {
                    "question": result.question,
                    "isschat_response": result.isschat_response,
                    "expected_answer": result.expected_answer,
                    "evaluation": result.evaluation,
                }
            )

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    def print_summary(self) -> None:
        """Print evaluation summary"""
        if not self.results:
            logger.warning("No results to summarize")
            return

        total_questions = len(self.results)
        good_responses = sum(1 for r in self.results if r.evaluation == "rather good")
        bad_responses = sum(1 for r in self.results if r.evaluation == "rather bad")
        avg_response_time = sum(r.response_time for r in self.results) / total_questions
        errors = sum(1 for r in self.results if r.error)

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Questions: {total_questions}")
        print(f"Rather Good: {good_responses} ({good_responses / total_questions * 100:.1f}%)")
        print(f"Rather Bad: {bad_responses} ({bad_responses / total_questions * 100:.1f}%)")
        print(f"Errors: {errors}")
        print(f"Average Response Time: {avg_response_time:.2f}s")
        print("=" * 60)

        # Print detailed results
        print("\nDETAILED RESULTS:")
        print("-" * 60)
        for i, result in enumerate(self.results, 1):
            status_icon = "✅" if result.evaluation == "rather good" else "❌"
            print(f"{i:2d}. {status_icon} {result.evaluation.upper()}")
            print(f"    Q: {result.question[:80]}...")
            if result.error:
                print(f"    ERROR: {result.error}")
            print(f"    Time: {result.response_time:.2f}s")
            print()


def main():
    """Main execution function"""
    try:
        # Initialize evaluator
        evaluator = IsschatEvaluator()

        # Run evaluation
        evaluator.run_evaluation()

        # Save results
        evaluator.save_results()

        # Print summary
        evaluator.print_summary()

        logger.info("Evaluation completed successfully!")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
