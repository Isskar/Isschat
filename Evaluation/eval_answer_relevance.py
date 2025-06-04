#!/usr/bin/env python3
"""
Automated Evaluation System for Isschat - Concise OOP Implementation
"""

import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent / "src"))

from help_desk import HelpDesk
from config import get_config
from langchain_openai import ChatOpenAI
from langchain_core.utils.utils import convert_to_secret_str

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Evaluation result data structure"""

    question: str
    isschat_response: str
    expected_answer: str
    evaluation: str
    response_time: float
    error: str = ""

    @property
    def is_good(self) -> bool:
        return self.evaluation == "rather good"

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "isschat_response": self.isschat_response,
            "expected_answer": self.expected_answer,
            "evaluation": self.evaluation,
        }


class LLMJudge:
    """LLM-based evaluation judge"""

    EVALUATION_PROMPT = """Tu es un évaluateur expert pour un chatbot d'entreprise nommé Isschat.

QUESTION POSÉE: {question}
RÉPONSE D'ISSCHAT: {response}
COMPORTEMENT ATTENDU: {expected}

INSTRUCTIONS:
- Évalue si la réponse d'Isschat correspond au comportement attendu
- Réponds UNIQUEMENT par "rather good" ou "rather bad"

ÉVALUATION:"""

    def __init__(self):
        config = get_config()
        api_key = convert_to_secret_str(config.openrouter_api_key)
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found")

        self.llm = ChatOpenAI(
            model_name="anthropic/claude-3.5-sonnet",
            temperature=0.0,
            max_tokens=100,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )

    def evaluate(self, question: str, response: str, expected: str) -> str:
        """Evaluate response quality"""
        try:
            prompt = self.EVALUATION_PROMPT.format(question=question, response=response, expected=expected)
            result = self.llm.invoke(prompt).content.strip().lower()
            return "rather good" if "rather good" in result else "rather bad"
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return "rather bad"


class IsschatQuerier:
    """Handles Isschat queries and timing"""

    def __init__(self):
        self.isschat = HelpDesk(new_db=False)
        logger.info("Isschat initialized")

    def query(self, question: str) -> tuple[str, float]:
        """Query Isschat with timing"""
        start_time = time.time()
        try:
            response, sources = self.isschat.retrieval_qa_inference(question, verbose=False)
            response_time = time.time() - start_time
            full_response = f"{response}\n\nSources: {sources}" if sources else response
            return full_response, response_time
        except Exception as e:
            response_time = time.time() - start_time
            return f"ERROR: {str(e)}", response_time


class EvaluationSummary:
    """Handles evaluation summary and reporting"""

    def __init__(self, results: List[EvaluationResult]):
        self.results = results

    def print_summary(self) -> None:
        """Print comprehensive evaluation summary"""
        if not self.results:
            logger.warning("No results to summarize")
            return

        total = len(self.results)
        good = sum(1 for r in self.results if r.is_good)
        bad = total - good
        avg_time = sum(r.response_time for r in self.results) / total
        errors = sum(1 for r in self.results if r.error)

        print(f"\n{'=' * 60}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total Questions: {total}")
        print(f"Rather Good: {good} ({good / total * 100:.1f}%)")
        print(f"Rather Bad: {bad} ({bad / total * 100:.1f}%)")
        print(f"Errors: {errors}")
        print(f"Average Response Time: {avg_time:.2f}s")
        print(f"{'=' * 60}")

        print("\nDETAILED RESULTS:")
        print(f"{'-' * 60}")
        for i, result in enumerate(self.results, 1):
            icon = "✅" if result.is_good else "❌"
            print(f"{i:2d}. {icon} {result.evaluation.upper()}")
            print(f"    Q: {result.question[:80]}...")
            if result.error:
                print(f"    ERROR: {result.error}")
            print(f"    Time: {result.response_time:.2f}s\n")


class IsschatEvaluator:
    """Main evaluation orchestrator"""

    def __init__(self, dataset_path: str = "question_dataset.json"):
        self.dataset_path = Path(__file__).parent / dataset_path
        self.results: List[EvaluationResult] = []

        # Initialize components
        self.querier = IsschatQuerier()
        self.judge = LLMJudge()
        logger.info("Evaluator initialized")

    def load_dataset(self) -> List[Dict]:
        """Load evaluation dataset"""
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            logger.info(f"Loaded {len(dataset)} questions")
            return dataset
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Dataset loading failed: {e}")
            raise

    def evaluate_single(self, item: Dict, index: int, total: int) -> EvaluationResult:
        """Evaluate a single question"""
        question = item["question"]
        expected = item["expected_answer"]

        logger.info(f"Processing {index}/{total}: {question[:50]}...")

        # Query and evaluate
        response, response_time = self.querier.query(question)

        if response.startswith("ERROR:"):
            evaluation, error = "rather bad", response
        else:
            evaluation = self.judge.evaluate(question, response, expected)
            error = ""

        result = EvaluationResult(
            question=question,
            isschat_response=response,
            expected_answer=expected,
            evaluation=evaluation,
            response_time=response_time,
            error=error,
        )

        logger.info(f"Result: {evaluation}")
        return result

    def run_evaluation(self) -> List[EvaluationResult]:
        """Execute complete evaluation process"""
        logger.info("Starting evaluation...")
        dataset = self.load_dataset()

        for i, item in enumerate(dataset, 1):
            result = self.evaluate_single(item, i, len(dataset))
            self.results.append(result)
            time.sleep(1)  # Rate limiting

        logger.info("Evaluation completed")
        return self.results

    def save_results(self, output_path: Optional[str] = None) -> None:
        """Save results to JSON"""
        path = output_path or self.dataset_path
        try:
            data = [result.to_dict() for result in self.results]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {path}")
        except Exception as e:
            logger.error(f"Save failed: {e}")
            raise

    def get_summary(self) -> EvaluationSummary:
        """Get evaluation summary"""
        return EvaluationSummary(self.results)


def main():
    """Main execution"""
    try:
        evaluator = IsschatEvaluator()
        evaluator.run_evaluation()
        evaluator.save_results()
        evaluator.get_summary().print_summary()
        logger.info("Evaluation completed successfully!")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
