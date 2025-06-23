"""
LLM-based judge for evaluating Isschat responses
"""

import json
import logging
import re
from typing import Dict, Any

from src.core.config import get_config
from langchain_openai import ChatOpenAI
from langchain_core.utils.utils import convert_to_secret_str

logger = logging.getLogger(__name__)


class LLMJudge:
    """LLM-based evaluation judge for Isschat responses"""

    def __init__(self, config: Any):
        """Initialize LLM judge with configuration"""
        self.config = config

        # Get API key from config
        try:
            app_config = get_config()
            api_key = convert_to_secret_str(app_config.openrouter_api_key)
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in configuration")
        except Exception as e:
            raise ValueError(f"Failed to get API key: {e}")

        # Configure logging to suppress httpx INFO logs
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=config.judge_model,
            temperature=config.judge_temperature,
            max_tokens=512,  # Increased token limit
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )

    def evaluate_conversational(self, question: str, response: str, expected: str, context: str = "") -> Dict[str, Any]:
        """Evaluate conversational test response"""
        prompt = self.CONVERSATIONAL_PROMPT.format(
            context=context, question=question, response=response, expected=expected
        )
        return self._evaluate_with_prompt(prompt)

    def evaluate_performance(
        self, question: str, response: str, expected: str, response_time: float, complexity: str = "medium"
    ) -> Dict[str, Any]:
        """Evaluate performance test response"""
        prompt = self.PERFORMANCE_PROMPT.format(
            question=question, response=response, response_time=response_time, expected=expected, complexity=complexity
        )
        return self._evaluate_with_prompt(prompt)

    def evaluate_feedback(
        self, question: str, response: str, expected: str, feedback_type: str = "general"
    ) -> Dict[str, Any]:
        """Evaluate feedback test response"""
        prompt = self.FEEDBACK_PROMPT.format(
            question=question, response=response, feedback_type=feedback_type, expected=expected
        )
        return self._evaluate_with_prompt(prompt)

    def _evaluate_with_prompt(self, prompt: str) -> Dict[str, Any]:
        """Evaluate using the given prompt"""
        try:
            result = self.llm.invoke(prompt).content.strip()

            # FIXME: Clean up JSON response from markdown artifacts
            cleaned_result = self._clean_json_response(result)

            # Try to parse as JSON
            try:
                evaluation = json.loads(cleaned_result)

                # Validate required fields
                if not all(key in evaluation for key in ["score", "reasoning", "passes_criteria"]):
                    raise ValueError("Missing required fields in evaluation")

                # Ensure score is in valid range
                score = float(evaluation["score"])
                if not 0.0 <= score <= 1.0:
                    score = max(0.0, min(1.0, score))
                    evaluation["score"] = score

                return evaluation

            except (json.JSONDecodeError, ValueError):
                # Fallback: try to extract score from text
                return self._fallback_evaluation(result)

        except Exception as e:
            return {"score": 0.0, "reasoning": f"Evaluation error: {str(e)}", "passes_criteria": False}

    def _clean_json_response(self, response: str) -> str:
        """Extracts a JSON object from a string, even if it's embedded in other text.

        Args:
            response: The string containing the JSON object.

        Returns:
            The extracted JSON object as a string, or an empty dict string if not found.
        """
        # Find the start of the JSON object
        start_brace_index = response.find("{")
        if start_brace_index == -1:
            return "{}"  # No JSON object found

        # Find the end of the JSON object by matching braces
        open_braces = 0
        for i in range(start_brace_index, len(response)):
            if response[i] == "{":
                open_braces += 1
            elif response[i] == "}":
                open_braces -= 1
                if open_braces == 0:
                    # Found the matching closing brace
                    end_brace_index = i
                    return response[start_brace_index : end_brace_index + 1]

        return "{}"  # No valid JSON object found

    def _fallback_evaluation(self, result: str) -> Dict[str, Any]:
        """Fallback evaluation when JSON parsing fails - uses simplified LLM retry"""
        try:
            # Try a simplified prompt to get just a score
            simple_prompt = f"""Rate this evaluation result from 0.0 to 1.0: "{result[:100]}" Respond with only a number between 0.0 and 1.0:"""  # noqa

            simple_result = self.llm.invoke(simple_prompt).content.strip()

            # Extract numeric score

            score_match = re.search(r"(\d+\.?\d*)", simple_result)
            if score_match:
                score = float(score_match.group(1))
                # Ensure score is in valid range
                score = max(0.0, min(1.0, score))
            else:
                score = 0.5  # Default if no number found

        except Exception:
            # Ultimate fallback - neutral score
            score = 0.0

        passes = score >= 0.5

        return {
            "score": score,
            "reasoning": f"Fallback evaluation - simplified LLM scoring: {score}",
            "passes_criteria": passes,
        }

    def health_check(self) -> bool:
        """Check if LLM judge is working properly"""
        try:
            test_prompt = "Respond with: {'score': 1.0, 'reasoning': 'test', 'passes_criteria': true}"
            result = self.llm.invoke(test_prompt).content.strip()
            return "score" in result.lower()
        except Exception:
            return False
