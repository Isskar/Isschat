"""
LLM-based judge for evaluating Isschat responses
"""

from typing import Dict, Any

from src.config.settings import get_config
from langchain_openai import ChatOpenAI
from langchain_core.utils.utils import convert_to_secret_str


class LLMJudge:
    """LLM-based evaluation judge for Isschat responses"""

    def __init__(self, config: Any):
        """Initialize LLM judge with configuration"""
        self.config = config

        # Get API key and base URL from main Isschat config
        try:
            isschat_config = get_config()
            api_key = convert_to_secret_str(isschat_config.openrouter_api_key)
            base_url = isschat_config.openrouter_base_url
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in configuration")
            if not base_url:
                raise ValueError("OPENROUTER_BASE_URL not found in configuration")
        except Exception as e:
            raise ValueError(f"Failed to get API configuration: {e}")

        # Configure logging to suppress httpx INFO logs
        import logging

        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=config.judge_model,
            temperature=config.judge_temperature,
            max_tokens=config.judge_max_tokens,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.helicone.ai/api/v1",
            default_headers={"Helicone-Auth": f"Bearer {get_config().helicone_api_key}"},
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
            import json

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
        """Clean JSON response from markdown formatting and other artifacts"""
        import re
        import json

        response = re.sub(r"```json\s*", "", response)
        response = re.sub(r"```\s*$", "", response)
        response = response.strip()
        start_idx = response.find("{")
        end_idx = response.rfind("}")

        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            response = response[start_idx : end_idx + 1]

        # FIXME: Handle common JSON formatting issues
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError:
            # Add missing closing quote and brace if needed
            if response.count('"') % 2 == 1:
                response += '"'
            if response.count("{") > response.count("}"):
                response += "}"
            return response

    def _fallback_evaluation(self, result: str) -> Dict[str, Any]:
        """Fallback evaluation when JSON parsing fails - uses simplified LLM retry"""
        try:
            # Try a simplified prompt to get just a score
            simple_prompt = f"""Rate this evaluation result from 0.0 to 1.0: "{result[:100]}" Respond with only a number between 0.0 and 1.0:"""  # noqa

            simple_result = self.llm.invoke(simple_prompt).content.strip()

            # Extract numeric score
            import re

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
