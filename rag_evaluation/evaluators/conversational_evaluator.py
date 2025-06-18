"""
Conversational evaluator for testing context continuity and multi-turn conversations
"""

from rag_evaluation.core import BaseEvaluator

CONVERSATIONAL_PROMPT = """You are an expert evaluator for conversational AI systems.

CONVERSATION CONTEXT: {context}
CURRENT QUESTION: {question}
ISSCHAT RESPONSE: {response}
EXPECTED BEHAVIOR: {expected}

EVALUATION CRITERIA:
- Does the response globally maintain conversation context?
- Does it demonstrate conversational memory?
- Is the response coherent with the conversation flow?

Respond with a JSON object containing:
- "score": float between 0.0 and 1.0
- "reasoning": brief explanation of the score
- "passes_criteria": boolean indicating if it meets expectations

EVALUATION:"""


class ConversationalEvaluator(BaseEvaluator):
    """Evaluator for conversational history tests : NOT YET IMPLEMENTED"""

    def get_category(self) -> str:
        """Get the category this evaluator handles"""
        return "conversational"

    def evaluate_single(self, test_case):
        """Placeholder implementation"""
        import logging
        from rag_evaluation.core.base_evaluator import EvaluationResult, EvaluationStatus

        logger = logging.getLogger(__name__)

        # Log conversational test details
        test_type = test_case.metadata.get("test_type", "conversational")
        logger.info(f"Test {test_case.test_id} ({test_type}): SKIPPED - Not yet implemented")
        logger.info("LLM Judge Comment: Conversational evaluator not yet implemented")

        return EvaluationResult(
            test_id=test_case.test_id,
            category=test_case.category,
            test_name=test_case.test_name,
            question=test_case.question,
            response="Not implemented",
            expected_behavior=test_case.expected_behavior,
            status=EvaluationStatus.SKIPPED,
            score=0.0,
        )
