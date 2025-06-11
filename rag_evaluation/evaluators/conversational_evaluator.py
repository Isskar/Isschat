"""
Conversational evaluator for testing context continuity and multi-turn conversations
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.base_evaluator import BaseEvaluator, EvaluationResult, TestCase, TestCategory, EvaluationStatus
from core.isschat_client import IsschatClient
from core.llm_judge import LLMJudge


class ConversationalEvaluator(BaseEvaluator):
    """Evaluator for conversational history tests"""

    def __init__(self, config: Any):
        """Initialize conversational evaluator"""
        super().__init__(config)
        self.isschat_client = IsschatClient(conversation_memory=True)
        self.llm_judge = LLMJudge(config)
        self.conversation_state = {}  # Track conversation state across tests

    def get_category(self) -> TestCategory:
        """Get the category this evaluator handles"""
        return TestCategory.CONVERSATIONAL

    def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single conversational test case"""
        try:
            # Handle conversation context
            context_str = self._prepare_conversation_context(test_case)

            # Query Isschat with conversation context
            if test_case.conversation_context:
                response, response_time, sources = self.isschat_client.query_with_conversation_context(
                    test_case.question, test_case.conversation_context
                )
            else:
                response, response_time, sources = self.isschat_client.query(test_case.question)

            # Store response for potential use in subsequent tests
            self._store_conversation_state(test_case, response)

            # Check for errors
            if response.startswith("ERROR:"):
                return EvaluationResult(
                    test_id=test_case.test_id,
                    category=test_case.category,
                    test_name=test_case.test_name,
                    question=test_case.question,
                    response=response,
                    expected_behavior=test_case.expected_behavior,
                    status=EvaluationStatus.ERROR,
                    score=0.0,
                    response_time=response_time,
                    error_message=response,
                    sources=sources,
                )

            # Evaluate with LLM judge (including conversation context)
            evaluation = self.llm_judge.evaluate_conversational(
                test_case.question, response, test_case.expected_behavior, context_str
            )

            # Perform conversational-specific analysis
            conversational_analysis = self._analyze_conversational_aspects(test_case, response, context_str)

            # Combine evaluation details
            evaluation_details = {**evaluation, **conversational_analysis}

            # Calculate final score incorporating conversational metrics
            final_score = self._calculate_conversational_score(evaluation["score"], conversational_analysis, test_case)

            # Determine status
            status = (
                EvaluationStatus.PASSED
                if evaluation["passes_criteria"] and final_score >= 0.6
                else EvaluationStatus.FAILED
            )

            return EvaluationResult(
                test_id=test_case.test_id,
                category=test_case.category,
                test_name=test_case.test_name,
                question=test_case.question,
                response=response,
                expected_behavior=test_case.expected_behavior,
                status=status,
                score=final_score,
                evaluation_details=evaluation_details,
                response_time=response_time,
                sources=sources,
                metadata=test_case.metadata,
            )

        except Exception as e:
            return EvaluationResult(
                test_id=test_case.test_id,
                category=test_case.category,
                test_name=test_case.test_name,
                question=test_case.question,
                response="",
                expected_behavior=test_case.expected_behavior,
                status=EvaluationStatus.ERROR,
                score=0.0,
                error_message=str(e),
                metadata=test_case.metadata,
            )

    def _prepare_conversation_context(self, test_case: TestCase) -> str:
        """Prepare conversation context string for evaluation"""
        if not test_case.conversation_context:
            return ""

        context_parts = []
        for exchange in test_case.conversation_context:
            question = exchange.get("question", "")
            response = exchange.get("response", "")

            # Replace placeholders with actual stored responses if needed
            if response == "{{PREVIOUS_RESPONSE}}" and self.conversation_state:
                # Get the most recent response
                response = list(self.conversation_state.values())[-1] if self.conversation_state else ""
            elif response.startswith("{{") and response.endswith("}}"):
                # Handle other placeholders
                placeholder = response.strip("{}")
                response = self.conversation_state.get(placeholder, response)

            context_parts.append(f"Q: {question}\nA: {response}")

        return "\n\n".join(context_parts)

    def _store_conversation_state(self, test_case: TestCase, response: str):
        """Store conversation state for use in subsequent tests"""
        # Store by test_id for specific reference
        self.conversation_state[test_case.test_id] = response

        # Also store by sequence part if available
        sequence_part = test_case.metadata.get("sequence_part")
        if sequence_part:
            self.conversation_state[f"sequence_{sequence_part}"] = response

    def _analyze_conversational_aspects(self, test_case: TestCase, response: str, context: str) -> Dict[str, Any]:
        """Analyze conversational-specific aspects"""
        analysis = {}
        test_type = test_case.metadata.get("test_type", "")

        # Context continuity analysis
        if test_type == "context_continuity":
            analysis["context_continuity"] = self._analyze_context_continuity(test_case, response, context)

        # Context reference analysis
        elif test_type == "context_reference":
            analysis["context_reference"] = self._analyze_context_reference(test_case, response, context)

        # Memory recall analysis
        elif test_type == "memory_recall":
            analysis["memory_recall"] = self._analyze_memory_recall(test_case, response, context)

        # Pronoun resolution analysis
        elif test_type == "pronoun_resolution":
            analysis["pronoun_resolution"] = self._analyze_pronoun_resolution(test_case, response, context)

        # Topic transition analysis
        elif test_type == "topic_transition":
            analysis["topic_transition"] = self._analyze_topic_transition(test_case, response, context)

        # General conversational flow analysis
        analysis["conversational_flow"] = self._analyze_conversational_flow(test_case, response, context)

        return analysis

    def _analyze_context_continuity(self, test_case: TestCase, response: str, context: str) -> Dict[str, Any]:
        """Analyze context continuity (e.g., 'cite moi l'autre')"""
        response_lower = response.lower()

        # Check if response acknowledges the context
        context_indicators = [
            "autre",
            "également",
            "aussi",
            "en plus",
            "deuxième",
            "second",
            "précédent",
            "mentionné",
            "dit",
            "parlé",
        ]

        acknowledges_context = any(indicator in response_lower for indicator in context_indicators)

        # Check if it provides a different answer than the previous one
        provides_different_answer = len(response.split()) > 5  # Basic heuristic

        # Check for appropriate response structure
        has_appropriate_structure = not any(
            phrase in response_lower
            for phrase in ["je ne comprends pas", "pouvez-vous préciser", "que voulez-vous dire"]
        )

        return {
            "acknowledges_context": acknowledges_context,
            "provides_different_answer": provides_different_answer,
            "has_appropriate_structure": has_appropriate_structure,
            "context_indicators_found": [ind for ind in context_indicators if ind in response_lower],
            "passes": acknowledges_context and provides_different_answer,
        }

    def _analyze_context_reference(self, test_case: TestCase, response: str, context: str) -> Dict[str, Any]:
        """Analyze context reference (e.g., 'qui travaille dessus?')"""
        response_lower = response.lower()
        question_lower = test_case.question.lower()

        # Check for pronoun/reference resolution
        reference_words = ["dessus", "celui-ci", "celle-ci", "cela", "ça", "il", "elle", "ils", "elles"]
        has_reference = any(word in question_lower for word in reference_words)

        # Check if response shows understanding of the reference
        shows_understanding = not any(
            phrase in response_lower
            for phrase in ["de quoi parlez-vous", "que voulez-vous dire", "précisez", "je ne comprends pas"]
        )

        # Check if response is contextually relevant
        is_contextually_relevant = len(response.split()) > 10  # Basic heuristic

        return {
            "has_reference_in_question": has_reference,
            "shows_understanding": shows_understanding,
            "is_contextually_relevant": is_contextually_relevant,
            "passes": shows_understanding and is_contextually_relevant,
        }

    def _analyze_memory_recall(self, test_case: TestCase, response: str, context: str) -> Dict[str, Any]:
        """Analyze memory recall capabilities"""
        response_lower = response.lower()

        # Check if response attempts to recall previous conversation
        recall_indicators = [
            "au début",
            "précédemment",
            "avant",
            "parlé de",
            "mentionné",
            "discuté",
            "conversation",
            "échangé",
            "abordé",
        ]

        attempts_recall = any(indicator in response_lower for indicator in recall_indicators)

        # Check if it provides specific references
        provides_specifics = len(response.split()) > 15  # Longer response suggests detail

        return {
            "attempts_recall": attempts_recall,
            "provides_specifics": provides_specifics,
            "recall_indicators": [ind for ind in recall_indicators if ind in response_lower],
            "passes": attempts_recall,
        }

    def _analyze_pronoun_resolution(self, test_case: TestCase, response: str, context: str) -> Dict[str, Any]:
        """Analyze pronoun resolution"""
        response_lower = response.lower()
        question_lower = test_case.question.lower()

        # Check for pronouns in question
        pronouns = ["il", "elle", "ils", "elles", "celui", "celle", "ceux", "celles"]
        has_pronoun = any(pronoun in question_lower for pronoun in pronouns)

        # Check if response resolves the pronoun appropriately
        resolves_pronoun = not any(
            phrase in response_lower for phrase in ["qui", "de qui parlez-vous", "précisez la personne"]
        )

        # Check if response provides relevant information
        provides_relevant_info = len(response.split()) > 8

        return {
            "has_pronoun_in_question": has_pronoun,
            "resolves_pronoun": resolves_pronoun,
            "provides_relevant_info": provides_relevant_info,
            "passes": resolves_pronoun and provides_relevant_info,
        }

    def _analyze_topic_transition(self, test_case: TestCase, response: str, context: str) -> Dict[str, Any]:
        """Analyze topic transition handling"""
        response_lower = response.lower()
        question_lower = test_case.question.lower()

        # Check for transition indicators in question
        transition_indicators = ["revenons", "retournons", "parlons de", "à propos de"]
        has_transition = any(indicator in question_lower for indicator in transition_indicators)

        # Check if response acknowledges the transition
        acknowledges_transition = any(
            phrase in response_lower for phrase in ["effectivement", "bien sûr", "comme mentionné", "en effet"]
        )

        # Check if response is relevant to the requested topic
        is_topic_relevant = len(response.split()) > 10

        return {
            "has_transition_in_question": has_transition,
            "acknowledges_transition": acknowledges_transition,
            "is_topic_relevant": is_topic_relevant,
            "passes": is_topic_relevant,
        }

    def _analyze_conversational_flow(self, test_case: TestCase, response: str, context: str) -> Dict[str, Any]:
        """Analyze general conversational flow"""
        response_lower = response.lower()

        # Check for conversational markers
        conversational_markers = [
            "bonjour",
            "merci",
            "effectivement",
            "bien sûr",
            "en effet",
            "comme vous le savez",
            "comme mentionné",
            "précédemment",
        ]

        has_conversational_markers = any(marker in response_lower for marker in conversational_markers)

        # Check response length (conversational responses should be substantial)
        appropriate_length = 20 <= len(response.split()) <= 200

        # Check for natural language flow
        is_natural = not response.startswith("ERROR:") and len(response.strip()) > 0

        return {
            "has_conversational_markers": has_conversational_markers,
            "appropriate_length": appropriate_length,
            "is_natural": is_natural,
            "response_word_count": len(response.split()),
            "passes": is_natural and appropriate_length,
        }

    def _calculate_conversational_score(
        self, base_score: float, conversational_analysis: Dict[str, Any], test_case: TestCase
    ) -> float:
        """Calculate final conversational score"""
        # Start with base LLM judge score
        final_score = base_score

        # Apply bonuses/penalties based on conversational analysis
        for analysis_type, analysis_result in conversational_analysis.items():
            if isinstance(analysis_result, dict) and "passes" in analysis_result:
                if analysis_result["passes"]:
                    final_score += 0.1  # Bonus for good conversational handling
                else:
                    final_score -= 0.15  # Penalty for poor conversational handling

        # Ensure score stays within bounds
        return max(0.0, min(1.0, final_score))

    def reset_conversation_state(self):
        """Reset conversation state for new evaluation session"""
        self.conversation_state.clear()
        self.isschat_client.reset_conversation()

    def get_conversational_summary(self) -> Dict[str, Any]:
        """Get conversational-specific summary statistics"""
        if not self.results:
            return {}

        summary = self.get_summary_stats()

        # Add conversational-specific metrics
        context_continuity_tests = [r for r in self.results if r.metadata.get("test_type") == "context_continuity"]
        context_reference_tests = [r for r in self.results if r.metadata.get("test_type") == "context_reference"]
        memory_recall_tests = [r for r in self.results if r.metadata.get("test_type") == "memory_recall"]

        summary.update(
            {
                "context_continuity_pass_rate": len([r for r in context_continuity_tests if r.passed])
                / len(context_continuity_tests)
                if context_continuity_tests
                else 0,
                "context_reference_pass_rate": len([r for r in context_reference_tests if r.passed])
                / len(context_reference_tests)
                if context_reference_tests
                else 0,
                "memory_recall_pass_rate": len([r for r in memory_recall_tests if r.passed]) / len(memory_recall_tests)
                if memory_recall_tests
                else 0,
                "conversation_state_entries": len(self.conversation_state),
            }
        )

        return summary
