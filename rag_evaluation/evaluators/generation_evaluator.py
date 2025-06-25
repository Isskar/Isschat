"""
Generation evaluator for testing conversational generation capabilities
Uses LLM-based semantic evaluation for conversation flow and context handling
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from rag_evaluation.core.base_evaluator import TestCase
from rag_evaluation.core import IsschatClient, LLMJudge, BaseEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerationEvaluator(BaseEvaluator):
    """Evaluator for conversational generation tests using semantic LLM evaluation"""

    def __init__(self, config: Any):
        """Initialize generation evaluator"""
        super().__init__(config)
        self.isschat_client = IsschatClient(conversation_memory=True)
        self.llm_judge = LLMJudge(config)
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}

    def get_category(self) -> str:
        """Get the category this evaluator handles"""
        return "generation"

    def _query_system(self, test_case: TestCase) -> Tuple[str, float, List[str]]:
        """Query the system - handles conversation context for generation tests"""
        # Handle conversation context if present
        if test_case.conversation_context:
            self._setup_conversation_context(test_case)

        # Build context string for IsschatClient
        context_str = self._build_context_string(test_case)

        # Query Isschat with context
        response, response_time, sources = self.isschat_client.query(test_case.question, context=context_str)

        # Store response for future context
        self._store_conversation_turn(test_case, response)

        return response, response_time, sources

    def _evaluate_semantically(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Evaluate response semantically using generation-specific logic"""
        return self._evaluate_generation_semantically(test_case, response)

    def _setup_conversation_context(self, test_case: TestCase):
        """Setup conversation context for multi-turn conversations"""
        conversation_id = test_case.metadata.get("depends_on", test_case.test_id)

        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []

        # Add previous context to conversation history
        for context_turn in test_case.conversation_context:
            question = context_turn.get("question", "")
            response = context_turn.get("response", "")

            # Replace placeholders with actual responses if available
            if response == "{{PREVIOUS_RESPONSE}}" and self.conversation_history[conversation_id]:
                response = self.conversation_history[conversation_id][-1]["response"]
            elif response.startswith("{{RESPONSE_"):
                # Handle numbered response references
                response_num = int(response.replace("{{RESPONSE_", "").replace("}}", ""))
                if len(self.conversation_history[conversation_id]) >= response_num:
                    response = self.conversation_history[conversation_id][response_num - 1]["response"]

            self.conversation_history[conversation_id].append({"question": question, "response": response})

    def _store_conversation_turn(self, test_case: TestCase, response: str):
        """Store the current conversation turn for future reference"""
        conversation_id = test_case.metadata.get("depends_on", test_case.test_id)

        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []

        self.conversation_history[conversation_id].append({"question": test_case.question, "response": response})

    def _build_context_string(self, test_case: TestCase) -> Optional[str]:
        """Build context string from conversation context for IsschatClient"""
        if not test_case.conversation_context:
            return None

        context_parts = []
        conversation_id = test_case.metadata.get("depends_on", test_case.test_id)

        for turn in test_case.conversation_context:
            question = turn.get("question", "")
            response = turn.get("response", "")

            # Replace placeholders with actual responses if available
            if response == "{{PREVIOUS_RESPONSE}}" and conversation_id in self.conversation_history:
                if self.conversation_history[conversation_id]:
                    response = self.conversation_history[conversation_id][-1]["response"]
            elif response.startswith("{{RESPONSE_"):
                # Handle numbered response references
                try:
                    response_num = int(response.replace("{{RESPONSE_", "").replace("}}", ""))
                    if (
                        conversation_id in self.conversation_history
                        and len(self.conversation_history[conversation_id]) >= response_num
                    ):
                        response = self.conversation_history[conversation_id][response_num - 1]["response"]
                except ValueError:
                    pass

            if question and response:
                context_parts.append(f"Q: {question}")
                context_parts.append(f"R: {response}")

        return "\n".join(context_parts) if context_parts else None

    def _evaluate_generation_semantically(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Evaluate generation using semantic LLM analysis based on test type"""
        test_type = test_case.metadata.get("test_type", "")

        # Create specialized prompts based on test type
        if test_type == "context_continuity":
            return self._evaluate_context_continuity(test_case, response)
        elif test_type == "context_reference":
            return self._evaluate_context_reference(test_case, response)
        elif test_type == "memory_recall":
            return self._evaluate_memory_recall(test_case, response)
        elif test_type == "clarification":
            return self._evaluate_clarification(test_case, response)
        elif test_type == "topic_transition":
            return self._evaluate_topic_transition(test_case, response)
        elif test_type == "specific_reference":
            return self._evaluate_specific_reference(test_case, response)
        elif test_type in ["language_consistency", "project_inquiry", "technical_overview"]:
            return self._evaluate_language_and_content(test_case, response)
        else:
            # Generic conversational evaluation fallback
            return self._evaluate_generic_conversational(test_case, response)

    def _evaluate_context_continuity(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Evaluate context continuity in conversations"""
        prompt = f"""You are an expert evaluator for conversational AI context continuity.

QUESTION ASKED: {test_case.question}
ISSCHAT RESPONSE: {response}
EXPECTED BEHAVIOR: {test_case.expected_behavior}
CONVERSATION CONTEXT: {test_case.conversation_context}

DETAILED EVALUATION CRITERIA:
1. CONTEXT UNDERSTANDING (40%):
   - Does the response demonstrate understanding of previous conversation turns?
   - Are references to previous topics handled correctly?
   - Is the conversational flow maintained naturally?

2. CONTEXTUAL RELEVANCE (35%):
   - Is the response relevant to the current question in context?
   - Does it build appropriately on previous information?
   - Are implicit references understood and addressed?

3. CONVERSATIONAL COHERENCE (25%):
   - Is the response coherent within the conversation flow?
   - Does it maintain appropriate tone and style consistency?
   - Are transitions between topics handled smoothly?

SCORING RUBRIC:
- 0.9-1.0: Excellent context continuity with perfect understanding and natural flow
- 0.7-0.8: Good context handling with minor gaps in continuity
- 0.5-0.6: Adequate context awareness but some continuity issues
- 0.3-0.4: Basic context understanding but significant continuity problems
- 0.0-0.2: Poor context handling, breaks conversational flow

Respond with a short JSON object containing:
- "score": float between 0.0 and 1.0
- "reasoning": in one sentence, explain the score with specific examples
- "passes_criteria": boolean

EVALUATION:"""

        return self.llm_judge._evaluate_with_prompt(prompt)

    def _evaluate_context_reference(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Evaluate context reference handling"""
        prompt = f"""You are an expert evaluator for conversational AI context reference handling.

QUESTION ASKED: {test_case.question}
ISSCHAT RESPONSE: {response}
EXPECTED BEHAVIOR: {test_case.expected_behavior}
CONVERSATION CONTEXT: {test_case.conversation_context}

DETAILED EVALUATION CRITERIA:
1. REFERENCE RESOLUTION (45%):
   - Does the system correctly identify what pronouns/references point to?
   - Are implicit references (like "dessus", "ça", "celui-ci") properly resolved?
   - Is the referent from previous context correctly identified?

2. RESPONSE ACCURACY (35%):
   - Is the information provided accurate relative to the referenced context?
   - Does the response address the correct subject/object being referenced?
   - Is the level of detail appropriate for the reference?

3. NATURAL LANGUAGE HANDLING (20%):
   - Are French language references handled naturally?
   - Is the response fluent and contextually appropriate?
   - Does it maintain conversational naturalness?

SCORING RUBRIC:
- 0.9-1.0: Perfect reference resolution with accurate and natural responses
- 0.7-0.8: Good reference handling with mostly accurate responses
- 0.5-0.6: Adequate reference resolution but some accuracy issues
- 0.3-0.4: Basic reference understanding but significant resolution problems
- 0.0-0.2: Poor reference handling, incorrect or missing resolution

Respond with a short JSON object containing:
- "score": float between 0.0 and 1.0
- "reasoning": in one sentence, explain the score with specific examples
- "passes_criteria": boolean

EVALUATION:"""

        return self.llm_judge._evaluate_with_prompt(prompt)

    def _evaluate_memory_recall(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Evaluate memory recall capabilities"""
        prompt = f"""You are an expert evaluator for conversational AI memory and recall capabilities.

QUESTION ASKED: {test_case.question}
ISSCHAT RESPONSE: {response}
EXPECTED BEHAVIOR: {test_case.expected_behavior}
CONVERSATION CONTEXT: {test_case.conversation_context}

DETAILED EVALUATION CRITERIA:
1. MEMORY ACCURACY (40%):
   - Does the system accurately recall previous conversation topics?
   - Are the recalled details correct and complete?
   - Is the chronological order of topics maintained?

2. RECALL COMPLETENESS (35%):
   - Are all major topics from the conversation history mentioned?
   - Is the summary comprehensive without being overwhelming?
   - Are key details preserved in the recall?

3. ORGANIZATION AND CLARITY (25%):
   - Is the recalled information well-organized and clear?
   - Are topics presented in a logical sequence?
   - Is the summary easy to understand and follow?

SCORING RUBRIC:
- 0.9-1.0: Excellent memory recall with accurate, complete, and well-organized information
- 0.7-0.8: Good recall with mostly accurate information and clear organization
- 0.5-0.6: Adequate recall but some gaps in accuracy or organization
- 0.3-0.4: Basic recall present but significant issues with accuracy or completeness
- 0.0-0.2: Poor memory recall, inaccurate or missing information

Respond with a short JSON object containing:
- "score": float between 0.0 and 1.0
- "reasoning": in one sentence, explain the score with specific examples
- "passes_criteria": boolean

EVALUATION:"""

        return self.llm_judge._evaluate_with_prompt(prompt)

    def _evaluate_clarification(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Evaluate clarification request handling"""
        prompt = f"""You are an expert evaluator for conversational AI clarification handling.

QUESTION ASKED: {test_case.question}
ISSCHAT RESPONSE: {response}
EXPECTED BEHAVIOR: {test_case.expected_behavior}
CONVERSATION CONTEXT: {test_case.conversation_context}

DETAILED EVALUATION CRITERIA:
1. CLARIFICATION RECOGNITION (40%):
   - Does the system recognize the request for clarification?
   - Is the reference to "what you just said" properly understood?
   - Does it identify which part needs clarification?

2. CLARIFICATION QUALITY (35%):
   - Is the clarification helpful and informative?
   - Does it address potential confusion points?
   - Is additional detail or explanation provided appropriately?

3. RESPONSE COHERENCE (25%):
   - Is the clarification response well-structured and clear?
   - Does it maintain conversational flow?
   - Is the tone appropriate for providing clarification?

SCORING RUBRIC:
- 0.9-1.0: Excellent clarification with clear recognition and helpful detailed explanation
- 0.7-0.8: Good clarification with adequate recognition and useful information
- 0.5-0.6: Basic clarification but could be clearer or more helpful
- 0.3-0.4: Some clarification attempt but poor recognition or unhelpful response
- 0.0-0.2: Poor clarification handling, fails to recognize or provide useful information

Respond with a short JSON object containing:
- "score": float between 0.0 and 1.0
- "reasoning": in one sentence, explain the score with specific examples
- "passes_criteria": boolean

EVALUATION:"""

        return self.llm_judge._evaluate_with_prompt(prompt)

    def _evaluate_topic_transition(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Evaluate topic transition handling"""
        prompt = f"""You are an expert evaluator for conversational AI topic transition handling.

QUESTION ASKED: {test_case.question}
ISSCHAT RESPONSE: {response}
EXPECTED BEHAVIOR: {test_case.expected_behavior}
CONVERSATION CONTEXT: {test_case.conversation_context}

DETAILED EVALUATION CRITERIA:
1. TRANSITION RECOGNITION (40%):
   - Does the system recognize the request to return to a previous topic?
   - Is the reference to "sujet des collaborateurs" properly understood?
   - Does it identify which previous topic to return to?

2. TOPIC RETRIEVAL (35%):
   - Does the response successfully return to the requested topic?
   - Is relevant information from the previous discussion recalled?
   - Is the transition handled smoothly and naturally?

3. CONTEXTUAL INTEGRATION (25%):
   - Is the returned topic integrated well with current conversation state?
   - Does the response maintain conversational coherence?
   - Is the transition acknowledged appropriately?

SCORING RUBRIC:
- 0.9-1.0: Excellent topic transition with clear recognition and smooth return to previous topic
- 0.7-0.8: Good transition handling with adequate recognition and topic retrieval
- 0.5-0.6: Basic transition but some issues with recognition or topic return
- 0.3-0.4: Some transition attempt but poor recognition or execution
- 0.0-0.2: Poor topic transition handling, fails to recognize or return to requested topic

Respond with a short JSON object containing:
- "score": float between 0.0 and 1.0
- "reasoning": in one sentence, explain the score with specific examples
- "passes_criteria": boolean

EVALUATION:"""

        return self.llm_judge._evaluate_with_prompt(prompt)

    def _evaluate_specific_reference(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Evaluate specific reference handling"""
        prompt = f"""You are an expert evaluator for conversational AI specific reference handling.

QUESTION ASKED: {test_case.question}
ISSCHAT RESPONSE: {response}
EXPECTED BEHAVIOR: {test_case.expected_behavior}
CONVERSATION CONTEXT: {test_case.conversation_context}

DETAILED EVALUATION CRITERIA:
1. REFERENCE IDENTIFICATION (45%):
   - Does the system correctly identify "la première" refers to the first technology mentioned?
   - Is the specific item from the previous response accurately identified?
   - Is the ordinal reference ("première") properly processed?

2. INFORMATION RETRIEVAL (35%):
   - Is detailed information about the referenced item provided?
   - Is the information accurate and relevant?
   - Does the response expand appropriately on the referenced topic?

3. CONTEXTUAL ACCURACY (20%):
   - Is the reference resolved within the correct conversational context?
   - Does the response maintain logical flow from the previous exchange?
   - Is the level of detail appropriate for the specific request?

SCORING RUBRIC:
- 0.9-1.0: Perfect reference identification with accurate and detailed information
- 0.7-0.8: Good reference handling with mostly accurate information retrieval
- 0.5-0.6: Adequate reference processing but some accuracy or detail issues
- 0.3-0.4: Basic reference understanding but significant retrieval problems
- 0.0-0.2: Poor reference handling, incorrect identification or information

Respond with a short JSON object containing:
- "score": float between 0.0 and 1.0
- "reasoning": in one sentence, explain the score with specific examples
- "passes_criteria": boolean

EVALUATION:"""

        return self.llm_judge._evaluate_with_prompt(prompt)

    def _evaluate_language_and_content(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Evaluate language consistency and content quality"""
        prompt = f"""You are an expert evaluator for conversational AI language and content quality.

QUESTION ASKED: {test_case.question}
ISSCHAT RESPONSE: {response}
EXPECTED BEHAVIOR: {test_case.expected_behavior}

DETAILED EVALUATION CRITERIA:
1. LANGUAGE CONSISTENCY (40%):
   - Is the response entirely in French as expected?
   - Is the language register appropriate for enterprise context?
   - Are grammar and vocabulary correct and professional?

2. CONTENT ACCURACY (35%):
   - Is the information provided accurate and relevant?
   - Does the response address the question appropriately?
   - Is the level of detail suitable for the inquiry?

3. CONVERSATIONAL QUALITY (25%):
   - Is the response natural and engaging?
   - Does it maintain appropriate tone and style?
   - Is the structure clear and easy to follow?

SCORING RUBRIC:
- 0.9-1.0: Excellent language consistency with accurate content and natural conversation
- 0.7-0.8: Good language and content with minor issues
- 0.5-0.6: Adequate quality but some language or content problems
- 0.3-0.4: Basic quality with notable language or accuracy issues
- 0.0-0.2: Poor language consistency or content accuracy

Respond with a short JSON object containing:
- "score": float between 0.0 and 1.0
- "reasoning": in one sentence, explain the score with specific examples
- "passes_criteria": boolean

EVALUATION:"""

        return self.llm_judge._evaluate_with_prompt(prompt)

    def _evaluate_generic_conversational(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Generic conversational evaluation fallback"""
        prompt = f"""You are an expert evaluator for conversational AI generation quality.

QUESTION ASKED: {test_case.question}
ISSCHAT RESPONSE: {response}
EXPECTED BEHAVIOR: {test_case.expected_behavior}

DETAILED EVALUATION CRITERIA:
1. RESPONSE RELEVANCE (40%):
   - Does the response directly address the question asked?
   - Is the information provided relevant and useful?
   - Does it meet the expected behavior criteria?

2. CONVERSATIONAL QUALITY (35%):
   - Is the response natural and engaging?
   - Does it maintain appropriate conversational tone?
   - Is the language clear and professional?

3. COMPLETENESS AND ACCURACY (25%):
   - Is the response complete and informative?
   - Is the information accurate to the best of available knowledge?
   - Does it provide sufficient detail without being overwhelming?

SCORING RUBRIC:
- 0.9-1.0: Excellent conversational response with high relevance, quality, and accuracy
- 0.7-0.8: Good response with adequate quality and mostly accurate information
- 0.5-0.6: Acceptable response but some issues with relevance or quality
- 0.3-0.4: Basic response with notable problems in quality or accuracy
- 0.0-0.2: Poor response quality, irrelevant or inaccurate information

Respond with a short JSON object containing:
- "score": float between 0.0 and 1.0
- "reasoning": in one sentence, explain the score with specific examples
- "passes_criteria": boolean

EVALUATION:"""

        return self.llm_judge._evaluate_with_prompt(prompt)
