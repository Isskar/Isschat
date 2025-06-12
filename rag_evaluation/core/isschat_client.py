"""
Client interface for interacting with Isschat system
"""

import time
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Add src to path to import HelpDesk
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from help_desk import HelpDesk


class IsschatClient:
    """Client for interacting with Isschat system"""

    def __init__(self, conversation_memory: bool = False):
        """
        Initialize Isschat client

        Args:
            conversation_memory: Whether to maintain conversation history
        """
        self.conversation_memory = conversation_memory
        self.conversation_history: List[Dict[str, str]] = []

        # Initialize HelpDesk
        try:
            self.help_desk = HelpDesk(new_db=False)
            print("✅ Isschat client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Isschat client: {e}")
            raise

    def query(self, question: str, context: Optional[List[str]] = None) -> Tuple[str, float, List[str]]:
        """
        Query Isschat with a question

        Args:
            question: The question to ask
            context: Optional conversation context

        Returns:
            Tuple of (response, response_time, sources)
        """
        start_time = time.time()

        try:
            # If conversation memory is enabled and we have context,
            # we could modify the question to include context
            if self.conversation_memory and context:
                # For now, we'll just use the question as-is
                # In a more advanced implementation, we could modify the prompt
                pass

            # Query the help desk
            response, sources = self.help_desk.retrieval_qa_inference(question, verbose=False)

            response_time = time.time() - start_time

            # Store in conversation history if memory is enabled
            if self.conversation_memory:
                self.conversation_history.append(
                    {"question": question, "response": response, "sources": sources, "timestamp": time.time()}
                )

            # Format sources as list of strings
            source_list = []
            if sources:
                if isinstance(sources, str):
                    source_list = [sources]
                elif isinstance(sources, list):
                    source_list = sources
                else:
                    source_list = [str(sources)]

            return response, response_time, source_list

        except Exception as e:
            response_time = time.time() - start_time
            error_response = f"ERROR: {str(e)}"
            return error_response, response_time, []

    def query_with_conversation_context(
        self, question: str, previous_exchanges: List[Dict[str, str]]
    ) -> Tuple[str, float, List[str]]:
        """
        Query with explicit conversation context

        Args:
            question: Current question
            previous_exchanges: List of previous Q&A exchanges

        Returns:
            Tuple of (response, response_time, sources)
        """
        # For conversational tests, we might want to modify the question
        # to include context or use a different approach

        # Simple approach: just query normally for now
        # In a more advanced implementation, we could:
        # 1. Modify the prompt to include conversation history
        # 2. Use a different retrieval strategy
        # 3. Maintain state in the help_desk instance

        return self.query(question)

    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history.clear()

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get current conversation history"""
        return self.conversation_history.copy()

    def get_last_response_time(self) -> float:
        """Get response time of last query"""
        if self.conversation_history:
            # This would need to be tracked separately in a real implementation
            return 0.0
        return 0.0

    def health_check(self) -> bool:
        """Check if Isschat is responding properly"""
        try:
            test_question = "Hello"
            response, _, _ = self.query(test_question)
            return not response.startswith("ERROR:")
        except Exception:
            return False

    def get_system_info(self) -> Dict[str, str]:
        """Get system information"""
        return {
            "conversation_memory": str(self.conversation_memory),
            "conversation_length": str(len(self.conversation_history)),
            "status": "healthy" if self.health_check() else "unhealthy",
        }
