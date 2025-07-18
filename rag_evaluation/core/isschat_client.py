"""
Client interface for interacting with Isschat system
"""

import time
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Add src to path to import RAGPipeline
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.semantic_pipeline import SemanticRAGPipelineFactory


class IsschatClient:
    """Client for interacting with Isschat system"""

    def __init__(self, conversation_memory: bool = False):
        self.conversation_memory = conversation_memory
        self.conversation_history = []
        try:
            self.rag_pipeline = SemanticRAGPipelineFactory.create_semantic_pipeline()
            print("✅ Isschat client initialized successfully")
        except Exception as e:
            error_msg = str(e)
            if "configuration" in error_msg.lower() or "environment" in error_msg.lower():
                print(f"❌ Configuration error: {error_msg}")
                print("💡 Make sure you have a .env file with required environment variables")
                print("💡 Or run from the main project directory where configuration is available")
            else:
                print(f"❌ Failed to initialize Isschat client: {e}")
            raise

    def query(self, question: str, context: Optional[str] = None) -> Tuple[str, float, List[Dict[str, str]]]:
        start_time = time.time()
        try:
            # Use provided context or build contextual question
            contextual_question = question
            if context:
                contextual_question = f"Contexte de conversation:\n{context}\n\nQuestion actuelle: {question}"

            response, sources = self.rag_pipeline.process_query(contextual_question, verbose=False)
            response_time = time.time() - start_time

            # Store in conversation history if memory is enabled
            if self.conversation_memory:
                self.conversation_history.append({"question": question, "response": response})

            source_list = []
            if sources:
                source_list = self._parse_sources_string(sources)
            return response, response_time, source_list

        except Exception as e:
            response_time = time.time() - start_time
            error_response = f"ERROR: {str(e)}"
            return error_response, response_time, []

    def _parse_sources_string(self, sources_str: str) -> List[Dict[str, str]]:
        import re

        # FIXME: Pattern to match markdown links: [title](url) --> need to add a dataclass Source to handle all cases
        markdown_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        # Find all markdown links in the string
        markdown_matches = re.findall(markdown_pattern, sources_str)
        sources = []
        for title, url in markdown_matches:
            sources.append({"title": title, "url": url})
        return sources

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
