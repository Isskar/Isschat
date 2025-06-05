import streamlit as st
import os
import logging
import time
from datetime import datetime

# Import all feature modules
from src.conversation_analysis import ConversationAnalyzer
from src.performance_tracking import PerformanceTracker
from src.query_history import QueryHistory
from src.dashboard import AdminDashboard
from src.feedback_system import FeedbackSystem


class FeaturesManager:
    """Central manager for all advanced chatbot features"""

    def __init__(self, help_desk, user_id: str) -> None:
        """Initialize and integrate all features into the help_desk"""
        self.help_desk = help_desk
        self.user_id = user_id

        # Create necessary folders if they don't exist
        os.makedirs("./logs", exist_ok=True)
        os.makedirs("./data", exist_ok=True)
        os.makedirs("./cache", exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(f"./logs/chatbot_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler(),
            ],
        )

        self.logger = logging.getLogger("features_manager")
        self.logger.info(f"Initializing features manager for user {user_id}")

        # Create feature objects directly as attributes of this class
        self.analyzer = ConversationAnalyzer()
        self.performance_tracker = PerformanceTracker()
        self.query_history = QueryHistory()
        self.feedback_system = FeedbackSystem()

        # Create centralized dashboard (with new feedback system)
        self.dashboard = AdminDashboard(self.analyzer, self.feedback_system, self.performance_tracker)

        # Integrate necessary features
        self._integrate_selected_features()

    def _integrate_selected_features(self) -> None:
        """Integrate selected features into the help_desk"""
        try:
            # Modify ask_question method to record performance metrics
            self._wrap_ask_question()
            self.logger.info("Performance tracking integrated")

        except Exception as e:
            self.logger.error(f"Error integrating features: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())

    def _wrap_ask_question(self) -> None:
        """Wrap the ask_question method to record performance metrics"""
        original_ask = (
            self.help_desk.retrieval_qa_inference if hasattr(self.help_desk, "retrieval_qa_inference") else None
        )

        def wrapped_ask(question, verbose=False):
            # Measure total time
            start_time = time.time()

            # Call the original function
            answer, sources = original_ask(question, verbose) if original_ask else ("No answer", [])

            # Calculate total time
            total_time = (time.time() - start_time) * 1000  # in ms

            # Record the interaction in history
            self.query_history.add_query(
                user_id=self.user_id,
                question=question,
                answer=answer,
                sources=sources if isinstance(sources, list) else [sources],
            )

            # Record the interaction for conversation analysis (single source of truth)
            self.analyzer.log_interaction(
                user_id=self.user_id,
                question=question,
                answer=answer,
                sources=sources if isinstance(sources, list) else [sources],
                response_time=total_time,
            )

            # Performance tracker now reads from conversation logs - no separate logging needed

            return answer, sources

        # Replace the original method
        if hasattr(self.help_desk, "retrieval_qa_inference"):
            self.help_desk.retrieval_qa_inference = wrapped_ask

    def render_admin_dashboard(self, st) -> None:
        """Display the administration dashboard with all features"""
        # Use centralized dashboard
        self.dashboard.render_admin_dashboard(st)

    # Method removed - now handled by centralized AdminDashboard

    def process_question(
        self, question: str, show_suggestions: bool = True, show_feedback: bool = True
    ) -> tuple[str, list]:
        """Process a question and add user feedback system"""
        try:
            # 1. Get the answer
            start_time = datetime.now()
            answer, sources = self.help_desk.retrieval_qa_inference(question)
            response_time = (datetime.now() - start_time).total_seconds() * 1000

            # 2. Log performance metrics
            self.logger.info(f"Question processed in {response_time:.0f}ms: {question[:50]}...")

            # 3. Return the response
            return answer, sources

        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            self.logger.error(f"Error processing question: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            return "Sorry, an error occurred while processing your question.", []

    def add_feedback_widget(self, st, question: str, answer: str, sources: list, key_suffix: str = ""):
        """Add the new thumbs up/down feedback widget"""
        return self.feedback_system.render_feedback_widget(
            user_id=self.user_id, question=question, answer=answer, sources=sources, key_suffix=key_suffix
        )


# Function to integrate the feature manager into the main application
def setup_features(help_desk, user_id: str) -> FeaturesManager:
    """Configure and return a feature manager for the application"""
    return FeaturesManager(help_desk, user_id)
