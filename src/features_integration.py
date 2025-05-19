import streamlit as st
import os
import logging
import time
from datetime import datetime

# Import all feature modules
from src.conversation_analysis import ConversationAnalyzer
from src.response_tracking import ResponseTracker
from src.question_suggestion import QuestionSuggester
from src.performance_tracking import PerformanceTracker
from src.query_history import QueryHistory


class FeaturesManager:
    """Central manager for all advanced chatbot features"""

    def __init__(self, help_desk, user_id):
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
        self.response_tracker = ResponseTracker()
        self.question_suggester = QuestionSuggester()
        self.performance_tracker = PerformanceTracker()
        self.query_history = QueryHistory()

        # Integrate necessary features
        self._integrate_selected_features()

    def _integrate_selected_features(self):
        """Integrate selected features into the help_desk"""
        try:
            # Add user feedback system
            self._add_feedback_system()
            self.logger.info("User feedback system integrated")

            # Modify ask_question method to record performance metrics
            self._wrap_ask_question()
            self.logger.info("Performance tracking integrated")

        except Exception as e:
            self.logger.error(f"Error integrating features: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())

    def _add_feedback_system(self):
        """Add a user feedback system"""
        # Store the original method
        self.original_ask = self.help_desk.retrieval_qa_inference

    def _wrap_ask_question(self):
        """Wrap the ask_question method to record performance metrics"""
        original_ask = self.help_desk.retrieval_qa_inference

        def wrapped_ask(question, verbose=False):
            # Measure total time
            start_time = time.time()

            # Call the original function
            answer, sources = original_ask(question, verbose)

            # Calculate total time
            total_time = (time.time() - start_time) * 1000  # in ms

            # Record the interaction in history
            self.query_history.add_query(
                user_id=self.user_id,
                question=question,
                answer=answer,
                sources=sources if isinstance(sources, list) else [sources],
            )

            # Record performance metrics
            self.performance_tracker.track_query(
                question=question,
                retrieval_time=total_time * 0.3,  # Approximation
                generation_time=total_time * 0.7,  # Approximation
                num_docs_retrieved=len(sources) if isinstance(sources, list) else 1,
                total_time=total_time,
            )

            # Record the interaction for conversation analysis
            self.analyzer.log_interaction(
                user_id=self.user_id,
                question=question,
                answer=answer,
                sources=sources if isinstance(sources, list) else [sources],
                response_time=total_time,
            )

            return answer, sources

        # Replace the original method
        self.help_desk.retrieval_qa_inference = wrapped_ask

    def render_admin_dashboard(self, st):
        """Display the administration dashboard with all features"""
        st.title("Administration Dashboard")

        # Create tabs for each feature
        tabs = st.tabs(
            [
                "Conversation Analysis",
                "Response Tracking",
                "Performance",
                "General Statistics",
            ]
        )

        # Tab 1: Conversation Analysis
        with tabs[0]:
            try:
                self.analyzer.render_analysis_dashboard()
            except Exception as e:
                st.error(f"Error displaying conversation analysis: {str(e)}")

        # Tab 2: Response Tracking
        with tabs[1]:
            try:
                self.response_tracker.render_tracking_dashboard()
            except Exception as e:
                st.error(f"Error displaying response tracking: {str(e)}")

        # Tab 3: Performance
        with tabs[2]:
            try:
                self.performance_tracker.render_performance_dashboard()
            except Exception as e:
                st.error(f"Error displaying performance metrics: {str(e)}")

        # Tab 4: General Statistics
        with tabs[3]:
            self._render_general_statistics(st)

    def _render_general_statistics(self, st):
        """Display general statistics about chatbot usage"""
        st.subheader("General Statistics")

        # Collect statistics from different sources
        stats = {}

        try:
            # Conversation statistics
            conv_logs = self.analyzer.get_recent_logs(days=30)
            stats["total_conversations"] = len(conv_logs)

            # Performance statistics
            perf_logs = self.performance_tracker.get_performance_logs(days=30)
            if perf_logs:
                avg_time = sum(log.get("total_time_ms", 0) for log in perf_logs) / len(perf_logs)
                stats["avg_response_time"] = f"{avg_time:.0f} ms"
            else:
                stats["avg_response_time"] = "N/A"

            # Feedback statistics
            unsat_responses = self.response_tracker.get_unsatisfactory_responses(days=30)
            stats["unsatisfactory_responses"] = len(unsat_responses)

            # Display statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total conversations", stats["total_conversations"])

            with col2:
                st.metric("Average response time", stats["avg_response_time"])

            with col3:
                st.metric("Unsatisfactory responses", stats["unsatisfactory_responses"])

        except Exception as e:
            st.error(f"Error collecting statistics: {str(e)}")

    def process_question(self, question, show_suggestions=True, show_feedback=True):
        """Process a question and add user feedback system"""
        try:
            # 1. Get the answer
            start_time = datetime.now()
            answer, sources = self.help_desk.retrieval_qa_inference(question)
            response_time = (datetime.now() - start_time).total_seconds() * 1000

            # 2. Display the answer
            st.markdown(answer)

            # 3. Display sources
            if sources:
                st.write(sources)

            # 4. Display feedback widget if enabled
            if show_feedback:
                self._add_feedback_widget(st, question, answer, sources)

            # 5. Display question suggestions if enabled
            if show_suggestions:
                self._show_question_suggestions(st, question, answer)

            # Log performance metrics
            self.logger.info(f"Question processed in {response_time:.0f}ms: {question[:50]}...")

            return answer, sources

        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            self.logger.error(f"Error processing question: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            return "Sorry, an error occurred while processing your question.", []

    def _add_feedback_widget(self, st, question, answer, sources):
        """Add a feedback widget to evaluate the quality of the response"""
        st.write("---")
        st.write("### Rate this response")

        col1, col2 = st.columns([3, 1])

        with col1:
            feedback_score = st.slider("Response quality", 1, 5, 3, key="feedback_score")
            feedback_text = st.text_area(
                "Comment (optional)",
                key="feedback_text",
                placeholder="Tell us what you think about this response",
            )

        with col2:
            if st.button("Send feedback", key="send_feedback"):
                # Record the feedback
                self.response_tracker.log_response_quality(
                    user_id=self.user_id,
                    question=question,
                    answer=answer,
                    sources=sources if isinstance(sources, list) else [sources],
                    feedback_score=feedback_score,
                    feedback_text=feedback_text,
                )
                st.success("Thank you for your feedback!")

    def _show_question_suggestions(self, st, question, answer):
        """Display follow-up question suggestions"""
        try:
            suggestions = self.question_suggester.suggest_next_questions(question, answer)

            if suggestions:
                st.write("---")
                st.write("### Suggested questions")

                for i, suggestion in enumerate(suggestions):
                    if st.button(suggestion, key=f"suggest_{i}"):
                        # Store the question in the session for reuse
                        return suggestion
        except Exception as e:
            self.logger.error(f"Error displaying suggestions: {str(e)}")
            return None


# Function to integrate the feature manager into the main application
def setup_features(help_desk, user_id):
    """Configure and return a feature manager for the application"""
    return FeaturesManager(help_desk, user_id)
