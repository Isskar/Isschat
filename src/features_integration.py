import streamlit as st
import os
import logging
import time
from datetime import datetime

# Import all feature modules
from src.conversation_analysis import ConversationAnalyzer
from src.response_tracking import ResponseTracker
from src.performance_tracking import PerformanceTracker
from src.query_history import QueryHistory


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
        self.response_tracker = ResponseTracker()
        self.performance_tracker = PerformanceTracker()
        self.query_history = QueryHistory()

        # Integrate necessary features
        self._integrate_selected_features()

    def _integrate_selected_features(self) -> None:
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

    def _add_feedback_system(self) -> None:
        """Add a user feedback system"""
        # Store the original method
        self.original_ask = (
            self.help_desk.retrieval_qa_inference if hasattr(self.help_desk, "retrieval_qa_inference") else None
        )

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
        if hasattr(self.help_desk, "retrieval_qa_inference"):
            self.help_desk.retrieval_qa_inference = wrapped_ask

    def render_admin_dashboard(self, st) -> None:
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

    def _render_general_statistics(self, st) -> None:
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
                stats["avg_response_time"] = int(avg_time)
            else:
                stats["avg_response_time"] = 0

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

    def _add_feedback_widget(self, st, question: str, answer: str, sources: list) -> None:
        """Add a feedback widget to evaluate the quality of the response using thumbs up/down"""
        st.write("---")

        # Unique ID for this feedback instance
        feedback_id = f"feedback_{hash(question + answer)}"

        # Add debug info before feedback widget
        print("\n==== ADDING FEEDBACK WIDGET ====")
        print(f"Feedback ID: {feedback_id}")
        print(f"Question: {question[:50]}..." if len(question) > 50 else f"Question: {question}")

        # Store the feedback value in session state to track changes
        if f"prev_{feedback_id}" not in st.session_state:
            st.session_state[f"prev_{feedback_id}"] = None

        # Use Streamlit's feedback component with thumbs up/down
        feedback = st.feedback(
            options="thumbs",
            key=feedback_id,
            on_change=self._on_feedback_change,
            args=(question, answer, sources, feedback_id),
        )

        # Debug the current feedback value
        print(f"Current feedback value: {feedback}")
        print(f"Previous feedback value: {st.session_state.get(f'prev_{feedback_id}')}")

        # Check if feedback was given (0 = thumbs down, 1 = thumbs up, None = no feedback)
        if feedback is not None:
            # Convert to our scoring system: thumbs up (1) = 5, thumbs down (0) = 1
            feedback_score = 5 if feedback == 1 else 1
            feedback_text = None  # Streamlit feedback doesn't provide text directly

            # Debug information
            print(f"Feedback received: {feedback} (raw value)")
            print(f"Converted to score: {feedback_score}")

            # Check if this is a new feedback (different from previous)
            if feedback != st.session_state.get(f"prev_{feedback_id}"):
                print("New feedback detected! Processing...")
                st.session_state[f"prev_{feedback_id}"] = feedback

                # Log the feedback
                try:
                    self.response_tracker.log_response_quality(
                        user_id=self.user_id,
                        question=question,
                        answer=answer,
                        sources=sources if isinstance(sources, list) else [sources],
                        feedback_score=feedback_score,
                        feedback_text=feedback_text,
                    )
                    print("Feedback successfully saved!")
                except Exception as e:
                    print(f"Error saving feedback: {str(e)}")
                    import traceback

                    print(traceback.format_exc())
            else:
                print("This feedback was already processed. Skipping.")

        print("==== END ADDING FEEDBACK WIDGET ====")

    def _on_feedback_change(self, question, answer, sources, feedback_id):
        """Callback when feedback changes"""
        print("\n==== FEEDBACK CHANGED ====")
        print(f"Feedback ID: {feedback_id}")

        # Get the current feedback value
        feedback = st.session_state.get(feedback_id)
        print(f"New feedback value: {feedback}")

        if feedback is not None:
            # Convert to our scoring system: thumbs up (1) = 5, thumbs down (0) = 1
            feedback_score = 5 if feedback == 1 else 1
            feedback_text = None  # Streamlit feedback doesn't provide text directly

            # Log the feedback
            try:
                print(f"Logging feedback from callback: {feedback_score}")
                self.response_tracker.log_response_quality(
                    user_id=self.user_id,
                    question=question,
                    answer=answer,
                    sources=sources if isinstance(sources, list) else [sources],
                    feedback_score=feedback_score,
                    feedback_text=feedback_text,
                )
                print("Feedback successfully saved from callback!")
            except Exception as e:
                print(f"Error saving feedback from callback: {str(e)}")
                import traceback

                print(traceback.format_exc())

        print("==== END FEEDBACK CHANGED ====")

        # Store the comment in session state to preserve it between reruns
        if f"{feedback_id}_comment" not in st.session_state:
            st.session_state[f"{feedback_id}_comment"] = ""

        # If feedback is negative (thumbs down), show a comment field directly after the feedback widget
        if feedback is not None and feedback == 0:  # 0 = thumbs down
            # Comment section for negative feedback
            st.text_area(
                "Please tell us why this response wasn't helpful",
                key=f"{feedback_id}_comment",
                placeholder="Help us improve our responses...",
                value=st.session_state.get(f"{feedback_id}_comment", ""),
                on_change=self._on_comment_change,
                args=(feedback_id,),
            )

            # Submit button for the comment
            if st.button("Submit feedback", key=f"{feedback_id}_submit"):
                comment = st.session_state.get(f"{feedback_id}_comment", "")
                if comment.strip():
                    # Update the feedback with the comment
                    try:
                        print(f"Logging feedback with comment: {comment[:50]}...")
                        self.response_tracker.log_response_quality(
                            user_id=self.user_id,
                            question=question,
                            answer=answer,
                            sources=sources if isinstance(sources, list) else [sources],
                            feedback_score=1,  # thumbs down = 1 in our scoring system
                            feedback_text=comment,
                        )
                        st.success("Thank you for your feedback!")
                        print("Feedback with comment successfully saved!")

                        # Clear the comment field after submission
                        st.session_state[f"{feedback_id}_comment"] = ""
                        st.rerun()
                    except Exception as e:
                        print(f"Error saving feedback with comment: {str(e)}")
                        import traceback

                        print(traceback.format_exc())
                else:
                    st.warning("Please provide a comment before submitting.")

    def _on_comment_change(self, feedback_id):
        """Callback when comment changes"""
        print(f"Comment changed for {feedback_id}: {st.session_state.get(f'{feedback_id}_comment', '')}")
        # Just track the change, actual saving happens on submit


# Function to integrate the feature manager into the main application
def setup_features(help_desk, user_id: str) -> FeaturesManager:
    """Configure and return a feature manager for the application"""
    return FeaturesManager(help_desk, user_id)
