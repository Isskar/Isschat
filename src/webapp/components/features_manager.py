"""
Features manager for advanced chatbot functionality.
"""

import streamlit as st
from streamlit_feedback import streamlit_feedback
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List


class ConversationAnalyzer:
    """Analyzes conversation patterns and provides insights."""

    def __init__(self):
        self.conversations = []

    def analyze_conversation(self, query: str, response: str) -> Dict[str, Any]:
        """Analyze a conversation turn."""
        analysis = {
            "query_length": len(query),
            "response_length": len(response),
            "timestamp": datetime.now().isoformat(),
            "query_type": self._classify_query(query),
        }
        self.conversations.append(analysis)
        return analysis

    def _classify_query(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        if any(word in query_lower for word in ["how", "comment"]):
            return "how-to"
        elif any(word in query_lower for word in ["what", "qu'est-ce", "quoi"]):
            return "definition"
        elif any(word in query_lower for word in ["where", "où"]):
            return "location"
        elif any(word in query_lower for word in ["why", "pourquoi"]):
            return "explanation"
        else:
            return "general"

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        if not self.conversations:
            return {}

        return {
            "total_conversations": len(self.conversations),
            "avg_query_length": sum(c["query_length"] for c in self.conversations) / len(self.conversations),
            "avg_response_length": sum(c["response_length"] for c in self.conversations) / len(self.conversations),
            "query_types": {
                qtype: sum(1 for c in self.conversations if c["query_type"] == qtype)
                for qtype in set(c["query_type"] for c in self.conversations)
            },
        }


class FeedbackSystem:
    """Handles user feedback collection and storage."""

    def __init__(self, storage_service=None, feedback_file: Optional[str] = None):
        # Get storage service from config if not provided
        if storage_service is None:
            from src.core.config import _ensure_config_initialized

            config_manager = _ensure_config_initialized()
            self.storage_service = config_manager.get_storage_service()
        else:
            self.storage_service = storage_service

        # Remove redundant feedback file creation - use only data_manager
        # The feedback_file parameter is kept for backward compatibility but not used
        self.feedback_file = feedback_file

    def _load_feedback_data(self):
        """Load feedback data from data_manager (JSONL format)"""
        import logging

        logger = logging.getLogger(__name__)
        all_feedback = []

        try:
            from src.core.data_manager import get_data_manager

            data_manager = get_data_manager()

            # Use data_manager to get feedback data directly
            logger.info("Loading feedbacks from data_manager")
            all_feedback = data_manager.get_feedback_data()

            logger.info(f"Total: {len(all_feedback)} feedbacks loaded from data_manager")
            return all_feedback

        except Exception as e:
            logger.error(f"Error loading feedback data from data_manager: {e}")
            return []

    def render_feedback_widget(
        self, user_id: str, question: str, answer: str, sources: str, conversation_id: str, key_suffix: str = ""
    ) -> Optional[Dict]:
        """
        Renders a Streamlit feedback widget for user responses.

        Args:
            user_id: The ID of the user providing feedback.
            question: The original question.
            answer: Answer given.
            sources: Sources used.
            conversation_id: Conversation ID.
            key_suffix: Suffix for unique widget key.

        Returns:
            Dict: Feedback response if provided, otherwise None.
        """
        content_hash = abs(hash(question + answer))

        def feedback_callback(response):
            print(f"Feedback received: {response}")
            try:
                # Sauvegarder dans le data manager d'abord
                try:
                    from src.core.data_manager import get_data_manager

                    data_manager = get_data_manager()

                    rating = 3  # neutral default
                    comment = ""

                    if isinstance(response, dict):
                        score = response.get("score")
                        if score == 1 or score == "👍":
                            rating = 5
                        elif score == 0 or score == "👎":
                            rating = 1
                        comment = response.get("text", "")

                    # Save feedback via data_manager (JSONL format)
                    data_manager.save_feedback(
                        user_id=user_id,
                        conversation_id=conversation_id,  # Pass conversation_id
                        rating=rating,
                        comment=comment,
                        metadata={
                            "original_feedback": response,
                            "question": question,
                            "answer": answer,
                            "sources": sources,
                        },
                    )
                    print(
                        f"Feedback saved via data_manager: rating={rating}, conversation_id={conversation_id}, user_id={user_id}"  # noqa
                    )
                except Exception as e:
                    print(f"Error saving feedback via data_manager: {e}")

            except Exception as e:
                print(f"Error saving feedback via data_manager: {e}")

            # Force a rerun to update the interface
            st.rerun()

        try:
            feedback_response = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="Commentaire (optionnel):",
                align="flex-start",
                key=f"feedback_widget_{content_hash}",
                on_submit=feedback_callback,
            )

            return feedback_response
        except Exception as e:
            st.error(f"Erreur avec le widget de feedback: {e}")
            return None

    def get_feedback_statistics(self, days: int = 30) -> Dict:
        """
        Get feedback statistics using both data manager and file system

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with statistics
        """
        # Try data manager first
        try:
            from src.core.data_manager import get_data_manager

            data_manager = get_data_manager()
            feedback_data = data_manager.get_feedback_data(limit=1000)

            if feedback_data:
                return self._process_feedback_data(feedback_data, days)
        except Exception as e:
            print(f"Data manager feedback failed: {e}")

        # Fallback to file system
        try:
            feedback_data = self._load_feedback_data()
            if feedback_data:
                return self._process_feedback_data(feedback_data, days)
        except Exception as e:
            print(f"File system feedback failed: {e}")

        # Return empty stats if both methods fail
        return {"total_feedback": 0, "positive_feedback": 0, "negative_feedback": 0, "satisfaction_rate": 0.0}

    def _process_feedback_data(self, feedback_data: List[Dict], days: int = 30) -> Dict:
        """Process feedback data and return statistics."""
        if not feedback_data:
            return {"total_feedback": 0, "positive_feedback": 0, "negative_feedback": 0, "satisfaction_rate": 0.0}

        # Filter by date if needed
        if days < 30:
            from datetime import datetime, timedelta

            cutoff_date = datetime.now() - timedelta(days=days)

            recent_feedback = []
            for entry in feedback_data:
                try:
                    entry_date = datetime.fromisoformat(entry["timestamp"])
                    if entry_date >= cutoff_date:
                        recent_feedback.append(entry)
                except (KeyError, ValueError):
                    recent_feedback.append(entry)
        else:
            recent_feedback = feedback_data

        total = len(recent_feedback)
        positive = 0
        negative = 0

        for entry in recent_feedback:
            # Handle both old format (feedback.score) and new format (rating)
            rating = entry.get("rating", 0)
            feedback_obj = entry.get("feedback", {})
            score = feedback_obj.get("score") if feedback_obj else None

            # Handle different rating formats
            if rating >= 4 or score == 1 or score == "👍":
                positive += 1
            elif rating <= 2 or score == 0 or score == "👎":
                negative += 1

        satisfaction_rate = (positive / total * 100) if total > 0 else 0.0

        return {
            "total_feedback": total,
            "positive_feedback": positive,
            "negative_feedback": negative,
            "satisfaction_rate": satisfaction_rate,
        }

    def _get_feedback_statistics_fallback(self, days: int = 30) -> Dict:
        """Fallback method using old file loading"""
        feedback_data = self._load_feedback_data()

        if not feedback_data:
            return {"total_feedback": 0, "positive_feedback": 0, "negative_feedback": 0, "satisfaction_rate": 0.0}

        # Filter by date if needed
        if days < 30:
            from datetime import datetime, timedelta

            cutoff_date = datetime.now() - timedelta(days=days)

            recent_feedback = []
            for entry in feedback_data:
                try:
                    entry_date = datetime.fromisoformat(entry["timestamp"])
                    if entry_date >= cutoff_date:
                        recent_feedback.append(entry)
                except (KeyError, ValueError):
                    recent_feedback.append(entry)
        else:
            recent_feedback = feedback_data

        total = len(recent_feedback)
        positive = 0
        negative = 0

        for entry in recent_feedback:
            feedback_obj = entry.get("feedback", {})
            score = feedback_obj.get("score")

            # Handle emoji ('👍'/'👎') and numeric (1/0) formats
            if score == 1 or score == "👍":
                positive += 1
            elif score == 0 or score == "👎":
                negative += 1

        satisfaction_rate = (positive / total * 100) if total > 0 else 0.0

        return {
            "total_feedback": total,
            "positive_feedback": positive,
            "negative_feedback": negative,
            "satisfaction_rate": satisfaction_rate,
        }

    def get_all_feedback(self):
        """Get all recorded feedbacks"""
        return self._load_feedback_data()

    def get_feedback_by_user(self, user_id: str):
        """Get feedbacks from a specific user"""
        all_feedback = self._load_feedback_data()
        return [entry for entry in all_feedback if entry.get("user_id") == user_id]


class ResponseTracker:
    """Tracks response quality and user feedback."""

    def __init__(self):
        self.responses = []
        self.feedback_data = []

    def track_response(self, query: str, response: str, response_time: float) -> str:
        """Track a response."""
        response_id = f"resp_{len(self.responses)}_{int(time.time())}"

        response_data = {
            "id": response_id,
            "query": query,
            "response": response,
            "response_time": response_time,
            "timestamp": datetime.now(),
            "feedback": None,
        }

        self.responses.append(response_data)
        return response_id

    def add_feedback(self, response_id: str, feedback: Dict[str, Any]):
        """Add user feedback for a response."""
        for response in self.responses:
            if response["id"] == response_id:
                response["feedback"] = feedback
                break

        self.feedback_data.append({"response_id": response_id, "feedback": feedback, "timestamp": datetime.now()})

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        if not self.feedback_data:
            return {}

        ratings = [f["feedback"].get("rating", 0) for f in self.feedback_data if f["feedback"].get("rating")]

        return {
            "total_feedback": len(self.feedback_data),
            "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
            "positive_feedback": sum(1 for r in ratings if r >= 4),
            "negative_feedback": sum(1 for r in ratings if r <= 2),
        }


class PerformanceTracker:
    """Tracks system performance metrics."""

    def __init__(self):
        self.performance_data = []

    def track_performance(self, operation: str, duration: float, **kwargs):
        """Track performance of an operation."""
        perf_data = {"operation": operation, "duration": duration, "timestamp": datetime.now(), **kwargs}
        self.performance_data.append(perf_data)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_data:
            return {}

        operations = {}
        for data in self.performance_data:
            op = data["operation"]
            if op not in operations:
                operations[op] = []
            operations[op].append(data["duration"])

        return {
            "total_operations": len(self.performance_data),
            "avg_durations": {op: sum(durations) / len(durations) for op, durations in operations.items()},
            "operation_counts": {op: len(durations) for op, durations in operations.items()},
        }


class QueryHistory:
    """Manages query history and patterns."""

    def __init__(self):
        self.history = []

    def add_query(self, query: str, response: str, user_id: str = "anonymous"):
        """Add a query to history."""
        history_entry = {"query": query, "response": response, "user_id": user_id, "timestamp": datetime.now()}
        self.history.append(history_entry)

    def get_user_history(self, user_id: str, limit: int = 10):
        """Get history for a specific user."""
        user_queries = [h for h in self.history if h["user_id"] == user_id]
        return sorted(user_queries, key=lambda x: x["timestamp"], reverse=True)[:limit]

    def get_recent_queries(self, limit: int = 20):
        """Get recent queries across all users."""
        return sorted(self.history, key=lambda x: x["timestamp"], reverse=True)[:limit]


class FeaturesManager:
    """Central manager for all advanced chatbot features."""

    def __init__(self, user_id: str = "anonymous", storage_service=None):
        """Initialize and integrate all features."""
        self.user_id = user_id

        # Get storage service from config if not provided
        if storage_service is None:
            from src.core.config import _ensure_config_initialized

            config_manager = _ensure_config_initialized()
            self.storage_service = config_manager.get_storage_service()
        else:
            self.storage_service = storage_service

        # Create necessary directories using storage service
        directories = ["logs", "logs/feedback", "data", "cache"]
        for directory in directories:
            if hasattr(self.storage_service._storage, "create_directory"):
                self.storage_service._storage.create_directory(directory)

        # Configure logging - use only console logging to avoid local file creation
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
            ],
        )

        self.logger = logging.getLogger("features_manager")
        self.logger.info(f"Initializing features manager for user {user_id}")

        # Initialize feature components with storage service
        self.analyzer = ConversationAnalyzer()
        self.response_tracker = ResponseTracker()
        self.performance_tracker = PerformanceTracker()
        self.query_history = QueryHistory()
        self.feedback_system = FeedbackSystem(storage_service=self.storage_service)

        # Initialize session state for features
        if "features_data" not in st.session_state:
            st.session_state.features_data = {"conversations": [], "responses": [], "performance": [], "history": []}

    def process_query_response(self, query: str, response: str, response_time: float, conversation_id: str) -> str:
        """Process a query-response pair through all features."""
        # Analyze conversation
        analysis = self.analyzer.analyze_conversation(query, response)

        # Track response
        response_id = self.response_tracker.track_response(query, response, response_time)

        # Track performance
        self.performance_tracker.track_performance("query_processing", response_time)

        # Add to history
        self.query_history.add_query(query, response, self.user_id)

        # Save to data manager
        try:
            from src.core.data_manager import get_data_manager

            data_manager = get_data_manager()

            # Save conversation
            data_manager.save_conversation(
                user_id=self.user_id,
                conversation_id=conversation_id,
                question=query,
                answer=response,
                response_time_ms=response_time,
                sources=None,  # Could be enhanced to include sources
                metadata={"analysis": analysis},
            )

            # Save performance metric
            data_manager.save_performance(
                operation="query_processing",
                duration_ms=response_time,
                user_id=self.user_id,
                metadata={"query_length": len(query), "response_length": len(response)},
            )

        except Exception as e:
            self.logger.error(f"Error saving to data manager: {e}")

        # Log the interaction
        self.logger.info(
            f"Processed query for user {self.user_id}: {len(query)} chars -> {len(response)} chars in {response_time:.2f}s"  # noqa
        )

        return response_id

    def add_user_feedback(self, response_id: str, rating: int, comment: str = ""):
        """Add user feedback for a response."""
        feedback = {"rating": rating, "comment": comment, "user_id": self.user_id}
        self.response_tracker.add_feedback(response_id, feedback)
        self.logger.info(f"Received feedback from user {self.user_id}: rating={rating}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for admin dashboard."""
        return {
            "conversation_stats": self.analyzer.get_conversation_stats(),
            "feedback_stats": self.response_tracker.get_feedback_stats(),
            "performance_stats": self.performance_tracker.get_performance_stats(),
            "recent_queries": self.query_history.get_recent_queries(10),
            "user_history": self.query_history.get_user_history(self.user_id, 5),
        }

    def render_feedback_widget(
        self, question: str, answer: str, sources: str = "", conversation_id: str = "", key_suffix: str = ""
    ):
        """Render feedback widget with thumbs (👍/👎) for a response."""
        # If conversation_id is not provided, use the hash of the question (legacy behavior)
        if not conversation_id:
            conversation_id = f"conv_{abs(hash(question))}"

        return self.feedback_system.render_feedback_widget(
            user_id=self.user_id,
            question=question,
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
            key_suffix=key_suffix,
        )

    def get_feedback_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get feedback statistics from the thumbs feedback system."""
        return self.feedback_system.get_feedback_statistics(days)

    def add_feedback_widget(
        self, st, question: str, answer: str, sources: str, conversation_id: Optional[str] = None, key_suffix: str = ""
    ):
        """Compatible method with old version to add a feedback widget"""
        # If conversation_id is not provided, use the hash of the question (legacy behavior)
        if conversation_id is None:
            conversation_id = f"conv_{abs(hash(question))}"

        return self.feedback_system.render_feedback_widget(
            user_id=self.user_id,
            question=question,
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
            key_suffix=key_suffix,
        )
