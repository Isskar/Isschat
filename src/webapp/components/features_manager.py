"""
Features manager for advanced chatbot functionality.
"""

import streamlit as st
import os
import logging
import time
from datetime import datetime
from typing import Dict, Any


class ConversationAnalyzer:
    """Analyzes conversation patterns and provides insights."""

    def __init__(self):
        self.conversations = []

    def analyze_conversation(self, query: str, response: str) -> Dict[str, Any]:
        """Analyze a conversation turn."""
        analysis = {
            "query_length": len(query),
            "response_length": len(response),
            "timestamp": datetime.now(),
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

    def __init__(self, user_id: str = "anonymous"):
        """Initialize and integrate all features."""
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

        # Initialize feature components
        self.analyzer = ConversationAnalyzer()
        self.response_tracker = ResponseTracker()
        self.performance_tracker = PerformanceTracker()
        self.query_history = QueryHistory()

        # Initialize session state for features
        if "features_data" not in st.session_state:
            st.session_state.features_data = {"conversations": [], "responses": [], "performance": [], "history": []}

    def process_query_response(self, query: str, response: str, response_time: float) -> str:
        """Process a query-response pair through all features."""
        # Analyze conversation
        # analysis = self.analyzer.analyze_conversation(query, response)

        # Track response
        response_id = self.response_tracker.track_response(query, response, response_time)

        # Track performance
        self.performance_tracker.track_performance("query_processing", response_time)

        # Add to history
        self.query_history.add_query(query, response, self.user_id)

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

    def render_feedback_widget(self, response_id: str):
        """Render feedback widget for a response."""
        st.markdown("---")
        st.markdown("**Was this response helpful?**")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            rating = st.select_slider(
                "Rating", options=[1, 2, 3, 4, 5], value=3, format_func=lambda x: "⭐" * x, key=f"rating_{response_id}"
            )

            comment = st.text_area("Additional comments (optional)", key=f"comment_{response_id}", height=68)

            if st.button("Submit Feedback", key=f"submit_{response_id}"):
                self.add_user_feedback(response_id, rating, comment)
                st.success("Thank you for your feedback!")
                st.rerun()
