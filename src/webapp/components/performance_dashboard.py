"""
Chatbot Performance Dashboard for Isschat
Focused on conversation analysis and performance tracking
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List
from collections import Counter


class PerformanceDashboard:
    """Performance dashboard focused on conversation and performance metrics."""

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "success": "#F18F01",
            "warning": "#C73E1D",
            "info": "#6A994E",
        }

    def render_dashboard(self):
        """Main dashboard rendering with core metrics."""
        st.title("Performance Dashboard")

        # Force refresh button to clear cache
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("Refresh"):
                st.cache_data.clear()
                st.rerun()

        # Main dashboard tabs
        tab1, tab2, tab3 = st.tabs(["Conversations", "Performance", "Feedback"])

        with tab1:
            self._render_conversation_analytics()

        with tab2:
            self._render_performance_tracking()

        with tab3:
            self._render_feedback_analytics()

    def _render_conversation_analytics(self):
        """Render conversation analytics."""
        st.subheader("Conversation Analytics")

        conversations = self.data_manager.get_conversation_history(limit=200)

        if not conversations:
            st.info("No conversation data available")
            return

        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Conversations", len(conversations))
        with col2:
            avg_length = sum(len(c.get("answer", "")) for c in conversations) / len(conversations)
            st.metric("Avg Answer Length", f"{avg_length:.0f} chars")
        with col3:
            avg_time = sum(c.get("response_time_ms", 0) for c in conversations) / len(conversations)
            st.metric("Avg Response Time", f"{avg_time:.0f}ms")

        # Volume chart
        self._render_conversation_volume(conversations)

        # Popular topics
        self._render_popular_topics(conversations)

    def _render_performance_tracking(self):
        """Render performance tracking."""
        st.subheader("Performance Tracking")

        performance_data = self.data_manager.get_performance_metrics(limit=200)

        if not performance_data:
            st.info("No performance data available")
            return

        # Basic stats
        col1, col2 = st.columns(2)
        with col1:
            avg_time = sum(p.get("duration_ms", 0) for p in performance_data) / len(performance_data)
            st.metric("Avg Response Time", f"{avg_time:.0f}ms")
        with col2:
            st.metric("Total Requests", len(performance_data))

        # Timeline chart
        self._render_performance_timeline(performance_data)

    def _render_conversation_volume(self, conversations: List[Dict]):
        """Render conversation volume chart."""
        df = pd.DataFrame(conversations)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        daily_counts = df.groupby("date").size().reset_index(name="count")

        fig = px.bar(
            daily_counts,
            x="date",
            y="count",
            title="Daily Conversation Volume",
            color_discrete_sequence=[self.colors["primary"]],
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_popular_topics(self, conversations: List[Dict]):
        """Render popular topics analysis."""
        all_words = []
        for conv in conversations:
            words = conv.get("question", "").lower().split()
            filtered_words = [
                word.strip(".,!?;:")
                for word in words
                if len(word) > 3 and word not in ["what", "how", "when", "where", "why"]
            ]
            all_words.extend(filtered_words)

        if all_words:
            word_counts = Counter(all_words)
            top_words = word_counts.most_common(5)

            st.markdown("#### Popular Topics")
            for word, count in top_words:
                st.write(f"- {word.title()} ({count} mentions)")

    def _render_performance_timeline(self, performance_data: List[Dict]):
        """Render performance timeline."""
        df = pd.DataFrame(performance_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        fig = px.line(
            df,
            x="timestamp",
            y="duration_ms",
            title="Response Time Over Time",
            color_discrete_sequence=[self.colors["primary"]],
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_feedback_analytics(self):
        """Render feedback analytics section."""
        st.subheader("Feedback Analysis")

        # Get feedback data from the features manager feedback system
        try:
            from ..components.features_manager import FeedbackSystem

            feedback_system = FeedbackSystem()
            feedback_stats = feedback_system.get_feedback_statistics()

            if feedback_stats.get("total_feedback", 0) == 0:
                st.info("No feedback data available")
                st.markdown("""
                **Possible causes:**
                - Users haven't provided feedback yet
                - Feedback saving issues
                - Cache preventing data reload
                """)
                return

            # Display feedback metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Feedback", feedback_stats["total_feedback"])
            with col2:
                st.metric("Positive Feedback", feedback_stats["positive_feedback"])
            with col3:
                st.metric("Negative Feedback", feedback_stats["negative_feedback"])
            with col4:
                st.metric("Satisfaction Rate", f"{feedback_stats['satisfaction_rate']:.1f}%")

            # Feedback distribution chart
            if feedback_stats["total_feedback"] > 0:
                import plotly.graph_objects as go

                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=["Positive", "Negative"],
                            y=[feedback_stats["positive_feedback"], feedback_stats["negative_feedback"]],
                            marker_color=[self.colors["success"], self.colors["warning"]],
                        )
                    ]
                )
                fig.update_layout(
                    title="Feedback Distribution",
                    xaxis_title="Feedback Type",
                    yaxis_title="Count",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading feedback data: {e}")


def render_performance_dashboard(data_manager):
    """Utility function to render the dashboard."""
    dashboard = PerformanceDashboard(data_manager)
    dashboard.render_dashboard()
