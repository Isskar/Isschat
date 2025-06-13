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
        st.markdown(
            "<h1 style='text-align: center; color: #2E86AB; margin-bottom: 2rem;'>"
            "🤖 Chatbot Performance Dashboard</h1>",
            unsafe_allow_html=True,
        )

        # Main dashboard tabs
        tab1, tab2 = st.tabs(["💬 Conversation Analytics", "⚡ Performance Tracking"])

        with tab1:
            self._render_conversation_analytics()

        with tab2:
            self._render_performance_tracking()

    def _render_conversation_analytics(self):
        """Render conversation analytics."""
        st.markdown("### Conversation Analytics")

        conversations = self.data_manager.get_conversation_history(limit=200)

        if not conversations:
            st.info("No conversation data available")
            return

        # Basic stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Conversations", len(conversations))
        with col2:
            avg_length = sum(len(c.get("answer", "")) for c in conversations) / len(conversations)
            st.metric("Avg Answer Length", f"{avg_length:.0f} chars")

        # Volume chart
        self._render_conversation_volume(conversations)

        # Popular topics
        self._render_popular_topics(conversations)

    def _render_performance_tracking(self):
        """Render performance tracking."""
        st.markdown("### Performance Tracking")

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


def render_performance_dashboard(data_manager):
    """Utility function to render the dashboard."""
    dashboard = PerformanceDashboard(data_manager)
    dashboard.render_dashboard()
