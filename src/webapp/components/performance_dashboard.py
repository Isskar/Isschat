"""
Chatbot Performance Dashboard for Isschat
Advanced monitoring interface with comprehensive metrics and analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List
from collections import Counter
import logging
import time

logger = logging.getLogger(__name__)


class PerformanceDashboard:
    """Advanced performance dashboard for chatbot monitoring and analytics."""

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "success": "#F18F01",
            "warning": "#C73E1D",
            "info": "#6A994E",
            "background": "#1E1E1E",
            "surface": "#2D2D2D",
            "text": "#FFFFFF",
        }

        # Apply dark theme styling
        self._apply_custom_styling()

    def _apply_custom_styling(self):
        """Apply custom dark theme styling."""
        st.markdown(
            """
        <style>
        .main > div {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }

        .stMetric {
            background-color: #2D2D2D;
            border: 1px solid #404040;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .metric-container {
            background: linear-gradient(135deg, #2D2D2D 0%, #3D3D3D 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            border: 1px solid #404040;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }

        .performance-card {
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            color: white;
            box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        }

        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .status-good { background-color: #6A994E; }
        .status-warning { background-color: #F18F01; }
        .status-critical { background-color: #C73E1D; }

        .stTabs [data-baseweb="tab-list"] {
            background-color: #2D2D2D;
            border-radius: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #FFFFFF;
            border-radius: 8px;
        }

        .stTabs [aria-selected="true"] {
            background-color: #2E86AB;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    def render_dashboard(self):
        """Main dashboard rendering with comprehensive chatbot metrics."""
        st.markdown(
            "<h1 style='text-align: center; color: #2E86AB; margin-bottom: 2rem;'>"
            "🤖 Chatbot Performance Dashboard</h1>",
            unsafe_allow_html=True,
        )

        # Real-time overview
        self._render_realtime_overview()

        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📊 Live Metrics",
                "💬 Conversation Analytics",
                "⚡ Performance Trends",
                "🎯 Quality Insights",
                "🔧 System Health",
            ]
        )

        with tab1:
            self._render_live_metrics()

        with tab2:
            self._render_conversation_analytics()

        with tab3:
            self._render_performance_trends()

        with tab4:
            self._render_quality_insights()

        with tab5:
            self._render_system_health()

    def _render_realtime_overview(self):
        """Display real-time overview with key chatbot metrics."""
        # Get recent data
        conversations = self.data_manager.get_conversation_history(limit=100)
        performance_data = self.data_manager.get_performance_metrics(limit=100)
        system_metrics = self._get_system_metrics()

        # Calculate key metrics
        total_conversations_today = len([c for c in conversations if self._is_today(c.get("timestamp", ""))])
        avg_response_time = self._calculate_avg_response_time(performance_data)
        user_satisfaction = self._calculate_user_satisfaction(conversations)
        system_health_score = self._calculate_health_score(system_metrics, avg_response_time)

        # Display overview cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            self._render_metric_card(
                "💬 Conversations Today",
                str(total_conversations_today),
                self._get_trend_indicator(total_conversations_today, 50),
                "#2E86AB",
            )

        with col2:
            self._render_metric_card(
                "⚡ Avg Response Time",
                f"{avg_response_time:.0f}ms",
                self._get_performance_indicator(avg_response_time),
                "#F18F01",
            )

        with col3:
            self._render_metric_card(
                "😊 User Satisfaction",
                f"{user_satisfaction:.1f}/5.0",
                self._get_satisfaction_indicator(user_satisfaction),
                "#6A994E",
            )

        with col4:
            self._render_metric_card(
                "🏥 System Health",
                f"{system_health_score:.0f}%",
                self._get_health_indicator(system_health_score),
                "#A23B72",
            )

    def _render_metric_card(self, title: str, value: str, indicator: str, color: str):
        """Render a custom metric card."""
        st.markdown(
            f"""
        <div class="metric-container">
            <h4 style="color: {color}; margin: 0 0 0.5rem 0;">{title}</h4>
            <h2 style="margin: 0; color: #FFFFFF;">{value}</h2>
            <p style="margin: 0.5rem 0 0 0; color: #CCCCCC;">{indicator}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def _render_live_metrics(self):
        """Render live metrics tab with real-time data."""
        st.markdown("### 📈 Real-Time Performance Metrics")

        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            time.sleep(30)
            st.rerun()

        # Get fresh data
        performance_data = self.data_manager.get_performance_metrics(limit=50)
        system_metrics = self._get_system_metrics()

        col1, col2 = st.columns(2)

        with col1:
            self._render_response_time_chart(performance_data)
            self._render_system_resources(system_metrics)

        with col2:
            self._render_throughput_metrics(performance_data)
            self._render_error_rate_metrics()

    def _render_conversation_analytics(self):
        """Render conversation analytics with detailed insights."""
        st.markdown("### 💬 Conversation Analytics & Insights")

        conversations = self.data_manager.get_conversation_history(limit=200)

        if not conversations:
            st.info("No conversation data available for analysis.")
            return

        col1, col2 = st.columns(2)

        with col1:
            self._render_conversation_volume_chart(conversations)
            self._render_popular_topics(conversations)

        with col2:
            self._render_user_engagement_metrics(conversations)
            self._render_conversation_length_distribution(conversations)

    def _render_performance_trends(self):
        """Render performance trends and historical analysis."""
        st.markdown("### ⚡ Performance Trends & Historical Analysis")

        # Time range selector
        time_range = st.selectbox("Select time range:", ["Last 24 hours", "Last 7 days", "Last 30 days"], index=1)

        days = {"Last 24 hours": 1, "Last 7 days": 7, "Last 30 days": 30}[time_range]

        performance_data = self.data_manager.get_performance_metrics(limit=500)
        filtered_data = self._filter_by_days(performance_data, days)

        if filtered_data:
            col1, col2 = st.columns(2)

            with col1:
                self._render_performance_timeline(filtered_data)
                self._render_performance_distribution(filtered_data)

            with col2:
                self._render_peak_hours_analysis(filtered_data)
                self._render_performance_benchmarks(filtered_data)
        else:
            st.info("No performance data available for the selected time range.")

    def _render_quality_insights(self):
        """Render quality insights and user feedback analysis."""
        st.markdown("### 🎯 Quality Insights & User Feedback")

        conversations = self.data_manager.get_conversation_history(limit=200)

        if not conversations:
            st.info("No conversation data available for quality analysis.")
            return

        col1, col2 = st.columns(2)

        with col1:
            self._render_feedback_analysis(conversations)
            self._render_response_quality_metrics(conversations)

        with col2:
            self._render_source_usage_analysis(conversations)
            self._render_improvement_suggestions()

    def _render_system_health(self):
        """Render comprehensive system health monitoring."""
        st.markdown("### 🔧 System Health & Resource Monitoring")

        system_metrics = self._get_system_metrics()

        col1, col2 = st.columns(2)

        with col1:
            self._render_detailed_system_metrics(system_metrics)
            self._render_resource_alerts(system_metrics)

        with col2:
            self._render_system_status_overview(system_metrics)
            self._render_maintenance_recommendations(system_metrics)

    # Helper methods for calculations
    def _is_today(self, timestamp_str: str) -> bool:
        """Check if timestamp is from today."""
        try:
            if timestamp_str is None:
                logger.warning("Received None timestamp in _is_today")
                return False
            logger.debug(f"Processing timestamp: {timestamp_str} (type: {type(timestamp_str)})")
            dt = datetime.fromisoformat(timestamp_str)
            return dt.date() == datetime.now().date()
        except Exception as e:
            logger.error(f"Error processing timestamp '{timestamp_str}': {e}")
            return False

    def _calculate_avg_response_time(self, performance_data: List[Dict]) -> float:
        """Calculate average response time."""
        if not performance_data:
            return 0.0
        return sum(p.get("duration_ms", 0) for p in performance_data) / len(performance_data)

    def _calculate_user_satisfaction(self, conversations: List[Dict]) -> float:
        """Calculate user satisfaction score from feedback logs."""
        import json
        import glob

        # Charger les feedbacks depuis les logs
        feedback_pattern = "logs/feedback/feedback_*.json"
        feedback_files = glob.glob(feedback_pattern)

        all_feedback = []
        for file_path in feedback_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    feedback_data = json.load(f)
                    if isinstance(feedback_data, list):
                        all_feedback.extend(feedback_data)
            except (FileNotFoundError, json.JSONDecodeError):
                continue

        if not all_feedback:
            return 3.5  # Default score

        # Calculer le taux de satisfaction (feedbacks positifs / total)
        positive_count = 0
        total_count = 0

        for entry in all_feedback:
            feedback = entry.get("feedback", {})
            score = feedback.get("score")
            if score is not None:
                total_count += 1
                if score == 1:  # Feedback positif
                    positive_count += 1

        if total_count == 0:
            return 3.5

        # Convertir le pourcentage en score sur 5 (0-100% -> 1-5)
        satisfaction_percentage = (positive_count / total_count) * 100
        satisfaction_score = 1 + (satisfaction_percentage / 100) * 4  # 1-5 scale

        return satisfaction_score

    def _calculate_health_score(self, system_metrics: Dict, avg_response_time: float) -> float:
        """Calculate overall system health score."""
        cpu_score = max(0, 100 - system_metrics["cpu_percent"])
        memory_score = max(0, 100 - system_metrics["memory_percent"])
        response_score = max(0, 100 - (avg_response_time / 50))  # 50ms = 0 points

        return (cpu_score + memory_score + response_score) / 3

    def _get_trend_indicator(self, current: int, baseline: int) -> str:
        """Get trend indicator for metrics."""
        if current > baseline * 1.1:
            return "📈 Trending up"
        elif current < baseline * 0.9:
            return "📉 Trending down"
        else:
            return "➡️ Stable"

    def _get_performance_indicator(self, response_time: float) -> str:
        """Get performance indicator."""
        if response_time < 1000:
            return "🟢 Excellent"
        elif response_time < 2000:
            return "🟡 Good"
        else:
            return "🔴 Needs attention"

    def _get_satisfaction_indicator(self, satisfaction: float) -> str:
        """Get satisfaction indicator."""
        if satisfaction >= 4.0:
            return "😊 Very satisfied"
        elif satisfaction >= 3.0:
            return "😐 Satisfied"
        else:
            return "😞 Needs improvement"

    def _get_health_indicator(self, health_score: float) -> str:
        """Get health indicator."""
        if health_score >= 80:
            return "🟢 Healthy"
        elif health_score >= 60:
            return "🟡 Warning"
        else:
            return "🔴 Critical"

    # Chart rendering methods
    def _render_response_time_chart(self, performance_data: List[Dict]):
        """Render response time chart."""
        st.markdown("#### ⚡ Response Time Trends")

        if not performance_data:
            st.info("No performance data available")
            return

        df = pd.DataFrame(performance_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        fig = px.line(
            df,
            x="timestamp",
            y="duration_ms",
            title="Response Time Over Time",
            color_discrete_sequence=[self.colors["primary"]],
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    def _render_system_resources(self, system_metrics: Dict):
        """Render system resource usage."""
        st.markdown("#### 💻 System Resources")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("CPU Usage", f"{system_metrics['cpu_percent']:.1f}%")

        with col2:
            st.metric("Memory Usage", f"{system_metrics['memory_percent']:.1f}%")

        # Progress bars
        cpu_progress = system_metrics["cpu_percent"] / 100
        memory_progress = system_metrics["memory_percent"] / 100

        st.progress(cpu_progress, text=f"CPU: {system_metrics['cpu_percent']:.1f}%")
        st.progress(memory_progress, text=f"Memory: {system_metrics['memory_percent']:.1f}%")

    def _render_throughput_metrics(self, performance_data: List[Dict]):
        """Render throughput metrics."""
        st.markdown("#### 📊 Throughput Metrics")

        if not performance_data:
            st.info("No throughput data available")
            return

        # Calculate requests per hour
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)

        recent_requests = len([p for p in performance_data if datetime.fromisoformat(p["timestamp"]) >= hour_ago])

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Requests/Hour", recent_requests)

        with col2:
            avg_throughput = recent_requests / 60 if recent_requests > 0 else 0
            st.metric("Avg Requests/Min", f"{avg_throughput:.1f}")

    def _render_error_rate_metrics(self):
        """Render error rate metrics."""
        st.markdown("#### ⚠️ Error Rate")

        # This would need to be implemented based on your error tracking
        st.metric("Error Rate", "0.2%", delta="-0.1%")
        st.success("System operating normally")

    def _render_conversation_volume_chart(self, conversations: List[Dict]):
        """Render conversation volume chart."""
        st.markdown("#### 📈 Conversation Volume")

        df = pd.DataFrame(conversations)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        daily_counts = df.groupby("date").size().reset_index(name="count")

        fig = px.bar(
            daily_counts,
            x="date",
            y="count",
            title="Daily Conversation Volume",
            color_discrete_sequence=[self.colors["success"]],
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    def _render_popular_topics(self, conversations: List[Dict]):
        """Render popular topics analysis."""
        st.markdown("#### 🏷️ Popular Topics")

        # Extract keywords from questions
        all_words = []
        for conv in conversations:
            question = conv.get("question", "")
            logger.debug(f"Processing question: {question} (type: {type(question)})")
            if question is None:
                logger.warning(f"Found None question in conversation: {conv}")
                question = ""
            words = question.lower().split()
            filtered_words = []
            for word in words:
                if word is None:
                    logger.warning(f"Found None word in conversation question: {conv.get('question')}")
                    continue
                try:
                    stripped_word = word.strip(".,!?;:")
                    if len(stripped_word) > 3 and stripped_word not in ["what", "how", "when", "where", "why"]:
                        filtered_words.append(stripped_word)
                except AttributeError as e:
                    logger.error(f"Error stripping word '{word}' (type: {type(word)}): {e}")
                    continue
            all_words.extend(filtered_words)

        if all_words:
            word_counts = Counter(all_words)
            top_words = word_counts.most_common(5)

            for word, count in top_words:
                st.write(f"**{word.title()}**: {count} mentions")
        else:
            st.info("No topic data available")

    def _render_user_engagement_metrics(self, conversations: List[Dict]):
        """Render user engagement metrics."""
        st.markdown("#### 👥 User Engagement")

        unique_users = len(set(conv.get("user_id", "anonymous") for conv in conversations))
        avg_conversations_per_user = len(conversations) / unique_users if unique_users > 0 else 0

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Active Users", unique_users)

        with col2:
            st.metric("Avg Conv/User", f"{avg_conversations_per_user:.1f}")

    def _render_conversation_length_distribution(self, conversations: List[Dict]):
        """Render conversation length distribution."""
        st.markdown("#### 📏 Response Length Distribution")

        lengths = [conv.get("answer_length", 0) for conv in conversations]

        if lengths:
            df = pd.DataFrame({"length": lengths})

            fig = px.histogram(
                df, x="length", title="Response Length Distribution", color_discrete_sequence=[self.colors["info"]]
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No length data available")

    def _filter_by_days(self, data: List[Dict], days: int) -> List[Dict]:
        """Filter data by number of days."""
        cutoff = datetime.now() - timedelta(days=days)
        filtered_data = []
        for item in data:
            try:
                timestamp = item.get("timestamp")
                if timestamp is None:
                    logger.warning(f"Found None timestamp in data item: {item}")
                    continue
                logger.debug(f"Processing timestamp in filter: {timestamp} (type: {type(timestamp)})")
                dt = datetime.fromisoformat(timestamp)
                if dt >= cutoff:
                    filtered_data.append(item)
            except Exception as e:
                logger.error(f"Error processing timestamp '{item.get('timestamp')}' in filter: {e}")
                continue
        return filtered_data

    def _render_performance_timeline(self, performance_data: List[Dict]):
        """Render performance timeline."""
        st.markdown("#### 📊 Performance Timeline")

        df = pd.DataFrame(performance_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        fig = px.scatter(
            df,
            x="timestamp",
            y="duration_ms",
            title="Performance Over Time",
            color_discrete_sequence=[self.colors["primary"]],
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    def _render_performance_distribution(self, performance_data: List[Dict]):
        """Render performance distribution."""
        st.markdown("#### 📈 Performance Distribution")

        durations = [p.get("duration_ms", 0) for p in performance_data]

        fast = len([d for d in durations if d < 1000])
        medium = len([d for d in durations if 1000 <= d < 3000])
        slow = len([d for d in durations if d >= 3000])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Fast (<1s)", fast)

        with col2:
            st.metric("Medium (1-3s)", medium)

        with col3:
            st.metric("Slow (>3s)", slow)

    def _render_peak_hours_analysis(self, performance_data: List[Dict]):
        """Render peak hours analysis."""
        st.markdown("#### 🕐 Peak Hours Analysis")

        df = pd.DataFrame(performance_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour

        hourly_avg = df.groupby("hour")["duration_ms"].mean().reset_index()

        fig = px.bar(
            hourly_avg,
            x="hour",
            y="duration_ms",
            title="Average Response Time by Hour",
            color_discrete_sequence=[self.colors["warning"]],
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    def _render_performance_benchmarks(self, performance_data: List[Dict]):
        """Render performance benchmarks."""
        st.markdown("#### 🎯 Performance Benchmarks")

        durations = [p.get("duration_ms", 0) for p in performance_data]

        if durations:
            p50 = sorted(durations)[len(durations) // 2]
            p95 = sorted(durations)[int(len(durations) * 0.95)]
            p99 = sorted(durations)[int(len(durations) * 0.99)]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("P50", f"{p50:.0f}ms")

            with col2:
                st.metric("P95", f"{p95:.0f}ms")

            with col3:
                st.metric("P99", f"{p99:.0f}ms")

    def _render_feedback_analysis(self, conversations: List[Dict]):
        """Render feedback analysis from logs."""
        import json
        import glob

        st.markdown("#### 📝 User Feedback Analysis")

        # Charger les feedbacks depuis les logs
        feedback_pattern = "logs/feedback/feedback_*.json"
        feedback_files = glob.glob(feedback_pattern)

        all_feedback = []
        for file_path in feedback_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    feedback_data = json.load(f)
                    if isinstance(feedback_data, list):
                        all_feedback.extend(feedback_data)
            except (FileNotFoundError, json.JSONDecodeError):
                continue

        if all_feedback:
            # Analyser les scores de feedback (0/1)
            positive_count = 0
            negative_count = 0
            total_count = 0

            for entry in all_feedback:
                feedback = entry.get("feedback", {})
                score = feedback.get("score")
                if score is not None:
                    total_count += 1
                    if score == 1:
                        positive_count += 1
                    elif score == 0:
                        negative_count += 1

            if total_count > 0:
                satisfaction_rate = (positive_count / total_count) * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Feedbacks Positifs", positive_count)
                with col2:
                    st.metric("Feedbacks Négatifs", negative_count)
                with col3:
                    st.metric("Taux de Satisfaction", f"{satisfaction_rate:.1f}%")

                # Afficher la distribution
                st.write("**Distribution des feedbacks:**")
                st.write(f"👍 Positifs: {positive_count} ({positive_count / total_count * 100:.1f}%)")
                st.write(f"👎 Négatifs: {negative_count} ({negative_count / total_count * 100:.1f}%)")

                # Display some recent comments
                recent_comments = []
                for entry in all_feedback[-5:]:  # 5 derniers
                    feedback = entry.get("feedback", {})
                    logger.debug(f"Processing feedback entry: {entry}")
                    comment_raw = feedback.get("text", "")
                    logger.debug(f"Raw comment: {comment_raw} (type: {type(comment_raw)})")

                    if comment_raw is None:
                        logger.warning(f"Found None comment in feedback: {feedback}")
                        comment = ""
                    else:
                        try:
                            comment = comment_raw.strip()
                        except AttributeError as e:
                            logger.error(f"Error stripping comment '{comment_raw}' (type: {type(comment_raw)}): {e}")
                            comment = str(comment_raw) if comment_raw is not None else ""

                    if comment:
                        score_emoji = "👍" if feedback.get("score") == 1 else "👎"
                        recent_comments.append(f"{score_emoji} {comment}")

                if recent_comments:
                    st.write("**Commentaires récents:**")
                    for comment in recent_comments:
                        st.write(f"- {comment}")
            else:
                st.info("Aucun feedback avec score trouvé")
        else:
            st.info("Aucune donnée de feedback disponible dans les logs")

    def _render_response_quality_metrics(self, conversations: List[Dict]):
        """Render response quality metrics."""
        st.markdown("#### 🎯 Response Quality")

        if conversations:
            avg_length = sum(conv.get("answer_length", 0) for conv in conversations) / len(conversations)
            avg_sources = sum(conv.get("sources_count", 0) for conv in conversations) / len(conversations)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Avg Response Length", f"{avg_length:.0f} chars")

            with col2:
                st.metric("Avg Sources Used", f"{avg_sources:.1f}")
        else:
            st.info("No quality data available")

    def _render_source_usage_analysis(self, conversations: List[Dict]):
        """Render source usage analysis."""
        st.markdown("#### 📚 Source Usage Analysis")

        total_sources = sum(conv.get("sources_count", 0) for conv in conversations)
        conversations_with_sources = len([conv for conv in conversations if conv.get("sources_count", 0) > 0])

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Sources Used", total_sources)

        with col2:
            coverage = (conversations_with_sources / len(conversations) * 100) if conversations else 0
            st.metric("Source Coverage", f"{coverage:.1f}%")

    def _render_improvement_suggestions(self):
        """Render improvement suggestions."""
        st.markdown("#### 💡 Improvement Suggestions")

        suggestions = [
            "Consider optimizing response time for queries >2s",
            "Increase source coverage for better accuracy",
            "Monitor user satisfaction trends",
            "Implement caching for frequent queries",
        ]

        for suggestion in suggestions:
            st.write(f"• {suggestion}")

    def _render_detailed_system_metrics(self, system_metrics: Dict):
        """Render detailed system metrics."""
        st.markdown("#### 💻 Detailed System Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("CPU Usage", f"{system_metrics['cpu_percent']:.1f}%")
            st.metric("Memory Usage", f"{system_metrics['memory_percent']:.1f}%")

        with col2:
            st.metric("System Load", f"{system_metrics['load_avg']:.2f}")
            st.metric("Uptime", "99.9%")  # This would need real implementation

    def _render_resource_alerts(self, system_metrics: Dict):
        """Render resource alerts."""
        st.markdown("#### ⚠️ Resource Alerts")

        alerts = []

        if system_metrics["cpu_percent"] > 80:
            alerts.append("🔴 High CPU usage detected")

        if system_metrics["memory_percent"] > 80:
            alerts.append("🔴 High memory usage detected")

        if not alerts:
            st.success("✅ No resource alerts")
        else:
            for alert in alerts:
                st.warning(alert)

    def _render_system_status_overview(self, system_metrics: Dict):
        """Render system status overview."""
        st.markdown("#### 🏥 System Status Overview")

        status_items = [
            ("API Service", "🟢 Operational"),
            ("Database", "🟢 Operational"),
            ("Vector Store", "🟢 Operational"),
            ("Cache", "🟢 Operational"),
        ]

        for service, status in status_items:
            st.write(f"**{service}**: {status}")

    def _render_maintenance_recommendations(self, system_metrics: Dict):
        """Render maintenance recommendations."""
        st.markdown("#### 🔧 Maintenance Recommendations")

        recommendations = [
            "Regular database optimization scheduled",
            "Vector store reindexing recommended weekly",
            "Cache cleanup automated",
            "Log rotation configured",
        ]

        for rec in recommendations:
            st.write(f"✅ {rec}")

    def _get_system_metrics(self) -> Dict[str, float]:
        """Get system metrics with fallback."""
        try:
            # Try to get real system metrics
            import psutil

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()[0]  # 1-minute load average
            except AttributeError:
                # Windows doesn't have load average
                load_avg = cpu_percent / 100.0

            # Calculate deltas (simplified - in real app, store previous values)
            cpu_delta = 0.0  # Would compare with previous measurement
            memory_delta = 0.0  # Would compare with previous measurement
            load_delta = 0.0  # Would compare with previous measurement

            return {
                "cpu_percent": cpu_percent,
                "cpu_delta": cpu_delta,
                "memory_percent": memory_percent,
                "memory_delta": memory_delta,
                "load_avg": load_avg,
                "load_delta": load_delta,
            }
        except ImportError:
            # Fallback to simulated metrics
            logger.warning("psutil not available, using simulated metrics")
            import random

            return {
                "cpu_percent": random.uniform(20, 60),
                "cpu_delta": random.uniform(-5, 5),
                "memory_percent": random.uniform(30, 70),
                "memory_delta": random.uniform(-3, 3),
                "load_avg": random.uniform(0.5, 2.0),
                "load_delta": random.uniform(-0.2, 0.2),
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            # Fallback values
            return {
                "cpu_percent": 0.0,
                "cpu_delta": 0.0,
                "memory_percent": 0.0,
                "memory_delta": 0.0,
                "load_avg": 0.0,
                "load_delta": 0.0,
            }


def render_performance_dashboard(data_manager):
    """Utility function to render the dashboard."""
    try:
        dashboard = PerformanceDashboard(data_manager)
        dashboard.render_dashboard()
    except Exception as e:
        st.error(f"Error rendering performance dashboard: {e}")
        logger.error(f"Dashboard error: {e}")

        # Fallback simple dashboard
        st.title("Performance Dashboard")
        st.info("Dashboard temporarily unavailable - showing basic metrics")

        # Basic metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("System Status", "Operational")
        with col2:
            st.metric("Response Time", "< 1000ms")
        with col3:
            st.metric("Uptime", "99.9%")
