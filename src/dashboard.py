import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class AdminDashboard:
    """Centralized dashboard for all admin analytics and monitoring"""

    def __init__(self, conversation_analyzer, response_tracker, performance_tracker):
        self.conversation_analyzer = conversation_analyzer
        self.response_tracker = response_tracker
        self.performance_tracker = performance_tracker

    def render_admin_dashboard(self, st) -> None:
        """Display the complete administration dashboard with all features"""

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
                self.render_conversation_analysis_dashboard()
            except Exception as e:
                st.error(f"Error displaying conversation analysis: {str(e)}")

        # Tab 2: Response Tracking
        with tabs[1]:
            try:
                self.render_response_tracking_dashboard()
            except Exception as e:
                st.error(f"Error displaying response tracking: {str(e)}")

        # Tab 3: Performance
        with tabs[2]:
            try:
                self.render_performance_dashboard()
            except Exception as e:
                st.error(f"Error displaying performance metrics: {str(e)}")

        # Tab 4: General Statistics
        with tabs[3]:
            self.render_general_statistics()

    def render_conversation_analysis_dashboard(self) -> None:
        """Display the conversation analysis dashboard in Streamlit"""
        st.title("Conversation Analysis")

        # Period selection
        days = st.slider("Analysis period (days)", 1, 30, 7, key="conv_analysis_days")
        logs = self.conversation_analyzer.get_recent_logs(days)

        if not logs:
            st.warning("No data available for the selected period")
            return

        # Analysis
        analysis = self.conversation_analyzer.analyze_questions(logs)

        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total questions", analysis["total_questions"])
        with col2:
            st.metric("Average response time", f"{analysis['avg_response_time_ms']:.0f} ms")

        # Hourly distribution chart
        st.subheader("Question distribution by hour")
        hours = list(range(24))
        counts = [analysis["hour_distribution"].get(hour, 0) for hour in hours]

        hour_df = pd.DataFrame({"Hour": hours, "Number of questions": counts})
        st.bar_chart(hour_df.set_index("Hour"))

        # Most frequent words
        st.subheader("Most frequent keywords")
        if analysis["common_words"]:
            keywords_df = pd.DataFrame(analysis["common_words"], columns=["Word", "Frequency"])
            st.dataframe(keywords_df)
        else:
            st.info("Not enough data for keyword analysis")

    def render_response_tracking_dashboard(self):
        """Display the response tracking dashboard in Streamlit"""
        st.title("Response Tracking")

        # Period selection
        days = st.slider("Analysis period (days)", 1, 90, 30, key="response_tracking_days")
        unsatisfactory = self.response_tracker.get_unsatisfactory_responses(days)

        if not unsatisfactory:
            st.warning("No unsatisfactory responses found for the selected period")
            return

        # Basic metrics
        st.metric("Number of unsatisfactory responses", len(unsatisfactory))

        # Pattern analysis
        patterns = self.response_tracker.identify_patterns(unsatisfactory)

        if "error" in patterns:
            st.error(f"Analysis error: {patterns['error']}")
            return

        # Display clusters of similar questions
        if "clusters" in patterns and patterns["clusters"]:
            st.subheader("Groups of similar questions without satisfactory responses")

            for i, cluster in enumerate(patterns["clusters"]):
                with st.expander(f"Group {i + 1}: {cluster['main_question']} ({cluster['count']} questions)"):
                    for q in cluster["similar_questions"]:
                        st.write(f"- {q}")

        # Display common terms
        if "common_terms" in patterns and patterns["common_terms"]:
            st.subheader("Frequent terms in questions without satisfactory responses")
            terms_df = pd.DataFrame(patterns["common_terms"], columns=["Term", "Frequency"])
            st.dataframe(terms_df)

        # Table of questions without satisfactory responses
        st.subheader("Details of questions without satisfactory responses")

        # Convert to DataFrame for cleaner display
        df = pd.DataFrame(
            [
                {
                    "Date": datetime.fromisoformat(item["timestamp"]).strftime("%Y-%m-%d %H:%M"),
                    "Question": item["question"],
                    "Score": item["feedback_score"],
                    "Comment": item.get("feedback_text", ""),
                }
                for item in unsatisfactory
            ]
        )

        st.dataframe(df)

    def render_performance_dashboard(self):
        """Display the performance dashboard in Streamlit"""
        st.title("Performance Tracking")

        # Period selection
        days = st.slider("Analysis period (days)", 1, 30, 7, key="performance_days")
        logs = self.performance_tracker.get_performance_logs(days)

        if not logs:
            st.warning("No data available for the selected period")
            return

        # Analysis
        analysis = self.performance_tracker.analyze_performance(logs)

        # Display main metrics
        metrics = analysis["metrics"]
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Average response time", f"{metrics['avg_total_time_ms']:.0f} ms")
        with col2:
            st.metric("Average retrieval time", f"{metrics['avg_retrieval_time_ms']:.0f} ms")
        with col3:
            st.metric("Average generation time", f"{metrics['avg_generation_time_ms']:.0f} ms")

        # Time evolution chart
        st.subheader("Response Time Evolution")

        # Convert data for the chart
        daily_data = pd.DataFrame(analysis["daily_metrics"])
        if not daily_data.empty:
            daily_data["date"] = pd.to_datetime(daily_data["date"])
            daily_data = daily_data.sort_values("date")

            # Create chart
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Response times
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Time (ms)", color="tab:blue")
            ax1.plot(
                daily_data["date"],
                daily_data["total_time_ms"],
                "b-",
                label="Total time",
            )
            ax1.plot(
                daily_data["date"],
                daily_data["retrieval_time_ms"],
                "g--",
                label="Retrieval time",
            )
            ax1.plot(
                daily_data["date"],
                daily_data["generation_time_ms"],
                "r--",
                label="Generation time",
            )
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.legend(loc="upper left")

            # Number of queries
            ax2 = ax1.twinx()
            ax2.set_ylabel("Number of queries", color="tab:orange")
            ax2.bar(
                daily_data["date"],
                daily_data["query_count"],
                alpha=0.3,
                color="tab:orange",
                label="Queries",
            )
            ax2.tick_params(axis="y", labelcolor="tab:orange")
            ax2.legend(loc="upper right")

            fig.tight_layout()
            st.pyplot(fig)

        # Distribution by hour
        st.subheader("Performance distribution by hour")
        hourly_data = pd.DataFrame(analysis["hourly_metrics"])
        if not hourly_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="hour", y="total_time_ms", data=hourly_data, ax=ax, color="skyblue")
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Average response time (ms)")
            ax.set_title("Average response time by hour")
            st.pyplot(fig)

    def render_general_statistics(self) -> None:
        """Display general statistics about chatbot usage"""
        st.subheader("General Statistics")

        # Collect statistics from different sources
        stats = {}

        try:
            # Conversation statistics (single source of truth)
            conv_logs = self.conversation_analyzer.get_recent_logs(days=30)
            stats["total_conversations"] = len(conv_logs)

            # Performance statistics from conversation logs
            if conv_logs:
                avg_time = sum(log.get("response_time_ms", 0) for log in conv_logs) / len(conv_logs)
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
