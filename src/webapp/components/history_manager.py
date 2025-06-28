"""
Conversation History Manager for Streamlit interface.
Integrates with the new data management system.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import plotly.express as px
from itertools import groupby

from src.storage.data_manager import get_data_manager


class ConversationHistoryManager:
    """Conversation history manager for Streamlit interface."""

    def __init__(self):
        self.data_manager = get_data_manager()
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "success": "#F18F01",
            "warning": "#C73E1D",
            "info": "#6A994E",
        }
        self._apply_custom_styling()

    def _apply_custom_styling(self):
        """Apply custom dark theme styling."""
        st.markdown(
            """
        <style>
        .conversation-card {
            background: linear-gradient(135deg, #2D2D2D 0%, #3D3D3D 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid #404040;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }

        .search-highlight {
            background-color: #F18F01;
            color: #000000;
            padding: 2px 4px;
            border-radius: 3px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    def render_history_page(self, user_id: Optional[str] = None):
        """Render the conversation history page."""
        st.title("Conversation History")

        # Sidebar for filters
        with st.sidebar:
            st.subheader("Filters")

            # Period filter
            period_options = {
                "Today": 1,
                "Last 7 days": 7,
                "Last 30 days": 30,
                "All history": None,
            }

            selected_period = st.selectbox(
                "Time Period",
                options=list(period_options.keys()),
                index=3,  # Changed default to "All history" (index 3)
            )

            # User filter (for admins)
            show_all_users = st.checkbox("Show all users", value=False)
            if show_all_users:
                user_id = None

            # Conversation limit
            limit = st.slider("Max conversations", 10, 200, 50)

        # Get filtered data
        conversations = self._get_filtered_conversations(
            user_id=user_id, period_days=period_options[selected_period], limit=limit
        )

        if not conversations:
            st.info("No conversations found for the selected criteria.")
            return

        # Tabs for different views
        tab1, tab2 = st.tabs(["Conversation List", "Search"])

        with tab1:
            self._render_conversations_list(conversations)

        with tab2:
            self._render_search_interface(conversations)

    def _get_filtered_conversations(
        self, user_id: Optional[str] = None, period_days: Optional[int] = None, limit: int = 50
    ) -> List[Dict]:
        """Get filtered conversations."""
        conversations = self.data_manager.get_conversation_history(user_id=user_id, limit=limit)

        if period_days and conversations:
            cutoff_date = datetime.now() - timedelta(days=period_days)
            conversations = [conv for conv in conversations if datetime.fromisoformat(conv["timestamp"]) >= cutoff_date]

        return conversations

    def _render_conversations_list(self, conversations: List[Dict]):
        """Render the conversations list."""
        st.markdown("### Recent Conversations")

        if not conversations:
            st.info("No conversations to display.")
            return

        # Display options
        col1, col2 = st.columns([3, 1])
        with col1:
            show_details = st.checkbox("Show full details", value=False, key="show_details_history_list")
        with col2:
            export_btn = st.button("Export CSV", key="export_csv_history_list")

        if export_btn:
            self._export_conversations_csv(conversations)

        # Group conversations by conversation_id
        # First, sort by conversation_id and then by timestamp to ensure consistent grouping and order within groups
        conversations.sort(key=lambda x: (x.get("conversation_id", ""), x.get("timestamp", "")))

        grouped_conversations = {}
        for k, g in groupby(conversations, lambda x: x.get("conversation_id", "")):
            grouped_conversations[k] = list(g)

        # Sort conversation groups by the timestamp of their first message (most recent conversation first)
        sorted_conversation_groups = sorted(
            grouped_conversations.items(),
            key=lambda item: item[1][0].get("timestamp", ""),  # Use timestamp of the first message in the group
            reverse=True,
        )

        # Display conversations
        for i, (conversation_id, conv_entries) in enumerate(sorted_conversation_groups):
            # Get the first message for summary display
            first_entry = conv_entries[0]
            num_messages = len(conv_entries)
            total_response_time = sum(e.get("response_time_ms", 0) for e in conv_entries)

            with st.expander(
                f"💬 Conversation with {num_messages} messages "
                f"({self._format_timestamp(first_entry['timestamp'])}) - "
                f"👤 {first_entry.get('user_id', 'Anonymous')} - "
                f"⏱️ {total_response_time:.0f}ms",
                expanded=False,
            ):
                # Display all entries within this conversation
                for j, entry in enumerate(conv_entries):
                    st.markdown(f"**❓ Question ({self._format_timestamp(entry['timestamp'])}):**")
                    st.write(entry["question"])

                    st.markdown("**💡 Answer:**")
                    if show_details:
                        st.write(entry["answer"])
                    else:
                        preview = entry["answer"][:200] + "..." if len(entry["answer"]) > 200 else entry["answer"]
                        st.write(preview)

                    # Display metadata for each turn
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Response Length", f"{entry.get('answer_length', 0)} chars")
                    with col2:
                        st.metric("Sources", entry.get("sources_count", 0))
                    with col3:
                        st.metric("Response Time", f"{entry.get('response_time_ms', 0):.0f}ms")
                    with col4:
                        feedback = entry.get("feedback")
                        if feedback:
                            st.metric("Rating", f"{feedback.get('rating', 'N/A')}/5")
                        else:
                            st.metric("Rating", "Not rated")

                    if show_details and entry.get("sources"):
                        st.markdown("**Sources used:**")
                        for k, source in enumerate(entry["sources"][:3]):
                            title = source.get("title", "Unknown source")
                            url = source.get("url", "")
                            if url and url != "#":
                                st.markdown(f"{k + 1}. [{title}]({url})")
                            else:
                                st.write(f"{k + 1}. {title}")
                    st.markdown("---")  # Separator between turns

                # Button to continue this conversation
                if st.button("Continue this conversation", key=f"continue_conv_{conversation_id}_{i}"):
                    st.session_state["reuse_conversation_id"] = conversation_id
                    st.session_state["page"] = "chat"
                    st.rerun()

    def _render_statistics(self, conversations: List[Dict]):
        """Render conversation statistics."""
        st.markdown("### Conversation Statistics")

        if not conversations:
            st.info("No data available for statistics.")
            return

        # General metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Conversations", len(conversations))

        with col2:
            avg_response_time = sum(c.get("response_time_ms", 0) for c in conversations) / len(conversations)
            st.metric("Avg Response Time", f"{avg_response_time:.0f}ms")

        with col3:
            avg_length = sum(c.get("answer_length", 0) for c in conversations) / len(conversations)
            st.metric("Avg Length", f"{avg_length:.0f} chars")

        with col4:
            total_sources = sum(c.get("sources_count", 0) for c in conversations)
            st.metric("Total Sources Used", total_sources)

        # Charts
        st.markdown("### Trends")

        # Prepare data for charts
        df = pd.DataFrame(conversations)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["date"] = df["timestamp"].dt.date

            # Daily conversations chart
            daily_counts = df.groupby("date").size().reset_index(name="count")
            fig_daily = px.line(
                daily_counts,
                x="date",
                y="count",
                title="Daily Conversations",
                labels={"date": "Date", "count": "Number of conversations"},
                color_discrete_sequence=[self.colors["primary"]],
            )
            fig_daily.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_daily, use_container_width=True)

            # User engagement over time
            if len(set(conv.get("user_id") for conv in conversations)) > 1:
                user_daily = df.groupby(["date", "user_id"]).size().reset_index(name="count")
                fig_users = px.bar(
                    user_daily,
                    x="date",
                    y="count",
                    color="user_id",
                    title="User Engagement Over Time",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                fig_users.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig_users, use_container_width=True)

    def _render_search_interface(self, conversations: List[Dict]):
        """Render the search interface."""
        st.markdown("### Search Conversations")

        # Search interface
        search_term = st.text_input("Search in questions and answers:")

        if search_term:
            # Filter conversations
            filtered_conversations = []
            for conv in conversations:
                if (
                    search_term.lower() in conv.get("question", "").lower()
                    or search_term.lower() in conv.get("answer", "").lower()
                ):
                    filtered_conversations.append(conv)

            st.write(f"**{len(filtered_conversations)} result(s) found**")

            # Display results
            for i, conv in enumerate(filtered_conversations[:10]):  # Limit to 10 results
                with st.expander(
                    f"🕐 {self._format_timestamp(conv['timestamp'])} - 👤 {conv.get('user_id', 'Anonymous')}",
                    expanded=False,
                ):
                    st.markdown("**❓ Question:**")
                    question = conv["question"]
                    # Highlight search term
                    highlighted_question = question.replace(
                        search_term, f'<span class="search-highlight">{search_term}</span>'
                    )
                    st.markdown(highlighted_question, unsafe_allow_html=True)

                    st.markdown("**Answer:**")
                    answer = conv["answer"][:300] + "..." if len(conv["answer"]) > 300 else conv["answer"]
                    highlighted_answer = answer.replace(
                        search_term, f'<span class="search-highlight">{search_term}</span>'
                    )
                    st.markdown(highlighted_answer, unsafe_allow_html=True)

                    # Metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Response Time:** {conv.get('response_time_ms', 0):.0f}ms")
                    with col2:
                        st.write(f"**Sources:** {conv.get('sources_count', 0)}")
                    with col3:
                        feedback = conv.get("feedback")
                        rating = feedback.get("rating", "N/A") if feedback else "Not rated"
                        st.write(f"**Rating:** {rating}")

                    if st.button("Reuse", key=f"search_reuse_{i}"):
                        st.session_state["reuse_question"] = conv["question"]
                        st.session_state["page"] = "chat"
                        st.rerun()
        else:
            st.info("Enter a search term to find relevant conversations.")

            # Show popular topics when no search
            st.markdown("#### Popular Topics")
            self._render_popular_topics(conversations)

    def _render_popular_topics(self, conversations: List[Dict]):
        """Render popular topics analysis."""
        # Extract keywords from questions
        all_words = []
        for conv in conversations:
            words = conv.get("question", "").lower().split()
            # Filter short words and common words
            filtered_words = [
                word.strip(".,!?;:")
                for word in words
                if len(word) > 3 and word not in ["what", "how", "when", "where", "why", "with", "from", "this", "that"]
            ]
            all_words.extend(filtered_words)

        if all_words:
            from collections import Counter

            word_counts = Counter(all_words)
            top_words = word_counts.most_common(8)

            if top_words:
                col1, col2 = st.columns(2)
                for i, (word, count) in enumerate(top_words):
                    with col1 if i % 2 == 0 else col2:
                        st.write(f"**{word.title()}**: {count} mentions")
        else:
            st.info("No topic data available")

    def _format_timestamp(self, timestamp_str: str) -> str:
        """Format a timestamp for display."""
        try:
            dt = datetime.fromisoformat(timestamp_str)
            return dt.strftime("%d/%m/%Y %H:%M")
        except ValueError:
            return timestamp_str

    def _export_conversations_csv(self, conversations: List[Dict]):
        """Export conversations to CSV."""
        if not conversations:
            st.warning("No conversations to export.")
            return

        df = pd.DataFrame(conversations)
        csv = df.to_csv(index=False)

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    def save_conversation(
        self, user_id: str, question: str, answer: str, response_time_ms: float, sources: Optional[List[Dict]] = None
    ) -> bool:
        """Save a new conversation."""
        return self.data_manager.save_conversation(
            user_id=user_id, question=question, answer=answer, response_time_ms=response_time_ms, sources=sources
        )

    def add_feedback_to_conversation(self, conversation_id: str, user_id: str, rating: int, comment: str = "") -> bool:
        """Add feedback to a conversation."""
        return self.data_manager.save_feedback(
            user_id=user_id, conversation_id=conversation_id, rating=rating, comment=comment
        )


# Global instance of the history manager
_history_manager = None


def get_history_manager() -> Optional[ConversationHistoryManager]:
    """Return the global instance of the history manager."""
    global _history_manager
    if _history_manager is None:
        _history_manager = ConversationHistoryManager()
    return _history_manager
