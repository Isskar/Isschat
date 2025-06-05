import json
import os
from datetime import datetime
import streamlit as st


class FeedbackSystem:
    """New feedback system using thumbs up/down with st.feedback"""

    def __init__(self, log_path="./logs/feedback"):
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        self.current_log_file = os.path.join(log_path, f"feedback_log_{datetime.now().strftime('%Y%m%d')}.jsonl")

    def log_feedback(self, user_id: str, question: str, answer: str, documents: list, feedback_state: str):
        """Records user feedback in the log file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "documents_retrieved": documents if isinstance(documents, list) else [documents],
            "feedback_state": feedback_state,  # "satisfactory" or "unsatisfactory"
        }

        with open(self.current_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def render_feedback_widget(self, user_id: str, question: str, answer: str, sources: list, key_suffix: str = ""):
        """Render the thumbs up/down feedback widget using st.feedback"""

        # Create a unique key for this feedback widget
        feedback_key = f"feedback_{key_suffix}_{hash(question + answer)}"

        st.write("---")
        st.write("#### Rate this response")

        # Use st.feedback for thumbs up/down
        feedback = st.feedback("thumbs", key=feedback_key)

        if feedback is not None:
            # Convert feedback to our format
            feedback_state = "satisfactory" if feedback == 1 else "unsatisfactory"

            # Log the feedback
            self.log_feedback(
                user_id=user_id,
                question=question,
                answer=answer,
                documents=sources if isinstance(sources, list) else [sources],
                feedback_state=feedback_state,
            )

            # Show confirmation message
            if feedback == 1:
                st.success("👍 Thank you for your positive feedback!")
            else:
                st.info("👎 Thank you for your feedback. We'll work to improve!")

            return feedback_state

        return None

    def get_feedback_logs(self, days=30):
        """Retrieves feedback logs from the last n days"""
        logs = []

        # Find all log files in the specified period
        for filename in os.listdir(self.log_path):
            if filename.startswith("feedback_log_") and filename.endswith(".jsonl"):
                file_path = os.path.join(self.log_path, filename)
                with open(file_path, "r") as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            log_date = datetime.fromisoformat(log_entry["timestamp"])
                            days_ago = (datetime.now() - log_date).days
                            if days_ago <= days:
                                logs.append(log_entry)
                        except json.JSONDecodeError:
                            continue

        return logs

    def get_feedback_statistics(self, days=30):
        """Get feedback statistics for the dashboard"""
        logs = self.get_feedback_logs(days)

        if not logs:
            return {"total_feedback": 0, "satisfactory": 0, "unsatisfactory": 0, "satisfaction_rate": 0}

        satisfactory_count = len([log for log in logs if log["feedback_state"] == "satisfactory"])
        unsatisfactory_count = len([log for log in logs if log["feedback_state"] == "unsatisfactory"])

        return {
            "total_feedback": len(logs),
            "satisfactory": satisfactory_count,
            "unsatisfactory": unsatisfactory_count,
            "satisfaction_rate": (satisfactory_count / len(logs)) * 100 if logs else 0,
        }

    def render_feedback_dashboard(self):
        """Display the feedback dashboard in Streamlit"""
        st.title("Feedback Analysis")

        # Period selection
        days = st.slider("Analysis period (days)", 1, 90, 30, key="feedback_days")
        logs = self.get_feedback_logs(days)

        if not logs:
            st.warning("No feedback data available for the selected period")
            return

        # Get statistics
        stats = self.get_feedback_statistics(days)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Feedback", stats["total_feedback"])
        with col2:
            st.metric("👍 Satisfactory", stats["satisfactory"])
        with col3:
            st.metric("👎 Unsatisfactory", stats["unsatisfactory"])
        with col4:
            st.metric("Satisfaction Rate", f"{stats['satisfaction_rate']:.1f}%")

        # Show unsatisfactory feedback details
        if stats["unsatisfactory"] > 0:
            st.subheader("Unsatisfactory Responses")

            unsatisfactory_logs = [log for log in logs if log["feedback_state"] == "unsatisfactory"]

            for i, log in enumerate(unsatisfactory_logs):
                with st.expander(f"Feedback {i + 1}: {log['question'][:50]}..."):
                    st.write(f"**Question:** {log['question']}")
                    st.write(f"**Answer:** {log['answer'][:200]}...")
                    st.write(f"**Documents:** {', '.join(log['documents_retrieved'])}")
                    st.write(f"**Date:** {datetime.fromisoformat(log['timestamp']).strftime('%Y-%m-%d %H:%M')}")
