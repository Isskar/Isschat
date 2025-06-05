import json
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
import logging


class FeedbackSystem:
    """Unified feedback system for collecting and managing user feedback on chatbot responses"""

    def __init__(self, log_path: str = "./logs/feedback") -> None:
        """Initialize the feedback system with unified storage"""
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        self.current_log_file = os.path.join(log_path, f"feedback_log_{datetime.now().strftime('%Y%m%d')}.jsonl")

        # Setup logging
        self.logger = logging.getLogger("feedback_system")

    def log_feedback(
        self,
        user_id: str,
        question: str,
        answer: str,
        sources: List[str],
        feedback_type: str,  # "positive" or "negative"
        comment: Optional[str] = None,
        session_id: Optional[str] = None,
        version: Optional[str] = None,
    ) -> bool:
        """Log user feedback with structured format"""
        # Ensure sources is a list
        if isinstance(sources, str):
            sources = [sources]
        elif sources is None:
            sources = []

        print(f"DEBUG: log_feedback called with feedback_type: {feedback_type}")
        print(f"DEBUG: user_id: {user_id}, comment: {comment}")
        print(f"DEBUG: sources type: {type(sources)}, value: {sources}")

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "question": question,
            "answer_snippet": answer[:200] + "..." if len(answer) > 200 else answer,
            "sources_count": len(sources) if sources else 0,
            "feedback_type": feedback_type,  # "positive" or "negative"
            "comment": comment,
            "session_id": session_id,
            "version": version,
            "sources": sources[:3] if sources else [],  # Store first 3 sources for reference
        }

        print(f"DEBUG: log_entry created: {log_entry}")

        try:
            # Ensure the directory exists
            os.makedirs(self.log_path, exist_ok=True)

            with open(self.current_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                f.flush()  # Force write to disk

            self.logger.info(f"Feedback logged: {feedback_type} for user {user_id} to {self.current_log_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error logging feedback: {str(e)}")
            # Also print to console for debugging
            print(f"FEEDBACK ERROR: Failed to log feedback: {str(e)}")
            return False

    def get_feedback_logs(self, days: int = 30) -> List[Dict[str, Any]]:
        """Retrieve feedback logs from the last n days"""
        logs = []

        try:
            for filename in os.listdir(self.log_path):
                if filename.startswith("feedback_log_") and filename.endswith(".jsonl"):
                    file_path = os.path.join(self.log_path, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                log_entry = json.loads(line.strip())
                                log_date = datetime.fromisoformat(log_entry["timestamp"])
                                days_ago = (datetime.now() - log_date).days
                                if days_ago <= days:
                                    logs.append(log_entry)
                            except (json.JSONDecodeError, KeyError, ValueError):
                                continue
        except Exception as e:
            self.logger.error(f"Error reading feedback logs: {str(e)}")

        return logs

    def get_satisfaction_rate(self, days: int = 30) -> Dict[str, Any]:
        """Calculate satisfaction rate based on feedback"""
        logs = self.get_feedback_logs(days)

        if not logs:
            return {"positive": 0, "negative": 0, "total": 0, "satisfaction_rate": 0.0}

        positive_count = sum(1 for log in logs if log.get("feedback_type") == "positive")
        negative_count = sum(1 for log in logs if log.get("feedback_type") == "negative")
        total_count = len(logs)

        satisfaction_rate = (positive_count / total_count * 100) if total_count > 0 else 0.0

        return {
            "positive": positive_count,
            "negative": negative_count,
            "total": total_count,
            "satisfaction_rate": satisfaction_rate,
        }

    def get_negative_feedback(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get all negative feedback for analysis"""
        logs = self.get_feedback_logs(days)
        return [log for log in logs if log.get("feedback_type") == "negative"]

    def render_feedback_widget(
        self,
        st_instance,
        question: str,
        answer: str,
        sources: List[str],
        user_id: str,
        session_id: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        """Render simple and reliable feedback widget"""
        # Debug logging
        print(f"🔍 FEEDBACK WIDGET: Starting for user {user_id}")
        print(f"🔍 FEEDBACK WIDGET: Question length: {len(question)}")
        print(f"🔍 FEEDBACK WIDGET: Log file: {self.current_log_file}")

        # Create unique identifier for this Q&A pair
        qa_hash = str(hash(question + answer))
        feedback_given_key = f"feedback_given_{qa_hash}"

        # Always show feedback section header
        st_instance.markdown("---")
        st_instance.markdown("### 📝 How was this response?")

        # Check if feedback already given
        if st_instance.session_state.get(feedback_given_key, False):
            st_instance.success("✅ Thank you! Your feedback has been recorded.")
            return

        # Simple expander approach to avoid button disappearing issues
        with st_instance.expander("👍👎 Rate this response", expanded=True):
            st_instance.write("Please rate the quality of this response:")

            # Use columns for side-by-side buttons
            col1, col2 = st_instance.columns(2)

            # Positive feedback button
            with col1:
                if st_instance.button("👍 GOOD RESPONSE", key=f"pos_{qa_hash}", type="primary"):
                    print("🔍 POSITIVE BUTTON CLICKED!")

                    # Log immediately with verbose output
                    print(f"🔍 Logging positive feedback for user: {user_id}")
                    success = self.log_feedback(
                        user_id=user_id,
                        question=question,
                        answer=answer,
                        sources=sources,
                        feedback_type="positive",
                        comment=None,
                        session_id=session_id,
                        version=version,
                    )

                    if success:
                        print("🔍 SUCCESS: Positive feedback logged!")
                        st_instance.session_state[feedback_given_key] = True
                        st_instance.balloons()  # Fun visual feedback
                        st_instance.success("✅ Thank you for the positive feedback!")
                    else:
                        print("❌ FAILED: Could not log positive feedback")
                        st_instance.error("❌ Failed to save feedback. Please try again.")

            # Negative feedback button
            with col2:
                if st_instance.button("👎 NEEDS IMPROVEMENT", key=f"neg_{qa_hash}"):
                    print("🔍 NEGATIVE BUTTON CLICKED!")

                    # Show comment input immediately
                    st_instance.session_state[f"show_neg_comment_{qa_hash}"] = True

        # Handle negative feedback with comment
        if st_instance.session_state.get(f"show_neg_comment_{qa_hash}", False):
            st_instance.markdown("### 💬 Help us improve")
            st_instance.write("What could be improved? (optional)")

            # Comment input
            comment = st_instance.text_area(
                "Your feedback:",
                key=f"neg_comment_{qa_hash}",
                placeholder="Tell us specifically what could be better...",
                height=100,
            )

            # Submit buttons
            col1, col2 = st_instance.columns(2)
            with col1:
                if st_instance.button("📝 Submit with Comment", key=f"submit_comment_{qa_hash}", type="primary"):
                    print("🔍 SUBMIT WITH COMMENT CLICKED!")
                    print(f"🔍 Comment: '{comment}'")

                    success = self.log_feedback(
                        user_id=user_id,
                        question=question,
                        answer=answer,
                        sources=sources,
                        feedback_type="negative",
                        comment=comment,
                        session_id=session_id,
                        version=version,
                    )

                    if success:
                        print("🔍 SUCCESS: Negative feedback with comment logged!")
                        st_instance.session_state[feedback_given_key] = True
                        if f"show_neg_comment_{qa_hash}" in st_instance.session_state:
                            del st_instance.session_state[f"show_neg_comment_{qa_hash}"]
                        st_instance.success("✅ Thank you for the detailed feedback!")
                    else:
                        print("❌ FAILED: Could not log negative feedback with comment")
                        st_instance.error("❌ Failed to save feedback. Please try again.")

            with col2:
                if st_instance.button("⏭️ Submit without Comment", key=f"submit_no_comment_{qa_hash}"):
                    print("🔍 SUBMIT WITHOUT COMMENT CLICKED!")

                    success = self.log_feedback(
                        user_id=user_id,
                        question=question,
                        answer=answer,
                        sources=sources,
                        feedback_type="negative",
                        comment=None,
                        session_id=session_id,
                        version=version,
                    )

                    if success:
                        print("🔍 SUCCESS: Negative feedback without comment logged!")
                        st_instance.session_state[feedback_given_key] = True
                        if f"show_neg_comment_{qa_hash}" in st_instance.session_state:
                            del st_instance.session_state[f"show_neg_comment_{qa_hash}"]
                        st_instance.success("✅ Thank you for the feedback!")
                    else:
                        print("❌ FAILED: Could not log negative feedback without comment")
                        st_instance.error("❌ Failed to save feedback. Please try again.")

    def render_feedback_dashboard(self, st_instance) -> None:
        """Render feedback analytics dashboard"""
        st_instance.title("Feedback Analytics")

        # Period selection
        days = st_instance.slider("Analysis period (days)", 1, 90, 30, key="feedback_days")

        # Get satisfaction metrics
        satisfaction_data = self.get_satisfaction_rate(days)

        # Display metrics
        col1, col2, col3, col4 = st_instance.columns(4)

        with col1:
            st_instance.metric("Total Feedback", satisfaction_data["total"])
        with col2:
            st_instance.metric("👍 Positive", satisfaction_data["positive"])
        with col3:
            st_instance.metric("👎 Negative", satisfaction_data["negative"])
        with col4:
            st_instance.metric("Satisfaction Rate", f"{satisfaction_data['satisfaction_rate']:.1f}%")

        # Negative feedback analysis
        negative_feedback = self.get_negative_feedback(days)

        if negative_feedback:
            st_instance.subheader("Recent Negative Feedback")

            for feedback in negative_feedback[-10:]:  # Show last 10
                with st_instance.expander(f"🕒 {feedback['timestamp'][:16]} - {feedback['question'][:50]}..."):
                    st_instance.write(f"**Question:** {feedback['question']}")
                    st_instance.write(f"**Answer snippet:** {feedback['answer_snippet']}")
                    if feedback.get("comment"):
                        st_instance.write(f"**User comment:** {feedback['comment']}")
                    st_instance.write(f"**Sources:** {feedback['sources_count']}")
        else:
            st_instance.info("No negative feedback found for the selected period.")
