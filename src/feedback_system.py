import streamlit as st
from streamlit_feedback import streamlit_feedback
import uuid
import json
import os
from datetime import datetime
from typing import Dict, List, Optional


class FeedbackSystem:
    """Fixed feedback system to avoid Streamlit rerun issues"""

    def __init__(self, feedback_file: Optional[str] = None):
        if feedback_file is None:
            # Create a filename with the current date
            date_str = datetime.now().strftime("%Y-%m-%d")
            feedback_file = f"./logs/feedback/feedback_{date_str}.json"
        self.feedback_file = feedback_file
        self._ensure_feedback_file_exists()

    def _ensure_feedback_file_exists(self):
        """Ensure the feedback file exists"""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, "w") as f:
                json.dump([], f)

    def _load_feedback_data(self) -> List[Dict]:
        """Load feedback data from all files of the last 30 days"""
        from datetime import datetime, timedelta

        all_feedback = []

        # Calculate the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        # Generate filenames for each day
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            feedback_file_path = f"./logs/feedback/feedback_{date_str}.json"

            # Load the file if it exists
            try:
                if os.path.exists(feedback_file_path):
                    with open(feedback_file_path, "r") as f:
                        daily_feedback = json.load(f)
                        if isinstance(daily_feedback, list):
                            all_feedback.extend(daily_feedback)
            except (json.JSONDecodeError, IOError) as e:
                # Ignore corrupted or inaccessible files
                print(f"Error loading {feedback_file_path}: {e}")
                continue

            current_date += timedelta(days=1)

        return all_feedback

    def _save_feedback_data(self, data: List[Dict]):
        """Save feedback data to file"""
        with open(self.feedback_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def render_feedback_widget(
        self, user_id: str, question: str, answer: str, sources: List[str], key_suffix: str = ""
    ) -> Optional[Dict]:
        """
        Display the fixed feedback widget to avoid rerun issues

        Args:
            user_id: User ID
            question: Question asked
            answer: Answer given
            sources: Sources used
            key_suffix: Suffix for the unique widget key

        Returns:
            Dictionary with feedback data if submitted
        """

        # Create a unique key based on content (more stable)
        content_hash = hash(f"{question}_{answer}_{key_suffix}")
        feedback_key = f"feedback_{content_hash}"

        # Persistent states for this specific feedback
        feedback_data_key = f"feedback_data_{feedback_key}"
        feedback_submitted_key = f"feedback_submitted_{feedback_key}"
        fbk_widget_key = f"fbk_widget_{feedback_key}"

        # Initialize feedback data if not done yet
        if feedback_data_key not in st.session_state:
            st.session_state[feedback_data_key] = {
                "user_id": user_id,
                "question": question,
                "answer": answer,
                "sources": sources,
                "timestamp": datetime.now().isoformat(),
                "feedback_submitted": False,
            }

        # Initialize submission state
        if feedback_submitted_key not in st.session_state:
            st.session_state[feedback_submitted_key] = False

        # Unique key for the widget
        if fbk_widget_key not in st.session_state:
            st.session_state[fbk_widget_key] = str(uuid.uuid4())

        def feedback_callback(response):
            """
            Callback to process submitted feedback

            Args:
                response: Response from the feedback widget
            """
            # Mark feedback as submitted
            st.session_state[feedback_submitted_key] = True

            # Update feedback data
            st.session_state[feedback_data_key]["feedback"] = response
            st.session_state[feedback_data_key]["feedback_submitted"] = True

            # Save to data file
            feedback_data = self._load_feedback_data()
            feedback_entry = {
                "user_id": user_id,
                "question": question,
                "answer": answer,
                "sources": sources,
                "feedback": response,
                "timestamp": datetime.now().isoformat(),
            }
            feedback_data.append(feedback_entry)
            self._save_feedback_data(feedback_data)

            # Generate a new key to avoid conflicts
            st.session_state[fbk_widget_key] = str(uuid.uuid4())

        # Display already submitted feedback if available
        if st.session_state[feedback_submitted_key]:
            return None

        # Display feedback widget if not yet submitted
        feedback_response = streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="[Optional] Comment:",
            align="flex-start",
            key=st.session_state[fbk_widget_key],
            on_submit=feedback_callback,
        )

        return feedback_response

    def get_feedback_statistics(self, days: int = 30) -> Dict:
        """
        Get feedback statistics for the last N days

        Args:
            days: Number of days to analyze (default 30, but _load_feedback_data already loads the last 30 days)

        Returns:
            Dictionary with statistics
        """
        feedback_data = self._load_feedback_data()

        if not feedback_data:
            return {"total_feedback": 0, "positive_feedback": 0, "negative_feedback": 0, "satisfaction_rate": 0.0}

        # If we want less than 30 days, filter by date
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
                    # Include entries without valid timestamp
                    recent_feedback.append(entry)
        else:
            # Use all data already filtered by _load_feedback_data
            recent_feedback = feedback_data

        total = len(recent_feedback)
        positive = 0
        negative = 0

        for entry in recent_feedback:
            feedback_obj = entry.get("feedback", {})
            score = feedback_obj.get("score")

            # Handle both emoji format ('👍'/'👎') and numeric format (1/0)
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

    def get_all_feedback(self) -> List[Dict]:
        """Get all recorded feedback"""
        return self._load_feedback_data()

    def get_feedback_by_user(self, user_id: str) -> List[Dict]:
        """Get feedback from a specific user"""
        all_feedback = self._load_feedback_data()
        return [entry for entry in all_feedback if entry.get("user_id") == user_id]
