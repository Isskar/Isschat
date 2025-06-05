import json
import os
from datetime import datetime
from collections import Counter


class ConversationAnalyzer:
    """Analyzes conversations and provides insights on user-chatbot interactions"""

    def __init__(self, log_path: str = "./logs/conversations") -> None:
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        self.current_log_file = os.path.join(log_path, f"conv_log_{datetime.now().strftime('%Y%m%d')}.jsonl")

    def log_interaction(
        self,
        user_id: str,
        question: str,
        answer: str,
        sources: list[str],
        response_time: float,
        feedback: str | None = None,
    ) -> None:
        """Records an interaction in the log file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "question": question,
            "answer_length": len(answer),
            "sources_count": len(sources) if sources else 0,
            "response_time_ms": response_time,
            "feedback": feedback,
        }

        with open(self.current_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def get_recent_logs(self, days: int = 7) -> list[dict]:
        """Retrieves logs from the last n days"""
        logs = []

        # Find all log files in the specified period
        for filename in os.listdir(self.log_path):
            if filename.startswith("conv_log_") and filename.endswith(".jsonl"):
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

    def analyze_questions(self, logs: list[dict] | None = None) -> dict:
        """Analyzes the questions asked"""
        if logs is None:
            logs = self.get_recent_logs()

        if not logs:
            return {"message": "No data available for analysis"}

        # Extract questions
        questions = [log["question"] for log in logs]

        # Simple keyword analysis
        words = []
        for q in questions:
            words.extend([w.lower() for w in q.split() if len(w) > 3])

        common_words = Counter(words).most_common(10)

        # Analysis of time of day
        hours = [datetime.fromisoformat(log["timestamp"]).hour for log in logs]
        hour_counts = Counter(hours)

        # Average response time
        avg_response_time = sum(log.get("response_time_ms", 0) for log in logs) / len(logs)

        return {
            "total_questions": len(questions),
            "common_words": common_words,
            "hour_distribution": dict(hour_counts),
            "avg_response_time_ms": avg_response_time,
        }

    # Dashboard rendering method moved to src/dashboard.py for centralization


# Function removed - conversation analysis is now integrated through FeaturesManager
# to avoid multiple wrapping of the ask_question method
