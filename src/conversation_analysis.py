import pandas as pd
import json
import os
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import streamlit as st

class ConversationAnalyzer:
    """Analyzes conversations and provides insights on user-chatbot interactions"""
    
    def __init__(self, log_path="./logs/conversations"):
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        self.current_log_file = os.path.join(log_path, f"conv_log_{datetime.now().strftime('%Y%m%d')}.jsonl")
    
    def log_interaction(self, user_id, question, answer, sources, response_time, feedback=None):
        """Records an interaction in the log file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "question": question,
            "answer_length": len(answer),
            "sources_count": len(sources) if sources else 0,
            "response_time_ms": response_time,
            "feedback": feedback
        }
        
        with open(self.current_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def get_recent_logs(self, days=7):
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
    
    def analyze_questions(self, logs=None):
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
            "avg_response_time_ms": avg_response_time
        }
    
    def render_analysis_dashboard(self):
        """Display the analysis dashboard in Streamlit"""
        st.title("Conversation Analysis")
        
        # Period selection
        days = st.slider("Analysis period (days)", 1, 30, 7)
        logs = self.get_recent_logs(days)
        
        if not logs:
            st.warning("No data available for the selected period")
            return
        
        # Analysis
        analysis = self.analyze_questions(logs)
        
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
        
        hour_df = pd.DataFrame({
            "Hour": hours,
            "Number of questions": counts
        })
        st.bar_chart(hour_df.set_index("Hour"))
        
        # Most frequent words
        st.subheader("Most frequent keywords")
        if analysis["common_words"]:
            keywords_df = pd.DataFrame(analysis["common_words"], columns=["Word", "Frequency"])
            st.dataframe(keywords_df)
        else:
            st.info("Not enough data for keyword analysis")


# Function to integrate the analyzer in the main application
def integrate_conversation_analyzer(help_desk, user_id):
    """Integrates the conversation analyzer into the help_desk"""
    analyzer = ConversationAnalyzer()
    
    # Wrapper function for ask_question that logs interactions
    original_ask = help_desk.ask_question
    
    def ask_with_logging(question, verbose=False):
        start_time = datetime.now()
        answer, sources = original_ask(question, verbose)
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds() * 1000  # in milliseconds
        
        # Log the interaction
        analyzer.log_interaction(
            user_id=user_id,
            question=question,
            answer=answer,
            sources=sources,
            response_time=response_time
        )
        
        return answer, sources
    
    # Replace the original method
    help_desk.ask_question = ask_with_logging
    
    # Add the analyzer as an attribute
    help_desk.analyzer = analyzer
    
    return help_desk
