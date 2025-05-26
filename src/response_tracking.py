import pandas as pd
import streamlit as st
import json
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from collections import Counter


class ResponseTracker:
    """Tracks questions without satisfactory responses and suggests improvements"""

    def __init__(self, log_path="./logs/responses"):
        # Convert relative path to absolute path
        if log_path.startswith("./"):
            import os

            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_path = os.path.join(base_dir, log_path[2:])  # Remove ./ prefix

        print(f"Response tracker initialized with log_path: {log_path}")

        # Ensure directory exists
        os.makedirs(log_path, exist_ok=True)

        # Set current log file path
        self.log_path = log_path
        self.current_log_file = os.path.join(log_path, f"response_log_{datetime.now().strftime('%Y%m%d')}.jsonl")
        print(f"Current log file: {self.current_log_file}")

        # Check if directory is writable
        try:
            test_file = os.path.join(log_path, "test_write.tmp")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print(f"Directory {log_path} is writable")
        except Exception as e:
            print(f"WARNING: Directory {log_path} may not be writable: {str(e)}")

        self.feedback_thresholds = {
            "negative": 2,  # Feedback score < 2 is considered negative
            "neutral": 3,  # Feedback score = 3 is considered neutral
            "positive": 4,  # Feedback score > 3 is considered positive
        }

    def log_response_quality(self, user_id, question, answer, sources, feedback_score, feedback_text=None):
        """Records the quality of a response based on user feedback

        Args:
            user_id: User identifier
            question: The question asked
            answer: The answer provided
            sources: The sources used for the answer
            feedback_score: 5 for thumbs up, 1 for thumbs down
            feedback_text: Optional feedback text
        """
        print("\n==== LOGGING FEEDBACK ====")
        print(f"Log file: {self.current_log_file}")
        print(f"User ID: {user_id}")
        print(f"Feedback score: {feedback_score}")

        # Determine if the response was satisfactory (thumbs up = satisfactory)
        is_satisfactory = feedback_score > 3  # thumbs up = 5, thumbs down = 1
        print(f"Is satisfactory: {is_satisfactory}")

        # Create the log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "question": question,
            "answer_snippet": answer[:100] + "..." if len(answer) > 100 else answer,
            "sources_count": len(sources) if sources else 0,
            "feedback_score": feedback_score,
            "feedback_text": feedback_text,
            "is_satisfactory": is_satisfactory,
            "feedback_type": "thumbs_up" if is_satisfactory else "thumbs_down",
        }

        print(f"Log entry created: {json.dumps(log_entry)[:150]}...")

        # Check if directory exists
        log_dir = os.path.dirname(self.current_log_file)
        if not os.path.exists(log_dir):
            print(f"Creating directory: {log_dir}")
            os.makedirs(log_dir, exist_ok=True)

        # Direct write approach for debugging
        try:
            print(f"Writing directly to file: {self.current_log_file}")
            with open(self.current_log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            print("✓ Successfully wrote feedback to file")
        except Exception as e:
            print(f"❌ Error writing to file: {str(e)}")
            import traceback

            print(traceback.format_exc())

            # Try with absolute path
            try:
                abs_path = os.path.abspath(self.current_log_file)
                print(f"Trying with absolute path: {abs_path}")
                with open(abs_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                print("✓ Successfully wrote feedback to file with absolute path")
            except Exception as e2:
                print(f"❌ Error writing to absolute path: {str(e2)}")

        print("==== END LOGGING FEEDBACK ====")

    def get_unsatisfactory_responses(self, days=30):
        """Retrieves unsatisfactory responses from the last n days"""
        unsatisfactory = []

        # Find all log files in the specified period
        for filename in os.listdir(self.log_path):
            if filename.startswith("response_log_") and filename.endswith(".jsonl"):
                file_path = os.path.join(self.log_path, filename)
                with open(file_path, "r") as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            log_date = datetime.fromisoformat(log_entry["timestamp"])
                            days_ago = (datetime.now() - log_date).days
                            if days_ago <= days and not log_entry.get("is_satisfactory", True):
                                unsatisfactory.append(log_entry)
                        except json.JSONDecodeError:
                            continue

        return unsatisfactory

    def identify_patterns(self, unsatisfactory_responses=None):
        """Identifies patterns in questions without satisfactory responses"""
        if unsatisfactory_responses is None:
            unsatisfactory_responses = self.get_unsatisfactory_responses()

        if not unsatisfactory_responses:
            return {"message": "No unsatisfactory responses found"}

        # extract the questions
        questions = [resp["question"] for resp in unsatisfactory_responses]

        # Use TF-IDF to find similarities
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(questions)

            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Find clusters of similar questions
            clusters = []
            processed = set()

            for i in range(len(questions)):
                if i in processed:
                    continue

                # Find similar questions (similarity threshold > 0.3)
                similar_indices = [j for j in range(len(questions)) if similarity_matrix[i, j] > 0.3 and i != j]

                if similar_indices:
                    cluster = [i] + similar_indices
                    clusters.append(
                        {
                            "main_question": questions[i],
                            "similar_questions": [questions[j] for j in similar_indices],
                            "count": len(similar_indices) + 1,
                        }
                    )
                    processed.update(cluster)
                else:
                    processed.add(i)

            # Extract the most important terms
            feature_names = vectorizer.get_feature_names_out()
            important_terms = []

            for i, row in enumerate(tfidf_matrix.toarray()):
                top_indices = row.argsort()[-5:][::-1]  # Top 5 terms
                terms = [feature_names[idx] for idx in top_indices if row[idx] > 0]
                if terms:
                    important_terms.extend(terms)

            return {
                "total_unsatisfactory": len(unsatisfactory_responses),
                "clusters": clusters,
                "common_terms": Counter(important_terms).most_common(10),
            }
        except Exception as e:
            return {"error": str(e), "message": "Error during pattern analysis"}  # Translated from French

    def render_tracking_dashboard(self):
        """Display the response tracking dashboard in Streamlit"""
        st.title("Suivi des réponses")

        # Period selection
        days = st.slider("Période d'analyse (jours)", 1, 90, 30, key="nonresponse_days")
        unsatisfactory = self.get_unsatisfactory_responses(days)

        if not unsatisfactory:
            st.warning("Aucune réponse insatisfaisante trouvée pour la période sélectionnée")
            return

        # Create columns for metrics
        col1, col2 = st.columns(2)

        # Basic metrics
        with col1:
            st.metric("Nombre de réponses insatisfaisantes", len(unsatisfactory))

        # Count feedback with comments
        with_comments = sum(1 for item in unsatisfactory if item.get("feedback_text"))
        with col2:
            st.metric("Réponses avec commentaires", with_comments)

        # Pattern analysis
        patterns = self.identify_patterns(unsatisfactory)

        if "error" in patterns:
            st.error(f"Erreur d'analyse: {patterns['error']}")
            return

        # Display clusters of similar questions
        if "clusters" in patterns and patterns["clusters"]:
            st.subheader("Groupes de questions similaires avec réponses insatisfaisantes")

            for i, cluster in enumerate(patterns["clusters"]):
                with st.expander(f"Groupe {i + 1}: {cluster['main_question']} ({cluster['count']} questions)"):
                    for q in cluster["similar_questions"]:
                        st.write(f"- {q}")

        # Display common terms
        if "common_terms" in patterns and patterns["common_terms"]:
            st.subheader("Termes fréquents dans les questions avec réponses insatisfaisantes")
            terms_df = pd.DataFrame(patterns["common_terms"], columns=["Terme", "Fréquence"])
            st.dataframe(terms_df)

        # Table of questions without satisfactory responses
        st.subheader("Détails des questions avec réponses insatisfaisantes")

        # Convert to DataFrame for cleaner display
        df = pd.DataFrame(
            [
                {
                    "Date": datetime.fromisoformat(item["timestamp"]).strftime("%Y-%m-%d %H:%M"),
                    "Question": item["question"],
                    "Feedback": "👎"
                    if item.get("feedback_type") == "thumbs_down"
                    else "👎",  # Should always be thumbs down
                    "Commentaire": item.get("feedback_text", ""),
                }
                for item in unsatisfactory
            ]
        )

        st.dataframe(df, use_container_width=True)


# Function to integrate the tracker in the main application
def integrate_response_tracker(help_desk, user_id):
    """Integrates the response tracker into the help_desk"""
    tracker = ResponseTracker()

    # Add the tracker as an attribute
    help_desk.response_tracker = tracker

    return help_desk
