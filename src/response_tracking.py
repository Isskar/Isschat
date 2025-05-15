import pandas as pd
import streamlit as st
import json
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResponseTracker:
    """Tracks questions without satisfactory responses and suggests improvements"""

    def __init__(self, log_path="./logs/responses"):
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        self.current_log_file = os.path.join(
            log_path, f"response_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        self.feedback_thresholds = {
            "negative": 2,  # Feedback score < 2 is considered negative
            "neutral": 3,  # Feedback score = 3 is considered neutral
            "positive": 4,  # Feedback score > 3 is considered positive
        }

    def log_response_quality(
        self, user_id, question, answer, sources, feedback_score, feedback_text=None
    ):
        """Records the quality of a response based on user feedback"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "question": question,
            "answer_snippet": answer[:100] + "..." if len(answer) > 100 else answer,
            "sources_count": len(sources) if sources else 0,
            "feedback_score": feedback_score,
            "feedback_text": feedback_text,
            "is_satisfactory": feedback_score >= self.feedback_thresholds["neutral"],
        }

        with open(self.current_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

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
                            if days_ago <= days and not log_entry.get(
                                "is_satisfactory", True
                            ):
                                unsatisfactory.append(log_entry)
                        except json.JSONDecodeError:
                            continue

        return unsatisfactory

    def identify_patterns(self, unsatisfactory_responses=None):
        """Identifies patterns in questions without satisfactory responses"""
        if unsatisfactory_responses is None:
            unsatisfactory_responses = self.get_unsatisfactory_responses()

        if not unsatisfactory_responses:
            return {"message": "Aucune réponse insatisfaisante trouvée"}

        # Extraire les questions
        questions = [resp["question"] for resp in unsatisfactory_responses]

        # Use TF-IDF to find similarities
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(questions)

            # Calculer la matrice de similarité
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Trouver des clusters de questions similaires
            clusters = []
            processed = set()

            for i in range(len(questions)):
                if i in processed:
                    continue

                # Trouver les questions similaires (seuil de similarité > 0.3)
                similar_indices = [
                    j
                    for j in range(len(questions))
                    if similarity_matrix[i, j] > 0.3 and i != j
                ]

                if similar_indices:
                    cluster = [i] + similar_indices
                    clusters.append(
                        {
                            "main_question": questions[i],
                            "similar_questions": [
                                questions[j] for j in similar_indices
                            ],
                            "count": len(similar_indices) + 1,
                        }
                    )
                    processed.update(cluster)
                else:
                    processed.add(i)

            # Extraire les termes les plus importants
            feature_names = vectorizer.get_feature_names_out()
            important_terms = []

            for i, row in enumerate(tfidf_matrix.toarray()):
                top_indices = row.argsort()[-5:][::-1]  # Top 5 termes
                terms = [feature_names[idx] for idx in top_indices if row[idx] > 0]
                if terms:
                    important_terms.extend(terms)

            return {
                "total_unsatisfactory": len(unsatisfactory_responses),
                "clusters": clusters,
                "common_terms": Counter(important_terms).most_common(10),
            }
        except Exception as e:
            return {"error": str(e), "message": "Erreur lors de l'analyse des patterns"}

    def render_tracking_dashboard(self):
        """Display the response tracking dashboard in Streamlit"""
        st.title("Response Tracking")

        # Period selection
        days = st.slider("Analysis period (days)", 1, 90, 30, key="nonresponse_days")
        unsatisfactory = self.get_unsatisfactory_responses(days)

        if not unsatisfactory:
            st.warning("No unsatisfactory responses found for the selected period")
            return

        # Basic metrics
        st.metric("Number of unsatisfactory responses", len(unsatisfactory))

        # Pattern analysis
        patterns = self.identify_patterns(unsatisfactory)

        if "error" in patterns:
            st.error(f"Analysis error: {patterns['error']}")
            return

        # Display clusters of similar questions
        if "clusters" in patterns and patterns["clusters"]:
            st.subheader("Groups of similar questions without satisfactory responses")

            for i, cluster in enumerate(patterns["clusters"]):
                with st.expander(
                    f"Group {i + 1}: {cluster['main_question']} ({cluster['count']} questions)"
                ):
                    for q in cluster["similar_questions"]:
                        st.write(f"- {q}")

        # Display common terms
        if "common_terms" in patterns and patterns["common_terms"]:
            st.subheader("Frequent terms in questions without satisfactory responses")
            terms_df = pd.DataFrame(
                patterns["common_terms"], columns=["Term", "Frequency"]
            )
            st.dataframe(terms_df)

        # Table of questions without satisfactory responses
        st.subheader("Details of questions without satisfactory responses")

        # Convert to DataFrame for cleaner display
        df = pd.DataFrame(
            [
                {
                    "Date": datetime.fromisoformat(item["timestamp"]).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "Question": item["question"],
                    "Score": item["feedback_score"],
                    "Comment": item.get("feedback_text", ""),
                }
                for item in unsatisfactory
            ]
        )

        st.dataframe(df)


# Function to integrate the tracker in the main application
def integrate_response_tracker(help_desk, user_id):
    """Integrates the response tracker into the help_desk"""
    tracker = ResponseTracker()

    # Add the tracker as an attribute
    help_desk.response_tracker = tracker

    return help_desk


# Function to add a feedback widget in Streamlit
def add_feedback_widget(st, help_desk, user_id, question, answer, sources):
    """Adds a feedback widget for the response"""
    st.write("---")
    st.write("### Rate this response")

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        feedback_score = st.slider("Response quality", 1, 5, 3)

    with col2:
        if st.button("Send feedback"):
            feedback_text = st.session_state.get("feedback_text", "")
            help_desk.response_tracker.log_response_quality(
                user_id=user_id,
                question=question,
                answer=answer,
                sources=sources,
                feedback_score=feedback_score,
                feedback_text=feedback_text,
            )
            st.success("Thank you for your feedback!")

    with col3:
        if feedback_score <= 3:
            feedback_text = st.text_area(
                "Comment (optional)",
                key="feedback_text",
                placeholder="Tell us why this response was not satisfactory",
            )


from collections import Counter
