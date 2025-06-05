import json
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


class ResponseTracker:
    """Tracks questions without satisfactory responses and suggests improvements"""

    def __init__(self, log_path="./logs/responses"):
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        self.current_log_file = os.path.join(log_path, f"response_log_{datetime.now().strftime('%Y%m%d')}.jsonl")
        self.feedback_thresholds = {
            "negative": 2,  # Feedback score < 2 is considered negative
            "neutral": 3,  # Feedback score = 3 is considered neutral
            "positive": 4,  # Feedback score > 3 is considered positive
        }

    def log_response_quality(self, user_id, question, answer, sources, feedback_score, feedback_text=None):
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

    # Dashboard rendering method moved to src/dashboard.py for centralization


# Functions removed - response tracking is now integrated through FeaturesManager
# to avoid duplication and maintain consistency
