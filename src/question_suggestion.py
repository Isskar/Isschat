import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter


class QuestionSuggester:
    """Suggests relevant questions based on conversation context"""

    def __init__(
        self, log_path="./logs/conversations", suggestion_cache_path="./cache"
    ):
        self.log_path = log_path
        self.suggestion_cache_path = suggestion_cache_path
        os.makedirs(suggestion_cache_path, exist_ok=True)
        self.suggestion_cache_file = os.path.join(
            suggestion_cache_path, "question_suggestions.json"
        )
        self.popular_questions = self._load_popular_questions()

    def _load_popular_questions(self):
        """Loads popular questions from cache or creates an empty cache"""
        if os.path.exists(self.suggestion_cache_file):
            try:
                with open(self.suggestion_cache_file, "r") as f:
                    return json.load(f)
            except:
                return {"popular": [], "by_topic": {}}
        else:
            return {"popular": [], "by_topic": {}}

    def _save_popular_questions(self):
        """Saves popular questions to the cache"""
        with open(self.suggestion_cache_file, "w") as f:
            json.dump(self.popular_questions, f)

    def update_question_database(self):
        """Updates the questions database from logs"""
        all_questions = []

        # Go through all log files
        for filename in os.listdir(self.log_path):
            if filename.startswith("conv_log_") and filename.endswith(".jsonl"):
                file_path = os.path.join(self.log_path, filename)
                with open(file_path, "r") as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            if "question" in log_entry:
                                all_questions.append(log_entry["question"])
                        except:
                            continue

        if not all_questions:
            return

        # Identify the most popular questions
        question_counter = Counter(all_questions)
        popular = question_counter.most_common(20)
        self.popular_questions["popular"] = [q for q, count in popular if count > 1]

        # Group questions by theme
        if len(all_questions) > 10:  # Only if we have enough data
            vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
            try:
                tfidf_matrix = vectorizer.fit_transform(all_questions)
                feature_names = vectorizer.get_feature_names_out()

                # Extract main themes
                topics = {}
                for i, row in enumerate(tfidf_matrix.toarray()):
                    top_indices = row.argsort()[-3:][::-1]  # Top 3 terms
                    terms = [feature_names[idx] for idx in top_indices if row[idx] > 0]
                    if terms:
                        main_term = terms[0]
                        if main_term not in topics:
                            topics[main_term] = []
                        topics[main_term].append(all_questions[i])

                # Keep only themes with multiple questions
                for term, questions in list(topics.items()):
                    if len(questions) < 2:
                        del topics[term]
                    else:
                        # Limit to 5 questions per theme
                        topics[term] = questions[:5]

                self.popular_questions["by_topic"] = topics
            except:
                pass  # In case of error, keep the old data

        # Save the results
        self._save_popular_questions()

    def extract_keywords(self, text):
        """Extracts keywords from a text"""
        # Basic cleaning
        text = re.sub(r"[^\w\s]", "", text.lower())
        words = text.split()

        # Filter short words and stop words
        stopwords = set(
            [
                "the",
                "a",
                "an",
                "of",
                "and",
                "or",
                "to",
                "in",
                "on",
                "at",
                "by",
                "for",
                "with",
                "about",
                "against",
                "between",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "from",
                "up",
                "down",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "having",
                "do",
                "does",
                "did",
                "doing",
            ]
        )
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]

        return keywords

    def suggest_next_questions(
        self, current_question, current_answer, history=None, max_suggestions=3
    ):
        """Suggests follow-up questions based on the current question and answer"""
        suggestions = []

        # Extract keywords
        question_keywords = self.extract_keywords(current_question)
        answer_keywords = self.extract_keywords(current_answer)

        # Combine keywords, with more weight for those from the response
        all_keywords = question_keywords + answer_keywords * 2

        # 1. Check questions by theme
        for topic, questions in self.popular_questions["by_topic"].items():
            if topic in all_keywords:
                for q in questions:
                    if q != current_question and q not in suggestions:
                        suggestions.append(q)

        # 2. Generate follow-up questions based on patterns
        follow_up_templates = [
            "Can you explain {} in more detail?",
            "What are the advantages of {}?",
            "What are the limitations of {}?",
            "How can I use {} in my project?",
            "What is the difference between {} and alternatives?",
            "Are there any best practices for {}?",
        ]

        # Find important terms in the answer
        important_terms = [kw for kw in answer_keywords if len(kw) > 4]
        if important_terms:
            for term in important_terms[:2]:  # Limit to 2 terms
                template = np.random.choice(follow_up_templates)
                suggestions.append(template.format(term))

        # 3. Add popular questions if necessary
        if len(suggestions) < max_suggestions:
            for q in self.popular_questions["popular"]:
                if q != current_question and q not in suggestions:
                    suggestions.append(q)
                    if len(suggestions) >= max_suggestions:
                        break

        # Limit the number of suggestions
        return suggestions[:max_suggestions]

    def render_suggestions(self, st, current_question, current_answer, callback=None):
        """Displays question suggestions in Streamlit"""
        suggestions = self.suggest_next_questions(current_question, current_answer)

        if not suggestions:
            return

        st.write("---")
        st.write("### Suggested Questions")

        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggest_{hash(suggestion)}"):
                if callback:
                    callback(suggestion)


# Function to integrate the suggester into the main application
def integrate_question_suggester(help_desk):
    """Integrates the question suggester into the help_desk"""
    suggester = QuestionSuggester()

    # Update the questions database
    suggester.update_question_database()

    # Add the suggester as an attribute
    help_desk.question_suggester = suggester

    return help_desk
