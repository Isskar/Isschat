import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter


class QuestionSuggester:
    """Suggère des questions pertinentes basées sur le contexte de la conversation"""

    def __init__(self, log_path="./logs/conversations", suggestion_cache_path="./cache"):
        self.log_path = log_path
        self.suggestion_cache_path = suggestion_cache_path
        os.makedirs(suggestion_cache_path, exist_ok=True)
        self.suggestion_cache_file = os.path.join(suggestion_cache_path, "question_suggestions.json")
        self.popular_questions = self._load_popular_questions()

    def _load_popular_questions(self):
        """Charge les questions populaires depuis le cache ou crée un cache vide"""
        if os.path.exists(self.suggestion_cache_file):
            try:
                with open(self.suggestion_cache_file, "r") as f:
                    return json.load(f)
            except:
                return {"popular": [], "by_topic": {}}
        else:
            return {"popular": [], "by_topic": {}}

    def _save_popular_questions(self):
        """Sauvegarde les questions populaires dans le cache"""
        with open(self.suggestion_cache_file, "w") as f:
            json.dump(self.popular_questions, f)

    def update_question_database(self):
        """Met à jour la base de données des questions à partir des logs"""
        all_questions = []

        # Parcourir tous les fichiers de log
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

        # Identifier les questions les plus populaires
        question_counter = Counter(all_questions)
        popular = question_counter.most_common(20)
        self.popular_questions["popular"] = [q for q, count in popular if count > 1]

        # Regrouper les questions par thème
        if len(all_questions) > 10:  # Seulement si nous avons assez de données
            vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
            try:
                tfidf_matrix = vectorizer.fit_transform(all_questions)
                feature_names = vectorizer.get_feature_names_out()

                # Extraire les thèmes principaux
                topics = {}
                for i, row in enumerate(tfidf_matrix.toarray()):
                    top_indices = row.argsort()[-3:][::-1]  # Top 3 termes
                    terms = [feature_names[idx] for idx in top_indices if row[idx] > 0]
                    if terms:
                        main_term = terms[0]
                        if main_term not in topics:
                            topics[main_term] = []
                        topics[main_term].append(all_questions[i])

                # Garder seulement les thèmes avec plusieurs questions
                for term, questions in list(topics.items()):
                    if len(questions) < 2:
                        del topics[term]
                    else:
                        # Limiter à 5 questions par thème
                        topics[term] = questions[:5]

                self.popular_questions["by_topic"] = topics
            except:
                pass  # En cas d'erreur, on garde les anciennes données

        # Sauvegarder les résultats
        self._save_popular_questions()

    def extract_keywords(self, text):
        """Extrait les mots-clés d'un texte"""
        # Nettoyage basique
        text = re.sub(r"[^\w\s]", "", text.lower())
        words = text.split()

        # Filtrer les mots courts et les mots vides
        stopwords = set(
            [
                "le",
                "la",
                "les",
                "un",
                "une",
                "des",
                "et",
                "ou",
                "a",
                "à",
                "de",
                "du",
                "ce",
                "cette",
                "ces",
                "est",
                "sont",
                "comment",
                "pourquoi",
                "quand",
                "qui",
                "que",
                "quoi",
                "où",
                "dont",
                "pour",
                "par",
                "sur",
                "dans",
                "en",
                "avec",
                "sans",
            ]
        )
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]

        return keywords

    def suggest_next_questions(self, current_question, current_answer, history=None, max_suggestions=3):
        """Suggère des questions de suivi basées sur la question et la réponse actuelles"""
        suggestions = []

        # Extraire les mots-clés
        question_keywords = self.extract_keywords(current_question)
        answer_keywords = self.extract_keywords(current_answer)

        # Combiner les mots-clés, avec plus de poids pour ceux de la réponse
        all_keywords = question_keywords + answer_keywords * 2

        # 1. Vérifier les questions par thème
        for topic, questions in self.popular_questions["by_topic"].items():
            if topic in all_keywords:
                for q in questions:
                    if q != current_question and q not in suggestions:
                        suggestions.append(q)

        # 2. Générer des questions de suivi basées sur des patterns
        follow_up_templates = [
            "Pouvez-vous expliquer plus en détail {}?",
            "Quels sont les avantages de {}?",
            "Comment implémenter {}?",
            "Quelles sont les alternatives à {}?",
            "Quels sont les prérequis pour {}?",
        ]

        # Trouver les termes importants dans la réponse
        important_terms = [kw for kw in answer_keywords if len(kw) > 4]
        if important_terms:
            for term in important_terms[:2]:  # Limiter à 2 termes
                template = np.random.choice(follow_up_templates)
                suggestions.append(template.format(term))

        # 3. Ajouter des questions populaires si nécessaire
        if len(suggestions) < max_suggestions:
            for q in self.popular_questions["popular"]:
                if q != current_question and q not in suggestions:
                    suggestions.append(q)
                    if len(suggestions) >= max_suggestions:
                        break

        # Limiter le nombre de suggestions
        return suggestions[:max_suggestions]

    def render_suggestions(self, st, current_question, current_answer, callback=None):
        """Affiche les suggestions de questions dans Streamlit"""
        suggestions = self.suggest_next_questions(current_question, current_answer)

        if not suggestions:
            return

        st.write("---")
        st.write("### Questions suggérées")

        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggest_{hash(suggestion)}"):
                if callback:
                    callback(suggestion)


# Fonction pour intégrer le suggestionneur dans l'application principale
def integrate_question_suggester(help_desk):
    """Intègre le suggestionneur de questions au help_desk"""
    suggester = QuestionSuggester()

    # Mettre à jour la base de données des questions
    suggester.update_question_database()

    # Ajouter le suggestionneur comme attribut
    help_desk.question_suggester = suggester

    return help_desk
