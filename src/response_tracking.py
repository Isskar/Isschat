import pandas as pd
import streamlit as st
import json
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


class ResponseTracker:
    """Suit les questions sans réponses satisfaisantes et propose des améliorations"""

    def __init__(self, log_path="./logs/responses"):
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        self.current_log_file = os.path.join(log_path, f"response_log_{datetime.now().strftime('%Y%m%d')}.jsonl")
        self.feedback_thresholds = {
            "negative": 2,  # Score de feedback < 2 est considéré négatif
            "neutral": 3,  # Score de feedback = 3 est considéré neutre
            "positive": 4,  # Score de feedback > 3 est considéré positif
        }

    def log_response_quality(self, user_id, question, answer, sources, feedback_score, feedback_text=None):
        """Enregistre la qualité d'une réponse basée sur le feedback utilisateur"""
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
        """Récupère les réponses insatisfaisantes des n derniers jours"""
        unsatisfactory = []

        # Trouver tous les fichiers de log dans la période spécifiée
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
        """Identifie des modèles dans les questions sans réponses satisfaisantes"""
        if unsatisfactory_responses is None:
            unsatisfactory_responses = self.get_unsatisfactory_responses()

        if not unsatisfactory_responses:
            return {"message": "Aucune réponse insatisfaisante trouvée"}

        # Extraire les questions
        questions = [resp["question"] for resp in unsatisfactory_responses]

        # Utiliser TF-IDF pour trouver des similitudes
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
        """Affiche le tableau de bord de suivi des non-réponses dans Streamlit"""
        st.title("Suivi des Non-Réponses")

        # Sélection de la période
        days = st.slider("Période d'analyse (jours)", 1, 90, 30, key="nonresponse_days")
        unsatisfactory = self.get_unsatisfactory_responses(days)

        if not unsatisfactory:
            st.warning("Aucune réponse insatisfaisante trouvée pour la période sélectionnée")
            return

        # Métriques de base
        st.metric("Nombre de réponses insatisfaisantes", len(unsatisfactory))

        # Analyse des patterns
        patterns = self.identify_patterns(unsatisfactory)

        if "error" in patterns:
            st.error(f"Erreur d'analyse: {patterns['error']}")
            return

        # Afficher les clusters de questions similaires
        if "clusters" in patterns and patterns["clusters"]:
            st.subheader("Groupes de questions similaires sans réponses satisfaisantes")

            for i, cluster in enumerate(patterns["clusters"]):
                with st.expander(f"Groupe {i + 1}: {cluster['main_question']} ({cluster['count']} questions)"):
                    for q in cluster["similar_questions"]:
                        st.write(f"- {q}")

        # Afficher les termes communs
        if "common_terms" in patterns and patterns["common_terms"]:
            st.subheader("Termes fréquents dans les questions sans réponses")
            terms_df = pd.DataFrame(patterns["common_terms"], columns=["Terme", "Fréquence"])
            st.dataframe(terms_df)

        # Tableau des questions sans réponses
        st.subheader("Détail des questions sans réponses satisfaisantes")

        # Convertir en DataFrame pour un affichage plus propre
        df = pd.DataFrame(
            [
                {
                    "Date": datetime.fromisoformat(item["timestamp"]).strftime("%Y-%m-%d %H:%M"),
                    "Question": item["question"],
                    "Score": item["feedback_score"],
                    "Commentaire": item.get("feedback_text", ""),
                }
                for item in unsatisfactory
            ]
        )

        st.dataframe(df)


# Fonction pour intégrer le tracker dans l'application principale
def integrate_response_tracker(help_desk, user_id):
    """Intègre le tracker de réponses au help_desk"""
    tracker = ResponseTracker()

    # Ajouter le tracker comme attribut
    help_desk.response_tracker = tracker

    return help_desk


# Fonction pour ajouter un widget de feedback dans Streamlit
def add_feedback_widget(st, help_desk, user_id, question, answer, sources):
    """Ajoute un widget de feedback pour la réponse"""
    st.write("---")
    st.write("### Évaluez cette réponse")

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        feedback_score = st.slider("Qualité de la réponse", 1, 5, 3)

    with col2:
        if st.button("Envoyer feedback"):
            feedback_text = st.session_state.get("feedback_text", "")
            help_desk.response_tracker.log_response_quality(
                user_id=user_id,
                question=question,
                answer=answer,
                sources=sources,
                feedback_score=feedback_score,
                feedback_text=feedback_text,
            )
            st.success("Merci pour votre feedback!")

    with col3:
        if feedback_score <= 3:
            feedback_text = st.text_area(
                "Commentaire (optionnel)",
                key="feedback_text",
                placeholder="Dites-nous pourquoi cette réponse n'était pas satisfaisante",
            )
