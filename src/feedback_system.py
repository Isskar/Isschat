import streamlit as st
from streamlit_feedback import streamlit_feedback
import uuid
import json
import os
from datetime import datetime
from typing import Dict, List, Optional


class FeedbackSystem:
    """Système de feedback corrigé pour éviter les problèmes de rerun Streamlit"""

    def __init__(self, feedback_file: Optional[str] = None):
        if feedback_file is None:
            # Créer un nom de fichier avec la date actuelle
            date_str = datetime.now().strftime("%Y-%m-%d")
            feedback_file = f"./logs/feedback/feedback_{date_str}.json"
        self.feedback_file = feedback_file
        self._ensure_feedback_file_exists()

    def _ensure_feedback_file_exists(self):
        """S'assurer que le fichier de feedback existe"""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, "w") as f:
                json.dump([], f)

    def _load_feedback_data(self) -> List[Dict]:
        """Charger les données de feedback depuis le fichier"""
        try:
            with open(self.feedback_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_feedback_data(self, data: List[Dict]):
        """Sauvegarder les données de feedback dans le fichier"""
        with open(self.feedback_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def render_feedback_widget(
        self, user_id: str, question: str, answer: str, sources: List[str], key_suffix: str = ""
    ) -> Optional[Dict]:
        """
        Afficher le widget de feedback corrigé pour éviter les problèmes de rerun

        Args:
            user_id: ID de l'utilisateur
            question: Question posée
            answer: Réponse donnée
            sources: Sources utilisées
            key_suffix: Suffixe pour la clé unique du widget

        Returns:
            Dictionnaire avec les données de feedback si soumis
        """

        # Créer une clé unique basée sur le contenu (plus stable)
        content_hash = hash(f"{question}_{answer}_{key_suffix}")
        feedback_key = f"feedback_{content_hash}"

        # États persistants pour ce feedback spécifique
        feedback_data_key = f"feedback_data_{feedback_key}"
        feedback_submitted_key = f"feedback_submitted_{feedback_key}"
        fbk_widget_key = f"fbk_widget_{feedback_key}"

        # Initialiser les données de feedback si pas encore fait
        if feedback_data_key not in st.session_state:
            st.session_state[feedback_data_key] = {
                "user_id": user_id,
                "question": question,
                "answer": answer,
                "sources": sources,
                "timestamp": datetime.now().isoformat(),
                "feedback_submitted": False,
            }

        # Initialiser l'état de soumission
        if feedback_submitted_key not in st.session_state:
            st.session_state[feedback_submitted_key] = False

        # Clé unique pour le widget
        if fbk_widget_key not in st.session_state:
            st.session_state[fbk_widget_key] = str(uuid.uuid4())

        def feedback_callback(response):
            """
            Callback pour traiter le feedback soumis

            Args:
                response: Réponse du widget de feedback
            """
            # Marquer le feedback comme soumis
            st.session_state[feedback_submitted_key] = True

            # Mettre à jour les données de feedback
            st.session_state[feedback_data_key]["feedback"] = response
            st.session_state[feedback_data_key]["feedback_submitted"] = True

            # Sauvegarder dans le fichier de données
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

            # Générer une nouvelle clé pour éviter les conflits
            st.session_state[fbk_widget_key] = str(uuid.uuid4())

        # Afficher le feedback déjà soumis si disponible
        if st.session_state[feedback_submitted_key]:
            feedback_data = st.session_state[feedback_data_key]
            if "feedback" in feedback_data:
                feedback_type = "👍" if feedback_data["feedback"]["score"] == 1 else "👎"
                st.write(f"Votre feedback: {feedback_type}")
                if feedback_data["feedback"].get("text"):
                    st.write(f"Commentaire: {feedback_data['feedback']['text']}")
            return None

        # Afficher le widget de feedback si pas encore soumis
        feedback_response = streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="[Optionnel] Commentaire:",
            align="flex-start",
            key=st.session_state[fbk_widget_key],
            on_submit=feedback_callback,
        )

        return feedback_response

    def get_feedback_statistics(self, days: int = 30) -> Dict:
        """
        Obtenir les statistiques de feedback pour les N derniers jours

        Args:
            days: Nombre de jours à analyser

        Returns:
            Dictionnaire avec les statistiques
        """
        feedback_data = self._load_feedback_data()

        if not feedback_data:
            return {"total_feedback": 0, "positive_feedback": 0, "negative_feedback": 0, "satisfaction_rate": 0.0}

        # Filtrer par date si nécessaire
        from datetime import datetime, timedelta

        cutoff_date = datetime.now() - timedelta(days=days)

        recent_feedback = []
        for entry in feedback_data:
            try:
                entry_date = datetime.fromisoformat(entry["timestamp"])
                if entry_date >= cutoff_date:
                    recent_feedback.append(entry)
            except (KeyError, ValueError):
                # Inclure les entrées sans timestamp valide
                recent_feedback.append(entry)

        total = len(recent_feedback)
        positive = sum(1 for entry in recent_feedback if entry.get("feedback", {}).get("score") == 1)
        negative = sum(1 for entry in recent_feedback if entry.get("feedback", {}).get("score") == 0)

        satisfaction_rate = (positive / total * 100) if total > 0 else 0.0

        return {
            "total_feedback": total,
            "positive_feedback": positive,
            "negative_feedback": negative,
            "satisfaction_rate": satisfaction_rate,
        }

    def get_all_feedback(self) -> List[Dict]:
        """Obtenir tous les feedbacks enregistrés"""
        return self._load_feedback_data()

    def get_feedback_by_user(self, user_id: str) -> List[Dict]:
        """Obtenir les feedbacks d'un utilisateur spécifique"""
        all_feedback = self._load_feedback_data()
        return [entry for entry in all_feedback if entry.get("user_id") == user_id]
