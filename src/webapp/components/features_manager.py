"""
Features manager for advanced chatbot functionality.
"""

import streamlit as st
from streamlit_feedback import streamlit_feedback
import os
import logging
import time
import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional


class ConversationAnalyzer:
    """Analyzes conversation patterns and provides insights."""

    def __init__(self):
        self.conversations = []

    def analyze_conversation(self, query: str, response: str) -> Dict[str, Any]:
        """Analyze a conversation turn."""
        analysis = {
            "query_length": len(query),
            "response_length": len(response),
            "timestamp": datetime.now(),
            "query_type": self._classify_query(query),
        }
        self.conversations.append(analysis)
        return analysis

    def _classify_query(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        if any(word in query_lower for word in ["how", "comment"]):
            return "how-to"
        elif any(word in query_lower for word in ["what", "qu'est-ce", "quoi"]):
            return "definition"
        elif any(word in query_lower for word in ["where", "où"]):
            return "location"
        elif any(word in query_lower for word in ["why", "pourquoi"]):
            return "explanation"
        else:
            return "general"

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        if not self.conversations:
            return {}

        return {
            "total_conversations": len(self.conversations),
            "avg_query_length": sum(c["query_length"] for c in self.conversations) / len(self.conversations),
            "avg_response_length": sum(c["response_length"] for c in self.conversations) / len(self.conversations),
            "query_types": {
                qtype: sum(1 for c in self.conversations if c["query_type"] == qtype)
                for qtype in set(c["query_type"] for c in self.conversations)
            },
        }


class FeedbackSystem:
    """Système de feedback avec pouces (👍/👎) utilisant streamlit_feedback"""

    def __init__(self, feedback_file: Optional[str] = None):
        if feedback_file is None:
            # Créer un nom de fichier avec la date actuelle
            date_str = datetime.now().strftime("%Y-%m-%d")
            feedback_file = f"./data/logs/feedback/feedback_{date_str}.json"
        self.feedback_file = feedback_file
        self._ensure_feedback_file_exists()

    def _ensure_feedback_file_exists(self):
        """S'assurer que le fichier de feedback existe"""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, "w") as f:
                json.dump([], f)

    def _load_feedback_data(self):
        """Charger les données de feedback depuis tous les fichiers des 30 derniers jours"""
        from datetime import datetime, timedelta
        import logging

        logger = logging.getLogger(__name__)
        all_feedback = []

        # Calculer les 30 derniers jours
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        logger.info(f"Chargement des feedbacks du {start_date.date()} au {end_date.date()}")

        # Générer les noms de fichiers pour chaque jour
        current_date = start_date
        files_found = 0
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            feedback_file_path = f"logs/feedback/feedback_{date_str}.json"

            # Charger le fichier s'il existe
            try:
                if os.path.exists(feedback_file_path):
                    files_found += 1
                    logger.info(f"Fichier de feedback trouvé: {feedback_file_path}")
                    with open(feedback_file_path, "r") as f:
                        daily_feedback = json.load(f)
                        if isinstance(daily_feedback, list):
                            all_feedback.extend(daily_feedback)
                            logger.info(f"Chargé {len(daily_feedback)} feedbacks depuis {feedback_file_path}")
                else:
                    logger.debug(f"Fichier non trouvé: {feedback_file_path}")
            except (json.JSONDecodeError, IOError) as e:
                # Ignorer les fichiers corrompus ou inaccessibles
                logger.error(f"Erreur lors du chargement de {feedback_file_path}: {e}")
                continue

            current_date += timedelta(days=1)

        logger.info(f"Total: {files_found} fichiers trouvés, {len(all_feedback)} feedbacks chargés")
        return all_feedback

    def _save_feedback_data(self, data):
        """Sauvegarder les données de feedback dans le fichier"""
        with open(self.feedback_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def render_feedback_widget(
        self, user_id: str, question: str, answer: str, sources: str, key_suffix: str = ""
    ) -> Optional[Dict]:
        """
        Afficher le widget de feedback avec pouces pour éviter les problèmes de rerun

        Args:
            user_id: ID utilisateur
            question: Question posée
            answer: Réponse donnée
            sources: Sources utilisées
            key_suffix: Suffixe pour la clé unique du widget

        Returns:
            Dictionnaire avec les données de feedback si soumis
        """

        # Créer une clé unique basée sur le contenu (plus stable)
        content_hash = abs(hash(f"{question}_{answer}_{key_suffix}"))
        feedback_key = f"feedback_{content_hash}"

        # États persistants pour ce feedback spécifique
        feedback_submitted_key = f"feedback_submitted_{feedback_key}"

        # Initialiser l'état de soumission
        if feedback_submitted_key not in st.session_state:
            st.session_state[feedback_submitted_key] = False

        # Afficher le feedback déjà soumis si disponible
        if st.session_state[feedback_submitted_key]:
            return None

        # Callback simplifié pour traiter le feedback soumis
        def feedback_callback(response):
            """
            Callback pour traiter le feedback soumis

            Args:
                response: Réponse du widget de feedback
            """
            try:
                # Marquer le feedback comme soumis
                st.session_state[feedback_submitted_key] = True

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
                
                # Log pour debug
                print(f"Feedback sauvegardé: {feedback_entry}")
                
                # Forcer un rerun pour mettre à jour l'interface
                st.rerun()
                
            except Exception as e:
                print(f"Erreur lors de la sauvegarde du feedback: {e}")
                st.error(f"Erreur lors de la sauvegarde: {e}")

        # CSS targeted to make feedback widget background transparent
        st.markdown(f"""
        <style>
        /* Target all feedback widget containers */
        div[data-testid="stVerticalBlock"] div[style*="background"] {{
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}
        
        /* Target specifically black backgrounds in feedback widget */
        div[style*="rgb(0, 0, 0)"],
        div[style*="background-color: black"],
        div[style*="background-color: #000"],
        div[style*="background-color: #000000"] {{
            background: transparent !important;
            background-color: transparent !important;
        }}
        
        /* Target main streamlit-feedback container */
        .streamlit-feedback,
        .streamlit-feedback > div,
        .streamlit-feedback div {{
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}
        
        /* Force transparency on all dark background elements */
        [style*="background-color: rgb(0, 0, 0)"],
        [style*="background: rgb(0, 0, 0)"] {{
            background: transparent !important;
            background-color: transparent !important;
        }}
        
        /* Style pour le titre du feedback */
        .feedback-title {{
            color: #888;
            font-size: 0.8rem;
            font-weight: 400;
            margin: 0.5rem 0 0.3rem 0;
            padding-top: 0.5rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        /* Boutons du feedback - VISIBLES avec style */
        button[kind="secondary"] {{
            background: rgba(46, 134, 171, 0.1) !important;
            background-color: rgba(46, 134, 171, 0.1) !important;
            color: #2E86AB !important;
            border: 1px solid rgba(46, 134, 171, 0.3) !important;
            border-radius: 6px !important;
            padding: 0.4rem 0.8rem !important;
            margin: 0.2rem !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }}
        
        button[kind="secondary"]:hover {{
            background: rgba(46, 134, 171, 0.2) !important;
            background-color: rgba(46, 134, 171, 0.2) !important;
            color: #FFFFFF !important;
            border-color: #2E86AB !important;
            transform: translateY(-1px) !important;
        }}
        
        /* Boutons avec emojis (👍👎) */
        button[title*="👍"], button[title*="👎"] {{
            background: rgba(46, 134, 171, 0.1) !important;
            background-color: rgba(46, 134, 171, 0.1) !important;
            color: #2E86AB !important;
            border: 1px solid rgba(46, 134, 171, 0.3) !important;
            border-radius: 6px !important;
            padding: 0.4rem 0.8rem !important;
            margin: 0.2rem !important;
            font-size: 1rem !important;
            min-width: 50px !important;
            min-height: 35px !important;
        }}
        
        button[title*="👍"]:hover, button[title*="👎"]:hover {{
            background: rgba(46, 134, 171, 0.2) !important;
            background-color: rgba(46, 134, 171, 0.2) !important;
            border-color: #2E86AB !important;
            transform: translateY(-1px) !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        # Titre simple et intégré
        st.markdown('<div class="feedback-title">💬 Cette réponse vous a-t-elle été utile ?</div>', unsafe_allow_html=True)
        
        # Afficher le widget de feedback si pas encore soumis
        try:
            feedback_response = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="Commentaire (optionnel):",
                align="flex-start",
                key=f"feedback_widget_{content_hash}",
                on_submit=feedback_callback,
            )
            
            return feedback_response
        except Exception as e:
            st.error(f"Erreur avec le widget de feedback: {e}")
            return None

    def get_feedback_statistics(self, days: int = 30) -> Dict:
        """
        Obtenir les statistiques de feedback pour les N derniers jours

        Args:
            days: Nombre de jours à analyser (par défaut 30, mais _load_feedback_data charge déjà les 30 derniers jours)

        Returns:
            Dictionnaire avec les statistiques
        """
        feedback_data = self._load_feedback_data()

        if not feedback_data:
            return {"total_feedback": 0, "positive_feedback": 0, "negative_feedback": 0, "satisfaction_rate": 0.0}

        # Si on veut moins de 30 jours, filtrer par date
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
                    # Inclure les entrées sans timestamp valide
                    recent_feedback.append(entry)
        else:
            # Utiliser toutes les données déjà filtrées par _load_feedback_data
            recent_feedback = feedback_data

        total = len(recent_feedback)
        positive = 0
        negative = 0

        for entry in recent_feedback:
            feedback_obj = entry.get("feedback", {})
            score = feedback_obj.get("score")

            # Gérer les formats emoji ('👍'/'👎') et numérique (1/0)
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

    def get_all_feedback(self):
        """Obtenir tous les feedbacks enregistrés"""
        return self._load_feedback_data()

    def get_feedback_by_user(self, user_id: str):
        """Obtenir les feedbacks d'un utilisateur spécifique"""
        all_feedback = self._load_feedback_data()
        return [entry for entry in all_feedback if entry.get("user_id") == user_id]


class ResponseTracker:
    """Tracks response quality and user feedback."""

    def __init__(self):
        self.responses = []
        self.feedback_data = []

    def track_response(self, query: str, response: str, response_time: float) -> str:
        """Track a response."""
        response_id = f"resp_{len(self.responses)}_{int(time.time())}"

        response_data = {
            "id": response_id,
            "query": query,
            "response": response,
            "response_time": response_time,
            "timestamp": datetime.now(),
            "feedback": None,
        }

        self.responses.append(response_data)
        return response_id

    def add_feedback(self, response_id: str, feedback: Dict[str, Any]):
        """Add user feedback for a response."""
        for response in self.responses:
            if response["id"] == response_id:
                response["feedback"] = feedback
                break

        self.feedback_data.append({"response_id": response_id, "feedback": feedback, "timestamp": datetime.now()})

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        if not self.feedback_data:
            return {}

        ratings = [f["feedback"].get("rating", 0) for f in self.feedback_data if f["feedback"].get("rating")]

        return {
            "total_feedback": len(self.feedback_data),
            "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
            "positive_feedback": sum(1 for r in ratings if r >= 4),
            "negative_feedback": sum(1 for r in ratings if r <= 2),
        }


class PerformanceTracker:
    """Tracks system performance metrics."""

    def __init__(self):
        self.performance_data = []

    def track_performance(self, operation: str, duration: float, **kwargs):
        """Track performance of an operation."""
        perf_data = {"operation": operation, "duration": duration, "timestamp": datetime.now(), **kwargs}
        self.performance_data.append(perf_data)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_data:
            return {}

        operations = {}
        for data in self.performance_data:
            op = data["operation"]
            if op not in operations:
                operations[op] = []
            operations[op].append(data["duration"])

        return {
            "total_operations": len(self.performance_data),
            "avg_durations": {op: sum(durations) / len(durations) for op, durations in operations.items()},
            "operation_counts": {op: len(durations) for op, durations in operations.items()},
        }


class QueryHistory:
    """Manages query history and patterns."""

    def __init__(self):
        self.history = []

    def add_query(self, query: str, response: str, user_id: str = "anonymous"):
        """Add a query to history."""
        history_entry = {"query": query, "response": response, "user_id": user_id, "timestamp": datetime.now()}
        self.history.append(history_entry)

    def get_user_history(self, user_id: str, limit: int = 10):
        """Get history for a specific user."""
        user_queries = [h for h in self.history if h["user_id"] == user_id]
        return sorted(user_queries, key=lambda x: x["timestamp"], reverse=True)[:limit]

    def get_recent_queries(self, limit: int = 20):
        """Get recent queries across all users."""
        return sorted(self.history, key=lambda x: x["timestamp"], reverse=True)[:limit]


class FeaturesManager:
    """Central manager for all advanced chatbot features."""

    def __init__(self, user_id: str = "anonymous"):
        """Initialize and integrate all features."""
        self.user_id = user_id

        # Create necessary folders if they don't exist
        os.makedirs("./logs", exist_ok=True)
        os.makedirs("./logs/feedback", exist_ok=True)
        os.makedirs("./data", exist_ok=True)
        os.makedirs("./cache", exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(f"./logs/chatbot_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler(),
            ],
        )

        self.logger = logging.getLogger("features_manager")
        self.logger.info(f"Initializing features manager for user {user_id}")

        # Initialize feature components
        self.analyzer = ConversationAnalyzer()
        self.response_tracker = ResponseTracker()
        self.performance_tracker = PerformanceTracker()
        self.query_history = QueryHistory()
        self.feedback_system = FeedbackSystem()  # Nouveau système de feedback avec pouces

        # Initialize session state for features
        if "features_data" not in st.session_state:
            st.session_state.features_data = {"conversations": [], "responses": [], "performance": [], "history": []}

    def process_query_response(self, query: str, response: str, response_time: float) -> str:
        """Process a query-response pair through all features."""
        # Analyze conversation
        # analysis = self.analyzer.analyze_conversation(query, response)

        # Track response
        response_id = self.response_tracker.track_response(query, response, response_time)

        # Track performance
        self.performance_tracker.track_performance("query_processing", response_time)

        # Add to history
        self.query_history.add_query(query, response, self.user_id)

        # Log the interaction
        self.logger.info(
            f"Processed query for user {self.user_id}: {len(query)} chars -> {len(response)} chars in {response_time:.2f}s"  # noqa
        )

        return response_id

    def add_user_feedback(self, response_id: str, rating: int, comment: str = ""):
        """Add user feedback for a response."""
        feedback = {"rating": rating, "comment": comment, "user_id": self.user_id}
        self.response_tracker.add_feedback(response_id, feedback)
        self.logger.info(f"Received feedback from user {self.user_id}: rating={rating}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for admin dashboard."""
        return {
            "conversation_stats": self.analyzer.get_conversation_stats(),
            "feedback_stats": self.response_tracker.get_feedback_stats(),
            "performance_stats": self.performance_tracker.get_performance_stats(),
            "recent_queries": self.query_history.get_recent_queries(10),
            "user_history": self.query_history.get_user_history(self.user_id, 5),
        }

    def render_feedback_widget(self, question: str, answer: str, sources: str = "", key_suffix: str = ""):
        """Render feedback widget with thumbs (👍/👎) for a response."""
        return self.feedback_system.render_feedback_widget(
            user_id=self.user_id,
            question=question,
            answer=answer,
            sources=sources,
            key_suffix=key_suffix
        )

    def get_feedback_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get feedback statistics from the thumbs feedback system."""
        return self.feedback_system.get_feedback_statistics(days)

    def add_feedback_widget(self, st, question: str, answer: str, sources: str, key_suffix: str = ""):
        """Méthode compatible avec l'ancienne version pour ajouter un widget de feedback"""
        return self.feedback_system.render_feedback_widget(
            user_id=self.user_id,
            question=question,
            answer=answer,
            sources=sources,
            key_suffix=key_suffix
        )

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for admin dashboard."""
        return {
            "conversation_stats": self.analyzer.get_conversation_stats(),
            "feedback_stats": self.get_feedback_statistics(),  # Utilise le nouveau système
            "performance_stats": self.performance_tracker.get_performance_stats(),
            "recent_queries": self.query_history.get_recent_queries(10),
            "user_history": self.query_history.get_user_history(self.user_id, 5),
        }
