import streamlit as st
import os
import logging
import time
from datetime import datetime

# Importer tous les modules de fonctionnalités
from src.conversation_analysis import ConversationAnalyzer
from src.response_tracking import ResponseTracker
from src.question_suggestion import QuestionSuggester
from src.performance_tracking import PerformanceTracker
from src.query_history import QueryHistory


class FeaturesManager:
    """Gestionnaire central pour toutes les fonctionnalités avancées du chatbot"""

    def __init__(self, help_desk, user_id):
        """Initialise et intègre toutes les fonctionnalités au help_desk"""
        self.help_desk = help_desk
        self.user_id = user_id

        # Créer les dossiers nécessaires s'ils n'existent pas
        os.makedirs("./logs", exist_ok=True)
        os.makedirs("./data", exist_ok=True)
        os.makedirs("./cache", exist_ok=True)

        # Configurer le logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(f"./logs/chatbot_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler(),
            ],
        )

        self.logger = logging.getLogger("features_manager")
        self.logger.info(f"Initialisation du gestionnaire de fonctionnalités pour l'utilisateur {user_id}")

        # Créer les objets de fonctionnalités directement comme attributs de cette classe
        self.analyzer = ConversationAnalyzer()
        self.response_tracker = ResponseTracker()
        self.question_suggester = QuestionSuggester()
        self.performance_tracker = PerformanceTracker()
        self.query_history = QueryHistory()

        # Intégrer les fonctionnalités nécessaires
        self._integrate_selected_features()

    def _integrate_selected_features(self):
        """Intègre les fonctionnalités sélectionnées au help_desk"""
        try:
            # Ajouter le système de feedback utilisateur
            self._add_feedback_system()
            self.logger.info("Système de feedback utilisateur intégré")

            # Modifier la méthode ask_question pour enregistrer les métriques de performance
            self._wrap_ask_question()
            self.logger.info("Suivi des performances intégré")

        except Exception as e:
            self.logger.error(f"Erreur lors de l'intégration des fonctionnalités: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())

    def _add_feedback_system(self):
        """Ajoute un système de feedback utilisateur"""
        # Stocker la méthode originale
        self.original_ask = self.help_desk.retrieval_qa_inference

    def _wrap_ask_question(self):
        """Enveloppe la méthode ask_question pour enregistrer les métriques de performance"""
        original_ask = self.help_desk.retrieval_qa_inference

        def wrapped_ask(question, verbose=False):
            # Mesurer le temps total
            start_time = time.time()

            # Appeler la fonction originale
            answer, sources = original_ask(question, verbose)

            # Calculer le temps total
            total_time = (time.time() - start_time) * 1000  # en ms

            # Enregistrer l'interaction dans l'historique
            # query_id = self.query_history.add_query(
            #     user_id=self.user_id,
            #     question=question,
            #     answer=answer,
            #     sources=sources if isinstance(sources, list) else [sources],
            # )

            # Enregistrer les métriques de performance
            self.performance_tracker.track_query(
                question=question,
                retrieval_time=total_time * 0.3,  # Approximation
                generation_time=total_time * 0.7,  # Approximation
                num_docs_retrieved=len(sources) if isinstance(sources, list) else 1,
                total_time=total_time,
            )

            # Enregistrer l'interaction pour l'analyse conversationnelle
            self.analyzer.log_interaction(
                user_id=self.user_id,
                question=question,
                answer=answer,
                sources=sources if isinstance(sources, list) else [sources],
                response_time=total_time,
            )

            return answer, sources

        # Remplacer la méthode originale
        self.help_desk.retrieval_qa_inference = wrapped_ask

    def render_admin_dashboard(self, st):
        """Affiche le tableau de bord d'administration avec toutes les fonctionnalités"""
        st.title("Tableau de Bord d'Administration")

        # Créer des onglets pour chaque fonctionnalité
        tabs = st.tabs(
            [
                "Analyse Conversationnelle",
                "Suivi Non-Réponses",
                "Performances",
                "Statistiques Générales",
            ]
        )

        # Onglet 1: Analyse Conversationnelle
        with tabs[0]:
            try:
                self.analyzer.render_analysis_dashboard()
            except Exception as e:
                st.error(f"Erreur lors de l'affichage de l'analyse conversationnelle: {str(e)}")

        # Onglet 2: Suivi des Non-Réponses
        with tabs[1]:
            try:
                self.response_tracker.render_tracking_dashboard()
            except Exception as e:
                st.error(f"Erreur lors de l'affichage du suivi des non-réponses: {str(e)}")

        # Onglet 3: Performances
        with tabs[2]:
            try:
                self.performance_tracker.render_performance_dashboard()
            except Exception as e:
                st.error(f"Erreur lors de l'affichage des performances: {str(e)}")

        # Onglet 4: Statistiques Générales
        with tabs[3]:
            self._render_general_statistics(st)

    def _render_general_statistics(self, st):
        """Affiche des statistiques générales sur l'utilisation du chatbot"""
        st.subheader("Statistiques Générales")

        # Collecter les statistiques de différentes sources
        stats = {}

        try:
            # Statistiques de conversation
            conv_logs = self.analyzer.get_recent_logs(days=30)
            stats["total_conversations"] = len(conv_logs)

            # Statistiques de performance
            perf_logs = self.performance_tracker.get_performance_logs(days=30)
            if perf_logs:
                avg_time = sum(log.get("total_time_ms", 0) for log in perf_logs) / len(perf_logs)
                stats["avg_response_time"] = f"{avg_time:.0f} ms"
            else:
                stats["avg_response_time"] = "N/A"

            # Statistiques de feedback
            unsat_responses = self.response_tracker.get_unsatisfactory_responses(days=30)
            stats["unsatisfactory_responses"] = len(unsat_responses)

            # Afficher les statistiques
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total des conversations", stats["total_conversations"])

            with col2:
                st.metric("Temps de réponse moyen", stats["avg_response_time"])

            with col3:
                st.metric("Réponses insatisfaisantes", stats["unsatisfactory_responses"])

        except Exception as e:
            st.error(f"Erreur lors de la collecte des statistiques: {str(e)}")

    def process_question(self, question, show_suggestions=True, show_feedback=True):
        """Traite une question et ajoute un système de feedback utilisateur"""
        try:
            # 1. Obtenir la réponse
            start_time = datetime.now()
            answer, sources = self.help_desk.retrieval_qa_inference(question)
            response_time = (datetime.now() - start_time).total_seconds() * 1000

            # 2. Afficher la réponse
            st.markdown(answer)

            # 3. Afficher les sources
            if sources:
                st.write(sources)

            # 4. Afficher le widget de feedback si activé
            if show_feedback:
                self._add_feedback_widget(st, question, answer, sources)

            # 5. Afficher les suggestions de questions si activées
            if show_suggestions:
                self._show_question_suggestions(st, question, answer)

            # Enregistrer les métriques de performance
            self.logger.info(f"Question traitée en {response_time:.0f}ms: {question[:50]}...")

            return answer, sources

        except Exception as e:
            st.error(f"Erreur lors du traitement de la question: {str(e)}")
            self.logger.error(f"Erreur lors du traitement de la question: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            return (
                "Désolé, une erreur s'est produite lors du traitement de votre question.",
                [],
            )

    def _add_feedback_widget(self, st, question, answer, sources):
        """Ajoute un widget de feedback pour évaluer la qualité de la réponse"""
        st.write("---")
        st.write("### Évaluez cette réponse")

        col1, col2 = st.columns([3, 1])

        with col1:
            feedback_score = st.slider("Qualité de la réponse", 1, 5, 3, key="feedback_score")
            feedback_text = st.text_area(
                "Commentaire (optionnel)",
                key="feedback_text",
                placeholder="Dites-nous ce que vous pensez de cette réponse",
            )

        with col2:
            if st.button("Envoyer feedback", key="send_feedback"):
                # Enregistrer le feedback
                self.response_tracker.log_response_quality(
                    user_id=self.user_id,
                    question=question,
                    answer=answer,
                    sources=sources if isinstance(sources, list) else [sources],
                    feedback_score=feedback_score,
                    feedback_text=feedback_text,
                )
                st.success("Merci pour votre feedback!")

    def _show_question_suggestions(self, st, question, answer):
        """Affiche des suggestions de questions de suivi"""
        try:
            suggestions = self.question_suggester.suggest_next_questions(question, answer)

            if suggestions:
                st.write("---")
                st.write("### Questions suggérées")

                for i, suggestion in enumerate(suggestions):
                    if st.button(suggestion, key=f"suggest_{i}"):
                        # Stocker la question dans la session pour la réutiliser
                        return suggestion
        except Exception as e:
            self.logger.error(f"Erreur lors de l'affichage des suggestions: {str(e)}")
            return None


# Fonction pour intégrer le gestionnaire de fonctionnalités dans l'application principale
def setup_features(help_desk, user_id):
    """Configure et retourne un gestionnaire de fonctionnalités pour l'application"""
    return FeaturesManager(help_desk, user_id)
