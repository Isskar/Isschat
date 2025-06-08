"""
Gestionnaire d'historique des conversations pour l'interface Streamlit.
Intègre avec le nouveau système de données.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import plotly.express as px

from ...core.data_manager import get_data_manager


class ConversationHistoryManager:
    """Gestionnaire de l'historique des conversations pour Streamlit."""

    def __init__(self):
        self.data_manager = get_data_manager()

    def render_history_page(self, user_id: Optional[str] = None):
        """Rend la page d'historique des conversations."""
        st.title("📚 Historique des Conversations")

        # Sidebar pour les filtres
        with st.sidebar:
            st.subheader("🔍 Filtres")

            # Filtre par période
            period_options = {
                "Aujourd'hui": 1,
                "7 derniers jours": 7,
                "30 derniers jours": 30,
                "Tout l'historique": None,
            }

            selected_period = st.selectbox(
                "Période",
                options=list(period_options.keys()),
                index=1,  # 7 derniers jours par défaut
            )

            # Filtre par utilisateur (pour les admins)
            show_all_users = st.checkbox("Voir tous les utilisateurs", value=False)
            if show_all_users:
                user_id = None

            # Limite du nombre de conversations
            limit = st.slider("Nombre max de conversations", 10, 200, 50)

        # Récupération des données
        conversations = self._get_filtered_conversations(
            user_id=user_id, period_days=period_options[selected_period], limit=limit
        )

        if not conversations:
            st.info("Aucune conversation trouvée pour les critères sélectionnés.")
            return

        # Onglets pour différentes vues
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📋 Liste des Conversations", "📊 Statistiques", "🔍 Recherche", "📈 Analyse"]
        )

        with tab1:
            self._render_conversations_list(conversations)

        with tab2:
            self._render_statistics(conversations)

        with tab3:
            self._render_search_interface(conversations)

        with tab4:
            self._render_analysis(conversations)

    def _get_filtered_conversations(
        self, user_id: Optional[str] = None, period_days: Optional[int] = None, limit: int = 50
    ) -> List[Dict]:
        """Récupère les conversations filtrées."""
        conversations = self.data_manager.get_conversation_history(user_id, limit)

        if period_days and conversations:
            cutoff_date = datetime.now() - timedelta(days=period_days)
            conversations = [conv for conv in conversations if datetime.fromisoformat(conv["timestamp"]) >= cutoff_date]

        return conversations

    def _render_conversations_list(self, conversations: List[Dict]):
        """Rend la liste des conversations."""
        st.subheader("💬 Conversations Récentes")

        if not conversations:
            st.info("Aucune conversation à afficher.")
            return

        # Options d'affichage
        col1, col2 = st.columns([3, 1])
        with col1:
            show_details = st.checkbox("Afficher les détails complets", value=False)
        with col2:
            export_btn = st.button("📥 Exporter CSV")

        if export_btn:
            self._export_conversations_csv(conversations)

        # Affichage des conversations
        for i, conv in enumerate(conversations):
            with st.expander(
                f"🕐 {self._format_timestamp(conv['timestamp'])} - "
                f"👤 {conv.get('user_id', 'Anonyme')} - "
                f"⏱️ {conv.get('response_time_ms', 0):.0f}ms",
                expanded=False,
            ):
                # Question
                st.markdown("**❓ Question:**")
                st.write(conv["question"])

                # Réponse
                st.markdown("**💡 Réponse:**")
                if show_details:
                    st.write(conv["answer"])
                else:
                    # Aperçu tronqué
                    preview = conv["answer"][:200] + "..." if len(conv["answer"]) > 200 else conv["answer"]
                    st.write(preview)

                # Métadonnées
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Longueur réponse", f"{conv.get('answer_length', 0)} chars")
                with col2:
                    st.metric("Sources", conv.get("sources_count", 0))
                with col3:
                    st.metric("Temps réponse", f"{conv.get('response_time_ms', 0):.0f}ms")
                with col4:
                    feedback = conv.get("feedback")
                    if feedback:
                        st.metric("Note", f"{feedback.get('rating', 'N/A')}/5")
                    else:
                        st.metric("Note", "Non évaluée")

                # Sources si disponibles
                if show_details and conv.get("sources"):
                    st.markdown("**📚 Sources utilisées:**")
                    for j, source in enumerate(conv["sources"][:3]):  # Limiter à 3 sources
                        title = source.get("title", "Source inconnue")
                        url = source.get("url", "")

                        if url and url != "#":
                            # Créer un lien cliquable
                            st.markdown(f"{j + 1}. [{title}]({url})")
                        else:
                            st.write(f"{j + 1}. {title}")

                # Bouton pour réutiliser la question
                if st.button("🔄 Réutiliser cette question", key=f"reuse_{i}"):
                    st.session_state["reuse_question"] = conv["question"]
                    st.session_state["page"] = "chat"
                    st.rerun()

    def _render_statistics(self, conversations: List[Dict]):
        """Rend les statistiques des conversations."""
        st.subheader("📊 Statistiques des Conversations")

        if not conversations:
            st.info("Aucune donnée pour les statistiques.")
            return

        # Métriques générales
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total conversations", len(conversations))

        with col2:
            avg_response_time = sum(c.get("response_time_ms", 0) for c in conversations) / len(conversations)
            st.metric("Temps moyen", f"{avg_response_time:.0f}ms")

        with col3:
            avg_length = sum(c.get("answer_length", 0) for c in conversations) / len(conversations)
            st.metric("Longueur moyenne", f"{avg_length:.0f} chars")

        with col4:
            total_sources = sum(c.get("sources_count", 0) for c in conversations)
            st.metric("Sources utilisées", total_sources)

        # Graphiques
        st.subheader("📈 Tendances")

        # Préparation des données pour les graphiques
        df = pd.DataFrame(conversations)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["date"] = df["timestamp"].dt.date

            # Graphique des conversations par jour
            daily_counts = df.groupby("date").size().reset_index(name="count")
            fig_daily = px.line(
                daily_counts,
                x="date",
                y="count",
                title="Conversations par jour",
                labels={"date": "Date", "count": "Nombre de conversations"},
            )
            st.plotly_chart(fig_daily, use_container_width=True)

            # Distribution des temps de réponse
            if "response_time_ms" in df.columns:
                fig_response_time = px.histogram(
                    df,
                    x="response_time_ms",
                    title="Distribution des temps de réponse",
                    labels={"response_time_ms": "Temps de réponse (ms)", "count": "Fréquence"},
                )
                st.plotly_chart(fig_response_time, use_container_width=True)

    def _render_search_interface(self, conversations: List[Dict]):
        """Rend l'interface de recherche."""
        st.subheader("🔍 Recherche dans l'Historique")

        # Interface de recherche
        search_term = st.text_input("Rechercher dans les questions et réponses:")

        if search_term:
            # Filtrer les conversations
            filtered_conversations = []
            for conv in conversations:
                if (
                    search_term.lower() in conv.get("question", "").lower()
                    or search_term.lower() in conv.get("answer", "").lower()
                ):
                    filtered_conversations.append(conv)

            st.write(f"**{len(filtered_conversations)} résultat(s) trouvé(s)**")

            # Afficher les résultats
            for i, conv in enumerate(filtered_conversations[:10]):  # Limiter à 10 résultats
                with st.expander(
                    f"🕐 {self._format_timestamp(conv['timestamp'])} - 👤 {conv.get('user_id', 'Anonyme')}",
                    expanded=False,
                ):
                    st.markdown("**❓ Question:**")
                    question = conv["question"]
                    # Surligner le terme recherché
                    highlighted_question = question.replace(search_term, f"**{search_term}**")
                    st.markdown(highlighted_question)

                    st.markdown("**💡 Réponse:**")
                    answer = conv["answer"][:300] + "..." if len(conv["answer"]) > 300 else conv["answer"]
                    highlighted_answer = answer.replace(search_term, f"**{search_term}**")
                    st.markdown(highlighted_answer)

                    if st.button("🔄 Réutiliser", key=f"search_reuse_{i}"):
                        st.session_state["reuse_question"] = conv["question"]
                        st.session_state["page"] = "chat"
                        st.rerun()

    def _render_analysis(self, conversations: List[Dict]):
        """Rend l'analyse avancée des conversations."""
        st.subheader("📈 Analyse Avancée")

        if not conversations:
            st.info("Aucune donnée pour l'analyse.")
            return

        # Analyse des mots-clés les plus fréquents
        st.subheader("🏷️ Mots-clés Fréquents")

        # Extraire tous les mots des questions
        all_words = []
        for conv in conversations:
            words = conv.get("question", "").lower().split()
            # Filtrer les mots courts et les mots vides
            filtered_words = [
                word.strip(".,!?;:")
                for word in words
                if len(word) > 3 and word not in ["comment", "quest", "quoi", "dans", "avec", "pour", "cette", "cette"]
            ]
            all_words.extend(filtered_words)

        if all_words:
            from collections import Counter

            word_counts = Counter(all_words)
            top_words = word_counts.most_common(10)

            if top_words:
                words_df = pd.DataFrame(top_words, columns=["Mot", "Fréquence"])
                fig_words = px.bar(
                    words_df, x="Fréquence", y="Mot", orientation="h", title="Top 10 des mots-clés dans les questions"
                )
                st.plotly_chart(fig_words, use_container_width=True)

        # Analyse des performances par utilisateur
        if len(set(conv.get("user_id") for conv in conversations)) > 1:
            st.subheader("👥 Performance par Utilisateur")

            user_stats = {}
            for conv in conversations:
                user_id = conv.get("user_id", "Anonyme")
                if user_id not in user_stats:
                    user_stats[user_id] = {"conversations": 0, "total_response_time": 0, "total_length": 0}

                user_stats[user_id]["conversations"] += 1
                user_stats[user_id]["total_response_time"] += conv.get("response_time_ms", 0)
                user_stats[user_id]["total_length"] += conv.get("answer_length", 0)

            # Créer un DataFrame pour l'affichage
            user_df = []
            for user_id, stats in user_stats.items():
                user_df.append(
                    {
                        "Utilisateur": user_id,
                        "Conversations": stats["conversations"],
                        "Temps moyen (ms)": stats["total_response_time"] / stats["conversations"],
                        "Longueur moyenne": stats["total_length"] / stats["conversations"],
                    }
                )

            user_df = pd.DataFrame(user_df)
            st.dataframe(user_df, use_container_width=True)

        # Afficher les métriques de performance détaillées
        st.subheader("⚡ Métriques de Performance Détaillées")
        try:
            performance_data = self.data_manager.get_performance_metrics(limit=100)
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                perf_df["timestamp"] = pd.to_datetime(perf_df["timestamp"])

                # Graphique des temps de réponse dans le temps
                fig_perf = px.line(
                    perf_df,
                    x="timestamp",
                    y="duration_ms",
                    title="Évolution des temps de réponse",
                    labels={"timestamp": "Temps", "duration_ms": "Durée (ms)"},
                )
                st.plotly_chart(fig_perf, use_container_width=True)

                # Statistiques de performance
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_perf = perf_df["duration_ms"].mean()
                    st.metric("Temps moyen", f"{avg_perf:.0f}ms")
                with col2:
                    min_perf = perf_df["duration_ms"].min()
                    st.metric("Temps minimum", f"{min_perf:.0f}ms")
                with col3:
                    max_perf = perf_df["duration_ms"].max()
                    st.metric("Temps maximum", f"{max_perf:.0f}ms")
            else:
                st.info("Aucune donnée de performance disponible")
        except Exception as e:
            st.warning(f"Impossible de charger les métriques de performance: {e}")

    def _format_timestamp(self, timestamp_str: str) -> str:
        """Formate un timestamp pour l'affichage."""
        try:
            dt = datetime.fromisoformat(timestamp_str)
            return dt.strftime("%d/%m/%Y %H:%M")
        except ValueError:
            return timestamp_str

    def _export_conversations_csv(self, conversations: List[Dict]):
        """Exporte les conversations en CSV."""
        if not conversations:
            st.warning("Aucune conversation à exporter.")
            return

        df = pd.DataFrame(conversations)
        csv = df.to_csv(index=False)

        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name=f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    def save_conversation(
        self, user_id: str, question: str, answer: str, response_time_ms: float, sources: Optional[List[Dict]] = None
    ) -> bool:
        """Sauvegarde une nouvelle conversation."""
        return self.data_manager.save_conversation(
            user_id=user_id, question=question, answer=answer, response_time_ms=response_time_ms, sources=sources
        )

    def add_feedback_to_conversation(self, conversation_id: str, user_id: str, rating: int, comment: str = "") -> bool:
        """Ajoute un feedback à une conversation."""
        return self.data_manager.save_feedback(
            user_id=user_id, conversation_id=conversation_id, rating=rating, comment=comment
        )


# Instance globale du gestionnaire d'historique
_history_manager = None


def get_history_manager() -> Optional[ConversationHistoryManager]:
    """Retourne l'instance globale du gestionnaire d'historique."""
    global _history_manager
    if _history_manager is None:
        _history_manager = ConversationHistoryManager()
    return _history_manager
