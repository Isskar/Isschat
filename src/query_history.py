import json
import os
from datetime import datetime
import sqlite3


class QueryHistory:
    """Gère l'historique des requêtes des utilisateurs"""

    def __init__(self, db_path=None):
        if db_path is None:
            # Utiliser un chemin absolu par défaut
            self.db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "history.db",
            )
        else:
            self.db_path = db_path

        # Créer le dossier parent si nécessaire
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialise la base de données SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Créer la table des requêtes si elle n'existe pas
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp TEXT,
            question TEXT,
            answer TEXT,
            sources TEXT,
            feedback_score INTEGER DEFAULT NULL
        )
        """
        )

        # Créer la table des favoris si elle n'existe pas
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            query_id INTEGER,
            timestamp TEXT,
            FOREIGN KEY (query_id) REFERENCES query_history (id)
        )
        """
        )

        conn.commit()
        conn.close()

    def add_query(self, user_id, question, answer, sources):
        """Ajoute une requête à l'historique"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Convertir les sources en JSON
        sources_json = json.dumps([str(s) for s in sources]) if sources else "[]"

        cursor.execute(
            """
        INSERT INTO query_history (user_id, timestamp, question, answer, sources)
        VALUES (?, ?, ?, ?, ?)
        """,
            (user_id, datetime.now().isoformat(), question, answer, sources_json),
        )

        query_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return query_id

    def get_user_history(self, user_id, limit=50):
        """Récupère l'historique des requêtes d'un utilisateur"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Pour obtenir les résultats sous forme de dictionnaire
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT h.id, h.timestamp, h.question, h.answer, h.sources, h.feedback_score,
               CASE WHEN f.id IS NOT NULL THEN 1 ELSE 0 END as is_favorite
        FROM query_history h
        LEFT JOIN favorites f ON h.id = f.query_id AND f.user_id = ?
        WHERE h.user_id = ?
        ORDER BY h.timestamp DESC
        LIMIT ?
        """,
            (user_id, user_id, limit),
        )

        rows = cursor.fetchall()
        history = []

        for row in rows:
            try:
                sources = json.loads(row["sources"])
            except:
                sources = []

            history.append(
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "sources": sources,
                    "feedback_score": row["feedback_score"],
                    "is_favorite": bool(row["is_favorite"]),
                }
            )

        conn.close()
        return history

    def add_to_favorites(self, user_id, query_id):
        """Ajoute une requête aux favoris"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Vérifier si déjà dans les favoris
        cursor.execute(
            """
        SELECT id FROM favorites
        WHERE user_id = ? AND query_id = ?
        """,
            (user_id, query_id),
        )

        if cursor.fetchone() is None:
            cursor.execute(
                """
            INSERT INTO favorites (user_id, query_id, timestamp)
            VALUES (?, ?, ?)
            """,
                (user_id, query_id, datetime.now().isoformat()),
            )

        conn.commit()
        conn.close()

    def remove_from_favorites(self, user_id, query_id):
        """Retire une requête des favoris"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        DELETE FROM favorites
        WHERE user_id = ? AND query_id = ?
        """,
            (user_id, query_id),
        )

        conn.commit()
        conn.close()

    def get_favorites(self, user_id):
        """Récupère les requêtes favorites d'un utilisateur"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT h.id, h.timestamp, h.question, h.answer, h.sources, h.feedback_score
        FROM query_history h
        JOIN favorites f ON h.id = f.query_id
        WHERE f.user_id = ?
        ORDER BY f.timestamp DESC
        """,
            (user_id,),
        )

        rows = cursor.fetchall()
        favorites = []

        for row in rows:
            try:
                sources = json.loads(row["sources"])
            except:
                sources = []

            favorites.append(
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "sources": sources,
                    "feedback_score": row["feedback_score"],
                    "is_favorite": True,
                }
            )

        conn.close()
        return favorites

    def update_feedback(self, query_id, feedback_score):
        """Met à jour le score de feedback d'une requête"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        UPDATE query_history
        SET feedback_score = ?
        WHERE id = ?
        """,
            (feedback_score, query_id),
        )

        conn.commit()
        conn.close()

    def search_history(self, user_id, search_term):
        """Recherche dans l'historique des requêtes"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Recherche dans les questions et les réponses
        cursor.execute(
            """
        SELECT h.id, h.timestamp, h.question, h.answer, h.sources, h.feedback_score,
               CASE WHEN f.id IS NOT NULL THEN 1 ELSE 0 END as is_favorite
        FROM query_history h
        LEFT JOIN favorites f ON h.id = f.query_id AND f.user_id = ?
        WHERE h.user_id = ? AND (h.question LIKE ? OR h.answer LIKE ?)
        ORDER BY h.timestamp DESC
        """,
            (user_id, user_id, f"%{search_term}%", f"%{search_term}%"),
        )

        rows = cursor.fetchall()
        results = []

        for row in rows:
            try:
                sources = json.loads(row["sources"])
            except:
                sources = []

            results.append(
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "sources": sources,
                    "feedback_score": row["feedback_score"],
                    "is_favorite": bool(row["is_favorite"]),
                }
            )

        conn.close()
        return results

    def render_history_page(self, st, user_id):
        """Affiche la page d'historique des requêtes dans Streamlit"""
        st.title("Historique des Requêtes")

        # Onglets pour l'historique et les favoris
        tab1, tab2 = st.tabs(["Historique", "Favoris"])

        with tab1:
            # Barre de recherche
            search_term = st.text_input("Rechercher dans l'historique", key="history_search")

            if search_term:
                history = self.search_history(user_id, search_term)
                st.write(f"{len(history)} résultats trouvés pour '{search_term}'")
            else:
                history = self.get_user_history(user_id)

            if not history:
                st.info("Aucune requête dans l'historique")
                return

            # Afficher l'historique
            for item in history:
                with st.expander(
                    f"{item['question']} - {datetime.fromisoformat(item['timestamp']).strftime('%d/%m/%Y %H:%M')}"
                ):
                    st.write("**Réponse:**")
                    st.write(item["answer"])

                    if item["sources"]:
                        st.write("**Sources:**")
                        for source in item["sources"]:
                            st.write(f"- {source}")

                    # Actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if item["is_favorite"]:
                            if st.button("Retirer des favoris", key=f"unfav_{item['id']}"):
                                self.remove_from_favorites(user_id, item["id"])
                                st.rerun()
                        else:
                            if st.button("Ajouter aux favoris", key=f"fav_{item['id']}"):
                                self.add_to_favorites(user_id, item["id"])
                                st.rerun()

                    with col2:
                        if st.button("Reposer cette question", key=f"reask_{item['id']}"):
                            # Stocker la question dans la session pour la réutiliser
                            st.session_state.reuse_question = item["question"]
                            # Rediriger vers la page principale
                            st.session_state.page = "chat"
                            st.rerun()

        with tab2:
            favorites = self.get_favorites(user_id)

            if not favorites:
                st.info("Aucune requête dans vos favoris")
                return

            # Afficher les favoris
            for item in favorites:
                with st.expander(
                    f"{item['question']} - {datetime.fromisoformat(item['timestamp']).strftime('%d/%m/%Y %H:%M')}"
                ):
                    st.write("**Réponse:**")
                    st.write(item["answer"])

                    if item["sources"]:
                        st.write("**Sources:**")
                        for source in item["sources"]:
                            st.write(f"- {source}")

                    # Actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Retirer des favoris", key=f"unfav_fav_{item['id']}"):
                            self.remove_from_favorites(user_id, item["id"])
                            st.rerun()

                    with col2:
                        if st.button("Reposer cette question", key=f"reask_fav_{item['id']}"):
                            # Stocker la question dans la session pour la réutiliser
                            st.session_state.reuse_question = item["question"]
                            # Rediriger vers la page principale
                            st.session_state.page = "chat"
                            st.rerun()


# Fonction pour intégrer l'historique dans l'application principale
def integrate_query_history(help_desk, user_id):
    """Intègre l'historique des requêtes au help_desk"""
    history = QueryHistory()

    # Fonction wrapper pour ask_question qui enregistre les requêtes
    original_ask = help_desk.ask_question

    def ask_with_history(question, verbose=False):
        answer, sources = original_ask(question, verbose)

        # Enregistrer la requête dans l'historique
        query_id = history.add_query(user_id=user_id, question=question, answer=answer, sources=sources)

        return answer, sources, query_id

    # Remplacer la méthode originale
    help_desk.ask_question_with_history = ask_with_history

    # Ajouter l'historique comme attribut
    help_desk.query_history = history

    return help_desk
