import json
import os
from datetime import datetime
import sqlite3


class QueryHistory:
    """Manages user query history"""

    def __init__(self, db_path=None):
        if db_path is None:
            # Use an absolute path by default
            self.db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "history.db",
            )
        else:
            self.db_path = db_path

        # Create parent folder if necessary
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create the query table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp TEXT,
            question TEXT,
            answer TEXT,
            sources TEXT,
            feedback_score INTEGER DEFAULT NULL
        )
        """)

        # Create the favorites table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            query_id INTEGER,
            timestamp TEXT,
            FOREIGN KEY (query_id) REFERENCES query_history (id)
        )
        """)

        conn.commit()
        conn.close()

    def add_query(self, user_id, question, answer, sources):
        """Add a query to the history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Convert sources to JSON
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
        """Retrieve the query history for a user"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # To get results as dictionaries
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
            except Exception as e:
                print(f"Error parsing sources: {e}")
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
        """Add a query to favorites"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if already in favorites
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
        """Remove a query from favorites"""
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
        """Retrieve a user's favorite queries"""
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
            except Exception as e:
                print(f"Error parsing sources: {e}")
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
        """Updates the feedback score of a query"""
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
        """Searches in the query history"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Search in questions and answers
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
            except Exception as e:
                print(f"Error parsing sources: {e}")
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
        """Displays the query history page in Streamlit"""
        st.title("Query History")

        tab1, tab2 = st.tabs(["History", "Favorites"])

        with tab1:
            search_term = st.text_input("Search in history", key="history_search")

            if search_term:
                history = self.search_history(user_id, search_term)
                st.write(f"{len(history)} results found for '{search_term}'")
            else:
                history = self.get_user_history(user_id)

            if not history:
                st.info("No queries in history")
                return

            # Display history
            for item in history:
                with st.expander(
                    f"{item['question']} - {datetime.fromisoformat(item['timestamp']).strftime('%d/%m/%Y %H:%M')}"
                ):
                    st.write("**Answer:**")
                    st.write(item["answer"])

                    if item["sources"]:
                        st.write("**Sources:**")
                        for source in item["sources"]:
                            st.write(f"- {source}")

                    # Actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if item["is_favorite"]:
                            if st.button("Remove from favorites", key=f"unfav_{item['id']}"):
                                self.remove_from_favorites(user_id, item["id"])
                                st.rerun()
                        else:
                            if st.button("Add to favorites", key=f"fav_{item['id']}"):
                                self.add_to_favorites(user_id, item["id"])
                                st.rerun()

                    with col2:
                        if st.button("Ask this question again", key=f"reask_{item['id']}"):
                            # Store the question in the session for reuse
                            st.session_state.reuse_question = item["question"]
                            # Redirect to the main page
                            st.session_state.page = "chat"
                            st.rerun()

        with tab2:
            favorites = self.get_favorites(user_id)

            if not favorites:
                st.info("No queries in your favorites")
                return

            # Display favorites
            for item in favorites:
                with st.expander(
                    f"{item['question']} - {datetime.fromisoformat(item['timestamp']).strftime('%d/%m/%Y %H:%M')}"
                ):
                    st.write("**Answer:**")
                    st.write(item["answer"])

                    if item["sources"]:
                        st.write("**Sources:**")
                        for source in item["sources"]:
                            st.write(f"- {source}")

                    # Actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Remove from favorites", key=f"unfav_fav_{item['id']}"):
                            self.remove_from_favorites(user_id, item["id"])
                            st.rerun()

                    with col2:
                        if st.button("Ask this question again", key=f"reask_fav_{item['id']}"):
                            # Store the question in the session for reuse
                            st.session_state.reuse_question = item["question"]
                            # Redirect to the main page
                            st.session_state.page = "chat"
                            st.rerun()


# Function to integrate history into the main application
def integrate_query_history(help_desk, user_id):
    """Integrates query history into the help_desk"""
    history = QueryHistory()

    # Wrapper function for ask_question that records queries
    original_ask = help_desk.ask_question

    def ask_with_history(question, verbose=False):
        answer, sources = original_ask(question, verbose)

        # Record the query in history
        query_id = history.add_query(user_id=user_id, question=question, answer=answer, sources=sources)

        return answer, sources, query_id

    # Replace the original method
    help_desk.ask_question_with_history = ask_with_history

    # Add history as an attribute
    help_desk.query_history = history

    return help_desk
