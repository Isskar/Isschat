import os
import sqlite3
import hashlib
import streamlit as st
from pathlib import Path

# Path to the SQLite database
DB_PATH = Path(__file__).parent.parent / "data" / "users.db"


def init_auth_db() -> None:
    """Initialize the authentication database"""
    # Create the data directory if it doesn't exist
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Create the users table if it doesn't exist
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        is_admin BOOLEAN NOT NULL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Commit changes and close the connection
    conn.commit()
    conn.close()


def verify_user(email: str, password: str) -> dict | None:
    """Verify user credentials"""
    # Hash the provided password
    password_hash = hashlib.sha256(password.encode()).hexdigest()

    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Verify credentials
    c.execute(
        "SELECT id, is_admin FROM users WHERE email = ? AND password_hash = ?",
        (email, password_hash),
    )
    user = c.fetchone()
    conn.close()

    if user:
        return {"id": user[0], "email": email, "is_admin": user[1]}
    return None


def add_user(email: str, password: str, is_admin: bool = False) -> bool:
    """Add a new user to the database"""
    # Hash the password
    password_hash = hashlib.sha256(password.encode()).hexdigest()

    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Add user
        c.execute(
            "INSERT INTO users (email, password_hash, is_admin) VALUES (?, ?, ?)",
            (email, password_hash, is_admin),
            (email, password_hash, is_admin),
        )

        # Commit changes and close the connection
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def get_all_users() -> list[dict]:
    """Retrieve all users from the database"""
    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Retrieve all users
    c.execute("SELECT id, email, is_admin, created_at FROM users")
    users = c.fetchall()
    conn.close()

    # Format the results
    return [{"id": u[0], "email": u[1], "is_admin": u[2], "created_at": u[3]} for u in users]


def delete_user(user_id: int) -> None:
    """Delete a user from the database"""
    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Delete the user
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))

    # Commit changes and close the connection
    conn.commit()
    conn.close()


def login_required(func):
    """Decorator to protect pages that require authentication"""

    def wrapper(*args, **kwargs):
        if not st.session_state.get("user"):
            st.warning("You must log in to access this page.")
            st.stop()
        return func(*args, **kwargs)

    return wrapper


def admin_required(func):
    """Decorator to protect pages that require administrator rights"""

    def wrapper(*args, **kwargs):
        if not st.session_state.get("user") or not st.session_state["user"].get("is_admin"):
            st.error("You do not have administrator rights to access this page.")
            st.stop()
        return func(*args, **kwargs)

    return wrapper


def login_page() -> None:
    """Displays the login page"""
    st.title("Login")

    # Login form
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            user = verify_user(email, password)
            if user:
                st.session_state["user"] = user
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Incorrect email or password.")


def logout() -> None:
    """Logs out the user"""
    if "user" in st.session_state:
        del st.session_state["user"]
    st.rerun()


# Initialize the database at startup
init_auth_db()
