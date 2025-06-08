"""
Authentication manager for the webapp.
"""

import os
import sqlite3
import hashlib
import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any


class AuthManager:
    """Handles user authentication and session management."""

    def __init__(self):
        """Initialize the authentication manager."""
        self.db_path = Path(__file__).parent.parent.parent.parent / "data" / "users.db"
        self._init_auth_db()

    def _init_auth_db(self) -> None:
        """Initialize the authentication database."""
        # Create the data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Connect to the database
        conn = sqlite3.connect(self.db_path)
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

        # Create default admin user if no users exist
        c.execute("SELECT COUNT(*) FROM users")
        if c.fetchone()[0] == 0:
            admin_email = "admin@isschat.com"
            admin_password = "admin123"
            password_hash = self._hash_password(admin_password)

            c.execute(
                """
            INSERT INTO users (email, password_hash, is_admin)
            VALUES (?, ?, 1)
            """,
                (admin_email, password_hash),
            )

        conn.commit()
        conn.close()

    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user with email and password."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        password_hash = self._hash_password(password)
        c.execute(
            """
        SELECT id, email, is_admin FROM users
        WHERE email = ? AND password_hash = ?
        """,
            (email, password_hash),
        )

        result = c.fetchone()
        conn.close()

        if result:
            return {"id": result[0], "email": result[1], "is_admin": bool(result[2])}
        return None

    def register_user(self, email: str, password: str) -> bool:
        """Register a new user."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        try:
            password_hash = self._hash_password(password)
            c.execute(
                """
            INSERT INTO users (email, password_hash, is_admin)
            VALUES (?, ?, 0)
            """,
                (email, password_hash),
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False

    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return "user" in st.session_state and st.session_state.user is not None

    def is_admin(self) -> bool:
        """Check if current user is admin."""
        return self.is_authenticated() and st.session_state.user.get("is_admin", False)

    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current authenticated user."""
        if self.is_authenticated():
            return st.session_state.user
        return None

    def logout(self):
        """Logout current user."""
        if "user" in st.session_state:
            del st.session_state.user

    def render_login_form(self) -> bool:
        """Render login form and handle authentication."""
        if self.is_authenticated():
            return True

        st.title("🔐 ISSChat Login")

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")

                if submit:
                    user = self.authenticate_user(email, password)
                    if user:
                        st.session_state.user = user
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password")

        with tab2:
            with st.form("register_form"):
                reg_email = st.text_input("Email", key="reg_email")
                reg_password = st.text_input("Password", type="password", key="reg_password")
                reg_confirm = st.text_input("Confirm Password", type="password")
                reg_submit = st.form_submit_button("Register")

                if reg_submit:
                    if reg_password != reg_confirm:
                        st.error("Passwords don't match")
                    elif len(reg_password) < 6:
                        st.error("Password must be at least 6 characters")
                    elif self.register_user(reg_email, reg_password):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Email already exists")

        return False
