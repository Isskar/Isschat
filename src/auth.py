import os
import sqlite3
import hashlib
import streamlit as st
from pathlib import Path

# Chemin vers la base de données SQLite
DB_PATH = Path(__file__).parent.parent / "data" / "users.db"

def init_auth_db():
    """Initialise la base de données d'authentification"""
    # Créer le répertoire data s'il n'existe pas
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Connexion à la base de données
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Créer la table des utilisateurs si elle n'existe pas
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        is_admin BOOLEAN NOT NULL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Vérifier si l'utilisateur admin existe déjà
    c.execute("SELECT * FROM users WHERE email = ?", ("nicolas.lambropoulos@isskar.fr",))
    if not c.fetchone():
        # Ajouter l'utilisateur admin
        password_hash = hashlib.sha256("mdp".encode()).hexdigest()
        c.execute(
            "INSERT INTO users (email, password_hash, is_admin) VALUES (?, ?, ?)",
            ("nicolas.lambropoulos@isskar.fr", password_hash, True)
        )
        print("Utilisateur admin créé avec succès.")
    
    # Valider les changements et fermer la connexion
    conn.commit()
    conn.close()

def verify_user(email, password):
    """Vérifie les identifiants de l'utilisateur"""
    # Hasher le mot de passe fourni
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    # Connexion à la base de données
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Vérifier les identifiants
    c.execute("SELECT id, is_admin FROM users WHERE email = ? AND password_hash = ?", 
              (email, password_hash))
    user = c.fetchone()
    conn.close()
    
    if user:
        return {"id": user[0], "email": email, "is_admin": user[1]}
    return None

def add_user(email, password, is_admin=False):
    """Ajoute un nouvel utilisateur à la base de données"""
    # Hasher le mot de passe
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    try:
        # Connexion à la base de données
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Ajouter l'utilisateur
        c.execute(
            "INSERT INTO users (email, password_hash, is_admin) VALUES (?, ?, ?)",
            (email, password_hash, is_admin)
        )
        
        # Valider les changements et fermer la connexion
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        # L'email existe déjà
        return False

def get_all_users():
    """Récupère tous les utilisateurs de la base de données"""
    # Connexion à la base de données
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Récupérer tous les utilisateurs
    c.execute("SELECT id, email, is_admin, created_at FROM users")
    users = c.fetchall()
    conn.close()
    
    # Formater les résultats
    return [{"id": u[0], "email": u[1], "is_admin": u[2], "created_at": u[3]} for u in users]

def delete_user(user_id):
    """Supprime un utilisateur de la base de données"""
    # Connexion à la base de données
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Supprimer l'utilisateur
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    
    # Valider les changements et fermer la connexion
    conn.commit()
    conn.close()

def login_required(func):
    """Décorateur pour protéger les pages qui nécessitent une authentification"""
    def wrapper(*args, **kwargs):
        if not st.session_state.get("user"):
            st.warning("Vous devez vous connecter pour accéder à cette page.")
            st.stop()
        return func(*args, **kwargs)
    return wrapper

def admin_required(func):
    """Décorateur pour protéger les pages qui nécessitent des droits d'administrateur"""
    def wrapper(*args, **kwargs):
        if not st.session_state.get("user") or not st.session_state["user"].get("is_admin"):
            st.error("Vous n'avez pas les droits d'administrateur pour accéder à cette page.")
            st.stop()
        return func(*args, **kwargs)
    return wrapper

def login_page():
    """Affiche la page de connexion"""
    st.title("Connexion")
    
    # Formulaire de connexion
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Se connecter")
        
        if submit:
            user = verify_user(email, password)
            if user:
                st.session_state["user"] = user
                st.success("Connexion réussie!")
                st.rerun()
            else:
                st.error("Email ou mot de passe incorrect.")

def logout():
    """Déconnecte l'utilisateur"""
    if "user" in st.session_state:
        del st.session_state["user"]
    st.rerun()

# Initialiser la base de données au démarrage
init_auth_db()
