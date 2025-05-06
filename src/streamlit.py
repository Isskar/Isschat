# Streamlit
# Use QARetrieval to find informations about the Confluence knowledge base
# Enhanced with advanced features:
# - Conversation analysis
# - Response tracking
# - Question suggestion
# - Performance monitoring
# - Query history
# - Automatic reformulation

import streamlit as st
import time
import os
import sys
import uuid
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire parent au chemin de recherche Python
sys.path.append(str(Path(__file__).parent.parent))

# Import des modules personnalisés
from src.help_desk import HelpDesk
from src.auth import verify_user, logout, get_all_users, add_user, delete_user, login_page, login_required, admin_required

# Import des nouvelles fonctionnalités
from src.features_integration import setup_features, FeaturesManager

# Configuration de la page Streamlit - doit être la première commande Streamlit
st.set_page_config(page_title="Assistant Confluence", page_icon="🤖", layout="wide")

# Pas de cache pour forcer le rechargement à chaque lancement
def get_model(rebuild_db=False):
    # Afficher un spinner pendant le chargement
    with st.spinner("Chargement du modèle RAG..."):
        # Vérifier si le fichier index.faiss existe
        import os
        import sys
        from pathlib import Path
        from config import PERSIST_DIRECTORY
        
        # Afficher les informations de configuration pour le débogage
        st.sidebar.expander("Débogage", expanded=False).write(f"""
        **Configuration**:
        - Dossier de la base vectorielle: `{PERSIST_DIRECTORY}`
        - URL Confluence: `{os.getenv('CONFLUENCE_SPACE_NAME')}`
        - Clé d'espace: `{os.getenv('CONFLUENCE_SPACE_KEY')}`
        - Utilisateur: `{os.getenv('CONFLUENCE_EMAIL_ADRESS')}`
        - Clé API: `{'*'*5}{os.getenv('CONFLUENCE_PRIVATE_API_KEY')[-5:] if os.getenv('CONFLUENCE_PRIVATE_API_KEY') else 'Non définie'}`
        """)
        
        # Vérifier que le dossier existe
        if not os.path.exists(PERSIST_DIRECTORY):
            st.warning(f"Le dossier {PERSIST_DIRECTORY} n'existe pas. Tentative de création...")
            try:
                os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
            except Exception as e:
                st.error(f"Erreur lors de la création du dossier: {str(e)}")
        
        # Vérifier si les fichiers d'index existent
        index_faiss_path = os.path.join(PERSIST_DIRECTORY, "index.faiss")
        index_pkl_path = os.path.join(PERSIST_DIRECTORY, "index.pkl")
        
        if not os.path.exists(index_faiss_path) or not os.path.exists(index_pkl_path):
            st.warning("Base de données vectorielle non trouvée. Reconstruction en cours...")
            rebuild_db = True
        else:
            st.success("Base de données vectorielle trouvée!")
        
        # Créer le modèle
        try:
            model = HelpDesk(new_db=rebuild_db)
            return model
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")
            sys.exit(1)

# Initialisation de l'interface utilisateur
def main():
    # Sidebar pour la navigation et les options
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/confluence--v2.png", width=100)
        st.title("Assistant Confluence")
        
        # Afficher les informations de l'utilisateur connecté
        if "user" in st.session_state:
            st.success(f"Connecté en tant que: {st.session_state['user']['email']}")
            
            # Navigation principale
            st.subheader("Navigation")
            if st.button("Chat", key="nav_chat"):
                st.session_state["page"] = "chat"
                st.rerun()
                
            if st.button("Historique", key="nav_history"):
                st.session_state["page"] = "history"
                st.rerun()
                
            # Options d'administration
            if st.session_state["user"].get("is_admin"):
                st.divider()
                st.info("Statut: Administrateur")
                if st.button("Tableau de bord admin", key="nav_admin"):
                    st.session_state["page"] = "admin"
                    st.rerun()
                
                # Option pour reconstruire la base de données
                st.divider()
                st.subheader("Gestion de la base de données")
                
                # Ajouter un bouton pour forcer la reconstruction complète
                if st.button("Reconstruire depuis Confluence", type="primary"):
                    with st.spinner("Reconstruction de la base de données depuis Confluence en cours..."):
                        # Supprimer les fichiers existants
                        import os
                        import shutil
                        from config import PERSIST_DIRECTORY
                        
                        try:
                            if os.path.exists(PERSIST_DIRECTORY):
                                shutil.rmtree(PERSIST_DIRECTORY)
                                st.info(f"Dossier {PERSIST_DIRECTORY} supprimé avec succès.")
                            os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
                        except Exception as e:
                            st.error(f"Erreur lors de la suppression du dossier: {str(e)}")
                        
                        # Forcer le rechargement du modèle avec new_db=True
                        try:
                            st.cache_resource.clear()
                            model = get_model(rebuild_db=True)
                            st.success("Base de données reconstruite avec succès depuis Confluence!")
                            time.sleep(2)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erreur lors de la reconstruction: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc(), language="python")
            
            # Bouton de déconnexion (pour tous les utilisateurs)
            st.divider()
            if st.button("Déconnexion", key="nav_logout"):
                logout()
    
    # Déterminer quelle page afficher
    if "user" not in st.session_state:
        login_page()
    elif st.session_state.get("page") == "admin" and st.session_state["user"].get("is_admin"):
        admin_page()
    elif st.session_state.get("page") == "history":
        history_page()
    else:
        chat_page()

def login_required(func):
    """Décorateur pour vérifier si l'utilisateur est connecté"""
    def wrapper(*args, **kwargs):
        if "user" not in st.session_state:
            st.warning("Veuillez vous connecter pour accéder à cette page.")
            login_page()
            return None
        return func(*args, **kwargs)
    return wrapper

def admin_required(func):
    """Décorateur pour vérifier si l'utilisateur est admin"""
    def wrapper(*args, **kwargs):
        if "user" not in st.session_state or not st.session_state["user"].get("is_admin"):
            st.error("Vous n'avez pas les droits d'administration nécessaires.")
            st.session_state["page"] = "chat"
            chat_page()
            return None
        return func(*args, **kwargs)
    return wrapper

def login_page():
    """Affiche la page de connexion"""
    st.title("Connexion à l'Assistant Confluence")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Se connecter")
        
        if submit and email and password:
            user = verify_user(email, password)
            if user:
                # Utiliser l'ID utilisateur de la base de données pour assurer la persistance
                # Pas besoin de générer un UUID aléatoire à chaque session
                st.session_state["user"] = user
                
                # Utiliser l'ID utilisateur directement comme identifiant persistant
                st.session_state["user_id"] = f"user_{user['id']}"
                
                st.session_state["page"] = "chat"
                st.rerun()
            else:
                st.error("Email ou mot de passe incorrect.")

@login_required
def chat_page():
    # Réinitialiser la page si nécessaire
    st.session_state["page"] = "chat"
    
    # Charger le modèle
    model = get_model()
    
    # Initialiser le gestionnaire de fonctionnalités
    if "features_manager" not in st.session_state:
        user_id = st.session_state.get("user_id", f"user_{st.session_state['user']['id']}")
        st.session_state["features_manager"] = FeaturesManager(model, user_id)
    
    features = st.session_state["features_manager"]
    
    # Créer la mise en page
    st.title("Assistant de Recherche Confluence")
    
    # Barre latérale pour les options avancées
    with st.sidebar:
        st.subheader("Options avancées")
        show_suggestions = st.toggle("Suggestions de questions", value=True)
        show_feedback = st.toggle("Feedback sur les réponses", value=True)
        
        # Historique des requêtes
        if st.button("Historique des requêtes"):
            st.session_state["page"] = "history"
            st.rerun()
    
    # Afficher l'interface principale
    st.subheader("Posez vos questions sur la documentation Confluence")
    
    # Initialiser l'historique des messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Bonjour ! Comment puis-je vous aider aujourd'hui ?"}]
    
    # Afficher l'historique des messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # Vérifier s'il y a une question à réutiliser depuis l'historique
    if "reuse_question" in st.session_state:
        prompt = st.session_state.pop("reuse_question")
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Traiter la question avec toutes les fonctionnalités
        with st.spinner("Analyse en cours..."):
            result, sources = features.process_question(
                prompt,
                show_suggestions=show_suggestions,
                show_feedback=show_feedback
            )
        
        # Ajouter la réponse aux messages
        response_content = result
        if sources:
            response_content += '\n\n' + sources
            
        st.session_state.messages.append({"role": "assistant", "content": response_content})
    
    # Interface de chat
    if prompt := st.chat_input("Posez votre question ici..."):
        # Ajouter la question
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Traiter la question avec toutes les fonctionnalités
        with st.spinner("Analyse en cours..."):
            result, sources = features.process_question(
                prompt,
                show_suggestions=show_suggestions,
                show_feedback=show_feedback
            )
        
        # Ajouter la réponse aux messages
        response_content = result
        if sources:
            response_content += '\n\n' + sources
            
        st.chat_message("assistant").write(response_content)
        st.session_state.messages.append({"role": "assistant", "content": response_content})

@login_required
def history_page():
    """Affiche la page d'historique des requêtes"""
    # Réinitialiser la page si nécessaire
    st.session_state["page"] = "history"
    
    # Récupérer le gestionnaire de fonctionnalités
    if "features_manager" not in st.session_state:
        st.error("Erreur: Gestionnaire de fonctionnalités non initialisé")
        st.session_state["page"] = "chat"
        st.rerun()
    
    features = st.session_state["features_manager"]
    user_id = st.session_state.get("user_id", f"user_{st.session_state['user']['id']}")
    
    # Afficher l'historique des requêtes
    features.query_history.render_history_page(st, user_id)
    
    # Bouton de retour
    if st.button("Retour au chat", key="return_from_history"):
        st.session_state["page"] = "chat"
        st.rerun()

@admin_required
def admin_page():
    """Affiche la page d'administration"""
    st.title("Tableau de bord d'administration")
    
    # Créer des onglets pour séparer les différentes fonctionnalités
    tab1, tab2 = st.tabs(["Gestion des utilisateurs", "Analyse et performances"])
    
    with tab1:
        st.header("Gestion des utilisateurs")
        
        # Formulaire pour ajouter un utilisateur
        with st.expander("Ajouter un utilisateur", expanded=True):
            with st.form("add_user_form"):
                email = st.text_input("Email")
                password = st.text_input("Mot de passe", type="password")
                is_admin = st.checkbox("Administrateur")
                submit = st.form_submit_button("Ajouter")
                
                if submit and email and password:
                    success = add_user(email, password, is_admin)
                    if success:
                        st.success(f"Utilisateur {email} ajouté avec succès.")
                    else:
                        st.error(f"L'email {email} existe déjà.")
        
        # Liste des utilisateurs
        st.subheader("Liste des utilisateurs")
        users = get_all_users()
        
        if users:
            # Créer un tableau pour afficher les utilisateurs
            cols = st.columns([3, 2, 1, 1])
            cols[0].write("**Email**")
            cols[1].write("**Date de création**")
            cols[2].write("**Admin**")
            cols[3].write("**Actions**")
            
            for user in users:
                cols = st.columns([3, 2, 1, 1])
                cols[0].write(user["email"])
                cols[1].write(user["created_at"])
                cols[2].write("Oui" if user["is_admin"] else "Non")
                
                # Ne pas permettre de supprimer l'utilisateur connecté
                if user["id"] != st.session_state["user"]["id"]:
                    if cols[3].button("Supprimer", key=f"delete_{user['id']}"):
                        delete_user(user["id"])
                        st.success(f"Utilisateur {user['email']} supprimé.")
                        st.rerun()
                else:
                    cols[3].write("(Vous)")
        else:
            st.info("Aucun utilisateur trouvé.")
    
    with tab2:
        # Vérifier si le gestionnaire de fonctionnalités est initialisé
        if "features_manager" in st.session_state:
            features = st.session_state["features_manager"]
            features.render_admin_dashboard(st)
        else:
            st.warning("Veuillez d'abord interagir avec le chatbot pour initialiser les fonctionnalités d'analyse.")
    
    # Bouton de retour
    if st.button("Retour au chat"):
        st.session_state["page"] = "chat"
        st.rerun()

# Lancer l'application
if __name__ == "__main__":
    main()
