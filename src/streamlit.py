# Streamlit
# Use QARetrieval to find informations about the Octo Confluence
# Basic example with a improvementd:
# Add streaming
# Add Conversation history
# Optimize Splitter, Retriever,
# Try Open source models
import streamlit as st

# Configuration de la page Streamlit - doit être la première commande Streamlit
st.set_page_config(page_title="Assistant Confluence", page_icon="🤖")

import time
from help_desk import HelpDesk
from auth import login_page, login_required, admin_required, logout, get_all_users, add_user, delete_user


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
        - Utilisateur: `{os.getenv('EMAIL_ADRESS')}`
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
            if st.session_state["user"].get("is_admin"):
                st.info("Statut: Administrateur")
                if st.button("Gestion des utilisateurs"):
                    st.session_state["page"] = "admin"
            
            if st.button("Déconnexion"):
                logout()
            
            # Option pour reconstruire la base de données
            with st.expander("Options avancées"):
                rebuild_db = st.button("Reconstruire la base de données")
                if rebuild_db:
                    with st.spinner("Reconstruction de la base de données..."):
                        # Forcer le rechargement du modèle avec new_db=True
                        st.cache_resource.clear()
                        get_model(rebuild_db=True)
                        st.success("Base de données reconstruite avec succès!")
                        time.sleep(2)
                        st.rerun()
    
    # Déterminer quelle page afficher
    if "user" not in st.session_state:
        login_page()
    elif st.session_state.get("page") == "admin" and st.session_state["user"].get("is_admin"):
        admin_page()
    else:
        chat_page()

@login_required
def chat_page():
    # Réinitialiser la page si nécessaire
    st.session_state["page"] = "chat"
    
    # Charger le modèle
    model = get_model()
    
    # Titre et sous-titre
    st.title("Assistant de Recherche Confluence")
    st.subheader("Posez vos questions sur la documentation Confluence")
    
    # Initialiser l'historique des messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Bonjour ! Comment puis-je vous aider aujourd'hui ?"}]
    
    # Afficher l'historique des messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # Interface de chat
    if prompt := st.chat_input("Posez votre question ici..."):
        # Ajouter la question
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Obtenir la réponse avec un spinner
        with st.spinner("Recherche en cours..."):
            result, sources = model.retrieval_qa_inference(prompt)
        
        # Ajouter la réponse et les sources
        st.chat_message("assistant").write(result + '  \n  \n' + sources)
        st.session_state.messages.append({"role": "assistant", "content": result + '  \n  \n' + sources})

@admin_required
def admin_page():
    st.title("Gestion des utilisateurs")
    
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
    
    if st.button("Retour au chat"):
        st.session_state["page"] = "chat"
        st.rerun()

# Lancer l'application
if __name__ == "__main__":
    main()
