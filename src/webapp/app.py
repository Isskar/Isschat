"""
Application Streamlit pour Isschat - Interface de chat RAG
Version refactorisée avec architecture propre
"""

import streamlit as st
import time
import signal
import os
import sys
import asyncio
import logging
from pathlib import Path
import shutil
from typing import Optional, Tuple, Any

# === CONFIGURATION INITIALE ===
# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Streamlit - doit être la première commande Streamlit
st.set_page_config(page_title="Isschat", page_icon="🤖", layout="wide")


# === CONFIGURATION DU PROJET ===
def setup_project_paths():
    """Configure les chemins du projet de manière propre"""
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent

    # Ajout du chemin racine au PYTHONPATH si nécessaire
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.info(f"✅ Chemin projet ajouté: {project_root}")

    return project_root


def setup_environment():
    """Configure l'environnement système"""
    # Désactiver le parallélisme des tokenizers pour éviter les deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Configuration asyncio simplifiée pour Streamlit
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # Pas de boucle en cours, c'est normal pour Streamlit
        pass


# === IMPORTS SÉCURISÉS ===
def import_core_config():
    """Importe la configuration core"""
    try:
        from src.core.config import get_config, get_debug_info

        logger.info("✅ Configuration core importée")
        return get_config, get_debug_info
    except ImportError as e:
        logger.error(f"❌ Impossible d'importer la configuration: {e}")
        st.error("Erreur critique: Configuration non disponible")
        st.stop()


def import_auth_system():
    """Importe le système d'authentification"""
    try:
        from src.webapp.components.auth_manager import AuthManager

        logger.info("✅ Système d'auth importé")
        return AuthManager()
    except ImportError as e:
        logger.warning(f"⚠️ Auth non disponible: {e}")
        return None


def import_features_manager():
    """Importe le gestionnaire de fonctionnalités"""
    try:
        from src.webapp.components.features_manager import FeaturesManager

        logger.info("✅ FeaturesManager importé")
        return FeaturesManager
    except ImportError as e:
        logger.warning(f"⚠️ FeaturesManager non disponible: {e}")
        return None


def import_rag_pipeline():
    """Importe le pipeline RAG"""
    try:
        from src.rag_system.rag_pipeline import RAGPipelineFactory

        logger.info("✅ RAG Pipeline importé")
        return RAGPipelineFactory, True
    except ImportError as e:
        logger.error(f"❌ RAG Pipeline non disponible: {e}")
        return None, False


def import_history_components():
    """Importe les composants d'historique"""
    try:
        from src.webapp.components.history_manager import get_history_manager
        from src.core.data_manager import get_data_manager

        logger.info("✅ Composants d'historique importés")
        return get_history_manager, get_data_manager
    except ImportError as e:
        logger.warning(f"⚠️ Composants d'historique non disponibles: {e}")
        return None, None


# === INITIALISATION ===
setup_project_paths()
setup_environment()

# Imports
get_config, get_debug_info = import_core_config()
auth_manager = import_auth_system()
FeaturesManager = import_features_manager()
RAGPipelineFactory, NEW_RAG_AVAILABLE = import_rag_pipeline()
get_history_manager, get_data_manager = import_history_components()

logger.info(f"🚀 Initialisation terminée - RAG: {NEW_RAG_AVAILABLE}, Auth: {auth_manager is not None}")


# === FONCTIONS UTILITAIRES ===
def create_auth_decorators():
    """Crée les décorateurs d'authentification"""

    def login_required(func):
        """Décorateur pour les pages nécessitant une authentification"""

        def wrapper(*args, **kwargs):
            if auth_manager and not auth_manager.is_authenticated():
                if not auth_manager.render_login_form():
                    return
            return func(*args, **kwargs)

        return wrapper

    def admin_required(func):
        """Décorateur pour les pages admin"""

        def wrapper(*args, **kwargs):
            if auth_manager and not auth_manager.is_admin():
                st.error("Accès administrateur requis")
                return
            return func(*args, **kwargs)

        return wrapper

    return login_required, admin_required


login_required, admin_required = create_auth_decorators()


# === GESTION DU MODÈLE RAG ===
@st.cache_resource
def get_model(rebuild_db: bool = False):
    """Charge le modèle RAG avec gestion d'erreurs propre"""
    with st.spinner("Chargement du modèle RAG..."):
        config = get_config()
        debug_info = get_debug_info()

        # Affichage des informations de debug
        st.sidebar.expander("Debug", expanded=False).write(f"""
        **Configuration**:
        - Provider: `{debug_info["provider"]}`
        - Vector store: `{debug_info["persist_directory"]}`
        - Confluence URL: `{debug_info["confluence_url"]}`
        - Space key: `{debug_info["space_key"]}`
        - User: `{debug_info["user_email"]}`
        - API key: `{debug_info["confluence_api_key"]}`
        - OpenRouter key: `{debug_info["openrouter_api_key"]}`
        """)

        persist_path = Path(config.persist_directory)
        index_file = persist_path / "index.faiss"
        status_placeholder = st.empty()

        # Vérification si reconstruction nécessaire
        if not rebuild_db:
            if not persist_path.exists() or not index_file.exists():
                status_placeholder.info("🚀 Premier lancement - Création de la base vectorielle...")
                rebuild_db = True

        # Création du modèle
        try:
            if not NEW_RAG_AVAILABLE:
                st.error("❌ Architecture RAG non disponible!")
                return None

            # Création du pipeline RAG
            pipeline = RAGPipelineFactory.create_default_pipeline()

            if rebuild_db:
                status_placeholder.success("✅ Base vectorielle créée avec succès!")
                time.sleep(2)
                status_placeholder.empty()

            return pipeline

        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle: {str(e)}")
            import traceback

            st.code(traceback.format_exc(), language="python")
            return None


# === GESTION DES UTILISATEURS ===
def setup_user_session():
    """Configure la session utilisateur"""
    if "user" not in st.session_state:
        config = get_config()
        email = config.confluence_email_address or "admin@auto.login"

        if auth_manager:
            # Utilisation du système d'auth
            try:
                # Essayer d'authentifier avec le mot de passe par défaut
                user = auth_manager.authenticate_user(email, "admin123")
                if not user:
                    # Essayer de créer l'utilisateur
                    if auth_manager.register_user(email, "admin123"):
                        user = auth_manager.authenticate_user(email, "admin123")
                    else:
                        # L'utilisateur existe peut-être déjà, essayer avec le mot de passe par défaut de l'auth_manager
                        user = auth_manager.authenticate_user("admin@isschat.com", "admin123")

                if user:
                    st.session_state["user"] = user
                    st.session_state["user_id"] = f"user_{user['id']}"
                    logger.info(f"✅ Utilisateur connecté: {user['email']}")
                else:
                    # Fallback: créer un utilisateur en session sans base de données
                    logger.warning("⚠️ Fallback: création d'utilisateur en session")
                    st.session_state["user"] = {"email": email, "id": 1, "is_admin": True}
                    st.session_state["user_id"] = "user_1"
            except Exception as e:
                logger.error(f"❌ Erreur auth: {e}")
                # Fallback en cas d'erreur
                st.session_state["user"] = {"email": email, "id": 1, "is_admin": True}
                st.session_state["user_id"] = "user_1"
        else:
            # Fallback sans système d'auth
            st.session_state["user"] = {"email": email, "id": 1, "is_admin": True}
            st.session_state["user_id"] = "user_1"

        st.session_state["page"] = "chat"


# === TRAITEMENT DES QUESTIONS ===
def process_question(model, features_manager: Optional[Any], prompt: str) -> Tuple[str, str, float]:
    """Traite une question avec le modèle RAG"""
    start_time = time.time()

    try:
        if features_manager:
            # Utilisation du gestionnaire de fonctionnalités
            result, sources = features_manager.process_question(prompt)
        else:
            # Appel direct au modèle
            if hasattr(model, "process_query"):
                result, sources = model.process_query(prompt)
            elif hasattr(model, "query"):
                result = model.query(prompt)
                sources = ""
            else:
                result = "Modèle non disponible"
                sources = ""
    except Exception as e:
        result = f"Erreur: {str(e)}"
        sources = ""

    response_time_ms = (time.time() - start_time) * 1000
    return result, sources, response_time_ms


def save_conversation_data(prompt: str, result: str, sources: str, response_time_ms: float):
    """Sauvegarde les données de conversation"""
    if not (get_history_manager and get_data_manager):
        return

    try:
        history_manager = get_history_manager()
        data_manager = get_data_manager()
        user_id = st.session_state.get("user", {}).get("email", "anonymous")

        # Préparation des sources
        sources_data = []
        if sources and isinstance(sources, str):
            sources_data = [{"content": sources, "type": "text"}]
        elif sources and isinstance(sources, list):
            for source in sources:
                if hasattr(source, "metadata"):
                    source_data = {
                        "title": source.metadata.get("title", "Document"),
                        "url": source.metadata.get("url", ""),
                        "content": source.page_content[:200] + "..."
                        if len(source.page_content) > 200
                        else source.page_content,
                        "type": "confluence",
                    }
                elif isinstance(source, dict):
                    source_data = {
                        "title": source.get("title", source.get("metadata", {}).get("title", "Document")),
                        "url": source.get("url", source.get("metadata", {}).get("url", "")),
                        "content": source.get("content", "")[:200] + "..."
                        if len(source.get("content", "")) > 200
                        else source.get("content", ""),
                        "type": "confluence",
                    }
                else:
                    source_data = {"content": str(source), "type": "text"}
                sources_data.append(source_data)

        # Sauvegarde
        history_manager.save_conversation(
            user_id=user_id,
            question=prompt,
            answer=result,
            response_time_ms=response_time_ms,
            sources=sources_data,
        )

        data_manager.save_performance(
            operation="query_processing",
            duration_ms=response_time_ms,
            user_id=user_id,
            metadata={
                "question_length": len(prompt),
                "answer_length": len(result),
                "sources_count": len(sources_data),
                "has_sources": len(sources_data) > 0,
            },
        )

    except Exception as e:
        st.warning(f"Impossible de sauvegarder la conversation: {e}")


# === PAGES DE L'APPLICATION ===
@login_required
def chat_page():
    """Page principale de chat"""
    st.session_state["page"] = "chat"

    # Chargement du modèle
    model = get_model()
    if not model:
        st.error("❌ Aucun modèle RAG disponible")
        return

    # Initialisation du gestionnaire de fonctionnalités
    if "features_manager" not in st.session_state:
        user_id = st.session_state.get("user_id", f"user_{st.session_state['user']['id']}")
        if FeaturesManager:
            # Correction: FeaturesManager ne prend qu'un paramètre user_id
            st.session_state["features_manager"] = FeaturesManager(user_id)
        else:
            st.session_state["features_manager"] = None

    features = st.session_state["features_manager"]

    # Interface utilisateur
    st.title("Bienvenue sur ISSCHAT")

    # Sidebar pour les options avancées
    with st.sidebar:
        st.subheader("Options avancées")
        show_feedback = st.toggle("Feedback des réponses", value=True)

        # Statut du système RAG
        if NEW_RAG_AVAILABLE:
            st.subheader("🚀 Statut du système RAG")
            st.success("Nouvelle architecture active")

            if hasattr(model, "health_check"):
                try:
                    health = model.health_check()
                    if health.get("status") == "healthy":
                        st.caption("✅ Système en bonne santé")
                    else:
                        st.caption("⚠️ Problèmes détectés")
                        if health.get("issues"):
                            for issue in health["issues"]:
                                st.caption(f"• {issue}")
                except Exception:
                    st.caption("ℹ️ Vérification de santé indisponible")
        else:
            st.subheader("⚠️ Statut du système")
            st.warning("Architecture legacy")

        if st.button("Historique des requêtes"):
            st.session_state["page"] = "history"
            st.rerun()

    # Interface de chat
    st.subheader("Posez vos questions sur notre documentation Confluence")

    # Message de bienvenue
    user_email = st.session_state.get("user", {}).get("email", "")
    first_name = user_email.split("@")[0].split(".")[0].capitalize()
    welcome_message = f"Bonjour {first_name} ! Comment puis-je vous aider aujourd'hui ?"

    # Initialisation de l'historique des messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": welcome_message}]

    # Affichage de l'historique des messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Gestion des questions réutilisées depuis l'historique
    if "reuse_question" in st.session_state:
        prompt = st.session_state.pop("reuse_question")
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Traitement de la question
        with st.spinner("Analyse en cours..."):
            result, sources, response_time_ms = process_question(model, features, prompt)

            # Affichage de la réponse
            response_content = result
            if sources:
                response_content += "\n\n" + sources

            st.chat_message("assistant").markdown(result)
            if sources:
                st.chat_message("assistant").write(sources)

            # Sauvegarde
            save_conversation_data(prompt, result, sources, response_time_ms)

            # Widget de feedback
            if show_feedback and features:
                features.render_feedback_widget(f"resp_{len(st.session_state.messages)}")

            # Ajout à l'historique
            st.session_state.messages.append({"role": "assistant", "content": response_content})

    # Interface de saisie
    if prompt := st.chat_input("Posez votre question ici..."):
        # Ajout de la question
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Traitement de la question
        with st.spinner("Analyse en cours..."):
            result, sources, response_time_ms = process_question(model, features, prompt)

            # Affichage de la réponse
            response_content = result
            if sources:
                response_content += "\n\n" + sources

            st.chat_message("assistant").markdown(result)
            if sources:
                st.chat_message("assistant").write(sources)

            # Sauvegarde
            save_conversation_data(prompt, result, sources, response_time_ms)

            # Widget de feedback
            if show_feedback and features:
                features.render_feedback_widget(f"resp_{len(st.session_state.messages)}")

            # Ajout à l'historique
            st.session_state.messages.append({"role": "assistant", "content": response_content})


@login_required
def history_page():
    """Page d'historique des conversations"""
    try:
        if get_history_manager:
            history_manager = get_history_manager()
            user_id = st.session_state.get("user", {}).get("email", "anonymous")

            # Utiliser la méthode render_history_page du gestionnaire
            history_manager.render_history_page(user_id)
        else:
            st.title("📚 Historique des conversations")
            st.warning("Système d'historique non disponible")
    except Exception as e:
        st.title("📚 Historique des conversations")
        st.error(f"Erreur lors du chargement de l'historique: {e}")
        logger.error(f"Erreur historique: {e}")


@admin_required
def admin_page():
    """Page d'administration"""
    st.title("🔧 Administration")

    if auth_manager:
        st.subheader("Gestion des utilisateurs")

        # Affichage des utilisateurs (simulation)
        st.info("Fonctionnalité de gestion des utilisateurs à implémenter")
    else:
        st.warning("Système d'authentification non disponible")


# === INTERFACE PRINCIPALE ===
def render_sidebar():
    """Rendu de la barre latérale"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/confluence--v2.png", width=100)
        st.title("ISSCHAT")

        # Informations utilisateur
        if "user" in st.session_state:
            st.success(f"Connecté: {st.session_state['user']['email']}")

        # Navigation
        st.subheader("Navigation")
        if st.button("Chat", key="nav_chat"):
            st.session_state["page"] = "chat"
            st.rerun()

        if st.button("Historique", key="nav_history"):
            st.session_state["page"] = "history"
            st.rerun()

        # Options admin
        if st.session_state.get("user", {}).get("is_admin"):
            st.divider()
            st.info("Statut: Administrateur")
            if st.button("Tableau de bord admin", key="nav_admin"):
                st.session_state["page"] = "admin"
                st.rerun()

            # Gestion de la base de données
            st.divider()
            st.subheader("Gestion de la base")

            if st.button("Reconstruire depuis Confluence", type="primary"):
                with st.spinner("Reconstruction en cours..."):
                    config = get_config()

                    try:
                        if os.path.exists(config.persist_directory):
                            shutil.rmtree(config.persist_directory)
                            st.info(f"Répertoire {config.persist_directory} supprimé.")
                        os.makedirs(config.persist_directory, exist_ok=True)
                    except Exception as e:
                        st.error(f"Erreur lors de la suppression: {str(e)}")

                    try:
                        st.cache_resource.clear()
                        get_model(rebuild_db=True)
                        st.success("Base reconstruite avec succès!")
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur lors de la reconstruction: {str(e)}")

        # Bouton de fermeture
        st.divider()
        if st.button("Fermer l'application", key="nav_close"):
            st.warning("Arrêt de l'application...")
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGKILL)


def main():
    """Fonction principale de l'application"""
    # Configuration de la session utilisateur
    setup_user_session()

    # Rendu de la barre latérale
    render_sidebar()

    # Routage des pages
    page = st.session_state.get("page", "chat")

    if page == "admin" and st.session_state.get("user", {}).get("is_admin"):
        admin_page()
    elif page == "history":
        history_page()
    else:
        chat_page()


# === POINT D'ENTRÉE ===
if __name__ == "__main__":
    main()
