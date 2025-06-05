import streamlit as st
import time
import signal
import os
import sys
import asyncio
from pathlib import Path
import shutil

# Add the parent directory to the Python search path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config

# Set tokenizers parallelism to false to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix asyncio event loop issues with Streamlit
try:
    # Create and set a new event loop if needed
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Set the default policy
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
except Exception as e:
    print(f"Note: asyncio configuration: {str(e)}")
    pass  # Continue even if there's an issue with the event loop

# Import custom modules
from src.help_desk import HelpDesk
from src.auth import (
    login_required,
    admin_required,
)

# Import new features
from src.features_integration import FeaturesManager  # noqa: E402

# Streamlit page configuration - must be the first Streamlit command
st.set_page_config(page_title="Isschat", page_icon="🤖", layout="wide")


# No cache to force reload on each launch
def get_model(rebuild_db=False):
    # Display a spinner during loading
    with st.spinner("Loading RAG model..."):
        # Check if the index.faiss file exists
        import sys  # noqa: E402
        from config import get_debug_info  # noqa: E402
        from pathlib import Path  # noqa: E402

        # Get debug info
        config = get_config()
        debug_info = get_debug_info()

        st.sidebar.expander("Debug", expanded=False).write(f"""
                **Configuration**:
                - Provider: `{debug_info["provider"]}`
                - Vector store directory: `{debug_info["persist_directory"]}`
                - Confluence URL: `{debug_info["confluence_url"]}`
                - Space key: `{debug_info["space_key"]}`
                - User: `{debug_info["user_email"]}`
                - API key: `{debug_info["confluence_api_key"]}`
                - OpenRouter key: `{debug_info["openrouter_api_key"]}`
                """)

        persist_path = Path(config.persist_directory)
        index_file = persist_path / "index.faiss"
        if not rebuild_db:
            if not persist_path.exists() or not index_file.exists():
                st.info("🚀 First Launch Detected - Creating Vector DB...")
                rebuild_db = True

        # Create the model
        try:
            model = HelpDesk(new_db=rebuild_db)
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            import traceback

            st.code(traceback.format_exc(), language="python")
            sys.exit(1)


# User interface initialization
def main():
    if "user" not in st.session_state:
        # Create or retrieve admin user immediately
        config = get_config()
        email = config.confluence_email_address or "admin@auto.login"

        from src.auth import get_all_users, add_user

        # Check if user exists, create if needed
        users = get_all_users()
        user = next((u for u in users if u["email"] == email), None)

        if not user:
            add_user(email, "auto_generated_pwd", is_admin=True)
            users = get_all_users()
            user = next((u for u in users if u["email"] == email), None)

        if user:
            st.session_state["user"] = user
            st.session_state["user_id"] = f"user_{user['id']}"
            st.session_state["page"] = "chat"
        else:
            # This should never happen but just in case
            st.error("Critical error: Failed to create auto-login user")
            st.stop()

    # Sidebar for navigation and options
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/confluence--v2.png", width=100)
        st.title("ISSCHAT")

        # Always display user info
        st.success(f"Connected as: {st.session_state['user']['email']}")

        # Main navigation
        st.subheader("Navigation")
        if st.button("Chat", key="nav_chat"):
            st.session_state["page"] = "chat"
            st.rerun()

        if st.button("History", key="nav_history"):
            st.session_state["page"] = "history"
            st.rerun()

        # Admin options
        if st.session_state["user"].get("is_admin"):
            st.divider()
            st.info("Status: Administrator")
            if st.button("Admin Dashboard", key="nav_admin"):
                st.session_state["page"] = "admin"
                st.rerun()

            # Option to rebuild the database
            st.divider()
            st.subheader("Database Management")

            # Add button to force complete reconstruction
            if st.button("Rebuild from Confluence", type="primary"):
                with st.spinner("Rebuilding database from Confluence..."):
                    # Delete existing files
                    config = get_config()

                    try:
                        if os.path.exists(config.persist_directory):
                            shutil.rmtree(config.persist_directory)
                            st.info(f"Directory {config.persist_directory} successfully deleted.")
                        os.makedirs(config.persist_directory, exist_ok=True)
                    except Exception as e:
                        st.error(f"Error deleting directory: {str(e)}")

                    # Force model reload with new_db=True
                    try:
                        st.cache_resource.clear()
                        get_model(rebuild_db=True)
                        st.success("Database successfully rebuilt from Confluence!")
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during reconstruction: {str(e)}")
                        import traceback

                        st.code(traceback.format_exc(), language="python")

        # Close Button
        st.divider()
        if st.button("Close App", key="nav_close_app"):
            st.warning("Shutting down the Streamlit app...")
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGKILL)

    # Determine which page to display - user is already authenticated at the beginning of main()
    if st.session_state.get("page") == "admin" and st.session_state["user"].get("is_admin"):
        admin_page()
    elif st.session_state.get("page") == "history":
        history_page()
    else:
        # Default to chat page
        chat_page()


@login_required
def chat_page():
    # Reset page if necessary
    st.session_state["page"] = "chat"

    # Load the model
    model = get_model()

    # Initialize the features manager
    if "features_manager" not in st.session_state:
        user_id = st.session_state.get("user_id", f"user_{st.session_state['user']['id']}")
        st.session_state["features_manager"] = FeaturesManager(model, user_id)

    features = st.session_state["features_manager"]

    # Create layout
    st.title("Welcome to ISSCHAT")

    # Sidebar for advanced options
    with st.sidebar:
        st.subheader("Advanced Options")
        show_feedback = st.toggle("Enable feedback", value=True)

        # Query history
        if st.button("Query History"):
            st.session_state["page"] = "history"
            st.rerun()

    # Display main interface
    st.subheader("Ask questions about our Confluence documentation")

    # Extract first name from email (part before @)
    user_email = st.session_state.get("user", {}).get("email", "")
    first_name = user_email.split("@")[0].split(".")[0].capitalize()
    welcome_message = f"Bonjour {first_name} ! Comment puis-je vous aider aujourd'hui ?"

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": welcome_message,
            }
        ]  # ;)

    # Display message history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Check if there's a question to reuse from history
    if "reuse_question" in st.session_state:
        prompt = st.session_state.pop("reuse_question")
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Process the question with all features
        with st.spinner("Analysis in progress..."):
            result, sources = features.process_question(prompt)

            # Build the response content
            response_content = result
            if sources:
                response_content += "\n\n" + sources

            # Display the response
            st.chat_message("assistant").markdown(result)
            if sources:
                st.chat_message("assistant").write(sources)

            # Add new thumbs up/down feedback widget if enabled
            if show_feedback:
                features.add_feedback_widget(st, prompt, result, sources, key_suffix="reuse")

            # Add to message history
            st.session_state.messages.append({"role": "assistant", "content": response_content})

    # Chat interface
    if prompt := st.chat_input("Ask your question here..."):
        # Add the question
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Process the question with all features
        with st.spinner("Analysis in progress..."):
            result, sources = features.process_question(prompt)

            # Build the response content
            response_content = result
            if sources:
                response_content += "\n\n" + sources

            # Display the response
            st.chat_message("assistant").markdown(result)
            if sources:
                st.chat_message("assistant").write(sources)

            # Add new thumbs up/down feedback widget if enabled
            if show_feedback:
                features.add_feedback_widget(st, prompt, result, sources, key_suffix="chat")

            # Add to message history
            st.session_state.messages.append({"role": "assistant", "content": response_content})


@login_required
def history_page():
    """Display the query history page"""
    # Reset the page if necessary
    st.session_state["page"] = "history"

    # Get the features manager
    if "features_manager" not in st.session_state:
        st.error("Error: Features manager not initialized")
        st.session_state["page"] = "chat"
        st.rerun()

    features = st.session_state["features_manager"]
    user_id = st.session_state.get("user_id", f"user_{st.session_state['user']['id']}")

    # Display query history
    features.query_history.render_history_page(st, user_id)

    # Return button
    if st.button("Return to chat", key="return_from_history"):
        st.session_state["page"] = "chat"
        st.rerun()


@admin_required
def admin_page():
    """Display the administration page"""
    st.title("Administration Dashboard")

    # Check if the features manager is initialized
    if "features_manager" in st.session_state:
        features = st.session_state["features_manager"]
        features.render_admin_dashboard(st)
    else:
        st.warning("Please interact with the chatbot first to initialize the analytics features.")

    # Return button
    if st.button("Return to chat"):
        st.session_state["page"] = "chat"
        st.rerun()


# Launch the application
if __name__ == "__main__":
    main()
