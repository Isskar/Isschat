import streamlit as st
import time
import os
import sys
from pathlib import Path

# Add the parent directory to the Python search path
sys.path.append(str(Path(__file__).parent.parent))

# Import custom modules
from src.help_desk import HelpDesk  # noqa: E402
from src.auth import (  # noqa: E402
    logout,
    get_all_users,
    add_user,
    delete_user,
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
        from config import PERSIST_DIRECTORY  # noqa: E402

        # Display configuration information for debugging
        api_key = os.getenv("CONFLUENCE_PRIVATE_API_KEY")
        key_display = f"*****{api_key[-5:]}" if api_key else "Not defined"

        st.sidebar.expander("Debug", expanded=False).write(f"""
                **Configuration**:
                - Vector store directory: `{PERSIST_DIRECTORY}`
                - Confluence URL: `{os.getenv("CONFLUENCE_SPACE_NAME")}`
                - Space key: `{os.getenv("CONFLUENCE_SPACE_KEY")}`
                - User: `{os.getenv("CONFLUENCE_EMAIL_ADRESS")}`
                - API key: `{key_display}`
                """)

        # Check if directory exists
        if not os.path.exists(PERSIST_DIRECTORY):
            st.warning(f"Directory {PERSIST_DIRECTORY} does not exist. Attempting to create it...")
            try:
                os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
            except Exception as e:
                st.error(f"Error creating directory: {str(e)}")

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
    # Ensure user is always authenticated
    # Even before rendering sidebar, force user auth
    if "user" not in st.session_state:
        # Create or retrieve admin user immediately
        email = os.getenv("CONFLUENCE_EMAIL_ADRESS") or "admin@auto.login"
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
                    import shutil
                    from config import PERSIST_DIRECTORY

                    try:
                        if os.path.exists(PERSIST_DIRECTORY):
                            shutil.rmtree(PERSIST_DIRECTORY)
                            st.info(f"Directory {PERSIST_DIRECTORY} successfully deleted.")
                        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
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

        # Logout button (for all users)
        st.divider()
        if st.button("Logout", key="nav_logout"):
            logout()

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
        show_feedback = st.toggle("Response feedback", value=True)

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

            # Add feedback widget if enabled
            if show_feedback:
                features._add_feedback_widget(st, prompt, result, sources)

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

            # Add feedback widget if enabled
            if show_feedback:
                features._add_feedback_widget(st, prompt, result, sources)

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

    # Create tabs to separate different features
    tab1, tab2 = st.tabs(["User Management", "Analytics and Performance"])

    with tab1:
        st.header("User Management")

        # Form to add a user
        with st.expander("Add User", expanded=True):
            with st.form("add_user_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                is_admin = st.checkbox("Administrator")
                submit = st.form_submit_button("Add")

                if submit and email and password:
                    success = add_user(email, password, is_admin)
                    if success:
                        st.success(f"User {email} added successfully.")
                    else:
                        st.error(f"Email {email} already exists.")

        # User list
        st.subheader("User List")
        users = get_all_users()

        if users:
            # Create a table to display users
            cols = st.columns([3, 2, 1, 1])
            cols[0].write("**Email**")
            cols[1].write("**Creation Date**")
            cols[2].write("**Admin**")
            cols[3].write("**Actions**")

            for user in users:
                cols = st.columns([3, 2, 1, 1])
                cols[0].write(user["email"])
                cols[1].write(user["created_at"])
                cols[2].write("Yes" if user["is_admin"] else "No")

                # Don't allow deleting the currently logged in user
                if user["id"] != st.session_state["user"]["id"]:
                    if cols[3].button("Delete", key=f"delete_{user['id']}"):
                        delete_user(user["id"])
                        st.success(f"User {user['email']} deleted.")
                        st.rerun()
                else:
                    cols[3].write("(You)")
        else:
            st.info("No users found.")

    with tab2:
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
