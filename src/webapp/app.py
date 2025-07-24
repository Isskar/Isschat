import streamlit as st
import time
import signal
import os
import sys
import asyncio
from pathlib import Path
import traceback
import pandas as pd
from datetime import datetime, timedelta
import uuid
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent.parent))


os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    asyncio.get_running_loop()
except RuntimeError:
    pass

from src.rag.semantic_pipeline import SemanticRAGPipelineFactory
from src.webapp.components.features_manager import FeaturesManager
from src.webapp.components.history_manager import get_history_manager
from src.webapp.auth.azure_auth import AzureADAuth
from src.webapp.example_prompts import EXAMPLE_PROMPTS

# images paths
IMAGES = {
    "user": str(Path(__file__).parent.parent.parent / "Images" / "user.svg"),
    "bot": str(Path(__file__).parent.parent.parent / "Images" / "logo.png"),
    "panel": str(Path(__file__).parent.parent.parent / "Images" / "isschat.png"),
}


st.set_page_config(page_title="Isschat", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

# Add custom CSS for sidebar buttons
st.markdown(
    """
    <style>
    /* Sidebar button styling */
    .stSidebar .stButton > button {
        border-radius: 8px !important;
        width: 100% !important;
        text-align: left !important;
        justify-content: flex-start !important;
        display: flex !important;
        align-items: center !important;
        background-color: transparent !important;
        border: none !important;
    }
    .stSidebar .stButton > button:hover {
        background-color: #f0f0f0 !important;
    }
    .stSidebar .stButton > button > div {
        text-align: left !important;
        width: 100% !important;
    }
    div[data-testid="stChatMessage"] div img,
  .stChatMessage img {
      width: 40px !important;
      height: 40px !important;
      min-width: 40px !important;
      min-height: 40px !important;
      border-radius: 50% !important;
  }
    </style>
""",
    unsafe_allow_html=True,
)


# Initialize embedder at startup
@st.cache_resource
def initialize_embedder():
    """Initialize the embedding service at startup"""
    from src.embeddings.service import get_embedding_service

    try:
        embedder = get_embedding_service()
        # Force model loading by accessing the model property
        _ = embedder.model
        return embedder
    except Exception as e:
        st.error(f"Failed to initialize embedding service: {str(e)}")
        raise


# Cache the model loading
@st.cache_resource
def get_model(rebuild_db=False):
    # Display a spinner during loading
    with st.spinner("Loading RAG model..."):
        # Check if the index.faiss file exists
        from src.config.settings import get_debug_info

        # Get debug info
        debug_info = get_debug_info()

        st.sidebar.expander("Debug", expanded=False).write(f"""
                **Configuration**:
                debug_info: {debug_info}""")
        try:
            # Use semantic pipeline for enhanced understanding
            pipeline = SemanticRAGPipelineFactory.create_semantic_pipeline(use_semantic_features=True)
            return pipeline
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.code(traceback.format_exc(), language="python")
            sys.exit(1)


# Cache functions - removed caching to ensure fresh data
def get_cached_features_manager(user_id: str):
    """Get features manager (no longer cached to ensure fresh data)"""
    if FeaturesManager:
        return FeaturesManager(user_id)
    return None


# Authentication decorators
def login_required(func):
    """Decorator for pages requiring authentication"""

    def wrapper(*args, **kwargs):
        if "user" not in st.session_state:
            st.error("Authentication required")
            return
        return func(*args, **kwargs)

    return wrapper


def admin_required(func):
    """Decorator for admin pages"""

    def wrapper(*args, **kwargs):
        if not st.session_state.get("user", {}).get("is_admin"):
            st.error("Administrator access required")
            return
        return func(*args, **kwargs)

    return wrapper


# Initialize Azure AD authentication globally
@st.cache_resource
def get_auth():
    """Get Azure AD authentication instance"""
    try:
        return AzureADAuth()
    except ValueError as e:
        st.error(f"Authentication configuration error: {str(e)}")
        st.stop()


# User interface initialization
def main():
    # Get authentication instance
    auth = get_auth()

    # Check if user is authenticated
    if not auth.is_authenticated():
        auth.show_login_page()
        return

    # Initialize page if not set
    if "page" not in st.session_state:
        st.session_state["page"] = "chat"

    # Sidebar for navigation and options
    with st.sidebar:
        st.image(IMAGES["panel"])

        # Main navigation
        st.subheader("Navigation")
        if st.button("![Chat](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJmZWF0aGVyIGZlYXRoZXItbWVzc2FnZS1zcXVhcmUiPjxwYXRoIGQ9Ik0yMSAxNWEyIDIgMCAwIDEtMiAySDdsLTQgNFY1YTIgMiAwIDAgMSAyLTJoMTRhMiAyIDAgMCAxIDIgMnoiPjwvcGF0aD48L3N2Zz4=) Chat", key="nav_chat"):
            st.session_state["page"] = "chat"
            st.rerun()

        if st.button("![New Chat](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJmZWF0aGVyIGZlYXRoZXItcGx1cyI+PGxpbmUgeDE9IjEyIiB5MT0iNSIgeDI9IjEyIiB5Mj0iMTkiPjwvbGluZT48bGluZSB4MT0iNSIgeTE9IjEyIiB4Mj0iMTkiIHkyPSIxMiI+PC9saW5lPjwvc3ZnPg==) New Chat", key="new_chat_button_sidebar"):
            st.session_state["messages"] = []
            st.session_state["current_conversation_id"] = str(uuid.uuid4())
            st.session_state["page"] = "chat"  # Switch to chat page
            st.rerun()

        if st.button("![History](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJmZWF0aGVyIGZlYXRoZXItY2xvY2siPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIj48L2NpcmNsZT48cG9seWxpbmUgcG9pbnRzPSIxMiA2IDEyIDEyIDE2IDE0Ij48L3BvbHlsaW5lPjwvc3ZnPg==) History", key="nav_history"):
            st.session_state["page"] = "history"
            st.rerun()

        # Admin options
        if st.session_state["user"].get("is_admin"):
            st.divider()
            if st.button("![Admin Dashboard](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJmZWF0aGVyIGZlYXRoZXItc2V0dGluZ3MiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjMiPjwvY2lyY2xlPjxwYXRoIGQ9Ik0xOS40IDE1YTEuNjUgMS42NSAwIDAgMCAuMzMgMS44MmwuMDYuMDZhMiAyIDAgMCAxIDAgMi44MyAyIDIgMCAwIDEtMi44MyAwbC0uMDYtLjA2YTEuNjUgMS42NSAwIDAgMC0xLjgyLS4zMyAxLjY1IDEuNjUgMCAwIDAtMSAxLjUxVjIxYTIgMiAwIDAgMS0yIDIgMiAyIDAgMCAxLTItMnYtLjA5QTEuNjUgMS42NSAwIDAgMCA5IDE5LjRhMS42NSAxLjY1IDAgMCAwLTEuODIuMzNsLS4wNi4wNmEyIDIgMCAwIDEtMi44MyAwIDIgMiAwIDAgMSAwLTIuODNsLjA2LS4wNmExLjY1IDEuNjUgMCAwIDAgLjMzLTEuODIgMS42NSAxLjY1IDAgMCAwLTEuNTEtMUgzYTIgMiAwIDAgMS0yLTIgMiAyIDAgMCAxIDItMmguMDlBMS42NSAxLjY1IDAgMCAwIDQuNiA5YTEuNjUgMS42NSAwIDAgMC0uMzMtMS44MmwtLjA2LS4wNmEyIDIgMCAwIDEgMC0yLjgzIDIgMiAwIDAgMSAyLjgzIDBsLjA2LjA2YTEuNjUgMS42NSAwIDAgMCAxLjgyLjMzSDlhMS42NSAxLjY1IDAgMCAwIDEtMS41MVYzYTIgMiAwIDAgMSAyLTIgMiAyIDAgMCAxIDIgMnYuMDlhMS42NSAxLjY1IDAgMCAwIDEgMS41MSAxLjY1IDEuNjUgMCAwIDAgMS44Mi0uMzNsLjA2LS4wNmEyIDIgMCAwIDEgMi44MyAwIDIgMiAwIDAgMSAwIDIuODNsLS4wNi4wNmExLjY1IDEuNjUgMCAwIDAtLjMzIDEuODJWOWExLjY1IDEuNjUgMCAwIDAgMS41MSAxSDIxYTIgMiAwIDAgMSAyIDIgMiAyIDAgMCAxLTIgMmgtLjA5YTEuNjUgMS42NSAwIDAgMC0xLjUxIDF6Ij48L3BhdGg+PC9zdmc+) Admin Dashboard", key="nav_admin"):
                st.session_state["page"] = "admin"
                st.rerun()

            if st.button("![Performance Dashboard](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJmZWF0aGVyIGZlYXRoZXItYmFyLWNoYXJ0LTIiPjxsaW5lIHgxPSIxOCIgeTE9IjIwIiB4Mj0iMTgiIHkyPSIxMCI+PC9saW5lPjxsaW5lIHgxPSIxMiIgeTE9IjIwIiB4Mj0iMTIiIHkyPSI0Ij48L2xpbmU+PGxpbmUgeDE9IjYiIHkxPSIyMCIgeDI9IjYiIHkyPSIxNCI+PC9saW5lPjwvc3ZnPg==) Performance Dashboard", key="nav_dashboard"):
                st.session_state["page"] = "dashboard"
                st.rerun()

        # Logout Button
        st.divider()
        if st.button("![Logout](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJmZWF0aGVyIGZlYXRoZXItbG9nLW91dCI+PHBhdGggZD0iTTkgMjFINWEyIDIgMCAwIDEtMi0yVjVhMiAyIDAgMCAxIDItMmg0Ij48L3BhdGg+PHBvbHlsaW5lIHBvaW50cz0iMTYgMTcgMjEgMTIgMTYgNyI+PC9wb2x5bGluZT48bGluZSB4MT0iMjEiIHkxPSIxMiIgeDI9IjkiIHkyPSIxMiI+PC9saW5lPjwvc3ZnPg==) Logout", key="nav_logout", type="primary", use_container_width=True):
            auth.logout()
            st.rerun()

        # Close Button
        if st.button("![Close App](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJmZWF0aGVyIGZlYXRoZXIteC1jaXJjbGUiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIj48L2NpcmNsZT48bGluZSB4MT0iMTUiIHkxPSI5IiB4Mj0iOSIgeTI9IjE1Ij48L2xpbmU+PGxpbmUgeDE9IjkiIHkxPSI5IiB4Mj0iMTUiIHkyPSIxNSI+PC9saW5lPjwvc3ZnPg==) Close App", key="nav_close_app"):
            st.warning("Shutting down the Streamlit app...")
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGKILL)

        # User info at the bottom
        user_info = st.session_state.get("user", {})
        if user_info:
            st.markdown("---")
            st.success(f"Connected as : {user_info['email']}")

    # Determine which page to display - user is already authenticated at the beginning of main()
    if st.session_state.get("page") == "admin" and st.session_state["user"].get("is_admin"):
        admin_page()
    elif st.session_state.get("page") == "history":
        history_page()
    elif st.session_state.get("page") == "dashboard":
        dashboard_page()
    else:
        # Default to chat page
        chat_page()


@login_required
def chat_page():
    # Reset page if necessary
    st.session_state["page"] = "chat"

    # Initialize embedder first
    try:
        initialize_embedder()
    except Exception:
        st.error("Failed to initialize embedding service. Please check your configuration.")
        return

    # Load the model
    model = get_model()

    # Initialize the features manager (no caching to ensure fresh data)
    # Use email as the primary user_id everywhere for consistency
    user_id = st.session_state.get("user", {}).get("email", "anonymous")
    if FeaturesManager:
        st.session_state["features_manager"] = get_cached_features_manager(user_id)
    else:
        st.session_state["features_manager"] = None

    features = st.session_state["features_manager"]

    # Initialize current_conversation_id if not present or messages are empty (new chat)
    if "current_conversation_id" not in st.session_state or not st.session_state.get("messages"):
        if (
            st.session_state.get("messages")
            and len(st.session_state["messages"]) == 1
            and st.session_state["messages"][0]["role"] == "assistant"
        ):
            # If only welcome message is present, it's effectively a new chat
            pass  # Do not generate a new ID if it's just the welcome message
        else:
            st.session_state["current_conversation_id"] = str(uuid.uuid4())
            st.session_state["messages"] = []

    # Display main interface
    st.subheader("Welcome to Isschat, ask questions about our Confluence documentation !")

    # Extract first name from email (part before @)
    user_email = st.session_state.get("user", {}).get("email", "")
    first_name = user_email.split("@")[0].split(".")[0].capitalize()
    welcome_message = f"Hello {first_name}! How can I help you today?"

    # Initialize message history with welcome message if empty
    if not st.session_state["messages"]:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": welcome_message,
                "conversation_id": st.session_state["current_conversation_id"],
            }
        ]

    # Helper to format chat history for prompt
    def format_chat_history(conversation_id: str, max_turns=10):
        from src.storage.data_manager import get_data_manager

        data_manager = get_data_manager()
        # Fetch entries for the current conversation_id
        conversation_entries = data_manager.get_conversation_history(
            conversation_id=conversation_id, limit=max_turns * 2
        )

        conversation_entries.sort(key=lambda x: x.get("timestamp", ""))

        history = []
        for entry in conversation_entries:
            if entry.get("question") and entry.get("answer"):
                history.append(f"User: {entry['question']}")
                history.append(f"Assistant: {entry['answer']}")
        return "\n".join(history)

    # Display message history with feedback widgets
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.chat_message("user", avatar=IMAGES["user"]).write(msg["content"])
        else:
            with st.chat_message("assistant", avatar=IMAGES["bot"]):
                st.write(msg["content"])

                # Add feedback widget for assistant messages (except welcome)
                if i > 0 and features and "question_data" in msg:
                    features.add_feedback_widget(
                        st,
                        msg["question_data"]["question"],
                        msg["question_data"]["answer"],
                        msg["question_data"]["sources"],
                        conversation_id=msg.get("conversation_id"),  # Pass conversation_id
                        key_suffix=f"history_{i}",
                    )

    # Check if there's a question to reuse from history
    if "reuse_question" in st.session_state:
        # Start timing from the moment question is reused
        start_time = time.time()

        prompt = st.session_state.pop("reuse_question")
        # The conversation_id will be set by the history page when reusing a conversation
        # If it's a new session, ensure current_conversation_id is set
        if "current_conversation_id" not in st.session_state or st.session_state.get("reuse_conversation_id"):
            st.session_state["current_conversation_id"] = st.session_state.pop(
                "reuse_conversation_id", str(uuid.uuid4())
            )
            st.session_state["messages"] = []
            from src.storage.data_manager import get_data_manager

            data_manager = get_data_manager()
            existing_messages = data_manager.get_conversation_history(
                conversation_id=st.session_state["current_conversation_id"]
            )
            for entry in existing_messages:
                st.session_state.messages.append(
                    {"role": "user", "content": entry["question"], "conversation_id": entry["conversation_id"]}
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": entry["answer"],
                        "question_data": {
                            "question": entry["question"],
                            "answer": entry["answer"],
                            "sources": entry.get("sources"),
                        },
                        "conversation_id": entry["conversation_id"],
                    }
                )

        st.session_state.messages.append(
            {"role": "user", "content": prompt, "conversation_id": st.session_state["current_conversation_id"]}
        )
        st.chat_message("user", avatar=IMAGES["user"]).write(prompt)

        # Prepare chat history for context from the data manager
        chat_history = format_chat_history(st.session_state["current_conversation_id"])

        # Process the question with all features
        with st.spinner("Analysis in progress..."):
            result, sources = process_question_with_model(
                model, features, prompt, chat_history, st.session_state["current_conversation_id"], start_time
            )

            # Build the response content
            response_content = result
            if sources:
                response_content += "\n\n" + sources

            # Display the response
            st.chat_message("assistant", avatar=IMAGES["bot"]).markdown(result)
            if sources:
                st.chat_message("assistant", avatar=IMAGES["bot"]).write(sources)

            # Add to message history with question data for feedback
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_content,
                    "question_data": {"question": prompt, "answer": result, "sources": sources},
                    "conversation_id": st.session_state["current_conversation_id"],
                }
            )

            # Force rerun to display feedback widget in history
            st.rerun()

    # Display example prompts only on a new chat
    if len(st.session_state.messages) <= 1:
        with st.expander("Not sure where to start? Try these examples..."):
            # Function to handle prompt click
            def handle_prompt_click(prompt_text):
                st.session_state.clicked_prompt = prompt_text

            for example in EXAMPLE_PROMPTS:
                if st.button(
                    f"**{example['title']}** - {example['prompt']}",
                    key=f"example_{example['title']}",
                    on_click=handle_prompt_click,
                    args=[example["prompt"]],
                    use_container_width=True,
                ):
                    pass

    # Handle chat input, prioritizing clicked examples
    if "clicked_prompt" in st.session_state and st.session_state.clicked_prompt:
        prompt = st.session_state.pop("clicked_prompt")
    else:
        prompt = st.chat_input("Ask your question here...")

    if prompt:
        # Start timing from the moment user submits the question
        start_time = time.time()

        # Add the question
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "conversation_id": st.session_state["current_conversation_id"]}
        )
        st.chat_message("user", avatar=IMAGES["user"]).write(prompt)

        # Prepare chat history for context from the data manager
        chat_history = format_chat_history(st.session_state["current_conversation_id"])

        # Process the question with all features
        with st.spinner("Analysis in progress..."):
            result, sources = process_question_with_model(
                model, features, prompt, chat_history, st.session_state["current_conversation_id"], start_time
            )

            # Build the response content
            response_content = result
            if sources:
                response_content += "\n\n" + sources

            # Display the response
            st.chat_message("assistant", avatar=IMAGES["bot"]).markdown(result)
            if sources:
                st.chat_message("assistant", avatar=IMAGES["bot"]).write(sources)

            # Add to message history with question data for feedback
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_content,
                    "question_data": {"question": prompt, "answer": result, "sources": sources},
                    "conversation_id": st.session_state["current_conversation_id"],
                }
            )

            # Force rerun to display feedback widget in history
            st.rerun()


def process_question_with_model(
    model,
    features,
    prompt,
    chat_history=None,
    conversation_id: Optional[str] = None,
    start_time=None,
):
    """Process question with model and features"""
    try:
        # Use provided start_time or current time if not provided (for backward compatibility)
        if start_time is None:
            start_time = time.time()

        # Process with model directly
        if hasattr(model, "process_query"):
            if chat_history is not None:
                result, sources = model.process_query(prompt, history=chat_history)
            else:
                result, sources = model.process_query(prompt)
        elif hasattr(model, "query"):
            result = model.query(prompt)
            sources = ""
        else:
            result = "Model unavailable"
            sources = ""

        # Calculate total response time from user input to completion
        response_time = (time.time() - start_time) * 1000
        if features and result != "Model unavailable":
            features.process_query_response(prompt, result, response_time, conversation_id)
        return result, sources
    except Exception as e:
        return f"Error processing question: {str(e)}", ""


@login_required
def history_page():
    """Display the query history page"""
    # Reset the page if necessary
    st.session_state["page"] = "history"

    try:
        if get_history_manager:
            history_manager = get_history_manager()
            user_id = st.session_state.get("user", {}).get("email", "anonymous")

            # Render history page
            history_manager.render_history_page(user_id)
        else:
            st.info("History functionality not available")
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")

    # Return button
    if st.button("Return to chat", key="return_from_history"):
        st.session_state["page"] = "chat"
        st.rerun()


@login_required
def get_real_performance_data():
    """Get real performance data from data manager"""
    try:
        from src.storage.data_manager import get_data_manager

        data_manager = get_data_manager()

        # Get recent performance data
        performance_data = data_manager.get_performance_metrics(limit=100)
        conversations = data_manager.get_conversation_history(limit=100)

        if not performance_data and not conversations:
            return None

        # Calculate real metrics
        total_conversations = len(conversations)
        avg_response_time = 0
        if conversations:
            avg_response_time = sum(c.get("response_time_ms", 0) for c in conversations) / len(conversations)

        return {
            "avg_response_time": avg_response_time,
            "total_conversations": total_conversations,
            "conversations_today": len(
                [c for c in conversations if c.get("timestamp", "").startswith(datetime.now().strftime("%Y-%m-%d"))]
            ),
        }
    except Exception as e:
        print(f"Error getting performance data: {e}")
        return None


@login_required
def dashboard_page():
    """Performance dashboard page using the new PerformanceDashboard component"""
    st.session_state["page"] = "dashboard"

    try:
        # Use the new PerformanceDashboard component
        from src.storage.data_manager import get_data_manager
        from src.webapp.components.performance_dashboard import render_performance_dashboard

        data_manager = get_data_manager()
        render_performance_dashboard(data_manager)

    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")
        st.code(str(e))

    # Return button
    if st.button("Return to chat", key="return_from_dashboard"):
        st.session_state["page"] = "chat"
        st.rerun()


def render_performance_tracking():
    """Render performance tracking section"""
    # Period selection
    days = st.slider("Analysis period (days)", 1, 30, 7, key="performance_days")

    try:
        # Get real performance data
        perf_data = get_real_performance_data()

        if perf_data:
            # Display main metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Response Time", f"{perf_data.get('avg_response_time', 0):.0f} ms")
            with col2:
                st.metric("Total Conversations", perf_data.get("total_conversations", 0))
            with col3:
                st.metric("Conversations Today", perf_data.get("conversations_today", 0))

            # Get conversation data for charts
            from src.storage.data_manager import get_data_manager

            data_manager = get_data_manager()
            conversations = data_manager.get_conversation_history(limit=200)

            if conversations:
                # Convert to DataFrame for analysis
                df = pd.DataFrame(conversations)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["date"] = df["timestamp"].dt.date

                # Filter by selected period
                cutoff_date = datetime.now() - timedelta(days=days)
                recent_df = df[df["timestamp"] >= cutoff_date]

                if not recent_df.empty:
                    # Daily conversation count
                    daily_counts = recent_df.groupby("date").size().reset_index(name="count")
                    st.subheader("Daily Conversation Volume")
                    st.bar_chart(daily_counts.set_index("date")["count"])

                    # Response time evolution
                    if "response_time_ms" in recent_df.columns:
                        daily_response_times = recent_df.groupby("date")["response_time_ms"].mean().reset_index()
                        st.subheader("Average Response Time Evolution")
                        st.line_chart(daily_response_times.set_index("date")["response_time_ms"])
                else:
                    st.info(f"No conversations found in the last {days} days")
            else:
                st.info("No conversation data available")

        else:
            st.warning("No performance data available for the selected period")

    except Exception as e:
        st.error(f"Error displaying performance metrics: {str(e)}")


def render_conversation_analysis():
    """Render conversation analysis section"""
    try:
        # Get real conversation data
        from src.storage.data_manager import get_data_manager

        data_manager = get_data_manager()
        conversations = data_manager.get_conversation_history(limit=200)

        if conversations:
            # Display basic metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Conversations", len(conversations))
            with col2:
                avg_response_time = sum(c.get("response_time_ms", 0) for c in conversations) / len(conversations)
                st.metric("Average Response Time", f"{avg_response_time:.0f}ms")
            with col3:
                avg_length = sum(c.get("answer_length", 0) for c in conversations) / len(conversations)
                st.metric("Average Answer Length", f"{avg_length:.0f} chars")

            # Convert to DataFrame for analysis
            df = pd.DataFrame(conversations)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour

            # Hourly distribution
            st.subheader("Question Distribution by Hour")
            hourly_counts = df.groupby("hour").size().reset_index(name="count")

            # Fill missing hours with 0
            all_hours = pd.DataFrame({"hour": range(24)})
            hourly_counts = all_hours.merge(hourly_counts, on="hour", how="left").fillna(0)

            st.bar_chart(hourly_counts.set_index("hour")["count"])

            # User activity if multiple users
            unique_users = df["user_id"].nunique()
            if unique_users > 1:
                st.subheader("User Activity")
                user_counts = df["user_id"].value_counts().head(10)
                st.bar_chart(user_counts)
        else:
            st.info("No conversation data available")

    except Exception as e:
        st.error(f"Error displaying conversation analysis: {str(e)}")


def render_general_statistics():
    """Render general statistics section"""
    try:
        # Get statistics from various sources
        if "features_manager" in st.session_state and st.session_state["features_manager"]:
            features = st.session_state["features_manager"]

            # Get feedback statistics
            feedback_stats = features.get_feedback_statistics(days=30)

            # Display statistics in columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total conversations", feedback_stats.get("total_feedback", 0))

            with col2:
                st.metric("Average response time", "1200 ms")  # Sample data

            with col3:
                st.metric("Satisfaction rate", f"{feedback_stats.get('satisfaction_rate', 0):.1f}%")

            # Additional metrics
            col4, col5 = st.columns(2)

            with col4:
                st.metric("Positive feedback", feedback_stats.get("positive_feedback", 0))

            with col5:
                st.metric("Negative feedback", feedback_stats.get("negative_feedback", 0))

        else:
            st.warning("Statistics not available")

    except Exception as e:
        st.error(f"Error collecting statistics: {str(e)}")


@admin_required
def admin_page():
    """Display the administration page"""
    st.title("Administration Dashboard")

    # Check if the features manager is initialized
    if "features_manager" in st.session_state:
        features = st.session_state["features_manager"]
        if features and hasattr(features, "render_admin_dashboard"):
            features.render_admin_dashboard(st)
        else:
            st.info("Admin dashboard functionality not available")
    else:
        st.warning("Please interact with the chatbot first to initialize the analytics features.")

    # Return button
    if st.button("Return to chat"):
        st.session_state["page"] = "chat"
        st.rerun()


# Launch the application
if __name__ == "__main__":
    main()
