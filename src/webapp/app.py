import streamlit as st
import time
import signal
import os
import sys
import asyncio
from pathlib import Path
import shutil
import traceback
import pandas as pd
from datetime import datetime, timedelta
import uuid
from typing import Optional

# Add the parent directory to the Python search path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.config import get_config

# Set tokenizers parallelism to false to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix asyncio event loop issues with Streamlit
try:
    # Create and set a new event loop if needed
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop, create a new one
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

    # Set the default policy
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
except Exception as e:
    print(f"Note: asyncio configuration: {str(e)}")
    pass  # Continue even if there's an issue with the event loop

# Import custom modules
from src.rag_system.rag_pipeline import RAGPipelineFactory
from src.webapp.components.features_manager import FeaturesManager
from src.webapp.components.history_manager import get_history_manager

# Streamlit page configuration - must be the first Streamlit command
st.set_page_config(page_title="Isschat", page_icon="🤖", layout="wide")


# Cache the model loading
@st.cache_resource
def get_model(rebuild_db=False):
    # Display a spinner during loading
    with st.spinner("Loading RAG model..."):
        # Check if the index.faiss file exists
        from src.core.config import get_debug_info

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
                st.info("First Launch Detected - Creating Vector DB...")
                rebuild_db = True

        # Create the model
        try:
            pipeline = RAGPipelineFactory.create_default_pipeline()

            # 🔧 FIX: Force vector DB construction if needed
            if rebuild_db:
                st.info("🚀 Building vector database from Confluence...")
                # Force initialization of the retriever with rebuild
                if hasattr(pipeline.retriever, "_initialize_db"):
                    pipeline.retriever._initialize_db(force_rebuild=True)
                else:
                    # Fallback: trigger retrieval to force DB construction
                    try:
                        # Use invoke() method which is the correct LangChain retriever API
                        pipeline.retriever.invoke("test initialization")
                    except Exception as init_error:
                        st.error(f"Failed to initialize vector database: {init_error}")
                        raise
                st.success("✅ Vector database successfully built!")

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


# User interface initialization
def main():
    if "user" not in st.session_state:
        # Create or retrieve admin user immediately
        config = get_config()
        email = config.confluence_email_address or "admin@auto.login"

        # Auto-create user session (simplified approach like in reference file)
        st.session_state["user"] = {"email": email, "id": 1, "is_admin": True}
        st.session_state["user_id"] = "user_1"
        st.session_state["page"] = "chat"

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

        # New Chat button should be placed here, under 'Chat' button
        if st.session_state.get("page") == "chat" and st.button("New Chat", key="new_chat_button_sidebar"):
            st.session_state["messages"] = []  # Clear messages
            st.session_state["current_conversation_id"] = str(uuid.uuid4())  # Generate new conversation ID
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

            if st.button("Performance Dashboard", key="nav_dashboard"):
                st.session_state["page"] = "dashboard"
                st.rerun()

            # Option to rebuild the database
            st.divider()
            st.subheader("Database Management")

            # Add button to force complete reconstruction
            if st.button("Rebuild from Confluence", type="primary"):
                with st.spinner("Rebuilding database from Confluence..."):
                    try:
                        st.cache_resource.clear()

                        # Get the current pipeline to access vector store
                        pipeline = get_model(rebuild_db=False)  # Don't auto-rebuild, we'll do it manually

                        # Use the new rebuild method with proper validation
                        if hasattr(pipeline.vector_store, "rebuild_database"):
                            success = pipeline.vector_store.rebuild_database()
                            if success:
                                st.success("✅ Database successfully rebuilt from Confluence!")
                                st.cache_resource.clear()  # Clear cache after successful rebuild
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error("❌ Database rebuild failed. Check logs for details.")
                        else:
                            # Fallback to old method
                            config = get_config()
                            try:
                                if os.path.exists(config.persist_directory):
                                    shutil.rmtree(config.persist_directory)
                                    st.info(f"Directory {config.persist_directory} successfully deleted.")
                                os.makedirs(config.persist_directory, exist_ok=True)
                            except Exception as e:
                                st.error(f"Error deleting directory: {str(e)}")

                            get_model(rebuild_db=True)
                            st.success("✅ Database successfully rebuilt from Confluence!")
                            time.sleep(2)
                            st.rerun()

                    except Exception as e:
                        from src.core.exceptions import StorageAccessError, RebuildError

                        if isinstance(e, StorageAccessError):
                            st.error(f"🚫 **Erreur d'accès au stockage:**\n\n{str(e)}")
                        elif isinstance(e, RebuildError):
                            st.error(f"🚫 **Erreur de rebuild:**\n\n{str(e)}")
                        else:
                            st.error(f"❌ **Erreur inattendue lors du rebuild:**\n\n{str(e)}")
                            st.code(traceback.format_exc(), language="python")

                        st.info(
                            "💡 **Conseils de dépannage:**\n"
                            "- Vérifiez votre configuration Azure (USE_AZURE_STORAGE, AZURE_STORAGE_ACCOUNT)\n"
                            "- Vérifiez vos permissions Azure Storage\n"
                            "- Consultez les logs pour plus de détails"
                        )

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
    elif st.session_state.get("page") == "dashboard":
        dashboard_page()
    else:
        # Default to chat page
        chat_page()


@login_required
def chat_page():
    # Reset page if necessary
    st.session_state["page"] = "chat"

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

    # Create layout
    st.title("Welcome to ISSCHAT")

    # Sidebar for advanced options
    with st.sidebar:
        st.subheader("Advanced Options")
        show_feedback = st.toggle("Enable feedback", value=True)

        # New Chat button
        if st.button("New Chat", key="new_chat_button"):
            st.session_state["messages"] = []  # Clear messages
            st.session_state["current_conversation_id"] = str(uuid.uuid4())  # Generate new conversation ID
            st.rerun()

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
            st.session_state["messages"] = []  # Ensure messages are cleared for a truly new chat

    # Display main interface
    st.subheader("Ask questions about our Confluence documentation")

    # Extract first name from email (part before @)
    user_email = st.session_state.get("user", {}).get("email", "")
    first_name = user_email.split("@")[0].split(".")[0].capitalize()
    welcome_message = f"Bonjour {first_name} ! Comment puis-je vous aider aujourd'hui ?"

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
        from src.core.data_manager import get_data_manager

        data_manager = get_data_manager()
        # Fetch entries for the current conversation_id
        conversation_entries = data_manager.get_conversation_history(
            conversation_id=conversation_id, limit=max_turns * 2
        )

        # Sort by timestamp to ensure correct order
        conversation_entries.sort(key=lambda x: x.get("timestamp", ""))

        history = []
        for entry in conversation_entries:
            if entry.get("question"):
                history.append(f"User: {entry['question']}")
            if entry.get("answer"):
                history.append(f"Assistant: {entry['answer']}")
        return "\n".join(history)

    # Display message history with feedback widgets
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

                # Add feedback widget for assistant messages (except welcome)
                if i > 0 and show_feedback and features and "question_data" in msg:
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
            st.session_state["messages"] = []  # Clear messages if we are resuming an old conversation
            # Load existing messages for the resumed conversation
            from src.core.data_manager import get_data_manager

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
        st.chat_message("user").write(prompt)

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
            st.chat_message("assistant").markdown(result)
            if sources:
                st.chat_message("assistant").write(sources)

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

    # Chat interface
    if prompt := st.chat_input("Ask your question here..."):
        # Start timing from the moment user submits the question
        start_time = time.time()

        # Add the question
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "conversation_id": st.session_state["current_conversation_id"]}
        )
        st.chat_message("user").write(prompt)

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
            st.chat_message("assistant").markdown(result)
            if sources:
                st.chat_message("assistant").write(sources)

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
        response_time = (time.time() - start_time) * 1000  # in milliseconds

        # Process with features manager if available
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
        from src.core.data_manager import get_data_manager

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
        from src.core.data_manager import get_data_manager
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
            from src.core.data_manager import get_data_manager

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
        from src.core.data_manager import get_data_manager

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
