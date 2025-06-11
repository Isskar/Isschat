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
import numpy as np
from datetime import datetime, timedelta

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
from src.webapp.components.auth_manager import AuthManager
from src.webapp.components.history_manager import get_history_manager

auth_manager = AuthManager()
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


# Cache functions
@st.cache_resource
def get_cached_features_manager(user_id: str):
    """Get cached features manager"""
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

    # Initialize the features manager with caching
    if "features_manager" not in st.session_state:
        user_id = st.session_state.get("user_id", f"user_{st.session_state['user']['id']}")
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
        ]

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
                        key_suffix=f"history_{i}",
                    )

    # Check if there's a question to reuse from history
    if "reuse_question" in st.session_state:
        prompt = st.session_state.pop("reuse_question")
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Process the question with all features
        with st.spinner("Analysis in progress..."):
            result, sources = process_question_with_model(model, features, prompt)

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
                }
            )

            # Force rerun to display feedback widget in history
            st.rerun()

    # Chat interface
    if prompt := st.chat_input("Ask your question here..."):
        # Add the question
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Process the question with all features
        with st.spinner("Analysis in progress..."):
            result, sources = process_question_with_model(model, features, prompt)

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
                }
            )

            # Force rerun to display feedback widget in history
            st.rerun()


def process_question_with_model(model, features, prompt):
    """Process question with model and features"""
    try:
        start_time = time.time()

        # Process with model directly
        if hasattr(model, "process_query"):
            result, sources = model.process_query(prompt)
        elif hasattr(model, "query"):
            result = model.query(prompt)
            sources = ""
        else:
            result = "Model unavailable"
            sources = ""

        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # in milliseconds

        # Process with features manager if available
        if features and result != "Model unavailable":
            features.process_query_response(prompt, result, response_time)

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
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_cached_performance_data():
    """Get cached performance data"""
    return {"response_time": "1.2s", "queries_today": "247", "success_rate": "98.5%"}


@login_required
def dashboard_page():
    """Performance dashboard page - inspired by src/dashboard.py"""
    st.session_state["page"] = "dashboard"

    st.title("Performance Dashboard")

    try:
        # Create tabs for different dashboard sections
        tabs = st.tabs(["Performance Tracking", "Conversation Analysis", "General Statistics"])

        # Tab 1: Performance Tracking
        with tabs[0]:
            render_performance_tracking()

        # Tab 2: Conversation Analysis
        with tabs[1]:
            render_conversation_analysis()

        # Tab 3: General Statistics
        with tabs[2]:
            render_general_statistics()

    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")

    # Return button
    if st.button("Return to chat", key="return_from_dashboard"):
        st.session_state["page"] = "chat"
        st.rerun()


def render_performance_tracking():
    """Render performance tracking section"""
    # Period selection
    days = st.slider("Analysis period (days)", 1, 30, 7, key="performance_days")

    try:
        # Get performance data
        perf_data = get_cached_performance_data()

        if perf_data:
            # Display main metrics
            st.metric("Average response time", f"{perf_data.get('avg_response_time', 'N/A')} ms")

            # Create sample data for visualization

            # Generate sample daily data
            dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
            response_times = np.random.normal(1200, 200, days)  # Sample data
            query_counts = np.random.poisson(50, days)  # Sample data

            daily_data = pd.DataFrame({"date": dates, "response_time_ms": response_times, "query_count": query_counts})

            # Time evolution chart
            st.subheader("Response Time Evolution")
            st.line_chart(daily_data.set_index("date")[["response_time_ms"]])

            # Query count chart
            st.subheader("Daily Query Count")
            st.bar_chart(daily_data.set_index("date")[["query_count"]])

        else:
            st.warning("No performance data available for the selected period")

    except Exception as e:
        st.error(f"Error displaying performance metrics: {str(e)}")


def render_conversation_analysis():
    """Render conversation analysis section"""
    try:
        # Get conversation data from features manager
        if "features_manager" in st.session_state and st.session_state["features_manager"]:
            # Display basic metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total conversations", "156")  # Sample data
            with col2:
                st.metric("Average response time", "1.2s")  # Sample data

            # Sample hourly distribution
            st.subheader("Question distribution by hour")

            hours = list(range(24))
            counts = np.random.poisson(5, 24)  # Sample data
            hour_df = pd.DataFrame({"Hour": hours, "Questions": counts})
            st.bar_chart(hour_df.set_index("Hour"))

            # Most frequent keywords (sample)
            st.subheader("Most frequent keywords")
            keywords_data = [["confluence", 15], ["documentation", 12], ["access", 8], ["login", 6], ["help", 5]]
            keywords_df = pd.DataFrame(keywords_data, columns=["Keyword", "Frequency"])
            st.dataframe(keywords_df)

        else:
            st.warning("Conversation analysis not available")

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
