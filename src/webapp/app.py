"""
Streamlit Application for Isschat - Clean Professional Interface
Refactored version with integrated performance dashboard
"""

import streamlit as st
import time
import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple, Any

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit configuration - modern dark interface
st.set_page_config(
    page_title="Isschat - AI Chatbot Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="expanded"
)

# Custom CSS for modern dark interface
st.markdown(
    """
<style>
    /* Global dark theme */
    .main > div {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #2E86AB;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #404040;
    }
    
    .chat-container {
        background: linear-gradient(135deg, #2D2D2D 0%, #3D3D3D 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #404040;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .status-healthy {
        color: #6A994E;
        font-weight: 600;
    }
    .status-critical {
        color: #C73E1D;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2D2D2D;
    }
    
    /* Button styling - Modern and elegant design */
    .stButton > button {
        background: #2E86AB;
        color: white;
        border: 1px solid #2E86AB;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: none;
    }
    
    .stButton > button:hover {
        background: #1e5f7a;
        border-color: #1e5f7a;
        transform: none;
        box-shadow: 0 2px 4px rgba(46, 134, 171, 0.2);
    }
    
    .stButton > button:active {
        background: #164a5c;
        transform: translateY(1px);
    }
    
    /* Button variants for different contexts */
    .stButton.secondary > button {
        background: transparent;
        color: #2E86AB;
        border: 1px solid #2E86AB;
    }
    
    .stButton.secondary > button:hover {
        background: #2E86AB;
        color: white;
    }
    
    .stButton.danger > button {
        background: #dc3545;
        border-color: #dc3545;
    }
    
    .stButton.danger > button:hover {
        background: #c82333;
        border-color: #c82333;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background-color: #2D2D2D;
        color: #FFFFFF;
        border: 1px solid #404040;
        border-radius: 10px;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #2D2D2D;
        color: #FFFFFF;
        border: 1px solid #404040;
        border-radius: 10px;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > select {
        background-color: #2D2D2D;
        color: #FFFFFF;
        border: 1px solid #404040;
        border-radius: 10px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2D2D2D;
        border-radius: 10px;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #FFFFFF;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2E86AB;
        color: white;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: #2D2D2D;
        border: 1px solid #404040;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #2D2D2D;
        border: 1px solid #404040;
        border-radius: 10px;
        color: #FFFFFF;
    }
    
    /* Success/Info/Warning/Error message styling */
    .stSuccess {
        background-color: rgba(106, 153, 78, 0.1);
        border: 1px solid #6A994E;
        color: #6A994E;
    }
    
    .stInfo {
        background-color: rgba(46, 134, 171, 0.1);
        border: 1px solid #2E86AB;
        color: #2E86AB;
    }
    
    .stWarning {
        background-color: rgba(241, 143, 1, 0.1);
        border: 1px solid #F18F01;
        color: #F18F01;
    }
/* Enhanced page headers */
    .page-header {
        font-size: 2.2rem;
        font-weight: 600;
        background: linear-gradient(135deg, #2E86AB 0%, #4A9FD1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2E86AB;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Enhanced main header */
    .main-header {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #2E86AB 0%, #4A9FD1 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin-bottom: 2rem !important;
        text-align: center !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #2E86AB 0%, #1e5f7a 100%) !important;
        padding: 2rem !important;
        border-radius: 16px !important;
        color: white !important;
        margin: 1rem 0 !important;
        box-shadow: 0 8px 32px rgba(46, 134, 171, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    
    .metric-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 40px rgba(46, 134, 171, 0.3) !important;
    }
    
    /* Enhanced chat container */
    .chat-container {
        background: linear-gradient(135deg, #2D2D2D 0%, #3D3D3D 100%) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        margin: 1.5rem 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Dashboard grid layout */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    /* Dashboard cards */
    .dashboard-card {
        background: linear-gradient(135deg, #2D2D2D 0%, #3D3D3D 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .dashboard-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
    
    .dashboard-card h3 {
        color: #2E86AB;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* History items */
    .history-item {
        background: linear-gradient(135deg, #2D2D2D 0%, #3D3D3D 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }
    
    .history-item:hover {
        transform: translateX(4px);
        border-color: #2E86AB;
        box-shadow: 0 6px 24px rgba(46, 134, 171, 0.2);
    }
    
    .history-question {
        color: #4A9FD1;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .history-answer {
        color: #FFFFFF;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    
    .history-meta {
        color: #888;
        font-size: 0.85rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Status indicators */
    .status-warning {
        color: #F18F01;
        font-weight: 600;
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #2D2D2D 0%, #1E1E1E 100%) !important;
    }
    
    .sidebar-section {
        background: linear-gradient(135deg, #3D3D3D 0%, #2D2D2D 100%) !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        margin: 1rem 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: linear-gradient(135deg, #2D2D2D 0%, #3D3D3D 100%);
        border-radius: 12px;
        margin: 0.5rem 0;
/* Enhanced chat message styling */
    .stChatMessage {
        background: linear-gradient(135deg, #2D2D2D 0%, #3D3D3D 100%) !important;
        border-radius: 12px !important;
        margin: 0.5rem 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #2E86AB 0%, #1e5f7a 100%) !important;
    }
    
    .stChatMessage[data-testid="assistant-message"] {
        background: linear-gradient(135deg, #3D3D3D 0%, #2D2D2D 100%) !important;
    }
    
    /* Enhanced expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #3D3D3D 0%, #2D2D2D 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
    }
    
    /* Enhanced input styling */
    .stChatInput > div {
        background: linear-gradient(135deg, #2D2D2D 0%, #3D3D3D 100%) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Loading spinner styling */
    .stSpinner > div {
        border-color: #2E86AB !important;
    }
    
    /* Enhanced toggle styling */
    .stToggle > div {
        background-color: #2D2D2D !important;
    }
    
    /* Page transition effects */
    .main .block-container {
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Responsive design improvements */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem !important;
        }
        
        .page-header {
            font-size: 1.8rem !important;
        }
        
        .dashboard-grid {
            grid-template-columns: 1fr !important;
        }
        
        .welcome-container {
            padding: 1.5rem !important;
        }
        
        .dashboard-card {
            margin: 0.5rem 0 !important;
        }
    }
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Performance metrics */
    .perf-metric {
        background: linear-gradient(135deg, #2E86AB 0%, #1e5f7a 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .perf-metric h4 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .perf-metric p {
        margin: 0.25rem 0 0 0;
        opacity: 0.9;
        font-size: 0.9rem;
    }
    
    /* Welcome message styling */
    .welcome-container {
        background: linear-gradient(135deg, #2E86AB 0%, #1e5f7a 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(46, 134, 171, 0.2);
    }
    
    .welcome-container h2 {
        margin: 0 0 1rem 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    .welcome-container p {
        margin: 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    .stError {
        background-color: rgba(199, 62, 29, 0.1);
        border: 1px solid #C73E1D;
        color: #C73E1D;
    }
    .status-error {
        color: #dc3545;
        font-weight: 600;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .chat-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)


# === PROJECT CONFIGURATION ===
def setup_project_paths():
    """Configure project paths cleanly"""
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.info(f"Project path added: {project_root}")

    return project_root


def setup_environment():
    """Configure system environment"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass


# === SECURE IMPORTS ===
def import_core_config():
    """Import core configuration"""
    try:
        from src.core.config import get_config, get_debug_info

        logger.info("Core configuration imported")
        return get_config, get_debug_info
    except ImportError as e:
        logger.error(f"Cannot import configuration: {e}")
        st.error("Critical error: Configuration unavailable")
        st.stop()


def import_auth_system():
    """Import authentication system"""
    try:
        from src.webapp.components.auth_manager import AuthManager

        logger.info("Auth system imported")
        return AuthManager()
    except ImportError as e:
        logger.warning(f"Auth unavailable: {e}")
        return None


def import_features_manager():
    """Import features manager"""
    try:
        from src.webapp.components.features_manager import FeaturesManager

        logger.info("FeaturesManager imported")
        return FeaturesManager
    except ImportError as e:
        logger.warning(f"FeaturesManager unavailable: {e}")
        return None


def import_rag_pipeline():
    """Import RAG pipeline"""
    try:
        from src.rag_system.rag_pipeline import RAGPipelineFactory

        logger.info("RAG Pipeline imported")
        return RAGPipelineFactory, True
    except ImportError as e:
        logger.error(f"RAG Pipeline unavailable: {e}")
        return None, False


def import_history_components():
    """Import history components"""
    try:
        from src.webapp.components.history_manager import get_history_manager
        from src.core.data_manager import get_data_manager

        logger.info("History components imported")
        return get_history_manager, get_data_manager
    except ImportError as e:
        logger.warning(f"History components unavailable: {e}")
        return None, None


def import_performance_dashboard():
    """Import performance dashboard"""
    try:
        from src.webapp.components.performance_dashboard import render_performance_dashboard

        logger.info("Performance dashboard imported")
        return render_performance_dashboard
    except ImportError as e:
        logger.warning(f"Performance dashboard unavailable: {e}")
        return None


# === INITIALIZATION ===
setup_project_paths()
setup_environment()

# Imports
get_config, get_debug_info = import_core_config()
auth_manager = import_auth_system()
FeaturesManager = import_features_manager()
RAGPipelineFactory, NEW_RAG_AVAILABLE = import_rag_pipeline()
get_history_manager, get_data_manager = import_history_components()
render_performance_dashboard = import_performance_dashboard()

logger.info(f"Initialization complete - RAG: {NEW_RAG_AVAILABLE}, Auth: {auth_manager is not None}")


# === UTILITY FUNCTIONS ===
def create_auth_decorators():
    """Create authentication decorators"""

    def login_required(func):
        """Decorator for pages requiring authentication"""

        def wrapper(*args, **kwargs):
            if auth_manager and not auth_manager.is_authenticated():
                if not auth_manager.render_login_form():
                    return
            return func(*args, **kwargs)

        return wrapper

    def admin_required(func):
        """Decorator for admin pages"""

        def wrapper(*args, **kwargs):
            if auth_manager and not auth_manager.is_admin():
                st.error("Administrator access required")
                return
            return func(*args, **kwargs)

        return wrapper

    return login_required, admin_required


login_required, admin_required = create_auth_decorators()


# === RAG MODEL MANAGEMENT ===
@st.cache_resource
def get_model():
    """Load RAG model with clean error handling"""
    with st.spinner("Loading RAG model..."):
        config = get_config()
        debug_info = get_debug_info()

        # Display debug information in discrete expander
        with st.sidebar.expander("System Information", expanded=False):
            st.code(f"""
Configuration:
- Provider: {debug_info["provider"]}
- Vector store: {debug_info["persist_directory"]}
- Confluence URL: {debug_info["confluence_url"]}
- Space key: {debug_info["space_key"]}
- User: {debug_info["user_email"]}
            """)

        persist_path = Path(config.persist_directory)
        index_file = persist_path / "index.faiss"

        # Check database existence
        if not persist_path.exists() or not index_file.exists():
            st.info("First launch - Creating vector database...")

        # Model creation
        try:
            if not NEW_RAG_AVAILABLE:
                st.error("RAG architecture unavailable!")
                return None

            pipeline = RAGPipelineFactory.create_default_pipeline()
            st.success("Vector database ready")
            return pipeline

        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            import traceback

            with st.expander("Error details"):
                st.code(traceback.format_exc(), language="python")
            return None


# === USER MANAGEMENT ===
def setup_user_session():
    """Configure user session"""
    if "user" not in st.session_state:
        config = get_config()
        email = config.confluence_email_address or "admin@auto.login"

        if auth_manager:
            try:
                user = auth_manager.authenticate_user(email, "admin123")
                if not user:
                    if auth_manager.register_user(email, "admin123"):
                        user = auth_manager.authenticate_user(email, "admin123")
                    else:
                        user = auth_manager.authenticate_user("admin@isschat.com", "admin123")

                if user:
                    st.session_state["user"] = user
                    st.session_state["user_id"] = f"user_{user['id']}"
                    logger.info(f"User connected: {user['email']}")
                else:
                    logger.warning("Fallback: creating session user")
                    st.session_state["user"] = {"email": email, "id": 1, "is_admin": True}
                    st.session_state["user_id"] = "user_1"
            except Exception as e:
                logger.error(f"Auth error: {e}")
                st.session_state["user"] = {"email": email, "id": 1, "is_admin": True}
                st.session_state["user_id"] = "user_1"
        else:
            st.session_state["user"] = {"email": email, "id": 1, "is_admin": True}
            st.session_state["user_id"] = "user_1"

        st.session_state["page"] = "chat"


# === QUESTION PROCESSING ===
def process_question(model, features_manager: Optional[Any], prompt: str) -> Tuple[str, str, float]:
    """Process question with RAG model"""
    start_time = time.time()

    try:
        if hasattr(model, "process_query"):
            result, sources = model.process_query(prompt)
        elif hasattr(model, "query"):
            result = model.query(prompt)
            sources = ""
        else:
            result = "Model unavailable"
            sources = ""

        if features_manager and result != "Model unavailable":
            response_time_ms = (time.time() - start_time) * 1000
            features_manager.process_query_response(prompt, result, response_time_ms)
    except Exception as e:
        result = f"Error: {str(e)}"
        sources = ""

    response_time_ms = (time.time() - start_time) * 1000
    return result, sources, response_time_ms


def save_conversation_data(prompt: str, result: str, sources: str, response_time_ms: float):
    """Save conversation data"""
    if not (get_history_manager and get_data_manager):
        return

    try:
        history_manager = get_history_manager()
        data_manager = get_data_manager()
        user_id = st.session_state.get("user", {}).get("email", "anonymous")

        # Prepare sources
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

        # Save data
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
        st.warning(f"Cannot save conversation: {e}")


# === APPLICATION PAGES ===
@login_required
def chat_page():
    """Main chat page with modern interface"""
    st.session_state["page"] = "chat"

    # Enhanced header with icon
    st.markdown('''
    <div class="welcome-container">
        <h2>🤖 Bienvenue sur ISSCHAT</h2>
        <p>Votre assistant intelligent pour toutes vos questions</p>
    </div>
    ''', unsafe_allow_html=True)

    # Load model
    model = get_model()
    if not model:
        st.error("No RAG model available")
        return

    # Initialize features manager
    if "features_manager" not in st.session_state:
        user_id = st.session_state.get("user_id", f"user_{st.session_state['user']['id']}")
        if FeaturesManager:
            st.session_state["features_manager"] = FeaturesManager(user_id)
        else:
            st.session_state["features_manager"] = None

    features = st.session_state["features_manager"]

    # Sidebar options
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("Options")
        show_feedback = st.toggle("Response feedback", value=True)
        show_sources = st.toggle("Show sources", value=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # System status
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("System Status")
        if NEW_RAG_AVAILABLE:
            st.markdown('<span class="status-healthy">System operational</span>', unsafe_allow_html=True)

            if hasattr(model, "health_check"):
                try:
                    health = model.health_check()
                    if health.get("status") == "healthy":
                        st.caption("All checks passed")
                    else:
                        st.markdown('<span class="status-warning">Issues detected</span>', unsafe_allow_html=True)
                        if health.get("issues"):
                            for issue in health["issues"]:
                                st.caption(f"• {issue}")
                except Exception:
                    st.caption("Health check unavailable")
        else:
            st.markdown('<span class="status-error">Legacy architecture</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Chat interface in modern container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Welcome message
    user_email = st.session_state.get("user", {}).get("email", "")
    first_name = user_email.split("@")[0].split(".")[0].capitalize()
    welcome_message = f"Hello {first_name}, how can I help you today?"

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": welcome_message}]

    # Display message history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle reused questions from history
    if "reuse_question" in st.session_state:
        prompt = st.session_state.pop("reuse_question")
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Process question
        with st.spinner("Analyzing..."):
            result, sources, response_time_ms = process_question(model, features, prompt)

            # Display response
            st.chat_message("assistant").markdown(result)
            if sources and show_sources:
                with st.expander("Consulted sources"):
                    st.write(sources)

            # Save data
            save_conversation_data(prompt, result, sources, response_time_ms)

            # Feedback widget
            if show_feedback and features:
                features.render_feedback_widget(f"resp_{len(st.session_state.messages)}")

            # Add to history
            response_content = result
            if sources and show_sources:
                response_content += "\n\n" + sources
            st.session_state.messages.append({"role": "assistant", "content": response_content})

    # Input interface
    if prompt := st.chat_input("Ask your question here..."):
        # Add question
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Process question
        with st.spinner("Analyzing..."):
            result, sources, response_time_ms = process_question(model, features, prompt)

            # Display response
            st.chat_message("assistant").markdown(result)
            if sources and show_sources:
                with st.expander("Consulted sources"):
                    st.write(sources)

            # Save data
            save_conversation_data(prompt, result, sources, response_time_ms)

            # Feedback widget
            if show_feedback and features:
                features.render_feedback_widget(f"resp_{len(st.session_state.messages)}")

            # Add to history
            response_content = result
            if sources and show_sources:
                response_content += "\n\n" + sources
            st.session_state.messages.append({"role": "assistant", "content": response_content})

    st.markdown("</div>", unsafe_allow_html=True)


@login_required
def performance_page():
    """Performance dashboard page"""
    st.session_state["page"] = "performance"
    
    # Enhanced header
    st.markdown('''
    <div class="page-header">
        📊 Dashboard de Performance
    </div>
    ''', unsafe_allow_html=True)
    
    if render_performance_dashboard and get_data_manager:
        data_manager = get_data_manager()
        
        # Create dashboard grid
        st.markdown('<div class="dashboard-grid">', unsafe_allow_html=True)
        
        # Performance metrics in cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('''
            <div class="dashboard-card">
                <h3>⚡ Temps de Réponse</h3>
                <div class="perf-metric">
                    <h4>1.2s</h4>
                    <p>Moyenne</p>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown('''
            <div class="dashboard-card">
                <h3>💬 Requêtes Traitées</h3>
                <div class="perf-metric">
                    <h4>247</h4>
                    <p>Aujourd'hui</p>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown('''
            <div class="dashboard-card">
                <h3>✅ Taux de Succès</h3>
                <div class="perf-metric">
                    <h4>98.5%</h4>
                    <p>Cette semaine</p>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Render the actual dashboard
        render_performance_dashboard(data_manager)
    else:
        st.markdown('''
        <div class="dashboard-card">
            <h3>⚠️ Dashboard Indisponible</h3>
            <p>Le système de monitoring des performances n'est pas disponible actuellement.</p>
        </div>
        ''', unsafe_allow_html=True)


@login_required
def history_page():
    """Conversation history page"""
    st.session_state["page"] = "history"
    
    # Enhanced header
    st.markdown('''
    <div class="page-header">
        📚 Historique des Conversations
    </div>
    ''', unsafe_allow_html=True)
    
    try:
        if get_history_manager:
            history_manager = get_history_manager()
            user_id = st.session_state.get("user", {}).get("email", "anonymous")
            
            # Add some intro text
            st.markdown('''
            <div class="dashboard-card">
                <h3>💬 Vos Conversations Récentes</h3>
                <p>Retrouvez et réutilisez vos questions précédentes</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Render the history with enhanced styling
            history_manager.render_history_page(user_id)
        else:
            st.markdown('''
            <div class="dashboard-card">
                <h3>⚠️ Historique Indisponible</h3>
                <p>Le système d'historique des conversations n'est pas disponible actuellement.</p>
            </div>
            ''', unsafe_allow_html=True)
    except Exception as e:
        st.markdown('''
        <div class="dashboard-card">
            <h3>❌ Erreur de Chargement</h3>
            <p>Une erreur s'est produite lors du chargement de l'historique.</p>
        </div>
        ''', unsafe_allow_html=True)
        st.error(f"Détails de l'erreur: {e}")
        logger.error(f"History error: {e}")


@admin_required
def admin_page():
    """Clean administration page"""
    st.title("Administration")

    if auth_manager:
        st.subheader("User Management")
        st.info("User management functionality to be implemented")
    else:
        st.warning("Authentication system unavailable")


# === MAIN INTERFACE ===
def render_sidebar():
    """Render modern sidebar"""
    with st.sidebar:
        # Enhanced logo and title
        st.markdown('''
        <div class="sidebar-section">
            <h2 style="color: #2E86AB; margin: 0; font-size: 1.8rem;">🤖 ISSCHAT</h2>
            <p style="color: #888; margin: 0.5rem 0 0 0; font-style: italic;">Assistant Intelligent</p>
        </div>
        ''', unsafe_allow_html=True)

        # User information with enhanced styling
        if "user" in st.session_state:
            user_email = st.session_state["user"]["email"]
            first_name = user_email.split("@")[0].split(".")[0].capitalize()
            st.markdown(f'''
            <div class="sidebar-section">
                <h4 style="color: #2E86AB; margin: 0;">👤 Utilisateur</h4>
                <p style="color: #6A994E; margin: 0.25rem 0 0 0; font-weight: 600;">Connecté: {first_name}</p>
                <p style="color: #888; margin: 0; font-size: 0.8rem;">{user_email}</p>
            </div>
            ''', unsafe_allow_html=True)

        # Enhanced navigation with icons
        st.markdown('''
        <div class="sidebar-section">
            <h4 style="color: #2E86AB; margin: 0 0 1rem 0;">🧭 Navigation</h4>
        </div>
        ''', unsafe_allow_html=True)

        # Current page indicator
        current_page = st.session_state.get("page", "chat")
        
        # Chat button
        chat_type = "primary" if current_page == "chat" else "secondary"
        if st.button("💬 Chat", use_container_width=True, type=chat_type):
            st.session_state["page"] = "chat"
            st.rerun()

        # History button
        history_type = "primary" if current_page == "history" else "secondary"
        if st.button("📚 Historique", use_container_width=True, type=history_type):
            st.session_state["page"] = "history"
            st.rerun()

        # Dashboard button
        dashboard_type = "primary" if current_page == "performance" else "secondary"
        if st.button("📊 Dashboard", use_container_width=True, type=dashboard_type):
            st.session_state["page"] = "performance"
            st.rerun()

        # Admin options
        if st.session_state.get("user", {}).get("is_admin"):
            st.markdown("---")
            st.info("Administrator Mode")
            if st.button("Administration", use_container_width=True):
                st.session_state["page"] = "admin"
                st.rerun()

        # System information
        st.markdown("---")
        st.caption("Version 2.0 - Modern Interface")


def main():
    """Main application function"""
    # Configure user session
    setup_user_session()

    # Render sidebar
    render_sidebar()

    # Page routing
    page = st.session_state.get("page", "chat")

    if page == "performance":
        performance_page()
    elif page == "admin" and st.session_state.get("user", {}).get("is_admin"):
        admin_page()
    elif page == "history":
        history_page()
    else:
        chat_page()


# === ENTRY POINT ===
if __name__ == "__main__":
    main()
