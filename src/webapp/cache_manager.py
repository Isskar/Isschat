import streamlit as st
import hashlib
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VectorDBCacheManager:
    """
    Cache manager for per-user vector databases.
    Uses Streamlit cache to avoid recreation of vector DB.
    """

    @staticmethod
    def get_user_cache_key(user_email: str, config_hash: Optional[str] = None) -> str:
        """
        Generates a unique cache key for a user.

        Args:
        user_email: User email
        config_hash: Optional configuration hash

        Returns:
        Unique cache key
        """
        # Create a hash of user email for security
        user_hash = hashlib.md5(user_email.encode()).hexdigest()[:8]

        if config_hash:
            return f"vector_db_{user_hash}_{config_hash}"
        else:
            return f"vector_db_{user_hash}"

    @staticmethod
    def get_config_hash() -> str:
        """
        Generates a hash of the current configuration to invalidate the cache if necessary.

        Returns:
        Configuration hash
        """
        try:
            from src.core.config import get_config

            config = get_config()

            # Create a hash based on critical parameters
            config_str = f"{config.confluence_space_name}_{config.confluence_space_key}_{config.persist_directory}"
            return hashlib.md5(config_str.encode()).hexdigest()[:8]
        except Exception as e:
            logger.warning(f"Unable to generate config hash: {e}")
            return "default"

    @staticmethod
    def get_user_persist_directory(user_email: str, base_persist_dir: str) -> str:
        """
        Generate a user-specific persistence directory.

        Args:
            user_email: User email
            base_persist_dir: Base directory

        Returns:
            Path to user-specific directory
        """
        user_hash = hashlib.md5(user_email.encode()).hexdigest()[:8]
        user_dir = Path(base_persist_dir) / f"user_{user_hash}"
        user_dir.mkdir(parents=True, exist_ok=True)
        return str(user_dir)


@st.cache_resource
def get_user_vector_db(_user_email: str, _config_hash: str):
    """
    Streamlit cache for user-specific vector database.

    Args:
        _user_email: User email (prefixed with _ to avoid hashing by Streamlit)
        _config_hash: Configuration hash

    Returns:
        RAG pipeline initialized for the user
    """
    logger.info(f"Initializing vector DB for user: {_user_email[:10]}...")

    try:
        from src.rag_system.rag_pipeline import RAGPipelineFactory
        from src.core.config import get_config

        # Get base configuration
        config = get_config()

        # Create a user-specific persistence directory
        user_persist_dir = VectorDBCacheManager.get_user_persist_directory(_user_email, config.persist_directory)

        # Modifier temporairement la configuration pour cet utilisateur
        original_persist_dir = config.persist_directory
        config.persist_directory = user_persist_dir

        try:
            # Create pipeline with user configuration
            pipeline = RAGPipelineFactory.create_default_pipeline()
            logger.info(f"✅ Vector DB initialized for user in: {user_persist_dir}")
            return pipeline

        finally:
            # Restaurer la configuration originale
            config.persist_directory = original_persist_dir

    except Exception as e:
        logger.error(f"❌ Vector DB initialization error: {e}")
        raise


@st.cache_resource
def get_user_model(_user_email: str):
    """
    Simplified Streamlit cache for user-specific RAG model.
    Simplified version that uses only user email.

    Args:
        _user_email: User email

    Returns:
        RAG pipeline initialized for the user
    """
    config_hash = VectorDBCacheManager.get_config_hash()
    return get_user_vector_db(_user_email, config_hash)


def clear_user_cache(user_email: str):
    """
    Clear cache for a specific user.

    Args:
        user_email: User email
    """
    try:
        # Effacer le cache Streamlit pour cet utilisateur
        cache_key = VectorDBCacheManager.get_user_cache_key(user_email)

        # Streamlit doesn't allow easy deletion of specific entries
        # But we can force a reset by modifying the session
        if f"force_reload_{cache_key}" not in st.session_state:
            st.session_state[f"force_reload_{cache_key}"] = 0

        st.session_state[f"force_reload_{cache_key}"] += 1
        logger.info(f"Cache effacé pour l'utilisateur: {user_email[:10]}...")

    except Exception as e:
        logger.error(f"Erreur lors de l'effacement du cache: {e}")


def get_cache_stats() -> dict:
    """
    Get statistics about the current Streamlit cache.
    Returns:
        Dictionary with cache statistics
    """
    try:
        # Count cache entries in the session
        cache_entries = [key for key in st.session_state.keys() if key.startswith("vector_db_")]

        return {
            "cache_entries": len(cache_entries),
            "cache_keys": cache_entries[:5],  # Show only the first 5
            "total_session_keys": len(st.session_state.keys()),
        }
    except Exception as e:
        return {"error": str(e)}
