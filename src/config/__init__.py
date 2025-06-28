"""
Configuration unifiée pour Isschat.
Centralise embeddings, chunking, vectordb, RAG et storage.
"""

from .settings import IsschatConfig, get_config, get_debug_info
from .paths import PathManager, get_path_manager

__all__ = ["IsschatConfig", "get_config", "get_debug_info", "PathManager", "get_path_manager"]
