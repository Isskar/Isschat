"""
Robust path manager for Isschat.
Uses pathlib for cross-platform and robust paths.
"""

from pathlib import Path
from typing import Optional
from .settings import IsschatConfig


class PathManager:
    """Robust path manager for the entire project"""

    def __init__(self, config: Optional[IsschatConfig] = None):
        self.config = config
        # Project root path (goes up from src/config to root)
        self.project_root = Path(__file__).parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """Absolute and robust data path"""
        if self.config.data_dir.is_absolute():
            return self.config.data_dir
        return self.project_root / self.config.data_dir

    @property
    def logs_dir(self) -> Path:
        """Robust logs path"""
        return self.data_dir / "logs"

    @property
    def conversations_dir(self) -> Path:
        """Robust conversations path"""
        return self.logs_dir / "conversations"

    @property
    def feedback_dir(self) -> Path:
        """Robust feedback path"""
        return self.logs_dir / "feedback"

    @property
    def performance_dir(self) -> Path:
        """Robust performance path"""
        return self.logs_dir / "performance"

    @property
    def raw_data_dir(self) -> Path:
        """Robust raw data path"""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Robust processed data path"""
        return self.data_dir / "processed"

    def ensure_directories(self, storage_service=None) -> None:
        """Create all necessary directories using storage abstraction"""
        if storage_service is None:
            # Fallback to local creation if no storage service provided
            directories = [
                self.data_dir,
                self.logs_dir,
                self.conversations_dir,
                self.feedback_dir,
                self.performance_dir,
                self.raw_data_dir,
                self.processed_data_dir,
            ]
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
        else:
            # Use storage abstraction
            directory_names = [
                "logs",
                "logs/conversations",
                "logs/feedback",
                "logs/performance",
                "raw",
                "processed",
            ]
            for directory_name in directory_names:
                storage_service.create_directory(directory_name)

    def get_conversation_file(self, date_str: str) -> Path:
        """Conversation file path for a date"""
        return self.conversations_dir / f"conversations_{date_str}.jsonl"

    def get_feedback_file(self, date_str: str) -> Path:
        """Feedback file path for a date"""
        return self.feedback_dir / f"feedback_{date_str}.jsonl"

    def get_performance_file(self, date_str: str) -> Path:
        """Performance file path for a date"""
        return self.performance_dir / f"performance_{date_str}.jsonl"


# Global instance
_path_manager: Optional[PathManager] = None


def get_path_manager() -> Optional[PathManager]:
    """Get the global path manager instance"""
    global _path_manager
    if _path_manager is None:
        from .settings import get_config

        _path_manager = PathManager(get_config())
    return _path_manager


def reset_path_manager():
    """Reset for tests"""
    global _path_manager
    _path_manager = None
