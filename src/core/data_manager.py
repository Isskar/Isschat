"""
Centralized data manager for the new directory structure.
Manages conversation history, logs and data migration.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod


@dataclass
class ConversationEntry:
    """Structure of a conversation entry."""

    timestamp: str
    user_id: str
    conversation_id: str
    question: str
    answer: str
    answer_length: int
    sources_count: int
    response_time_ms: float
    feedback: Optional[Dict[str, Any]] = None
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceEntry:
    """Structure of a performance entry."""

    timestamp: str
    operation: str
    duration_ms: float
    user_id: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FeedbackEntry:
    """Structure of a feedback entry."""

    timestamp: str
    user_id: str
    conversation_id: str
    rating: int
    comment: str
    metadata: Optional[Dict[str, Any]] = None


class BaseDataStore(ABC):
    """Abstract interface for data stores."""

    @abstractmethod
    def save_entry(self, entry: Any) -> bool:
        """Save an entry."""
        pass

    @abstractmethod
    def load_entries(self, limit: Optional[int] = None) -> List[Any]:
        """Load entries."""
        pass

    @abstractmethod
    def get_entries_by_user(self, user_id: str, limit: Optional[int] = None) -> List[Any]:
        """Retrieve entries for a user."""
        pass


class JSONLDataStore(BaseDataStore):
    """Data store using JSONL format with storage service."""

    def __init__(self, storage_service, file_path: str, entry_class: type):
        """
        Initialize JSONL data store with storage service

        Args:
            storage_service: StorageService instance for file operations
            file_path: Path to the JSONL file (relative to storage base)
            entry_class: Class type for entries
        """
        self.storage_service = storage_service
        self.file_path = file_path
        self.entry_class = entry_class
        self._ensure_directory()

    def _ensure_directory(self):
        """Ensure that the directory exists using storage service."""
        # Extract directory path from file path
        directory_path = "/".join(self.file_path.split("/")[:-1])
        if directory_path:
            # Use storage service to create directory if it has the method
            if hasattr(self.storage_service._storage, "create_directory"):
                self.storage_service._storage.create_directory(directory_path)

    def save_entry(self, entry: Any) -> bool:
        """Save an entry in JSONL format using storage service."""
        try:
            # Convert entry to dict if needed
            if isinstance(entry, dict):
                entry_dict = entry
            else:
                entry_dict = asdict(entry)

            # Use storage service to append JSONL data
            return self.storage_service.append_jsonl_data(self.file_path, entry_dict)
        except Exception as e:
            logging.error(f"Error saving to {self.file_path}: {e}")
            return False

    def load_entries(self, limit: Optional[int] = None) -> List[Any]:
        """Load entries from JSONL file using storage service."""
        entries = []

        # Check if file exists using storage service
        if not self.storage_service.file_exists(self.file_path):
            return entries

        try:
            # Load file content using storage service
            content = self.storage_service.load_text_file(self.file_path)
            if not content:
                return entries

            # Parse JSONL content
            for line in content.split("\n"):
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        entries.append(data)
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines

            # Sort by timestamp (most recent first)
            entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            if limit:
                entries = entries[:limit]

            return entries
        except Exception as e:
            logging.error(f"Error loading from {self.file_path}: {e}")
            return []

    def get_entries_by_user(self, user_id: str, limit: Optional[int] = None) -> List[Any]:
        """Retrieve entries for a specific user."""
        all_entries = self.load_entries()
        user_entries = [entry for entry in all_entries if entry.get("user_id") == user_id]

        if limit:
            user_entries = user_entries[:limit]

        return user_entries


class DataManager:
    """Main data manager with new directory structure."""

    def __init__(self, base_data_dir: Optional[Path] = None, storage_service=None):
        """
        Initialize the data manager.

        Args:
            base_data_dir: Base directory for data. If None, uses config.
            storage_service: StorageService instance. If None, uses config.
        """
        if base_data_dir is None:
            # Use project directory as base
            project_root = Path(__file__).parent.parent.parent
            self.base_data_dir = project_root / "data"
        else:
            self.base_data_dir = Path(base_data_dir)

        # Get storage service from config if not provided
        if storage_service is None:
            from src.core.config import _ensure_config_initialized

            config_manager = _ensure_config_initialized()
            self.storage_service = config_manager.get_storage_service()
        else:
            self.storage_service = storage_service

        # Directory structure according to plan
        self.raw_dir = self.base_data_dir / "raw"
        self.processed_dir = self.base_data_dir / "processed"
        self.vector_db_dir = self.base_data_dir / "vector_db"

        # Directories for structured logs
        self.logs_dir = self.base_data_dir / "logs"
        self.conversations_dir = self.logs_dir / "conversations"
        self.performance_dir = self.logs_dir / "performance"
        self.feedback_dir = self.logs_dir / "feedback"

        self._create_directories()
        self._init_stores()

    def _create_directories(self):
        """Create directory structure using storage service."""
        directories = [
            "raw",
            "processed",
            "vector_db",
            "logs/conversations",
            "logs/performance",
            "logs/feedback",
        ]

        # Use storage service to create directories
        for directory in directories:
            if hasattr(self.storage_service._storage, "create_directory"):
                self.storage_service._storage.create_directory(directory)

    def _init_stores(self):
        """Initialize data stores using storage service."""
        today = datetime.now().strftime("%Y%m%d")

        # Stores for different data types using storage service
        self.conversation_store = JSONLDataStore(
            self.storage_service, f"logs/conversations/conversations_{today}.jsonl", ConversationEntry
        )

        self.performance_store = JSONLDataStore(
            self.storage_service, f"logs/performance/performance_{today}.jsonl", PerformanceEntry
        )

        self.feedback_store = JSONLDataStore(
            self.storage_service, f"logs/feedback/feedback_{today}.jsonl", FeedbackEntry
        )

    def save_conversation(
        self,
        user_id: str,
        conversation_id: str,
        question: str,
        answer: str,
        response_time_ms: float,
        sources: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Save conversation using unified storage system"""
        entry = ConversationEntry(
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            answer_length=len(answer),
            sources_count=len(sources) if sources else 0,
            response_time_ms=response_time_ms,
            sources=sources,
            metadata=metadata,
        )

        return self.conversation_store.save_entry(entry)

    def save_performance(
        self, operation: str, duration_ms: float, user_id: str, metadata: Optional[Dict] = None
    ) -> bool:
        """Sauvegarde une métrique de performance."""
        entry = PerformanceEntry(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            duration_ms=duration_ms,
            user_id=user_id,
            metadata=metadata,
        )

        return self.performance_store.save_entry(entry)

    def save_feedback(
        self, user_id: str, conversation_id: str, rating: int, comment: str = "", metadata: Optional[Dict] = None
    ) -> bool:
        """Save user feedback using unified storage system"""
        entry = FeedbackEntry(
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            conversation_id=conversation_id,
            rating=rating,
            comment=comment,
            metadata=metadata,
        )

        # Use the unified storage system via feedback_store
        return self.feedback_store.save_entry(entry)

    def get_conversation_history(
        self, user_id: Optional[str] = None, conversation_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict]:
        """Retrieve the conversation history, filtered by user_id and conversation_id."""
        all_entries = []

        try:
            # Use the conversation_store which already handles storage service properly
            all_entries = self.conversation_store.load_entries(limit=None)  # Load all first, then filter

            # Filter by user_id if provided
            if user_id:
                all_entries = [entry for entry in all_entries if entry.get("user_id") == user_id]

            # Filter by conversation_id if provided
            if conversation_id:
                all_entries = [entry for entry in all_entries if entry.get("conversation_id") == conversation_id]

            # Sort by timestamp (most recent first) - already done in load_entries but ensure it
            all_entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            # Apply limit
            if limit is not None and limit > 0:
                all_entries = all_entries[:limit]

        except Exception as e:
            logging.error(f"Error retrieving conversation history: {e}")
            return []

        return all_entries

    def get_performance_metrics(self, user_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Récupère les métriques de performance."""
        if user_id:
            return self.performance_store.get_entries_by_user(user_id, limit)
        else:
            return self.performance_store.load_entries(limit)

    def get_feedback_data(self, user_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Récupère les données de feedback."""
        if user_id:
            return self.feedback_store.get_entries_by_user(user_id, limit)
        else:
            return self.feedback_store.load_entries(limit)

    def get_data_structure_info(self) -> Dict[str, Any]:
        """Retourne des informations sur la structure de données."""
        return {
            "base_directory": str(self.base_data_dir),
            "directories": {
                "raw": str(self.raw_dir),
                "processed": str(self.processed_dir),
                "vector_db": str(self.vector_db_dir),
                "logs": {
                    "conversations": str(self.conversations_dir),
                    "performance": str(self.performance_dir),
                    "feedback": str(self.feedback_dir),
                },
            },
            "current_files": {
                "conversations": len(list(self.conversations_dir.glob("*.jsonl"))),
                "performance": len(list(self.performance_dir.glob("*.jsonl"))),
                "feedback": len(list(self.feedback_dir.glob("*.jsonl"))),
            },
        }


# Instance globale du gestionnaire de données
_data_manager = None


def get_data_manager() -> DataManager:
    """Retourne l'instance globale du gestionnaire de données."""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager  # type : ignore
