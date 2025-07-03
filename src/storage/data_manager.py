"""
Simplified data manager with robust paths.
Simplified version of data_manager for feedback and conversations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from ..config import get_path_manager
from .storage_factory import StorageFactory


@dataclass
class ConversationEntry:
    """Simplified conversation structure"""

    timestamp: str
    user_id: str
    conversation_id: str
    question: str
    answer: str
    response_time_ms: float
    sources: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None


@dataclass
class FeedbackEntry:
    """Simplified feedback structure"""

    timestamp: str
    user_id: str
    conversation_id: str
    rating: int
    comment: str
    metadata: Optional[Dict] = None


@dataclass
class PerformanceEntry:
    """Simplified performance structure"""

    timestamp: str
    operation: str
    duration_ms: float
    user_id: str
    metadata: Optional[Dict] = None


class DataManager:
    """Simplified data manager with robust paths"""

    def __init__(self, storage_service=None):
        """
        Initialize with storage service

        Args:
            storage_service: Storage service, otherwise use factory
        """
        self.path_manager = get_path_manager()
        self.logger = logging.getLogger(self.__class__.__name__)

        if storage_service is None:
            self.storage = self._create_storage_service()
        else:
            self.storage = storage_service

        self._ensure_directories()

    def _create_storage_service(self):
        """Create storage service based on config"""
        from ..config import get_config

        config = get_config()

        if config.use_azure_storage:
            if not config.azure_storage_account:
                raise ValueError("AZURE_STORAGE_ACCOUNT required if USE_AZURE_STORAGE=true")

            storage = StorageFactory.create_azure_storage(
                account_name=config.azure_storage_account, container_name=config.azure_blob_container
            )
            return storage
        else:
            storage = StorageFactory.create_local_storage(base_path=str(self.path_manager.data_dir))
            return storage

    def _ensure_directories(self):
        """Create directories with robust paths using storage abstraction"""
        try:
            self.path_manager.ensure_directories(storage_service=self.storage)
            self.logger.debug("Directories created successfully")
        except Exception as e:
            self.logger.error(f"Error creating directories: {e}")

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
        """Save conversation with robust paths"""
        entry = ConversationEntry(
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            response_time_ms=response_time_ms,
            sources=sources,
            metadata=metadata,
        )

        today = datetime.now().strftime("%Y%m%d")
        file_path = self.path_manager.get_conversation_file(today)

        return self._append_entry_to_file(file_path, entry)

    def save_feedback(
        self, user_id: str, conversation_id: str, rating: int, comment: str = "", metadata: Optional[Dict] = None
    ) -> bool:
        """Save feedback with robust paths"""
        entry = FeedbackEntry(
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            conversation_id=conversation_id,
            rating=rating,
            comment=comment,
            metadata=metadata,
        )

        today = datetime.now().strftime("%Y%m%d")
        file_path = self.path_manager.get_feedback_file(today)

        try:
            success = self._append_entry_to_file(file_path, entry)
            if success:
                self.logger.info(
                    f"✅ Feedback saved successfully: user={user_id}, conv={conversation_id}, rating={rating}"
                )
            else:
                self.logger.error(
                    f"❌ Failed to save feedback: user={user_id}, conv={conversation_id}, rating={rating}"
                )
            return success
        except Exception as e:
            self.logger.error(f"❌ Exception saving feedback: {e}")
            return False

    def save_performance(
        self, operation: str, duration_ms: float, user_id: str, metadata: Optional[Dict] = None
    ) -> bool:
        """Save performance with robust paths"""
        entry = PerformanceEntry(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            duration_ms=duration_ms,
            user_id=user_id,
            metadata=metadata,
        )

        today = datetime.now().strftime("%Y%m%d")
        file_path = self.path_manager.get_performance_file(today)

        return self._append_entry_to_file(file_path, entry)

    def _append_entry_to_file(self, file_path: Path, entry) -> bool:
        """Simple method to append JSONL with robust paths"""
        try:
            data = asdict(entry)

            relative_path = file_path.relative_to(self.path_manager.data_dir)
            success = self.storage.append_jsonl_data(str(relative_path), data)

            if success:
                self.logger.debug(f"Entry saved: {relative_path}")
            else:
                self.logger.warning(f"Save failed: {relative_path}")

            return success

        except Exception as e:
            self.logger.error(f"Error saving {file_path}: {e}")
            return False

    def get_conversation_history(
        self, user_id: Optional[str] = None, conversation_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict]:
        """Retrieve conversation history with robust paths"""
        try:
            conversations_dir = self.path_manager.conversations_dir
            self.logger.info(f"Loading conversations from: {conversations_dir}")
            all_entries = self._load_entries_from_directory(conversations_dir, "conversations_")
            self.logger.info(f"Found {len(all_entries)} total conversation entries")

            if user_id:
                all_entries = [e for e in all_entries if e.get("user_id") == user_id]
                self.logger.info(f"Filtered to {len(all_entries)} entries for user: {user_id}")

            if conversation_id:
                all_entries = [e for e in all_entries if e.get("conversation_id") == conversation_id]
                self.logger.info(f"Filtered to {len(all_entries)} entries for conversation: {conversation_id}")

            all_entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            if limit > 0:
                all_entries = all_entries[:limit]

            self.logger.info(f"Returning {len(all_entries)} conversation entries")
            return all_entries

        except Exception as e:
            self.logger.error(f"Error retrieving conversations: {e}")
            return []

    def get_feedback_data(self, user_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Retrieve feedback data with robust paths"""
        try:
            feedback_dir = self.path_manager.feedback_dir
            self.logger.info(f"Loading feedback from: {feedback_dir}")
            all_entries = self._load_entries_from_directory(feedback_dir, "feedback_")
            self.logger.info(f"Found {len(all_entries)} total feedback entries")

            if user_id:
                all_entries = [e for e in all_entries if e.get("user_id") == user_id]
                self.logger.info(f"Filtered to {len(all_entries)} feedback entries for user: {user_id}")

            all_entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            if limit > 0:
                all_entries = all_entries[:limit]

            self.logger.info(f"Returning {len(all_entries)} feedback entries")
            return all_entries

        except Exception as e:
            self.logger.error(f"Error retrieving feedback: {e}")
            return []

    def get_performance_metrics(self, user_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Retrieve performance metrics with robust paths"""
        try:
            performance_dir = self.path_manager.performance_dir
            all_entries = self._load_entries_from_directory(performance_dir, "performance_")

            if user_id:
                all_entries = [e for e in all_entries if e.get("user_id") == user_id]

            all_entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            if limit > 0:
                all_entries = all_entries[:limit]

            return all_entries

        except Exception as e:
            self.logger.error(f"Error retrieving performance: {e}")
            return []

    def _load_entries_from_directory(self, directory: Path, file_prefix: str) -> List[Dict]:
        """Load entries from directory with robust paths"""
        entries = []

        try:
            self.logger.info(f"Checking directory exists: {directory}")

            if directory.exists():
                files = list(directory.glob(f"{file_prefix}*.jsonl"))
                self.logger.info(f"Found {len(files)} files matching pattern '{file_prefix}*.jsonl'")

                for file_path in files:
                    relative_path = file_path.relative_to(self.path_manager.data_dir)
                    self.logger.info(f"Processing file: {relative_path}")

                    if self.storage.file_exists(str(relative_path)):
                        content = self.storage.load_text_file(str(relative_path))
                        if content:
                            lines = content.split("\n")
                            valid_lines = 0
                            for line in lines:
                                if line.strip():
                                    try:
                                        data = json.loads(line.strip())
                                        entries.append(data)
                                        valid_lines += 1
                                    except json.JSONDecodeError:
                                        continue
                            self.logger.info(f"Loaded {valid_lines} entries from {relative_path}")
                        else:
                            self.logger.warning(f"File {relative_path} exists but content is empty")
                    else:
                        self.logger.warning(f"File {relative_path} does not exist in storage")
            else:
                self.logger.warning(f"Directory {directory} does not exist")

        except Exception as e:
            self.logger.error(f"Error loading from {directory}: {e}")

        self.logger.info(f"Total entries loaded: {len(entries)}")
        return entries

    def get_info(self) -> Dict[str, Any]:
        """Information about the data manager"""
        try:
            conv_count = len(self.get_conversation_history(limit=1000))
            feedback_count = len(self.get_feedback_data(limit=1000))
            perf_count = len(self.get_performance_metrics(limit=1000))

            return {
                "storage_type": type(self.storage).__name__,
                "data_dir": str(self.path_manager.data_dir),
                "conversations_count": conv_count,
                "feedback_count": feedback_count,
                "performance_count": perf_count,
                "directories": {
                    "conversations": str(self.path_manager.conversations_dir),
                    "feedback": str(self.path_manager.feedback_dir),
                    "performance": str(self.path_manager.performance_dir),
                },
            }
        except Exception as e:
            return {"error": str(e)}


_data_manager: Optional[DataManager] = None


def get_data_manager() -> Optional[DataManager]:
    """Get the global data manager instance"""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager


def reset_data_manager():
    """Reset for tests"""
    global _data_manager
    _data_manager = None
