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
    """Data store using JSONL format."""

    def __init__(self, file_path: Path, entry_class: type):
        self.file_path = file_path
        self.entry_class = entry_class
        self._ensure_directory()

    def _ensure_directory(self):
        """Ensure that the directory exists."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def save_entry(self, entry: Any) -> bool:
        """Save an entry in JSONL format."""
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                if isinstance(entry, dict):
                    json.dump(entry, f, ensure_ascii=False)
                else:
                    json.dump(asdict(entry), f, ensure_ascii=False)
                f.write("\n")
            return True
        except Exception as e:
            logging.error(f"Error saving to {self.file_path}: {e}")
            return False

    def load_entries(self, limit: Optional[int] = None) -> List[Any]:
        """Load entries from JSONL file."""
        entries = []
        if not self.file_path.exists():
            return entries

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        entries.append(data)

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
    """Gestionnaire principal des données avec la nouvelle structure."""

    def __init__(self, base_data_dir: Optional[Path] = None):
        """
        Initialise le gestionnaire de données.

        Args:
            base_data_dir: Répertoire de base pour les données. Si None, utilise la config.
        """
        if base_data_dir is None:
            # Utiliser le répertoire du projet comme base
            project_root = Path(__file__).parent.parent.parent
            self.base_data_dir = project_root / "data"
        else:
            self.base_data_dir = Path(base_data_dir)

        # Structure des répertoires selon votre plan
        self.raw_dir = self.base_data_dir / "raw"
        self.processed_dir = self.base_data_dir / "processed"
        self.vector_db_dir = self.base_data_dir / "vector_db"

        # Répertoires pour les logs structurés
        self.logs_dir = self.base_data_dir / "logs"
        self.conversations_dir = self.logs_dir / "conversations"
        self.performance_dir = self.logs_dir / "performance"
        self.feedback_dir = self.logs_dir / "feedback"

        self._create_directories()
        self._init_stores()

    def _create_directories(self):
        """Crée la structure de répertoires."""
        directories = [
            self.raw_dir,
            self.processed_dir,
            self.vector_db_dir,
            self.conversations_dir,
            self.performance_dir,
            self.feedback_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _init_stores(self):
        """Initialise les stores de données."""
        today = datetime.now().strftime("%Y%m%d")

        # Stores pour les différents types de données
        self.conversation_store = JSONLDataStore(
            self.conversations_dir / f"conversations_{today}.jsonl", ConversationEntry
        )

        self.performance_store = JSONLDataStore(self.performance_dir / f"performance_{today}.jsonl", PerformanceEntry)

        self.feedback_store = JSONLDataStore(self.feedback_dir / f"feedback_{today}.jsonl", FeedbackEntry)

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
        """Sauvegarde une conversation."""
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
        """Sauvegarde un feedback utilisateur."""
        entry = FeedbackEntry(
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            conversation_id=conversation_id,
            rating=rating,
            comment=comment,
            metadata=metadata,
        )

        return self.feedback_store.save_entry(entry)

    def get_conversation_history(
        self, user_id: Optional[str] = None, conversation_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict]:
        """Récupère l'historique des conversations, éventuellement filtré par user_id ou conversation_id."""
        all_entries = []

        # List all conversation files (e.g., conversations_20231026.jsonl)
        conversation_files = sorted(self.conversations_dir.glob("conversations_*.jsonl"))

        for file_path in conversation_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if line.strip():
                            try:
                                entry = json.loads(line.strip())
                                all_entries.append(entry)
                            except json.JSONDecodeError as e:
                                logging.error(f"Error decoding JSON from {file_path}, line {i + 1}: {e}")
            except Exception as e:
                logging.error(f"Error reading conversation file {file_path}: {e}")

        # Filter by user_id if provided
        if user_id:
            filtered_entries = []
            for entry in all_entries:
                if entry.get("user_id") == user_id:
                    filtered_entries.append(entry)
            all_entries = filtered_entries

        # Filter by conversation_id if provided
        if conversation_id:
            filtered_entries = []
            for entry in all_entries:
                if entry.get("conversation_id") == conversation_id:
                    filtered_entries.append(entry)
            all_entries = filtered_entries

        # Sort by timestamp (most recent first)
        all_entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Apply limit
        if limit is not None and limit > 0:
            all_entries = all_entries[:limit]

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

    def migrate_legacy_logs(self, legacy_logs_dir: Path) -> Dict[str, int]:
        """
        Migre les anciens logs vers la nouvelle structure.

        Args:
            legacy_logs_dir: Répertoire contenant les anciens logs

        Returns:
            Dictionnaire avec le nombre d'entrées migrées par type
        """
        migration_stats = {"conversations": 0, "performance": 0, "feedback": 0, "errors": 0}

        # Migration des conversations
        conv_dir = legacy_logs_dir / "conversations"
        if conv_dir.exists():
            for conv_file in conv_dir.glob("*.jsonl"):
                try:
                    with open(conv_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line.strip())
                                # Adapter le format si nécessaire
                                if self.conversation_store.save_entry(data):
                                    migration_stats["conversations"] += 1
                except Exception as e:
                    logging.error(f"Erreur migration conversation {conv_file}: {e}")
                    migration_stats["errors"] += 1

        # Migration des performances
        perf_dir = legacy_logs_dir / "performance"
        if perf_dir.exists():
            for perf_file in perf_dir.glob("*.jsonl"):
                try:
                    with open(perf_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line.strip())
                                if self.performance_store.save_entry(data):
                                    migration_stats["performance"] += 1
                except Exception as e:
                    logging.error(f"Erreur migration performance {perf_file}: {e}")
                    migration_stats["errors"] += 1

        # Migration du feedback
        feedback_dir = legacy_logs_dir / "feedback"
        if feedback_dir.exists():
            for feedback_file in feedback_dir.glob("*.json"):
                try:
                    with open(feedback_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if self.feedback_store.save_entry(item):
                                    migration_stats["feedback"] += 1
                        else:
                            if self.feedback_store.save_entry(data):
                                migration_stats["feedback"] += 1
                except Exception as e:
                    logging.error(f"Erreur migration feedback {feedback_file}: {e}")
                    migration_stats["errors"] += 1

        return migration_stats

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
