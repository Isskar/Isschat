"""
Gestionnaire de chemins robustes pour Isschat.
Utilise pathlib pour des chemins cross-platform et robustes.
"""

from pathlib import Path
from typing import Optional
from .settings import IsschatConfig


class PathManager:
    """Gestionnaire de chemins robustes pour tout le projet"""

    def __init__(self, config: IsschatConfig):
        self.config = config
        # Chemin racine du projet (remonte de src/config vers la racine)
        self.project_root = Path(__file__).parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """Chemin data absolu et robuste"""
        if self.config.data_dir.is_absolute():
            return self.config.data_dir
        return self.project_root / self.config.data_dir

    @property
    def logs_dir(self) -> Path:
        """Chemin logs robuste"""
        return self.data_dir / "logs"

    @property
    def conversations_dir(self) -> Path:
        """Chemin conversations robuste"""
        return self.logs_dir / "conversations"

    @property
    def feedback_dir(self) -> Path:
        """Chemin feedback robuste"""
        return self.logs_dir / "feedback"

    @property
    def performance_dir(self) -> Path:
        """Chemin performance robuste"""
        return self.logs_dir / "performance"

    @property
    def raw_data_dir(self) -> Path:
        """Chemin données brutes robuste"""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Chemin données traitées robuste"""
        return self.data_dir / "processed"

    def ensure_directories(self) -> None:
        """Créer tous les répertoires nécessaires"""
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

    def get_conversation_file(self, date_str: str) -> Path:
        """Chemin fichier conversation pour une date"""
        return self.conversations_dir / f"conversations_{date_str}.jsonl"

    def get_feedback_file(self, date_str: str) -> Path:
        """Chemin fichier feedback pour une date"""
        return self.feedback_dir / f"feedback_{date_str}.jsonl"

    def get_performance_file(self, date_str: str) -> Path:
        """Chemin fichier performance pour une date"""
        return self.performance_dir / f"performance_{date_str}.jsonl"


# Instance globale
_path_manager: Optional[PathManager] = None


def get_path_manager() -> PathManager:
    """Obtenir l'instance globale du gestionnaire de chemins"""
    global _path_manager
    if _path_manager is None:
        from .settings import get_config

        _path_manager = PathManager(get_config())
    return _path_manager


def reset_path_manager():
    """Reset pour tests"""
    global _path_manager
    _path_manager = None
