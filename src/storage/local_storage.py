import os
import logging
from pathlib import Path
from typing import List

from src.storage.storage_interface import StorageInterface


class LocalStorage(StorageInterface):
    """Local file system storage implementation"""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logging.debug(f"LocalStorage initialized with base_path: {self.base_path}")

    def _get_full_path(self, file_path: str) -> Path:
        if os.path.isabs(file_path):
            return Path(file_path)
        return self.base_path / file_path

    def read_file(self, file_path: str) -> bytes:
        full_path = self._get_full_path(file_path)
        try:
            with open(full_path, "rb") as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error reading file {full_path}: {e}")
            raise

    def write_file(self, file_path: str, data: bytes) -> bool:
        full_path = self._get_full_path(file_path)
        try:
            # Create directory if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            with open(full_path, "wb") as f:
                f.write(data)

            logging.debug(f"Successfully wrote file: {full_path}")
            return True
        except Exception as e:
            logging.error(f"Error writing file {full_path}: {e}")
            return False

    def file_exists(self, file_path: str) -> bool:
        full_path = self._get_full_path(file_path)
        return full_path.exists()

    def delete_file(self, file_path: str) -> bool:
        full_path = self._get_full_path(file_path)
        try:
            if full_path.exists():
                full_path.unlink()
                logging.debug(f"Successfully deleted file: {full_path}")
            return True
        except Exception as e:
            logging.error(f"Error deleting file {full_path}: {e}")
            return False

    def list_files(self, directory_path: str, pattern: str = "*") -> List[str]:
        full_path = self._get_full_path(directory_path)
        try:
            if not full_path.exists():
                return []

            # Get all matching files
            files = list(full_path.glob(pattern))

            # Return relative paths from base_path
            return [str(f.relative_to(self.base_path)) for f in files if f.is_file()]
        except Exception as e:
            logging.error(f"Error listing files in {full_path}: {e}")
            return []

    def create_directory(self, directory_path: str) -> bool:
        full_path = self._get_full_path(directory_path)
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Successfully created directory: {full_path}")
            return True
        except Exception as e:
            logging.error(f"Error creating directory {full_path}: {e}")
            return False

    def directory_exists(self, directory_path: str) -> bool:
        full_path = self._get_full_path(directory_path)
        return full_path.exists() and full_path.is_dir()
