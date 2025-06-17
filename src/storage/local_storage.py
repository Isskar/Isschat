"""
Local file system storage implementation
"""

import os
import logging
from pathlib import Path
from typing import List

from .storage_interface import StorageInterface


class LocalStorage(StorageInterface):
    """Local file system storage implementation"""

    def __init__(self, base_path: str = "."):
        """
        Initialize local storage

        Args:
            base_path: Base directory for storage operations
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logging.debug(f"LocalStorage initialized with base_path: {self.base_path}")

    def _get_full_path(self, file_path: str) -> Path:
        """Get full path for a file"""
        if os.path.isabs(file_path):
            return Path(file_path)
        return self.base_path / file_path

    def read_file(self, file_path: str) -> bytes:
        """Read a file from local storage"""
        full_path = self._get_full_path(file_path)
        try:
            with open(full_path, "rb") as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error reading file {full_path}: {e}")
            raise

    def write_file(self, file_path: str, data: bytes) -> bool:
        """Write a file to local storage"""
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
        """Check if a file exists in local storage"""
        full_path = self._get_full_path(file_path)
        return full_path.exists()

    def delete_file(self, file_path: str) -> bool:
        """Delete a file from local storage"""
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
        """List files in a directory in local storage"""
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
