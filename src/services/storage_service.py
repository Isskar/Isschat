"""
Storage Service for high-level storage operations
"""

import json
import logging
from typing import Dict, List, Optional

from ..storage.storage_interface import StorageInterface


class StorageService:
    """High-level storage service with business logic"""

    def __init__(self, storage: StorageInterface):
        """
        Initialize storage service with dependency injection

        Args:
            storage: Storage implementation to use
        """
        self._storage = storage
        logging.debug(f"StorageService initialized with {type(storage).__name__}")

    def save_json_data(self, file_path: str, data: Dict) -> bool:
        """
        Save data as JSON file

        Args:
            file_path: Path to save the JSON file
            data: Data to save

        Returns:
            True if successful, False otherwise
        """
        try:
            json_data = json.dumps(data, indent=2, default=str).encode("utf-8")
            return self._storage.write_file(file_path, json_data)
        except Exception as e:
            logging.error(f"Error saving JSON data to {file_path}: {e}")
            return False

    def load_json_data(self, file_path: str) -> Dict:
        """
        Load data from JSON file

        Args:
            file_path: Path to the JSON file

        Returns:
            Loaded data or empty dict if error
        """
        if not self._storage.file_exists(file_path):
            return {}

        try:
            data = self._storage.read_file(file_path)
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            logging.error(f"Error loading JSON data from {file_path}: {e}")
            return {}

    def append_jsonl_data(self, file_path: str, data: Dict) -> bool:
        """
        Append a line to a JSONL file

        Args:
            file_path: Path to the JSONL file
            data: Data to append

        Returns:
            True if successful, False otherwise
        """
        try:
            line = json.dumps(data, default=str) + "\n"

            # Read existing data if file exists
            existing_data = b""
            if self._storage.file_exists(file_path):
                existing_data = self._storage.read_file(file_path)

            # Append new line
            new_data = existing_data + line.encode("utf-8")
            return self._storage.write_file(file_path, new_data)
        except Exception as e:
            logging.error(f"Error appending JSONL data to {file_path}: {e}")
            return False

    def save_text_file(self, file_path: str, content: str) -> bool:
        """
        Save text content to file

        Args:
            file_path: Path to save the file
            content: Text content to save

        Returns:
            True if successful, False otherwise
        """
        try:
            data = content.encode("utf-8")
            return self._storage.write_file(file_path, data)
        except Exception as e:
            logging.error(f"Error saving text file {file_path}: {e}")
            return False

    def load_text_file(self, file_path: str) -> Optional[str]:
        """
        Load text content from file

        Args:
            file_path: Path to the file

        Returns:
            Text content or None if error
        """
        if not self._storage.file_exists(file_path):
            return None

        try:
            data = self._storage.read_file(file_path)
            return data.decode("utf-8")
        except Exception as e:
            logging.error(f"Error loading text file {file_path}: {e}")
            return None

    def copy_file(self, source_path: str, dest_path: str) -> bool:
        """
        Copy a file within the same storage

        Args:
            source_path: Source file path
            dest_path: Destination file path

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._storage.file_exists(source_path):
                logging.error(f"Source file does not exist: {source_path}")
                return False

            data = self._storage.read_file(source_path)
            return self._storage.write_file(dest_path, data)
        except Exception as e:
            logging.error(f"Error copying file {source_path} to {dest_path}: {e}")
            return False

    def get_file_list(self, directory_path: str, pattern: str = "*") -> List[str]:
        """
        Get list of files in directory

        Args:
            directory_path: Directory to list
            pattern: File pattern to match

        Returns:
            List of file paths
        """
        try:
            return self._storage.list_files(directory_path, pattern)
        except Exception as e:
            logging.error(f"Error listing files in {directory_path}: {e}")
            return []

    def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists

        Args:
            file_path: Path to check

        Returns:
            True if file exists, False otherwise
        """
        return self._storage.file_exists(file_path)

    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file

        Args:
            file_path: Path to the file to delete

        Returns:
            True if successful, False otherwise
        """
        return self._storage.delete_file(file_path)

    def read_binary_file(self, file_path: str) -> Optional[bytes]:
        """
        Read binary file content

        Args:
            file_path: Path to the file

        Returns:
            Binary content or None if error
        """
        if not self._storage.file_exists(file_path):
            return None

        try:
            return self._storage.read_file(file_path)
        except Exception as e:
            logging.error(f"Error reading binary file {file_path}: {e}")
            return None

    def write_binary_file(self, file_path: str, data: bytes) -> bool:
        """
        Write binary data to file

        Args:
            file_path: Path to save the file
            data: Binary data to save

        Returns:
            True if successful, False otherwise
        """
        return self._storage.write_file(file_path, data)
