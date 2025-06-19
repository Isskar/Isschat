import json
import logging
from typing import Dict, List, Optional

from src.storage import StorageInterface


class StorageService:
    """High-level storage service with business logic"""

    def __init__(self, storage: StorageInterface):
        self._storage = storage
        logging.debug(f"StorageService initialized with {type(storage).__name__}")

    def save_json_data(self, file_path: str, data: Dict) -> bool:
        try:
            json_data = json.dumps(data, indent=2, default=str).encode("utf-8")
            return self._storage.write_file(file_path, json_data)
        except Exception as e:
            logging.error(f"Error saving JSON data to {file_path}: {e}")
            return False

    def load_json_data(self, file_path: str) -> Dict:
        if not self._storage.file_exists(file_path):
            return {}

        try:
            data = self._storage.read_file(file_path)
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            logging.error(f"Error loading JSON data from {file_path}: {e}")
            return {}

    def append_jsonl_data(self, file_path: str, data: Dict) -> bool:
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
        try:
            data = content.encode("utf-8")
            return self._storage.write_file(file_path, data)
        except Exception as e:
            logging.error(f"Error saving text file {file_path}: {e}")
            return False

    def load_text_file(self, file_path: str) -> Optional[str]:
        if not self._storage.file_exists(file_path):
            return None

        try:
            data = self._storage.read_file(file_path)
            return data.decode("utf-8")
        except Exception as e:
            logging.error(f"Error loading text file {file_path}: {e}")
            return None

    def copy_file(self, source_path: str, dest_path: str) -> bool:
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
        try:
            return self._storage.list_files(directory_path, pattern)
        except Exception as e:
            logging.error(f"Error listing files in {directory_path}: {e}")
            return []

    def file_exists(self, file_path: str) -> bool:
        return self._storage.file_exists(file_path)

    def delete_file(self, file_path: str) -> bool:
        return self._storage.delete_file(file_path)

    def read_binary_file(self, file_path: str) -> Optional[bytes]:
        if not self._storage.file_exists(file_path):
            return None

        try:
            return self._storage.read_file(file_path)
        except Exception as e:
            logging.error(f"Error reading binary file {file_path}: {e}")
            return None

    def write_binary_file(self, file_path: str, data: bytes) -> bool:
        return self._storage.write_file(file_path, data)
