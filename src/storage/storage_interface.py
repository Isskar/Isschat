from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json


class StorageInterface(ABC):
    """Abstract interface for storage operations"""

    @abstractmethod
    def read_file(self, file_path: str) -> bytes:
        pass

    @abstractmethod
    def write_file(self, file_path: str, data: bytes) -> bool:
        pass

    @abstractmethod
    def file_exists(self, file_path: str) -> bool:
        pass

    @abstractmethod
    def delete_file(self, file_path: str) -> bool:
        pass

    @abstractmethod
    def list_files(self, directory_path: str, pattern: str = "*") -> List[str]:
        pass

    @abstractmethod
    def create_directory(self, directory_path: str) -> bool:
        pass

    @abstractmethod
    def directory_exists(self, directory_path: str) -> bool:
        pass

    def load_text_file(self, file_path: str) -> Optional[str]:
        """Load text file content"""
        try:
            data = self.read_file(file_path)
            return data.decode("utf-8")
        except Exception:
            return None

    def append_jsonl_data(self, file_path: str, data: Dict[str, Any]) -> bool:
        """Append JSON data as a line to a JSONL file"""
        try:
            # Read existing content if file exists
            existing_content = ""
            if self.file_exists(file_path):
                existing_data = self.read_file(file_path)
                existing_content = existing_data.decode("utf-8")

            # Append new data
            json_line = json.dumps(data, ensure_ascii=False) + "\n"
            new_content = existing_content + json_line

            # Write back
            return self.write_file(file_path, new_content.encode("utf-8"))
        except Exception:
            return False
