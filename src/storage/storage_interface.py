from abc import ABC, abstractmethod
from typing import List


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
