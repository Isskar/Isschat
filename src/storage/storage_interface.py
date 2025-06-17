"""
Storage interface for abstracting different storage backends
"""

from abc import ABC, abstractmethod
from typing import List


class StorageInterface(ABC):
    """Abstract interface for storage operations"""

    @abstractmethod
    def read_file(self, file_path: str) -> bytes:
        """
        Read a file from storage

        Args:
            file_path: Path to the file

        Returns:
            File content as bytes

        Raises:
            Exception: If file cannot be read
        """
        pass

    @abstractmethod
    def write_file(self, file_path: str, data: bytes) -> bool:
        """
        Write a file to storage

        Args:
            file_path: Path where to store the file
            data: File content as bytes

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in storage

        Args:
            file_path: Path to check

        Returns:
            True if file exists, False otherwise
        """
        pass

    @abstractmethod
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from storage

        Args:
            file_path: Path to the file to delete

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def list_files(self, directory_path: str, pattern: str = "*") -> List[str]:
        """
        List files in a directory

        Args:
            directory_path: Directory to list
            pattern: File pattern to match

        Returns:
            List of file paths
        """
        pass
