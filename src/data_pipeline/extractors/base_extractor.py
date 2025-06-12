"""
Abstract interface for data extraction.
"""

from abc import ABC, abstractmethod
from typing import List
from src.core.interfaces import Document


class BaseExtractor(ABC):
    """Interface for data extraction."""

    @abstractmethod
    def extract(self) -> List[Document]:
        """
        Extract documents from the data source.

        Returns:
            List[Document]: List of extracted documents
        """
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Validate connection to the data source.

        Returns:
            bool: True if connection is valid
        """
        pass
