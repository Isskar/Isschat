"""
Interface abstraite pour l'extraction de données.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class Document:
    """Représentation d'un document extrait."""

    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        # Support both content and page_content for compatibility
        self.content = content
        self.page_content = content  # Alias for compatibility with interfaces
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        # Return both formats for maximum compatibility
        return {"content": self.content, "page_content": self.page_content, "metadata": self.metadata}


class BaseExtractor(ABC):
    """Interface pour l'extraction de données."""

    @abstractmethod
    def extract(self) -> List[Document]:
        """
        Extrait les documents de la source de données.

        Returns:
            List[Document]: Liste des documents extraits
        """
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Valide la connexion à la source de données.

        Returns:
            bool: True si la connexion est valide
        """
        pass
