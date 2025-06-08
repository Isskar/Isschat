"""
Filtrage de qualité des documents.
"""

from typing import List, Dict, Any, Optional

# Use absolute imports with fallbacks
try:
    from data_pipeline.extractors.base_extractor import Document
except ImportError:
    try:
        from src.data_pipeline.extractors.base_extractor import Document
    except ImportError:
        from ..extractors.base_extractor import Document


class DocumentFilter:
    """Filtre les documents selon des critères de qualité."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise le filtre de documents.

        Args:
            config: Configuration du filtrage
        """
        self.config = config or {}
        self.min_length = self.config.get("min_length", 50)
        self.max_length = self.config.get("max_length", 10000)
        self.excluded_patterns = self.config.get("excluded_patterns", [])

    def filter_documents(self, documents: List[Document]) -> List[Document]:
        """
        Filtre les documents selon les critères de qualité.

        Args:
            documents: Liste des documents à filtrer

        Returns:
            List[Document]: Documents filtrés
        """
        filtered_docs = []

        for doc in documents:
            if self._is_valid_document(doc):
                filtered_docs.append(doc)

        return filtered_docs

    def _is_valid_document(self, document: Document) -> bool:
        """
        Vérifie si un document respecte les critères de qualité.

        Args:
            document: Document à vérifier

        Returns:
            bool: True si le document est valide
        """
        content = document.content.strip()

        # Vérifier la longueur
        if len(content) < self.min_length or len(content) > self.max_length:
            return False

        # Vérifier les patterns exclus
        for pattern in self.excluded_patterns:
            if pattern.lower() in content.lower():
                return False

        # Vérifier que le contenu n'est pas vide ou uniquement des espaces
        if not content or content.isspace():
            return False

        return True

    def get_filter_stats(self, original_docs: List[Document], filtered_docs: List[Document]) -> Dict[str, Any]:
        """
        Retourne les statistiques de filtrage.

        Args:
            original_docs: Documents originaux
            filtered_docs: Documents filtrés

        Returns:
            Dict: Statistiques de filtrage
        """
        return {
            "original_count": len(original_docs),
            "filtered_count": len(filtered_docs),
            "removed_count": len(original_docs) - len(filtered_docs),
            "retention_rate": len(filtered_docs) / len(original_docs) if original_docs else 0,
        }
