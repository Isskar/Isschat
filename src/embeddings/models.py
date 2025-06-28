"""
Modèles d'embedding supportés avec leurs métadonnées.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelInfo:
    """Informations sur un modèle d'embedding"""

    name: str
    dimension: int
    max_sequence_length: int
    description: str
    recommended_device: str = "cpu"


# Modèles supportés avec leurs dimensions
SUPPORTED_MODELS: Dict[str, ModelInfo] = {
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": ModelInfo(
        name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        dimension=768,
        max_sequence_length=512,
        description="Modèle multilingue haute qualité, recommandé pour Isschat",
        recommended_device="cpu",
    ),
    "sentence-transformers/all-MiniLM-L12-v2": ModelInfo(
        name="sentence-transformers/all-MiniLM-L12-v2",
        dimension=384,
        max_sequence_length=512,
        description="Modèle compact et rapide, bon compromis performance/taille",
    ),
    "sentence-transformers/all-mpnet-base-v2": ModelInfo(
        name="sentence-transformers/all-mpnet-base-v2",
        dimension=768,
        max_sequence_length=512,
        description="Modèle anglais haute performance",
    ),
    "sentence-transformers/all-MiniLM-L6-v2": ModelInfo(
        name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        max_sequence_length=512,
        description="Modèle très léger pour des tests rapides",
    ),
}


def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """Obtenir les informations d'un modèle"""
    return SUPPORTED_MODELS.get(model_name)


def get_model_dimension(model_name: str) -> int:
    """Obtenir la dimension d'un modèle"""
    model_info = get_model_info(model_name)
    if model_info:
        return model_info.dimension

    # Fallback pour modèles non référencés
    # Essayer de déduire depuis le nom
    if "384" in model_name or "MiniLM" in model_name:
        return 384
    elif "768" in model_name or "mpnet" in model_name or "multilingual" in model_name:
        return 768
    else:
        # Valeur par défaut conservative
        return 384


def validate_model_name(model_name: str) -> bool:
    """Valider qu'un modèle est supporté"""
    return model_name in SUPPORTED_MODELS
