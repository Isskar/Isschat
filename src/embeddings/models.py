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
    "intfloat/multilingual-e5-large": ModelInfo(
        name="intfloat/multilingual-e5-large",
        dimension=1024,
        max_sequence_length=512,
        description="",
        recommended_device="cpu",
    ),
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": ModelInfo(
        name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        dimension=768,
        max_sequence_length=512,
        description="",
        recommended_device="cpu",
    ),
    "sentence-transformers/all-MiniLM-L12-v2": ModelInfo(
        name="sentence-transformers/all-MiniLM-L12-v2",
        dimension=384,
        max_sequence_length=512,
        description="",
    ),
    "sentence-transformers/all-mpnet-base-v2": ModelInfo(
        name="sentence-transformers/all-mpnet-base-v2",
        dimension=768,
        max_sequence_length=512,
        description="",
    ),
    "sentence-transformers/all-MiniLM-L6-v2": ModelInfo(
        name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        max_sequence_length=512,
        description="",
    ),
    "intfloat/multilingual-e5-small": ModelInfo(
        name="intfloat/multilingual-e5-small",
        dimension=384,
        max_sequence_length=512,
        description="",
        recommended_device="cpu",
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
    from src.core.exceptions import EmbeddingModelNotSelectedError

    raise EmbeddingModelNotSelectedError(list(SUPPORTED_MODELS.keys()))


def validate_model_name(model_name: str) -> bool:
    """Valider qu'un modèle est supporté"""
    return model_name in SUPPORTED_MODELS
