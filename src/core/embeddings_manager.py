"""
Centralized embeddings manager.
Avoids duplication of embedding model configuration throughout the code.
"""

from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from src.core.config import get_config
from src.core.exceptions import ConfigurationError


class EmbeddingsManager:
    """Centralized embeddings manager - Singleton pattern"""

    _instance: Optional[HuggingFaceEmbeddings] = None
    _config_hash: Optional[str] = None

    @classmethod
    def get_embeddings(cls, config=None) -> HuggingFaceEmbeddings:
        """
        Get embeddings instance - Singleton to avoid reloading the model.

        Args:
            config: Optional configuration (uses get_config() by default)

        Returns:
            Configured HuggingFaceEmbeddings instance
        """
        config = config or get_config()

        # Create a simple hash of the config to detect changes
        config_hash = f"{config.embeddings_model}_{config.embeddings_device}_{config.embeddings_batch_size}_{config.embeddings_normalize}_{config.embeddings_trust_remote_code}"  # noqa

        # Recreate instance if config changed or doesn't exist
        if cls._instance is None or cls._config_hash != config_hash:
            try:
                cls._instance = HuggingFaceEmbeddings(
                    model_name=config.embeddings_model,
                    model_kwargs={
                        "device": config.embeddings_device,
                        "trust_remote_code": config.embeddings_trust_remote_code,
                    },
                    encode_kwargs={
                        "normalize_embeddings": config.embeddings_normalize,
                        "batch_size": config.embeddings_batch_size,
                    },
                )
                cls._config_hash = config_hash
                print(f"✅ Embeddings loaded: {config.embeddings_model} on {config.embeddings_device}")

            except Exception as e:
                raise ConfigurationError(f"Failed to initialize embeddings: {str(e)}")

        return cls._instance

    @classmethod
    def reset(cls):
        """Reset embeddings instance - useful for tests"""
        cls._instance = None
        cls._config_hash = None

    @classmethod
    def get_model_info(cls) -> dict:
        """Get information about the current embeddings model"""
        if cls._instance is None:
            return {"status": "not_initialized"}

        config = get_config()
        return {
            "status": "initialized",
            "model_name": config.embeddings_model,
            "device": config.embeddings_device,
            "batch_size": config.embeddings_batch_size,
            "normalize": config.embeddings_normalize,
            "config_hash": cls._config_hash,
        }
