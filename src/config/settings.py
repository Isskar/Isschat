import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


@dataclass
class IsschatConfig:
    data_dir: Path = Path("data")

    embeddings_model: str = "intfloat/multilingual-e5-large"
    embeddings_device: str = "cpu"
    embeddings_batch_size: int = 32
    embeddings_normalize: bool = True

    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100

    vectordb_collection: str = "isschat_docs"
    vectordb_index_type: str = "hnsw"
    vectordb_host: str = "localhost"
    vectordb_port: int = 8080

    llm_model: str = "google/gemini-2.5-flash-lite-preview-06-17"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 512
    search_k: int = 3
    search_fetch_k: int = 5

    confluence_api_key: str = ""
    confluence_space_key: str = ""
    confluence_space_name: str = ""
    confluence_email: str = ""
    openrouter_api_key: str = ""

    use_azure_storage: bool = False
    azure_storage_account: str = ""
    azure_blob_container: str = ""

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "IsschatConfig":
        """Charger depuis env avec chemins robustes"""
        if os.path.exists(env_file):
            load_dotenv(env_file)

        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            embeddings_model=os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large"),
            embeddings_device=os.getenv("EMBEDDINGS_DEVICE", "cpu"),
            embeddings_batch_size=int(os.getenv("EMBEDDINGS_BATCH_SIZE", "32")),
            embeddings_normalize=os.getenv("EMBEDDINGS_NORMALIZE", "true").lower() == "true",
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "100")),
            vectordb_collection=os.getenv("VECTORDB_COLLECTION", "isschat_docs"),
            vectordb_index_type=os.getenv("VECTORDB_INDEX_TYPE", "hnsw"),
            vectordb_host=os.getenv("VECTORDB_HOST", "localhost"),
            vectordb_port=int(os.getenv("VECTORDB_PORT", "8080")),
            llm_model=os.getenv("LLM_MODEL", "google/gemini-2.5-flash-lite-preview-06-17"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
            search_k=int(os.getenv("SEARCH_K", "3")),
            search_fetch_k=int(os.getenv("SEARCH_FETCH_K", "5")),
            confluence_api_key=os.getenv("CONFLUENCE_PRIVATE_API_KEY", ""),
            confluence_space_key=os.getenv("CONFLUENCE_SPACE_KEY", ""),
            confluence_space_name=os.getenv("CONFLUENCE_SPACE_NAME", ""),
            confluence_email=os.getenv("CONFLUENCE_EMAIL_ADDRESS", ""),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            use_azure_storage=os.getenv("USE_AZURE_STORAGE", "false").lower() == "true",
            azure_storage_account=os.getenv("AZURE_STORAGE_ACCOUNT", ""),
            azure_blob_container=os.getenv("AZURE_BLOB_CONTAINER_NAME", ""),
        )

    def validate(self) -> bool:
        """Valider la configuration"""
        if not self.data_dir:
            raise ValueError("data_dir est requis")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size doit être positif")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap doit être < chunk_size")

        if self.vectordb_index_type not in ["hnsw", "flat"]:
            raise ValueError("vectordb_index_type doit être 'hnsw' ou 'flat'")

        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY est requis")

        return True

    @property
    def confluence_url(self) -> str:
        """URL Confluence complète"""
        return f"{self.confluence_space_name}/wiki"


_config: Optional[IsschatConfig] = None


def get_config() -> Optional[IsschatConfig]:
    """Obtenir l'instance globale de configuration"""
    global _config
    if _config is None:
        _config = IsschatConfig.from_env()
        _config.validate()
    return _config


def get_debug_info() -> dict:
    """Obtenir informations de debug sur la configuration (compatibilité)"""
    try:
        config = get_config()
        return {
            "embeddings_model": config.embeddings_model,
            "llm_model": config.llm_model,
            "vectordb_collection": config.vectordb_collection,
            "chunk_size": config.chunk_size,
            "search_k": config.search_k,
            "confluence_api_key_configured": bool(config.confluence_api_key),
            "openrouter_api_key_configured": bool(config.openrouter_api_key),
            "use_azure_storage": config.use_azure_storage,
            "data_dir": str(config.data_dir),
        }
    except Exception as e:
        return {"error": str(e)}


def reset_config():
    """Reset pour tests"""
    global _config
    _config = None
