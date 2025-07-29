import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from . import secrets


@dataclass
class IsschatConfig:
    data_dir: Path = Path("data")

    embeddings_model: str = "intfloat/multilingual-e5-small"
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

    # Semantic understanding configuration
    use_semantic_features: bool = True
    semantic_expansion_enabled: bool = True
    semantic_reranking_enabled: bool = True
    semantic_similarity_threshold: float = 0.7
    query_expansion_max_variations: int = 5
    intent_classification_enabled: bool = True

    # Query reformulation configuration
    force_reformulate_all_queries: bool = True

    # Source filtering configuration
    source_filtering_enabled: bool = True
    min_source_score_threshold: float = 0.3  # Réduit de 0.4 → 0.3
    min_source_relevance_threshold: float = 0.2  # Réduit de 0.3 → 0.2
    use_flexible_filtering: bool = True  # Enable flexible multi-criteria filtering

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
        """Load configuration from environment with robust defaults"""
        if os.path.exists(env_file):
            load_dotenv(env_file)

        # Create instance with defaults first
        defaults = cls()

        return cls(
            data_dir=Path(os.getenv("DATA_DIR", str(defaults.data_dir))),
            embeddings_model=os.getenv("EMBEDDINGS_MODEL", defaults.embeddings_model),
            embeddings_device=os.getenv("EMBEDDINGS_DEVICE", defaults.embeddings_device),
            embeddings_batch_size=int(os.getenv("EMBEDDINGS_BATCH_SIZE", str(defaults.embeddings_batch_size))),
            embeddings_normalize=os.getenv("EMBEDDINGS_NORMALIZE", str(defaults.embeddings_normalize)).lower()
            == "true",
            chunk_size=int(os.getenv("CHUNK_SIZE", str(defaults.chunk_size))),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", str(defaults.chunk_overlap))),
            min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", str(defaults.min_chunk_size))),
            vectordb_collection=os.getenv("VECTORDB_COLLECTION", defaults.vectordb_collection),
            vectordb_index_type=os.getenv("VECTORDB_INDEX_TYPE", defaults.vectordb_index_type),
            vectordb_host=os.getenv("VECTORDB_HOST", defaults.vectordb_host),
            vectordb_port=int(os.getenv("VECTORDB_PORT", str(defaults.vectordb_port))),
            llm_model=os.getenv("LLM_MODEL", defaults.llm_model),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", str(defaults.llm_temperature))),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", str(defaults.llm_max_tokens))),
            search_k=int(os.getenv("SEARCH_K", str(defaults.search_k))),
            search_fetch_k=int(os.getenv("SEARCH_FETCH_K", str(defaults.search_fetch_k))),
            use_semantic_features=os.getenv("USE_SEMANTIC_FEATURES", str(defaults.use_semantic_features)).lower()
            == "true",
            semantic_expansion_enabled=os.getenv(
                "SEMANTIC_EXPANSION_ENABLED", str(defaults.semantic_expansion_enabled)
            ).lower()
            == "true",
            semantic_reranking_enabled=os.getenv(
                "SEMANTIC_RERANKING_ENABLED", str(defaults.semantic_reranking_enabled)
            ).lower()
            == "true",
            semantic_similarity_threshold=float(
                os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", str(defaults.semantic_similarity_threshold))
            ),
            query_expansion_max_variations=int(
                os.getenv("QUERY_EXPANSION_MAX_VARIATIONS", str(defaults.query_expansion_max_variations))
            ),
            intent_classification_enabled=os.getenv(
                "INTENT_CLASSIFICATION_ENABLED", str(defaults.intent_classification_enabled)
            ).lower()
            == "true",
            force_reformulate_all_queries=os.getenv(
                "FORCE_REFORMULATE_ALL_QUERIES", str(defaults.force_reformulate_all_queries)
            ).lower()
            == "true",
            source_filtering_enabled=os.getenv(
                "SOURCE_FILTERING_ENABLED", str(defaults.source_filtering_enabled)
            ).lower()
            == "true",
            min_source_score_threshold=float(
                os.getenv("MIN_SOURCE_SCORE_THRESHOLD", str(defaults.min_source_score_threshold))
            ),
            min_source_relevance_threshold=float(
                os.getenv("MIN_SOURCE_RELEVANCE_THRESHOLD", str(defaults.min_source_relevance_threshold))
            ),
            use_flexible_filtering=os.getenv("USE_FLEXIBLE_FILTERING", str(defaults.use_flexible_filtering)).lower()
            == "true",
            confluence_api_key=secrets.get_confluence_api_key() or defaults.confluence_api_key,
            confluence_space_key=secrets.get_confluence_space_key() or defaults.confluence_space_key,
            confluence_space_name=secrets.get_confluence_space_name() or defaults.confluence_space_name,
            confluence_email=secrets.get_confluence_email() or defaults.confluence_email,
            openrouter_api_key=secrets.get_openrouter_api_key() or defaults.openrouter_api_key,
            use_azure_storage=os.getenv("USE_AZURE_STORAGE", str(defaults.use_azure_storage)).lower() == "true",
            azure_storage_account=secrets.get_azure_storage_account() or defaults.azure_storage_account,
            azure_blob_container=secrets.get_azure_blob_container() or defaults.azure_blob_container,
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
            "semantic_features": {
                "use_semantic_features": config.use_semantic_features,
                "semantic_expansion_enabled": config.semantic_expansion_enabled,
                "semantic_reranking_enabled": config.semantic_reranking_enabled,
                "semantic_similarity_threshold": config.semantic_similarity_threshold,
                "query_expansion_max_variations": config.query_expansion_max_variations,
                "intent_classification_enabled": config.intent_classification_enabled,
            },
        }
    except Exception as e:
        return {"error": str(e)}


def reset_config():
    """Reset pour tests"""
    global _config
    _config = None
