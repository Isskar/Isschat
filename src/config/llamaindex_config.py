"""
Configuration settings for LlamaIndex integration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LlamaIndexConfig:
    """Configuration for LlamaIndex RAG pipeline"""

    # Pipeline type selection
    pipeline_type: str = "hyde"  # Options: "hyde", "decompose", "hybrid", "simple"

    # Query transformation settings
    use_hyde: bool = True
    use_decompose: bool = False
    include_original_query: bool = True

    # Memory settings - adaptive token management
    memory_token_limit: Optional[int] = None  # Auto-calculated from LLM max tokens
    memory_enabled: bool = True
    adaptive_memory: bool = True  # Enable adaptive token limit adjustment
    memory_reserve_ratio: float = 0.3  # Reserve 30% of max tokens for memory by default
    min_memory_tokens: int = 1000  # Minimum tokens reserved for memory
    max_memory_tokens: int = 8000  # Maximum tokens reserved for memory

    # HyDE specific settings
    hyde_prompt_template: Optional[str] = None  # Use LlamaIndex default

    # Query decomposition settings
    decompose_max_questions: int = 3
    decompose_verbose: bool = False

    # Retrieval settings
    retrieval_top_k: int = 5
    retrieval_similarity_threshold: float = 0.7

    # Response generation settings
    response_mode: str = "compact"  # Options: "compact", "tree_summarize", "accumulate"
    streaming: bool = False

    def __post_init__(self):
        """Validate configuration"""
        valid_pipeline_types = ["hyde", "decompose", "hybrid", "simple"]
        if self.pipeline_type not in valid_pipeline_types:
            raise ValueError(f"Pipeline type must be one of {valid_pipeline_types}")

        # Auto-configure based on pipeline type
        if self.pipeline_type == "hyde":
            self.use_hyde = True
            self.use_decompose = False
        elif self.pipeline_type == "decompose":
            self.use_hyde = False
            self.use_decompose = True
        elif self.pipeline_type == "hybrid":
            self.use_hyde = True
            self.use_decompose = True
        elif self.pipeline_type == "simple":
            self.use_hyde = False
            self.use_decompose = False


def get_llamaindex_config() -> LlamaIndexConfig:
    """Get LlamaIndex configuration with environment variable overrides"""
    import os

    config = LlamaIndexConfig()

    # Override from environment variables
    if os.getenv("LLAMAINDEX_PIPELINE_TYPE"):
        config.pipeline_type = os.getenv("LLAMAINDEX_PIPELINE_TYPE")

    if os.getenv("LLAMAINDEX_USE_HYDE"):
        config.use_hyde = os.getenv("LLAMAINDEX_USE_HYDE").lower() == "true"

    if os.getenv("LLAMAINDEX_USE_DECOMPOSE"):
        config.use_decompose = os.getenv("LLAMAINDEX_USE_DECOMPOSE").lower() == "true"

    if os.getenv("LLAMAINDEX_MEMORY_ENABLED"):
        config.memory_enabled = os.getenv("LLAMAINDEX_MEMORY_ENABLED").lower() == "true"

    if os.getenv("LLAMAINDEX_RETRIEVAL_TOP_K"):
        config.retrieval_top_k = int(os.getenv("LLAMAINDEX_RETRIEVAL_TOP_K"))

    if os.getenv("LLAMAINDEX_SIMILARITY_THRESHOLD"):
        config.retrieval_similarity_threshold = float(os.getenv("LLAMAINDEX_SIMILARITY_THRESHOLD"))

    if os.getenv("LLAMAINDEX_ADAPTIVE_MEMORY"):
        config.adaptive_memory = os.getenv("LLAMAINDEX_ADAPTIVE_MEMORY").lower() == "true"

    if os.getenv("LLAMAINDEX_MEMORY_RESERVE_RATIO"):
        config.memory_reserve_ratio = float(os.getenv("LLAMAINDEX_MEMORY_RESERVE_RATIO"))

    # Re-run validation after environment overrides
    config.__post_init__()

    return config
