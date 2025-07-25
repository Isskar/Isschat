"""
LlamaIndex-based RAG pipeline with optimized query processing and memory management.
Replaces the custom semantic pipeline with proven LlamaIndex components.
"""

import logging
import time
from typing import Tuple, Dict, Any, Optional, List

from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_pipeline import QueryPipeline

# Query Transforms
from llama_index.core.indices.query.query_transform import HyDEQueryTransform, DecomposeQueryTransform

# Memory
from llama_index.core.memory import ChatMemoryBuffer

# LLM
from llama_index.llms.openrouter import OpenRouter

from ..config import get_config
from ..storage.data_manager import get_data_manager
from ..vectordb import VectorDBFactory


class LlamaIndexRAGPipeline:
    """
    Enhanced RAG pipeline using LlamaIndex components.
    Provides query transformation, memory management, and optimized retrieval.
    """

    def __init__(self, use_hyde: bool = True, use_decompose: bool = False):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_manager = get_data_manager()

        # Configuration
        self.use_hyde = use_hyde
        self.use_decompose = use_decompose

        # Initialize LlamaIndex Settings
        self._setup_llama_index()

        # Initialize components
        self._setup_vector_store()
        self._setup_memory()
        self._setup_query_transforms()
        self._setup_pipeline()

        self.logger.info("✅ LlamaIndex RAG pipeline initialized")

    def _setup_llama_index(self):
        """Configure global LlamaIndex settings"""
        # Set up LLM
        Settings.llm = OpenRouter(
            api_key=self.config.openrouter_api_key,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )

        # Set up embedding model - reuse existing service
        from ..embeddings import get_embedding_service

        self.embedding_service = get_embedding_service()

        # Custom wrapper to make embedding service compatible with LlamaIndex
        class IsschatEmbedding:
            def __init__(self, embedding_service):
                self._embedding_service = embedding_service

            def get_text_embedding(self, text: str) -> List[float]:
                return self._embedding_service.encode_query(text)

            def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                embeddings = self._embedding_service.encode_texts(texts)
                return embeddings.tolist()

        Settings.embed_model = IsschatEmbedding(self.embedding_service)

        self.logger.debug("LlamaIndex settings configured")

    def _setup_vector_store(self):
        """Initialize vector store and index"""
        try:
            # Use existing Weaviate vector store
            self.vector_db = VectorDBFactory.create_from_config()

            # Create LlamaIndex wrapper for existing vector store
            # Note: This is a simplified approach - in production you might want
            # to use LlamaIndex's native Weaviate integration

            class IsschatVectorStoreWrapper:
                """Wrapper to make Isschat vector store compatible with LlamaIndex"""

                def __init__(self, vector_db, embedding_service):
                    self.vector_db = vector_db
                    self.embedding_service = embedding_service

                def query(self, query_str: str, similarity_top_k: int = None) -> List[Dict]:
                    """Query the vector store and return results in LlamaIndex format"""
                    k = similarity_top_k or self.config.search_k
                    query_embedding = self.embedding_service.encode_query(query_str)

                    search_results = self.vector_db.search(query_embedding=query_embedding, k=k)

                    # Convert to LlamaIndex format
                    nodes = []
                    for result in search_results:
                        node = {
                            "text": result.document.content,
                            "metadata": result.document.metadata or {},
                            "score": result.score,
                            "id_": result.document.id,
                        }
                        nodes.append(node)

                    return nodes

            # For now, we'll use a simplified approach and create the retriever directly
            self.retriever = VectorIndexRetriever(
                index=None,  # We'll handle this manually
                similarity_top_k=self.config.search_k,
            )

            self.logger.debug("Vector store wrapper created")

        except Exception as e:
            raise RuntimeError(f"Failed to setup vector store: {e}")

    def _setup_memory(self):
        """Initialize unified memory management"""
        self.memory = ChatMemoryBuffer.from_defaults(
            token_limit=self.config.llm_max_tokens // 2,  # Reserve half tokens for context
            tokenizer_fn=lambda text: len(text.split()),  # Simple tokenizer
        )

        self.logger.debug("ChatMemoryBuffer initialized")

    def _setup_query_transforms(self):
        """Initialize query transformation components"""
        self.query_transforms = []

        if self.use_hyde:
            # HyDE: Generate hypothetical document and use for retrieval
            self.hyde_transform = HyDEQueryTransform(
                llm=Settings.llm,
                include_original=True,  # Include original query as well
            )
            self.query_transforms.append(("hyde", self.hyde_transform))

        if self.use_decompose:
            # Decompose: Break complex queries into sub-questions
            self.decompose_transform = DecomposeQueryTransform(
                llm=Settings.llm,
                verbose=False,
            )
            self.query_transforms.append(("decompose", self.decompose_transform))

        self.logger.debug(f"Query transforms configured: {[name for name, _ in self.query_transforms]}")

    def _setup_pipeline(self):
        """Setup the query processing pipeline"""
        try:
            # Create query engine with memory
            self.query_engine = RetrieverQueryEngine.from_args(
                retriever=self.retriever,
                memory=self.memory,
                verbose=False,
            )

            # Create query pipeline if transforms are enabled
            if self.query_transforms:
                self.pipeline = QueryPipeline(verbose=False)

                # Add components to pipeline
                for name, transform in self.query_transforms:
                    self.pipeline.add_modules({name: transform})

                # Add query engine
                self.pipeline.add_modules({"query_engine": self.query_engine})

                # Connect components
                if len(self.query_transforms) == 1:
                    # Single transform
                    transform_name = self.query_transforms[0][0]
                    self.pipeline.add_link(transform_name, "query_engine")
                else:
                    # Multiple transforms - chain them
                    prev_name = self.query_transforms[0][0]
                    for name, _ in self.query_transforms[1:]:
                        self.pipeline.add_link(prev_name, name)
                        prev_name = name
                    self.pipeline.add_link(prev_name, "query_engine")

            self.logger.debug("Query pipeline configured")

        except Exception as e:
            raise RuntimeError(f"Failed to setup pipeline: {e}")

    def process_query(
        self,
        query: str,
        user_id: str = "anonymous",
        conversation_id: Optional[str] = None,
        verbose: bool = False,
    ) -> Tuple[str, str]:
        """
        Process a query with LlamaIndex components.

        Args:
            query: User question
            user_id: User ID for logs
            conversation_id: Conversation ID for logs
            verbose: Detailed display

        Returns:
            Tuple (answer, sources)
        """
        start_time = time.time()

        try:
            if verbose:
                self.logger.info(f"🔍 Processing query with LlamaIndex: '{query[:100]}...'")

            # Add current query to memory context
            self.memory.put(f"Human: {query}")

            # Process query through pipeline or directly
            if hasattr(self, "pipeline") and self.pipeline:
                if verbose:
                    self.logger.info("🔄 Using query transformation pipeline")
                response = self.pipeline.run(query=query)
            else:
                if verbose:
                    self.logger.info("🎯 Using direct query engine")
                response = self.query_engine.query(query)

            answer = str(response.response) if hasattr(response, "response") else str(response)

            # Add response to memory
            self.memory.put(f"Assistant: {answer}")

            # Extract sources
            sources = self._format_sources(response)

            response_time = (time.time() - start_time) * 1000  # in ms

            if verbose:
                self.logger.info(f"✅ Response generated in {response_time:.0f}ms")

            # Save conversation with enhanced metadata
            try:
                conv_id = conversation_id or f"conv_{int(time.time())}"
                metadata = {
                    "pipeline_type": "llama_index_rag",
                    "query_transforms_used": [name for name, _ in self.query_transforms],
                    "memory_enabled": True,
                    "response_time_ms": response_time,
                }

                self.data_manager.save_conversation(
                    user_id=user_id,
                    conversation_id=conv_id,
                    question=query,
                    answer=answer,
                    response_time_ms=response_time,
                    sources=sources,
                    metadata=metadata,
                )
            except Exception as e:
                self.logger.warning(f"Failed to save conversation: {e}")

            return answer, sources

        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            response_time = (time.time() - start_time) * 1000

            error_answer = f"Sorry, an error occurred while processing your question: {str(e)}"

            # Still try to save the failed attempt
            try:
                conv_id = conversation_id or f"conv_{int(time.time())}"
                self.data_manager.save_conversation(
                    user_id=user_id,
                    conversation_id=conv_id,
                    question=query,
                    answer=error_answer,
                    response_time_ms=response_time,
                    sources="",
                    metadata={"pipeline_type": "llama_index_rag", "error": str(e)},
                )
            except Exception:
                pass

            return error_answer, ""

    def _format_sources(self, response) -> str:
        """Format sources from LlamaIndex response"""
        try:
            if hasattr(response, "source_nodes") and response.source_nodes:
                sources = []
                seen_sources = set()

                for node in response.source_nodes:
                    # Extract metadata
                    metadata = getattr(node.node, "metadata", {}) if hasattr(node, "node") else {}
                    title = metadata.get("title", "Document")
                    url = metadata.get("url", "")

                    # Create source link
                    if url:
                        source_link = f"[{title}]({url})"
                    else:
                        source_link = title

                    # Deduplicate
                    if source_link not in seen_sources:
                        seen_sources.add(source_link)
                        sources.append(source_link)

                return " • ".join(sources)

            return ""

        except Exception as e:
            self.logger.warning(f"Failed to format sources: {e}")
            return ""

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.reset()
        self.logger.debug("Memory cleared")

    def get_memory_summary(self) -> str:
        """Get current memory content summary"""
        try:
            messages = self.memory.get_all()
            return f"Memory contains {len(messages)} messages"
        except Exception:
            return "Memory summary unavailable"

    def is_ready(self) -> bool:
        """Check if pipeline is ready"""
        try:
            return (
                hasattr(self, "query_engine")
                and self.query_engine is not None
                and self.vector_db.exists()
                and self.vector_db.count() > 0
            )
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "type": "llama_index_pipeline",
            "ready": self.is_ready(),
            "query_transforms": [name for name, _ in self.query_transforms],
            "memory_enabled": True,
            "memory_summary": self.get_memory_summary(),
            "vector_store_count": self.vector_db.count() if hasattr(self, "vector_db") else 0,
        }


class LlamaIndexRAGPipelineFactory:
    """Factory for creating LlamaIndex RAG pipelines with different configurations"""

    @staticmethod
    def create_hyde_pipeline() -> LlamaIndexRAGPipeline:
        """Create pipeline with HyDE query transformation"""
        return LlamaIndexRAGPipeline(use_hyde=True, use_decompose=False)

    @staticmethod
    def create_decompose_pipeline() -> LlamaIndexRAGPipeline:
        """Create pipeline with query decomposition"""
        return LlamaIndexRAGPipeline(use_hyde=False, use_decompose=True)

    @staticmethod
    def create_hybrid_pipeline() -> LlamaIndexRAGPipeline:
        """Create pipeline with both HyDE and decomposition"""
        return LlamaIndexRAGPipeline(use_hyde=True, use_decompose=True)

    @staticmethod
    def create_simple_pipeline() -> LlamaIndexRAGPipeline:
        """Create simple pipeline without query transformation"""
        return LlamaIndexRAGPipeline(use_hyde=False, use_decompose=False)
