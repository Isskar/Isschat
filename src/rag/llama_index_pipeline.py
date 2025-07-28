"""
LlamaIndex-based RAG pipeline with optimized query processing and memory management.
Replaces the custom semantic pipeline with proven LlamaIndex components.
"""

import logging
import time
from typing import Tuple, Dict, Any, Optional

from llama_index.core import Settings

# Query Transforms
from llama_index.core.indices.query.query_transform import HyDEQueryTransform, DecomposeQueryTransform

# Memory
from llama_index.core.memory import ChatMemoryBuffer

# LLM
from llama_index.llms.openrouter import OpenRouter

from ..config import get_config
from ..config.llamaindex_config import get_llamaindex_config
from ..storage.data_manager import get_data_manager
from ..vectordb import VectorDBFactory


class LlamaIndexRAGPipeline:
    """
    Enhanced RAG pipeline using LlamaIndex components.
    Provides query transformation, memory management, and optimized retrieval.
    """

    def __init__(self, use_hyde: bool = True, use_decompose: bool = False):
        self.config = get_config()
        self.llama_config = get_llamaindex_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_manager = get_data_manager()

        # Configuration
        self.use_hyde = use_hyde
        self.use_decompose = use_decompose

        # Adaptive memory tracking
        self.conversation_lengths = []  # Track recent conversation lengths
        self.current_context_size = 0

        # Conversation history management
        self.current_conversation_id = None
        self.conversation_history_loaded = False

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

        # For now, skip LlamaIndex embedding integration to maintain compatibility
        # We'll use existing Isschat embedding service directly in retrieval
        # Settings.embed_model = self.embedding_service  # Skip this for compatibility

        self.logger.debug("LlamaIndex settings configured")

    def _setup_vector_store(self):
        """Initialize vector store - using existing Isschat infrastructure"""
        try:
            # Use existing Weaviate vector store (keeping current architecture)
            self.vector_db = VectorDBFactory.create_from_config()

            # Note: We'll integrate LlamaIndex components gradually
            # For now, we keep the existing vector store and add LlamaIndex
            # memory management and query transforms on top

            self.logger.debug("Vector store initialized (existing Isschat infrastructure)")

        except Exception as e:
            raise RuntimeError(f"Failed to setup vector store: {e}")

    def _setup_memory(self):
        """Initialize unified memory management with adaptive token limits"""
        # Calculate initial memory token limit
        initial_limit = self._calculate_adaptive_memory_limit()

        self.memory = ChatMemoryBuffer.from_defaults(
            token_limit=initial_limit,
            tokenizer_fn=self._estimate_tokens,  # More accurate tokenizer
        )

        self.logger.debug(f"ChatMemoryBuffer initialized with adaptive limit: {initial_limit} tokens")

    def _calculate_adaptive_memory_limit(self) -> int:
        """Calculate optimal memory token limit based on context and configuration"""
        max_tokens = self.config.llm_max_tokens

        if not self.llama_config.adaptive_memory:
            # Static allocation if adaptive is disabled
            if self.llama_config.memory_token_limit:
                return self.llama_config.memory_token_limit
            return max_tokens // 2

        # Adaptive logic based on conversation history
        base_limit = int(max_tokens * self.llama_config.memory_reserve_ratio)

        # Adjust based on recent conversation patterns
        if len(self.conversation_lengths) >= 3:
            avg_length = sum(self.conversation_lengths[-5:]) / min(5, len(self.conversation_lengths))

            if avg_length > 500:  # Long conversations need more memory
                adjustment_factor = 1.3
            elif avg_length < 100:  # Short conversations need less memory
                adjustment_factor = 0.7
            else:
                adjustment_factor = 1.0

            base_limit = int(base_limit * adjustment_factor)

        # Apply bounds
        return max(self.llama_config.min_memory_tokens, min(base_limit, self.llama_config.max_memory_tokens))

    def _estimate_tokens(self, text: str) -> int:
        """More accurate token estimation"""
        # Simple heuristic: ~4 characters per token for most languages
        return max(1, len(text) // 4)

    def _update_memory_limit_if_needed(self):
        """Dynamically adjust memory limit if adaptive mode is enabled"""
        if not self.llama_config.adaptive_memory:
            return

        new_limit = self._calculate_adaptive_memory_limit()
        current_limit = self.memory.token_limit

        # Only update if the change is significant (>20% difference)
        if abs(new_limit - current_limit) / current_limit > 0.2:
            self.memory.token_limit = new_limit
            self.logger.debug(f"Memory limit adapted: {current_limit} -> {new_limit} tokens")

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
        """Setup simplified pipeline - hybrid approach"""
        try:
            # For now, we create a simpler integration that combines:
            # 1. LlamaIndex memory management (ChatMemoryBuffer) ✅
            # 2. LlamaIndex query transforms (HyDE) ✅
            # 3. Existing Isschat retrieval + generation ✅

            # This is a pragmatic approach that doesn't break existing architecture
            # while adding the LlamaIndex benefits we want

            self.logger.debug("Hybrid pipeline configured (LlamaIndex + Isschat)")

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

            # Handle conversation context setup
            if conversation_id:
                if self.current_conversation_id != conversation_id:
                    # Switch to different conversation - load its history
                    if not self.continue_conversation(conversation_id, user_id):
                        self.logger.warning(f"Failed to load history for conversation: {conversation_id}")
                elif not self.conversation_history_loaded:
                    # Same conversation but history not loaded yet
                    self.continue_conversation(conversation_id, user_id)
            else:
                # No conversation_id provided - generate one for tracking
                conversation_id = f"conv_{int(time.time())}_{user_id}"
                self.start_new_conversation(conversation_id)

            # Track conversation length for adaptive memory
            query_length = self._estimate_tokens(query)
            self.conversation_lengths.append(query_length)
            if len(self.conversation_lengths) > 10:  # Keep only recent 10 conversations
                self.conversation_lengths.pop(0)

            # Update memory limit adaptively before processing
            self._update_memory_limit_if_needed()

            # Add current query to memory context
            self.memory.put(f"Human: {query}")

            # Apply query transforms (HyDE/Decompose) if enabled
            processed_query = query
            if self.query_transforms:
                if verbose:
                    self.logger.info("🔄 Applying LlamaIndex query transforms")

                for name, transform in self.query_transforms:
                    try:
                        if name == "hyde":
                            # Apply HyDE transformation
                            hyde_result = transform(processed_query)
                            processed_query = str(hyde_result) if hyde_result else processed_query
                            if verbose:
                                self.logger.info(f"HyDE query: {processed_query[:100]}...")
                        elif name == "decompose":
                            # Apply decomposition (for complex queries)
                            decompose_result = transform(processed_query)
                            processed_query = str(decompose_result) if decompose_result else processed_query
                    except Exception as e:
                        self.logger.warning(f"Query transform {name} failed: {e}")
                        continue

            # Use existing Isschat retrieval + generation pipeline
            if verbose:
                self.logger.info("🎯 Using Isschat retrieval + generation")

            # Import existing tools here to avoid circular imports
            from .tools.semantic_retrieval_tool import SemanticRetrievalTool
            from .tools.generation_tool import GenerationTool

            # Create tools if not already available
            if not hasattr(self, "_retrieval_tool"):
                self._retrieval_tool = SemanticRetrievalTool()
            if not hasattr(self, "_generation_tool"):
                self._generation_tool = GenerationTool()

            # Retrieve relevant documents (using HyDE-enhanced query, no need for semantic expansion)
            retrieved_docs = self._retrieval_tool.retrieve(
                query=processed_query,
                k=self.config.search_k,
                use_semantic_expansion=False,  # Disabled: HyDE replaces semantic expansion
                use_semantic_reranking=True,  # Keep reranking for score adjustment
            )

            # Get memory context for generation
            memory_context = ""
            try:
                memory_messages = self.memory.get_all()
                if memory_messages:
                    # Take recent messages for context
                    recent_messages = memory_messages[-6:]  # Last 3 exchanges
                    memory_context = "\n".join([str(msg) for msg in recent_messages])
            except Exception as e:
                self.logger.warning(f"Failed to get memory context: {e}")

            # Generate answer using existing generation tool
            generation_result = self._generation_tool.generate(
                query=query,  # Use original query for generation
                documents=retrieved_docs,
                history=memory_context,
            )

            answer = generation_result.get("answer", "No answer generated")

            # Track response length for adaptive memory
            response_length = self._estimate_tokens(answer)
            self.conversation_lengths[-1] += response_length  # Add to last query's length

            # Add response to memory
            self.memory.put(f"Assistant: {answer}")

            # Extract sources from retrieved documents
            sources = self._format_sources_from_docs(retrieved_docs)

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
                    "adaptive_memory": self.llama_config.adaptive_memory,
                    "current_memory_limit": self.memory.token_limit,
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

    def _format_sources_from_docs(self, retrieved_docs) -> str:
        """Format sources from Isschat retrieved documents"""
        try:
            if not retrieved_docs:
                return ""

            sources = []
            seen_sources = set()

            for doc in retrieved_docs:
                # Extract metadata from Isschat document format
                metadata = doc.metadata or {}
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

        except Exception as e:
            self.logger.warning(f"Failed to format sources: {e}")
            return ""

    def load_conversation_history(self, conversation_id: str, user_id: str = "anonymous") -> bool:
        """
        Load conversation history from persistent storage into ChatMemoryBuffer

        Args:
            conversation_id: ID of conversation to load
            user_id: User ID for filtering (optional)

        Returns:
            True if history was loaded successfully
        """
        try:
            # Don't reload if already loaded for this conversation
            if self.current_conversation_id == conversation_id and self.conversation_history_loaded:
                self.logger.debug(f"History already loaded for conversation: {conversation_id}")
                return True

            # Clear existing memory
            self.memory.reset()

            # Load conversation history from DataManager
            history_entries = self.data_manager.get_conversation_history(
                user_id=user_id,
                conversation_id=conversation_id,
                limit=20,  # Load last 20 exchanges to respect memory limits
            )

            if not history_entries:
                self.logger.debug(f"No history found for conversation: {conversation_id}")
                self.current_conversation_id = conversation_id
                self.conversation_history_loaded = True
                return True

            # Sort by timestamp to maintain chronological order
            history_entries.sort(key=lambda x: x.get("timestamp", ""))

            # Load history into memory buffer
            loaded_count = 0
            for entry in history_entries:
                try:
                    question = entry.get("question", "")
                    answer = entry.get("answer", "")

                    if question and answer:
                        self.memory.put(f"Human: {question}")
                        self.memory.put(f"Assistant: {answer}")
                        loaded_count += 1

                except Exception as e:
                    self.logger.warning(f"Failed to load history entry: {e}")
                    continue

            self.current_conversation_id = conversation_id
            self.conversation_history_loaded = True

            self.logger.info(f"✅ Loaded {loaded_count} conversation exchanges for: {conversation_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load conversation history: {e}")
            return False

    def start_new_conversation(self, conversation_id: str) -> bool:
        """
        Start a new conversation, clearing memory and setting up tracking

        Args:
            conversation_id: New conversation ID

        Returns:
            True if successfully initialized
        """
        try:
            # Clear memory for fresh start
            self.memory.reset()

            # Reset tracking variables
            self.current_conversation_id = conversation_id
            self.conversation_history_loaded = True
            self.conversation_lengths = []

            self.logger.info(f"🆕 Started new conversation: {conversation_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start new conversation: {e}")
            return False

    def continue_conversation(self, conversation_id: str, user_id: str = "anonymous") -> bool:
        """
        Continue an existing conversation by loading its history

        Args:
            conversation_id: Conversation to continue
            user_id: User ID for filtering

        Returns:
            True if conversation was set up successfully
        """
        return self.load_conversation_history(conversation_id, user_id)

    def clear_memory(self):
        """Clear conversation memory and reset tracking"""
        self.memory.reset()
        self.current_conversation_id = None
        self.conversation_history_loaded = False
        self.conversation_lengths = []
        self.logger.debug("Memory cleared and tracking reset")

    def get_memory_summary(self) -> str:
        """Get current memory content summary"""
        try:
            messages = self.memory.get_all()
            return f"Memory contains {len(messages)} messages (Conversation: {self.current_conversation_id or 'None'})"
        except Exception:
            return f"Memory summary unavailable (Conversation: {self.current_conversation_id or 'None'})"

    def is_ready(self) -> bool:
        """Check if pipeline is ready"""
        try:
            # Simple check: vector DB exists and has documents, memory is initialized
            return (
                hasattr(self, "vector_db")
                and hasattr(self, "memory")
                and self.vector_db.exists()
                and self.vector_db.count() > 0
            )
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        avg_conversation_length = (
            sum(self.conversation_lengths) / len(self.conversation_lengths) if self.conversation_lengths else 0
        )

        return {
            "type": "llama_index_pipeline",
            "ready": self.is_ready(),
            "query_transforms": [name for name, _ in self.query_transforms],
            "memory_enabled": True,
            "memory_summary": self.get_memory_summary(),
            "adaptive_memory": {
                "enabled": self.llama_config.adaptive_memory,
                "current_limit": self.memory.token_limit,
                "reserve_ratio": self.llama_config.memory_reserve_ratio,
                "min_tokens": self.llama_config.min_memory_tokens,
                "max_tokens": self.llama_config.max_memory_tokens,
                "avg_conversation_length": round(avg_conversation_length, 1),
                "recent_conversations": len(self.conversation_lengths),
            },
            "conversation_management": {
                "current_conversation_id": self.current_conversation_id,
                "history_loaded": self.conversation_history_loaded,
                "memory_messages": len(self.memory.get_all()) if hasattr(self.memory, "get_all") else 0,
            },
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
