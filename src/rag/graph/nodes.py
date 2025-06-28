"""
LangGraph nodes for RAG workflow.
Each node represents a step in the RAG pipeline.
"""

import re
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer

from .state import RAGState
from ...config import get_config
from ...vectordb import VectorDBFactory


class QueryProcessorNode:
    """LangGraph node for query processing and enhancement"""

    def __init__(self):
        self.config = get_config()

    def __call__(self, state: RAGState) -> Dict[str, Any]:
        """Process and enhance the user query"""
        try:
            query = state["query"].strip()

            # Basic query processing
            processed_query = self._process_query(query)

            # Query analysis
            metadata = {
                "original_length": len(query),
                "processed_length": len(processed_query),
                "has_question_mark": "?" in query,
                "word_count": len(query.split()),
            }

            return {
                "processed_query": processed_query,
                "query_metadata": metadata,
                "should_retrieve": True,  # Always retrieve for now
                "step": "query_processed",
            }

        except Exception as e:
            return {"error": f"Query processing failed: {str(e)}", "should_retrieve": False, "step": "query_error"}

    def _process_query(self, query: str) -> str:
        """Process the query (normalize, clean, etc.)"""
        # Remove extra whitespace
        query = re.sub(r"\s+", " ", query).strip()

        # Remove trailing punctuation if not meaningful
        if query.endswith(".") and not query.endswith("..."):
            query = query[:-1]

        # Add question mark if it's clearly a question but missing one
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        if any(query.lower().startswith(word) for word in question_words):
            if not query.endswith("?"):
                query += "?"

        return query


class RetrieverNode:
    """LangGraph node for document retrieval"""

    def __init__(self):
        self.config = get_config()
        self.embeddings_model = None
        self.vector_db = None
        self._initialize()

    def _initialize(self):
        """Initialize embeddings model and vector database"""
        try:
            # Initialize embeddings model
            self.embeddings_model = SentenceTransformer(self.config.embeddings_model)

            # Initialize vector database
            self.vector_db = VectorDBFactory.create(
                db_type=self.config.vector_db_type,
                collection_name=self.config.collection_name,
                path=self.config.vector_db_path,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to initialize retriever: {e}")

    def __call__(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve relevant documents"""
        try:
            if not state.get("should_retrieve", False):
                return {"step": "retrieval_skipped"}

            query = state.get("processed_query", state["query"])

            # Generate query embedding
            query_embedding = self.embeddings_model.encode(query).tolist()

            # Search vector database
            search_results = self.vector_db.search(query_embedding=query_embedding, k=self.config.search_k)

            # Extract documents and scores
            documents = [result.document for result in search_results]
            scores = [result.score for result in search_results]

            # Create context from retrieved documents
            context = self._create_context(documents)

            return {
                "retrieved_documents": documents,
                "retrieval_scores": scores,
                "context": context,
                "should_generate": len(documents) > 0,
                "step": "retrieval_completed",
            }

        except Exception as e:
            return {
                "error": f"Retrieval failed: {str(e)}",
                "retrieved_documents": [],
                "retrieval_scores": [],
                "context": "",
                "should_generate": False,
                "step": "retrieval_error",
            }

    def _create_context(self, documents: List) -> str:
        """Create context string from retrieved documents"""
        if not documents:
            return ""

        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get("title", "Document")
            content = doc.content[:500]  # Limit content length
            context_parts.append(f"Document {i} ({title}):\n{content}")

        return "\n\n".join(context_parts)


class GeneratorNode:
    """LangGraph node for answer generation"""

    def __init__(self):
        self.config = get_config()
        self.generator = None
        self._initialize()

    def _initialize(self):
        """Initialize the LLM generator"""
        try:
            from ...rag.tools.unified_generation_tool import UnifiedGenerationTool

            self.generator = UnifiedGenerationTool()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize generator: {e}")

    def __call__(self, state: RAGState) -> Dict[str, Any]:
        """Generate answer from context"""
        try:
            if not state.get("should_generate", False):
                return {
                    "answer": "Je ne peux pas répondre à cette question car aucun document pertinent n'a été trouvé.",
                    "sources": "Aucune source disponible",
                    "step": "generation_skipped",
                }

            query = state["query"]
            context = state.get("context", "")
            history = state.get("history", "")

            # Generate answer using unified generation tool
            documents = state.get("retrieved_documents", [])
            scores = state.get("retrieval_scores", [])

            generation_result = self.generator.generate(
                query=query, documents=documents, scores=scores, history=history
            )

            return {
                "answer": generation_result["answer"],
                "sources": generation_result["sources"],
                "generation_metadata": {
                    "token_count": generation_result["token_count"],
                    "generation_time": generation_result["generation_time"],
                },
                "step": "generation_completed",
            }

        except Exception as e:
            return {
                "answer": f"Erreur lors de la génération: {str(e)}",
                "sources": "Erreur",
                "error": f"Generation failed: {str(e)}",
                "step": "generation_error",
            }
