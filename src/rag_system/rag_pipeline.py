"""
Main RAG pipeline orchestrating retrieval and generation.
"""

from typing import Dict, Any, Tuple

from src.retrieval.base_retriever import BaseRetriever
from src.generation.base_generator import BaseGenerator


class RAGPipeline:
    """
    Main RAG pipeline that orchestrates retrieval and generation.
    This is the new architecture equivalent of HelpDesk.
    """

    def __init__(self, retriever: BaseRetriever, generator: BaseGenerator):
        """
        Initialize RAG pipeline.

        Args:
            retriever: Retriever component
            generator: Generator component
        """
        self.retriever = retriever
        self.generator = generator

    def process_query(self, query: str, top_k: int = 3, verbose: bool = True) -> Tuple[str, str]:
        """
        Process a query through the RAG pipeline.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            verbose: Whether to print verbose output

        Returns:
            Tuple of (answer, sources)
        """
        try:
            if verbose:
                print(f"\n🔍 Processing query: '{query}'")
                print("=" * 50)

            # Step 1: Retrieve relevant documents
            retrieval_result = self.retriever.retrieve(query, top_k=top_k)

            if verbose:
                print(f"\n📚 Retrieved {len(retrieval_result.documents)} documents:")
                for i, doc in enumerate(retrieval_result.documents):
                    print(f"  Document {i + 1}:")
                    print(f"    Title: {doc.metadata.get('title', 'N/A')}")
                    print(f"    Source: {doc.metadata.get('source', 'N/A')}")
                    print(f"    Content: {doc.page_content[:150]}...")
                    if i < len(retrieval_result.scores):
                        print(f"    Score: {retrieval_result.scores[i]:.4f}")
                    print()

            # Step 2: Generate answer
            generation_result = self.generator.generate(query, retrieval_result)

            if verbose:
                print(f"⚡ Generation completed in {generation_result.generation_time:.2f}s")
                print(f"📝 Token count: {generation_result.token_count}")
                print(f"\n📖 Sources:\n{generation_result.sources}")
                print("=" * 50)

            return generation_result.answer, generation_result.sources

        except Exception as e:
            error_msg = f"RAG pipeline failed: {str(e)}"
            if verbose:
                print(f"❌ Error: {error_msg}")
            return (
                f"Désolé, une erreur s'est produite lors du traitement de votre question: {error_msg}",
                "Aucune source disponible",
            )

    def retrieval_qa_inference(self, question: str, verbose: bool = True) -> Tuple[str, str]:
        """
        Legacy-compatible method name for HelpDesk replacement.

        Args:
            question: User question
            verbose: Whether to print verbose output

        Returns:
            Tuple of (answer, sources)
        """
        return self.process_query(question, verbose=verbose)

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return {
            "pipeline_type": "Modern RAG",
            "retriever_stats": self.retriever.get_stats(),
            "generator_stats": self.generator.get_stats(),
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the pipeline."""
        health = {"status": "healthy", "components": {}, "issues": []}

        try:
            # Check retriever
            retriever_stats = self.retriever.get_stats()
            health["components"]["retriever"] = {
                "status": retriever_stats.get("status", "unknown"),
                "type": retriever_stats.get("retriever_type", "unknown"),
            }

            # Check generator
            generator_stats = self.generator.get_stats()
            health["components"]["generator"] = {
                "status": generator_stats.get("status", "unknown"),
                "type": generator_stats.get("generator_type", "unknown"),
            }

            # Test with a simple query
            test_result = self.retriever.retrieve("test", top_k=1)
            if not test_result.documents:
                health["issues"].append("No documents retrieved for test query")

        except Exception as e:
            health["status"] = "unhealthy"
            health["issues"].append(f"Health check failed: {str(e)}")

        return health


class RAGPipelineFactory:
    """Factory for creating RAG pipelines with different configurations."""

    @staticmethod
    def create_default_pipeline() -> RAGPipeline:
        """
        Create a default RAG pipeline with FAISS retriever and OpenRouter generator.

        Returns:
            Configured RAGPipeline
        """
        # Use absolute imports with fallbacks
        try:
            from retrieval.simple_retriever import SimpleRetriever
            from generation.openrouter_generator import OpenRouterGenerator
        except ImportError:
            try:
                from src.retrieval.simple_retriever import SimpleRetriever
                from src.generation.openrouter_generator import OpenRouterGenerator
            except ImportError:
                from ..retrieval.simple_retriever import SimpleRetriever
                from ..generation.openrouter_generator import OpenRouterGenerator

        retriever = SimpleRetriever()
        generator = OpenRouterGenerator()

        return RAGPipeline(retriever, generator)

    @staticmethod
    def create_pipeline(retriever_type: str = "faiss", generator_type: str = "openrouter", **kwargs) -> RAGPipeline:
        """
        Create a RAG pipeline with specified components.

        Args:
            retriever_type: Type of retriever ("faiss")
            generator_type: Type of generator ("openrouter")
            **kwargs: Additional configuration

        Returns:
            Configured RAGPipeline
        """
        # Import here to avoid circular imports
        try:
            from retrieval.retriever_factory import RetrieverFactory
            from generation.openrouter_generator import OpenRouterGenerator
        except ImportError:
            try:
                from src.retrieval.retriever_factory import RetrieverFactory
                from src.generation.openrouter_generator import OpenRouterGenerator
            except ImportError:
                from ..retrieval.retriever_factory import RetrieverFactory
                from ..generation.openrouter_generator import OpenRouterGenerator

        # Create retriever
        retriever = RetrieverFactory.create_retriever(retriever_type, **kwargs.get("retriever_config", {}))

        # Create generator
        if generator_type.lower() == "openrouter":
            generator = OpenRouterGenerator(**kwargs.get("generator_config", {}))
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")

        return RAGPipeline(retriever, generator)
