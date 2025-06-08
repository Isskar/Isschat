"""
Main evaluator for the RAG system.
"""

from typing import List, Dict, Any, Optional

# Use absolute imports with fallbacks
try:
    from rag_system.rag_pipeline import RAGPipeline
except ImportError:
    try:
        from src.rag_system.rag_pipeline import RAGPipeline
    except ImportError:
        from ..rag_system.rag_pipeline import RAGPipeline


class RAGEvaluator:
    """Main evaluator for RAG system performance."""

    def __init__(self, rag_pipeline: RAGPipeline, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluator.

        Args:
            rag_pipeline: RAG pipeline to evaluate
            config: Evaluation configuration
        """
        self.rag_pipeline = rag_pipeline
        self.config = config or {}
        self.metrics_collector = None

    def evaluate_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the RAG system on a dataset.

        Args:
            dataset: List of evaluation examples with 'query' and 'expected_answer'

        Returns:
            Dict: Evaluation results
        """
        results = []

        for example in dataset:
            query = example.get("query")
            expected_answer = example.get("expected_answer")

            if not query:
                continue

            # Generate response
            try:
                response = self.rag_pipeline.process_query(query)

                # Evaluate response
                evaluation = self._evaluate_response(query, response, expected_answer)
                evaluation["query"] = query
                evaluation["response"] = response
                evaluation["expected_answer"] = expected_answer

                results.append(evaluation)

            except Exception as e:
                results.append(
                    {
                        "query": query,
                        "response": None,
                        "expected_answer": expected_answer,
                        "error": str(e),
                        "success": False,
                    }
                )

        # Aggregate results
        return self._aggregate_results(results)

    def _evaluate_response(self, query: str, response: str, expected_answer: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single response.

        Args:
            query: Original query
            response: Generated response
            expected_answer: Expected answer (if available)

        Returns:
            Dict: Evaluation metrics for this response
        """
        evaluation = {
            "success": response is not None,
            "response_length": len(response) if response else 0,
        }

        if expected_answer and response:
            # Simple similarity metrics
            evaluation["exact_match"] = response.strip().lower() == expected_answer.strip().lower()
            evaluation["contains_expected"] = expected_answer.lower() in response.lower()
            evaluation["length_ratio"] = len(response) / len(expected_answer) if expected_answer else 0

        return evaluation

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate evaluation results.

        Args:
            results: List of individual evaluation results

        Returns:
            Dict: Aggregated metrics
        """
        if not results:
            return {"total_queries": 0}

        total_queries = len(results)
        successful_queries = sum(1 for r in results if r.get("success", False))

        aggregated = {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate": successful_queries / total_queries,
            "avg_response_length": sum(r.get("response_length", 0) for r in results) / total_queries,
        }

        # Aggregate metrics that require expected answers
        exact_matches = [r for r in results if r.get("exact_match") is not None]
        if exact_matches:
            aggregated["exact_match_rate"] = sum(r["exact_match"] for r in exact_matches) / len(exact_matches)
            aggregated["contains_expected_rate"] = sum(r["contains_expected"] for r in exact_matches) / len(
                exact_matches
            )
            aggregated["avg_length_ratio"] = sum(r["length_ratio"] for r in exact_matches) / len(exact_matches)

        return aggregated

    def evaluate_single_query(self, query: str, expected_answer: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single query.

        Args:
            query: Query to evaluate
            expected_answer: Expected answer (optional)

        Returns:
            Dict: Evaluation result
        """
        try:
            response = self.rag_pipeline.process_query(query)
            evaluation = self._evaluate_response(query, response, expected_answer)
            evaluation.update({"query": query, "response": response, "expected_answer": expected_answer})
            return evaluation
        except Exception as e:
            return {
                "query": query,
                "response": None,
                "expected_answer": expected_answer,
                "error": str(e),
                "success": False,
            }
