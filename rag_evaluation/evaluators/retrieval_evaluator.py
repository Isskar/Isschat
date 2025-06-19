"""
Retrieval evaluator for testing document retrieval performance
"""

import time
import logging
import math
from typing import List, Dict, Any

from rag_evaluation.core.base_evaluator import TestCase, EvaluationStatus
from rag_evaluation.core import IsschatClient, BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class RetrievalEvaluator(BaseEvaluator):
    """Evaluator for retrieval performance tests"""

    def __init__(self, config: Any):
        super().__init__(config)
        self.client = IsschatClient()

    def get_category(self) -> str:
        return "retrieval"

    def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        start_time = time.time()

        try:
            # Get retrieval results from Isschat
            response = self.client.query(test_case.question)
            response_time = (time.time() - start_time) * 1000

            # Extract retrieved documents
            _, _, retrieved_docs = response

            metrics = self._calculate_retrieval_metrics(
                test_case.metadata.get("expected_documents", []), retrieved_docs
            )

            # Only use pass/fail in CI mode, otherwise just measure metrics
            if self.config.ci_mode:
                status = (
                    EvaluationStatus.PASSED
                    if metrics["overall_score"] >= self.config.ci_threshold
                    else EvaluationStatus.FAILED
                )
            else:
                status = EvaluationStatus.MEASURED

            logger.info(f"Test {test_case.test_id}: score={metrics['overall_score']:.3f}, status={status.value}")

            return EvaluationResult(
                test_id=test_case.test_id,
                category=test_case.category,
                test_name=test_case.test_name,
                question=test_case.question,
                response=str(retrieved_docs),
                expected_behavior=test_case.expected_behavior,
                status=status,
                score=metrics["overall_score"],
                evaluation_details=metrics,
                response_time=response_time,
                sources=[doc.get("url", "") for doc in retrieved_docs],
                metadata=test_case.metadata,
            )

        except Exception as e:
            logger.error(f"Error evaluating {test_case.test_id}: {e}")
            return EvaluationResult(
                test_id=test_case.test_id,
                category=test_case.category,
                test_name=test_case.test_name,
                question=test_case.question,
                response="",
                expected_behavior=test_case.expected_behavior,
                status=EvaluationStatus.ERROR,
                score=0.0,
                error_message=str(e),
                response_time=(time.time() - start_time) * 1000,
            )

    def _calculate_retrieval_metrics(self, expected_docs: List[Dict], retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive retrieval performance metrics"""

        if not expected_docs:
            # For tests expecting no results
            return {
                "precision": 1.0 if not retrieved_docs else 0.0,
                "recall": 1.0,
                "f1_score": 1.0 if not retrieved_docs else 0.0,
                "overall_score": 1.0 if not retrieved_docs else 0.0,
                "retrieved_count": len(retrieved_docs),
                "expected_count": 0,
                "precision_at_1": 1.0 if not retrieved_docs else 0.0,
                "precision_at_3": 1.0 if not retrieved_docs else 0.0,
                "precision_at_5": 1.0 if not retrieved_docs else 0.0,
                "recall_at_1": 1.0,
                "recall_at_3": 1.0,
                "recall_at_5": 1.0,
                "mrr": 1.0 if not retrieved_docs else 0.0,
                "map": 1.0 if not retrieved_docs else 0.0,
                "ndcg_at_5": 1.0 if not retrieved_docs else 0.0,
                "ndcg_at_10": 1.0 if not retrieved_docs else 0.0,
            }

        # Extract URLs for comparison
        expected_urls = {doc["url"] for doc in expected_docs}
        retrieved_urls = [doc.get("url", "") for doc in retrieved_docs]

        # Calculate basic precision and recall
        relevant_retrieved = expected_urls.intersection(set(retrieved_urls))
        precision = len(relevant_retrieved) / len(retrieved_urls) if retrieved_urls else 0.0
        recall = len(relevant_retrieved) / len(expected_urls) if expected_urls else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Calculate Precision@K and Recall@K
        precision_at_k = self._calculate_precision_at_k(expected_urls, retrieved_urls)
        recall_at_k = self._calculate_recall_at_k(expected_urls, retrieved_urls)

        # Calculate Mean Reciprocal Rank (MRR)
        mrr = self._calculate_mrr(expected_urls, retrieved_urls)

        # Calculate Mean Average Precision (MAP)
        map_score = self._calculate_map(expected_urls, retrieved_urls)

        # Calculate NDCG@K
        ndcg_scores = self._calculate_ndcg(expected_docs, retrieved_docs)

        # Calculate ranking score if expected ranks are provided
        ranking_score = self._calculate_ranking_score(expected_docs, retrieved_docs)

        # Overall score is simply recall@5
        overall_score = recall_at_k["recall_at_5"]

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "ranking_score": ranking_score,
            "overall_score": overall_score,
            "retrieved_count": len(retrieved_docs),
            "expected_count": len(expected_docs),
            "relevant_retrieved": len(relevant_retrieved),
            **precision_at_k,
            **recall_at_k,
            "mrr": mrr,
            "map": map_score,
            **ndcg_scores,
        }

    def _calculate_precision_at_k(self, expected_urls: set, retrieved_urls: List[str]) -> Dict[str, float]:
        """Calculate Precision@K for K=1,3,5,10"""
        precision_at_k = {}

        for k in [1, 3, 5, 10]:
            if len(retrieved_urls) >= k:
                relevant_at_k = sum(1 for url in retrieved_urls[:k] if url in expected_urls)
                precision_at_k[f"precision_at_{k}"] = relevant_at_k / k
            else:
                # If we have fewer than k results, calculate based on what we have
                relevant_at_k = sum(1 for url in retrieved_urls if url in expected_urls)
                precision_at_k[f"precision_at_{k}"] = relevant_at_k / len(retrieved_urls) if retrieved_urls else 0.0

        return precision_at_k

    def _calculate_recall_at_k(self, expected_urls: set, retrieved_urls: List[str]) -> Dict[str, float]:
        """Calculate Recall@K for K=1,3,5,10"""
        recall_at_k = {}

        for k in [1, 3, 5, 10]:
            relevant_at_k = sum(1 for url in retrieved_urls[:k] if url in expected_urls)
            recall_at_k[f"recall_at_{k}"] = relevant_at_k / len(expected_urls) if expected_urls else 0.0

        return recall_at_k

    def _calculate_mrr(self, expected_urls: set, retrieved_urls: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, url in enumerate(retrieved_urls):
            if url in expected_urls:
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_map(self, expected_urls: set, retrieved_urls: List[str]) -> float:
        """Calculate Mean Average Precision"""
        if not expected_urls:
            return 0.0

        relevant_count = 0
        precision_sum = 0.0

        for i, url in enumerate(retrieved_urls):
            if url in expected_urls:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i

        return precision_sum / len(expected_urls) if expected_urls else 0.0

    def _calculate_ndcg(self, expected_docs: List[Dict], retrieved_docs: List[Dict]) -> Dict[str, float]:
        """Calculate Normalized Discounted Cumulative Gain at K=5,10"""
        ndcg_scores = {}

        # Create relevance mapping (1 for relevant, 0 for not relevant)
        expected_urls = {doc["url"] for doc in expected_docs}

        for k in [5, 10]:
            dcg = 0.0
            for i, doc in enumerate(retrieved_docs[:k]):
                url = doc.get("url", "")
                relevance = 1.0 if url in expected_urls else 0.0
                if i == 0:
                    dcg += relevance
                else:
                    dcg += relevance / math.log2(i + 1)

            # Ideal DCG (assuming all relevant docs are at the top)
            idcg = 0.0
            relevant_count = min(len(expected_docs), k)
            for i in range(relevant_count):
                if i == 0:
                    idcg += 1.0
                else:
                    idcg += 1.0 / math.log2(i + 1)

            ndcg_scores[f"ndcg_at_{k}"] = dcg / idcg if idcg > 0 else 0.0

        return ndcg_scores

    def _calculate_ranking_score(self, expected_docs: List[Dict], retrieved_docs: List[Dict]) -> float:
        """Calculate ranking quality score"""

        if not expected_docs or not retrieved_docs:
            return 0.0

        # Check if expected ranks are provided
        has_ranking_info = any("expected_rank" in doc for doc in expected_docs)
        if not has_ranking_info:
            return 1.0  # No ranking requirements

        # Create mapping of URL to expected rank
        expected_ranks = {doc["url"]: doc.get("expected_rank", float("inf")) for doc in expected_docs}

        # Calculate NDCG-like score
        dcg = 0.0
        for i, retrieved_doc in enumerate(retrieved_docs[:10]):  # Top 10
            url = retrieved_doc.get("url", "")
            if url in expected_ranks:
                relevance = 1.0 / expected_ranks[url]  # Higher relevance for lower expected rank
                dcg += relevance / (i + 1)  # Discounted by position

        # Ideal DCG (if documents were in perfect order)
        sorted_expected = sorted(expected_docs, key=lambda x: x.get("expected_rank", float("inf")))
        idcg = sum(1.0 / doc.get("expected_rank", 1) / (i + 1) for i, doc in enumerate(sorted_expected[:10]))

        return dcg / idcg if idcg > 0 else 0.0

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics for retrieval evaluation"""
        base_stats = super().get_summary_stats()

        if self.results:
            num_results = len(self.results)
            avg_retrieval_time = sum(r.response_time for r in self.results) / num_results
            base_stats["average_retrieval_time_ms"] = avg_retrieval_time

            # Basic retrieval metrics
            avg_precision = sum(r.evaluation_details.get("precision", 0) for r in self.results) / num_results
            avg_recall = sum(r.evaluation_details.get("recall", 0) for r in self.results) / num_results
            avg_f1 = sum(r.evaluation_details.get("f1_score", 0) for r in self.results) / num_results

            # Precision@K metrics
            avg_precision_at_1 = sum(r.evaluation_details.get("precision_at_1", 0) for r in self.results) / num_results
            avg_precision_at_3 = sum(r.evaluation_details.get("precision_at_3", 0) for r in self.results) / num_results
            avg_precision_at_5 = sum(r.evaluation_details.get("precision_at_5", 0) for r in self.results) / num_results
            avg_precision_at_10 = (
                sum(r.evaluation_details.get("precision_at_10", 0) for r in self.results) / num_results
            )

            # Recall@K metrics
            avg_recall_at_1 = sum(r.evaluation_details.get("recall_at_1", 0) for r in self.results) / num_results
            avg_recall_at_3 = sum(r.evaluation_details.get("recall_at_3", 0) for r in self.results) / num_results
            avg_recall_at_5 = sum(r.evaluation_details.get("recall_at_5", 0) for r in self.results) / num_results
            avg_recall_at_10 = sum(r.evaluation_details.get("recall_at_10", 0) for r in self.results) / num_results

            # Advanced metrics
            avg_mrr = sum(r.evaluation_details.get("mrr", 0) for r in self.results) / num_results
            avg_map = sum(r.evaluation_details.get("map", 0) for r in self.results) / num_results
            avg_ndcg_at_5 = sum(r.evaluation_details.get("ndcg_at_5", 0) for r in self.results) / num_results
            avg_ndcg_at_10 = sum(r.evaluation_details.get("ndcg_at_10", 0) for r in self.results) / num_results

            base_stats.update(
                {
                    # Basic metrics
                    "average_precision": avg_precision,
                    "average_recall": avg_recall,
                    "average_f1_score": avg_f1,
                    # Precision@K metrics
                    "average_precision_at_1": avg_precision_at_1,
                    "average_precision_at_3": avg_precision_at_3,
                    "average_precision_at_5": avg_precision_at_5,
                    "average_precision_at_10": avg_precision_at_10,
                    # Recall@K metrics
                    "average_recall_at_1": avg_recall_at_1,
                    "average_recall_at_3": avg_recall_at_3,
                    "average_recall_at_5": avg_recall_at_5,
                    "average_recall_at_10": avg_recall_at_10,
                    # Advanced metrics
                    "average_mrr": avg_mrr,
                    "average_map": avg_map,
                    "average_ndcg_at_5": avg_ndcg_at_5,
                    "average_ndcg_at_10": avg_ndcg_at_10,
                }
            )

        return base_stats

    def format_detailed_summary(self) -> str:
        """Format detailed retrieval metrics summary"""
        if not self.results:
            return ""

        summary = self.get_summary_stats()

        lines = [
            "  📊 MÉTRIQUES DÉTAILLÉES DE RETRIEVAL:",
            "  " + "-" * 50,
        ]

        # Basic metrics
        precision = summary.get("average_precision", 0)
        recall = summary.get("average_recall", 0)
        f1 = summary.get("average_f1_score", 0)
        lines.append(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1:.3f}")

        # Precision@K metrics
        p_at_1 = summary.get("average_precision_at_1", 0)
        p_at_3 = summary.get("average_precision_at_3", 0)
        p_at_5 = summary.get("average_precision_at_5", 0)
        p_at_10 = summary.get("average_precision_at_10", 0)
        lines.append(f"  Precision@K: P@1={p_at_1:.3f} | P@3={p_at_3:.3f} | P@5={p_at_5:.3f} | P@10={p_at_10:.3f}")

        # Recall@K metrics
        r_at_1 = summary.get("average_recall_at_1", 0)
        r_at_3 = summary.get("average_recall_at_3", 0)
        r_at_5 = summary.get("average_recall_at_5", 0)
        r_at_10 = summary.get("average_recall_at_10", 0)
        lines.append(f"  Recall@K:    R@1={r_at_1:.3f} | R@3={r_at_3:.3f} | R@5={r_at_5:.3f} | R@10={r_at_10:.3f}")

        # Advanced metrics
        mrr = summary.get("average_mrr", 0)
        map_score = summary.get("average_map", 0)
        ndcg_5 = summary.get("average_ndcg_at_5", 0)
        ndcg_10 = summary.get("average_ndcg_at_10", 0)
        lines.append(f"  Métriques avancées: MRR={mrr:.3f} | MAP={map_score:.3f}")
        lines.append(f"  NDCG: NDCG@5={ndcg_5:.3f} | NDCG@10={ndcg_10:.3f}")

        # Performance metrics
        avg_time = summary.get("average_retrieval_time_ms", 0)
        lines.append(f"  Temps moyen de retrieval: {avg_time:.1f}ms")
        lines.append("  " + "-" * 50)

        return "\n".join(lines)
