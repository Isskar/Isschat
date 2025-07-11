"""
Retrieval evaluator for testing document retrieval performance
"""

import time
import logging
import math
import re
from typing import List, Dict, Any, Tuple, Optional

from rag_evaluation.core.base_evaluator import TestCase, EvaluationStatus, EvaluationResult
from rag_evaluation.core import IsschatClient, BaseEvaluator

logger = logging.getLogger(__name__)


class RetrievalEvaluator(BaseEvaluator):
    """Evaluator for retrieval performance tests"""

    def __init__(self, config: Any):
        super().__init__(config)
        self.client = IsschatClient()

    def get_category(self) -> str:
        return "retrieval"

    def _query_system(self, test_case: TestCase) -> Tuple[str, float, List[str]]:
        """Query the system and return response, response_time, and sources"""
        start_time = time.time()

        try:
            # Get retrieval results from Isschat
            response = self.client.query(test_case.question)
            response_time = (time.time() - start_time) * 1000

            # Extract retrieved documents
            _, _, retrieved_docs = response

            # Convert retrieved docs to string response
            response_str = str(retrieved_docs)

            # Extract sources
            sources = [doc.get("url", "") for doc in retrieved_docs]

            # Store retrieved docs in metadata for evaluation
            self._retrieved_docs = retrieved_docs

            return response_str, response_time, sources

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return f"ERROR: {str(e)}", response_time, []

    def _evaluate_semantically(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Evaluate retrieval performance using metrics"""
        # Get retrieved docs from the stored metadata
        retrieved_docs = getattr(self, "_retrieved_docs", [])
        expected_docs = test_case.metadata.get("expected_documents", [])

        # Calculate retrieval metrics
        metrics = self._calculate_retrieval_metrics(expected_docs, retrieved_docs)

        # Determine if test passes criteria: at least one expected document was retrieved
        passes_criteria = self._has_at_least_one_expected_document(expected_docs, retrieved_docs)

        return {
            "passes_criteria": passes_criteria,
            "score": metrics["overall_score"],
            "reasoning": f"Retrieval metrics calculated: overall_score={metrics['overall_score']:.3f}",
            **metrics,
        }

    def _create_success_result(
        self,
        test_case: TestCase,
        response: str,
        evaluation: Dict[str, Any],
        response_time: float,
        sources: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Create a successful evaluation result with PASSED/FAILED status based on document retrieval"""
        # Always use PASSED/FAILED based on whether at least one expected document was retrieved
        status = EvaluationStatus.PASSED if evaluation["passes_criteria"] else EvaluationStatus.FAILED

        return EvaluationResult(
            test_id=test_case.test_id,
            category=test_case.category,
            test_name=test_case.test_name,
            question=test_case.question,
            response=response,
            expected_behavior=test_case.expected_behavior,
            status=status,
            score=evaluation["score"],
            evaluation_details=evaluation,
            response_time=response_time,
            sources=sources or [],
            metadata=test_case.metadata,
        )

    def _extract_page_id(self, url: str) -> str:
        """Extract page ID from Confluence URL using regex"""
        if not url:
            return ""

        # Extract pages/number from URL
        match = re.search(r"pages/(\d+)", url)
        return match.group(1) if match else url

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
            }

        # Extract page IDs for comparison
        expected_page_ids = {self._extract_page_id(doc["url"]) for doc in expected_docs}
        retrieved_page_ids = [self._extract_page_id(doc.get("url", "")) for doc in retrieved_docs]

        # Calculate basic precision and recall
        relevant_retrieved = expected_page_ids.intersection(set(retrieved_page_ids))
        precision = len(relevant_retrieved) / len(retrieved_page_ids) if retrieved_page_ids else 0.0
        recall = len(relevant_retrieved) / len(expected_page_ids) if expected_page_ids else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Calculate Precision@K and Recall@K
        precision_at_k = self._calculate_precision_at_k(expected_page_ids, retrieved_page_ids)
        recall_at_k = self._calculate_recall_at_k(expected_page_ids, retrieved_page_ids)

        # Calculate Mean Reciprocal Rank (MRR)
        mrr = self._calculate_mrr(expected_page_ids, retrieved_page_ids)

        # Calculate Mean Average Precision (MAP)
        map_score = self._calculate_map(expected_page_ids, retrieved_page_ids)

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

    def _calculate_precision_at_k(self, expected_page_ids: set, retrieved_page_ids: List[str]) -> Dict[str, float]:
        """Calculate Precision@K for K=1,3,5"""
        precision_at_k = {}

        for k in [1, 3, 5]:
            if len(retrieved_page_ids) >= k:
                relevant_at_k = sum(1 for page_id in retrieved_page_ids[:k] if page_id in expected_page_ids)
                precision_at_k[f"precision_at_{k}"] = relevant_at_k / k
            else:
                # If we have fewer than k results, calculate based on what we have
                relevant_at_k = sum(1 for page_id in retrieved_page_ids if page_id in expected_page_ids)
                precision_at_k[f"precision_at_{k}"] = (
                    relevant_at_k / len(retrieved_page_ids) if retrieved_page_ids else 0.0
                )

        return precision_at_k

    def _calculate_recall_at_k(self, expected_page_ids: set, retrieved_page_ids: List[str]) -> Dict[str, float]:
        """Calculate Recall@K for K=1,3,5"""
        recall_at_k = {}

        for k in [1, 3, 5]:
            relevant_at_k = sum(1 for page_id in retrieved_page_ids[:k] if page_id in expected_page_ids)
            recall_at_k[f"recall_at_{k}"] = relevant_at_k / len(expected_page_ids) if expected_page_ids else 0.0

        return recall_at_k

    def _calculate_mrr(self, expected_page_ids: set, retrieved_page_ids: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, page_id in enumerate(retrieved_page_ids):
            if page_id in expected_page_ids:
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_map(self, expected_page_ids: set, retrieved_page_ids: List[str]) -> float:
        """Calculate Mean Average Precision"""
        if not expected_page_ids:
            return 0.0

        relevant_count = 0
        precision_sum = 0.0

        for i, page_id in enumerate(retrieved_page_ids):
            if page_id in expected_page_ids:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i

        return precision_sum / len(expected_page_ids) if expected_page_ids else 0.0

    def _calculate_ndcg(self, expected_docs: List[Dict], retrieved_docs: List[Dict]) -> Dict[str, float]:
        """Calculate Normalized Discounted Cumulative Gain at K=5"""
        ndcg_scores = {}

        # Create relevance mapping (1 for relevant, 0 for not relevant)
        expected_page_ids = {self._extract_page_id(doc["url"]) for doc in expected_docs}

        for k in [5]:
            dcg = 0.0
            for i, doc in enumerate(retrieved_docs[:k]):
                page_id = self._extract_page_id(doc.get("url", ""))
                relevance = 1.0 if page_id in expected_page_ids else 0.0
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

    def _has_at_least_one_expected_document(self, expected_docs: List[Dict], retrieved_docs: List[Dict]) -> bool:
        """Check if at least one expected document was retrieved using flexible URL matching"""
        if not expected_docs:
            # If no expected documents, consider it a pass (empty test case)
            return True

        if not retrieved_docs:
            # If no documents retrieved but some were expected, it's a fail
            return False

        # Get expected URLs
        expected_urls = [doc.get("url", "") for doc in expected_docs]

        # Get retrieved URLs
        retrieved_urls = [doc.get("url", "") for doc in retrieved_docs]

        # Check for flexible matches
        for expected_url in expected_urls:
            for retrieved_url in retrieved_urls:
                if self._urls_match_flexibly(expected_url, retrieved_url):
                    return True

        return False

    def _urls_match_flexibly(self, url1: str, url2: str) -> bool:
        """Check if two URLs match using flexible criteria"""
        if not url1 or not url2:
            return False

        # Exact match
        if url1 == url2:
            return True

        # One URL contains the other (handles cases like /pages/123 vs /pages/123/Title)
        if url1 in url2 or url2 in url1:
            return True

        # Same page ID (extract page ID and compare)
        page_id1 = self._extract_page_id(url1)
        page_id2 = self._extract_page_id(url2)

        if page_id1 and page_id2 and page_id1 == page_id2:
            return True

        return False

    def _calculate_ranking_score(self, expected_docs: List[Dict], retrieved_docs: List[Dict]) -> float:
        """Calculate ranking quality score"""

        if not expected_docs or not retrieved_docs:
            return 0.0

        # Check if expected ranks are provided
        has_ranking_info = any("expected_rank" in doc for doc in expected_docs)
        if not has_ranking_info:
            return 1.0  # No ranking requirements

        # Create mapping of page ID to expected rank
        expected_ranks = {
            self._extract_page_id(doc["url"]): doc.get("expected_rank", float("inf")) for doc in expected_docs
        }

        # Calculate NDCG-like score
        dcg = 0.0
        for i, retrieved_doc in enumerate(retrieved_docs[:5]):  # Top 5
            page_id = self._extract_page_id(retrieved_doc.get("url", ""))
            if page_id in expected_ranks:
                relevance = 1.0 / expected_ranks[page_id]  # Higher relevance for lower expected rank
                dcg += relevance / (i + 1)  # Discounted by position

        # Ideal DCG (if documents were in perfect order)
        sorted_expected = sorted(expected_docs, key=lambda x: x.get("expected_rank", float("inf")))
        idcg = sum(1.0 / doc.get("expected_rank", 1) / (i + 1) for i, doc in enumerate(sorted_expected[:5]))

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

            # Recall@K metrics
            avg_recall_at_1 = sum(r.evaluation_details.get("recall_at_1", 0) for r in self.results) / num_results
            avg_recall_at_3 = sum(r.evaluation_details.get("recall_at_3", 0) for r in self.results) / num_results
            avg_recall_at_5 = sum(r.evaluation_details.get("recall_at_5", 0) for r in self.results) / num_results

            # Advanced metrics
            avg_mrr = sum(r.evaluation_details.get("mrr", 0) for r in self.results) / num_results
            avg_map = sum(r.evaluation_details.get("map", 0) for r in self.results) / num_results
            avg_ndcg_at_5 = sum(r.evaluation_details.get("ndcg_at_5", 0) for r in self.results) / num_results

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
                    # Recall@K metrics
                    "average_recall_at_1": avg_recall_at_1,
                    "average_recall_at_3": avg_recall_at_3,
                    "average_recall_at_5": avg_recall_at_5,
                    # Advanced metrics
                    "average_mrr": avg_mrr,
                    "average_map": avg_map,
                    "average_ndcg_at_5": avg_ndcg_at_5,
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
        lines.append(f"  Precision@K: P@1={p_at_1:.3f} | P@3={p_at_3:.3f} | P@5={p_at_5:.3f}")

        # Recall@K metrics
        r_at_1 = summary.get("average_recall_at_1", 0)
        r_at_3 = summary.get("average_recall_at_3", 0)
        r_at_5 = summary.get("average_recall_at_5", 0)
        lines.append(f"  Recall@K:    R@1={r_at_1:.3f} | R@3={r_at_3:.3f} | R@5={r_at_5:.3f}")

        # Advanced metrics
        mrr = summary.get("average_mrr", 0)
        map_score = summary.get("average_map", 0)
        ndcg_5 = summary.get("average_ndcg_at_5", 0)
        lines.append(f"  Métriques avancées: MRR={mrr:.3f} | MAP={map_score:.3f}")
        lines.append(f"  NDCG: NDCG@5={ndcg_5:.3f}")

        # Performance metrics
        avg_time = summary.get("average_retrieval_time_ms", 0)
        lines.append(f"  Temps moyen de retrieval: {avg_time:.1f}ms")
        lines.append("  " + "-" * 50)

        return "\n".join(lines)
