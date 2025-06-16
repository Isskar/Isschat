"""
Document relevance evaluator for testing if Isschat retrieves expected documents
"""

import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, List
from urllib.parse import urlparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ..core.base_evaluator import BaseEvaluator, EvaluationResult, TestCase, TestCategory, EvaluationStatus

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentRelevanceEvaluator(BaseEvaluator):
    """Evaluator for document relevance tests"""

    def __init__(self, config: Any):
        """Initialize document relevance evaluator"""
        super().__init__(config)

    def get_category(self) -> TestCategory:
        """Get the category this evaluator handles"""
        return TestCategory.ROBUSTNESS  # We'll use robustness category for now

    def evaluate_document_relevance(
        self,
        test_case: TestCase,
        retrieved_sources: List[str] = None,  # ty : ignore
    ) -> Dict[str, Any]:  # ty : ignore
        """
        Evaluate document relevance for a test case

        Args:
            test_case: The test case with expected documents
            retrieved_sources: List of source URLs/titles retrieved by Isschat (optional)
                              If None, will use retrieved_documents from test_case metadata

        Returns:
            Dictionary with relevance evaluation results
        """
        # Save retrieved sources to JSON file if provided
        if retrieved_sources is not None:
            self._save_retrieved_sources_to_json(test_case, retrieved_sources)
        expected_documents = test_case.metadata.get("expected_documents", [])

        # Determine retrieved sources - either from parameter or from test case metadata
        if retrieved_sources is None:
            # Use pre-defined retrieved documents from test case
            retrieved_docs = test_case.metadata.get("retrieved_documents", [])
            retrieved_sources = []
            for doc in retrieved_docs:
                if isinstance(doc, dict) and "url" in doc:
                    retrieved_sources.append(doc["url"])
                elif isinstance(doc, str):
                    retrieved_sources.append(doc)

        # If no expected documents, this test doesn't require document retrieval
        if not expected_documents:
            return {
                "score": 1.0,
                "passes_criteria": True,
                "expected_count": 0,
                "retrieved_count": len(retrieved_sources),
                "matched_count": 0,
                "matched_documents": [],
                "reasoning": "No documents expected for this question",
                "evaluation_mode": "static"
                if retrieved_sources != test_case.metadata.get("retrieved_documents", [])
                else "dynamic",
            }

        # Extract URLs from expected documents
        expected_urls = set()
        for doc in expected_documents:
            if isinstance(doc, dict) and "url" in doc:
                expected_urls.add(doc["url"])
            elif isinstance(doc, str):
                expected_urls.add(doc)

        # Find matches between retrieved sources and expected documents
        matched_documents = []
        matched_urls = set()

        for source in retrieved_sources:
            for expected_url in expected_urls:
                if self._urls_match(source, expected_url):
                    matched_documents.append({"retrieved_source": source, "expected_url": expected_url})
                    matched_urls.add(expected_url)
                    break

        # Calculate score and pass criteria
        expected_count = len(expected_documents)
        matched_count = len(matched_urls)

        # Score is the ratio of matched documents to expected documents
        score = matched_count / expected_count if expected_count > 0 else 1.0

        # Pass if at least one expected document was retrieved
        passes_criteria = matched_count > 0

        reasoning = f"Retrieved {matched_count}/{expected_count} expected documents"
        if matched_count == 0:
            reasoning += ". No expected documents were found in retrieved sources."
        elif matched_count == expected_count:
            reasoning += ". All expected documents were successfully retrieved."
        else:
            reasoning += f". Missing {expected_count - matched_count} expected documents."

        return {
            "score": score,
            "passes_criteria": passes_criteria,
            "expected_count": expected_count,
            "retrieved_count": len(retrieved_sources),
            "matched_count": matched_count,
            "matched_documents": matched_documents,
            "expected_urls": list(expected_urls),
            "retrieved_sources": retrieved_sources,
            "reasoning": reasoning,
        }

    def _urls_match(self, retrieved_source: str, expected_url: str) -> bool:
        """
        Check if a retrieved source matches an expected URL

        Args:
            retrieved_source: Source returned by Isschat
            expected_url: Expected document URL

        Returns:
            True if they match, False otherwise
        """
        # Direct string match
        if retrieved_source == expected_url:
            return True

        # Check if expected URL is contained in retrieved source
        if expected_url in retrieved_source:
            return True

        # Check if retrieved source is contained in expected URL
        if retrieved_source in expected_url:
            return True

        # Parse URLs and compare components
        try:
            retrieved_parsed = urlparse(retrieved_source)
            expected_parsed = urlparse(expected_url)

            # Compare domain and path
            if retrieved_parsed.netloc == expected_parsed.netloc and retrieved_parsed.path == expected_parsed.path:
                return True

            # Check if the path contains the expected page ID (for Confluence)
            if expected_parsed.path and expected_parsed.path in retrieved_parsed.path:
                return True

        except Exception:
            # If URL parsing fails, fall back to string comparison
            pass

        return False

    def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """
        This method is required by BaseEvaluator but won't be used directly
        Document relevance evaluation will be called separately with retrieved sources
        """
        return EvaluationResult(
            test_id=test_case.test_id,
            category=test_case.category,
            test_name=test_case.test_name,
            question=test_case.question,
            response="",
            expected_behavior=test_case.expected_behavior,
            status=EvaluationStatus.SKIPPED,
            score=0.0,
            error_message="Document relevance evaluation requires retrieved sources",
        )

    def log_document_relevance_result(self, test_id: str, evaluation: Dict[str, Any]):
        """
        Log document relevance evaluation result in the same format as other evaluators

        Args:
            test_id: Test identifier
            evaluation: Document relevance evaluation results
        """
        score_display = f"{evaluation['matched_count']}/{evaluation['expected_count']}"
        passes = evaluation["passes_criteria"]

        logger.info(f"Test {test_id}: Document relevance score={score_display}, passes={passes}")

    def _save_retrieved_sources_to_json(self, test_case: TestCase, retrieved_sources: List[str]):
        """
        Save retrieved sources back to the JSON test dataset file

        Args:
            test_case: The test case being evaluated
            retrieved_sources: List of source URLs retrieved by Isschat
        """
        try:
            # Determine which JSON file to update based on test category
            if test_case.category == TestCategory.CONVERSATIONAL:
                json_file_path = Path(__file__).parent.parent / "config" / "test_datasets" / "conversational_tests.json"
            elif test_case.category == TestCategory.ROBUSTNESS:
                json_file_path = Path(__file__).parent.parent / "config" / "test_datasets" / "robustness_tests.json"
            else:
                logger.warning(f"Unknown test category {test_case.category}, cannot save retrieved sources")
                return

            # Read current JSON file
            if not json_file_path.exists():
                logger.error(f"JSON file not found: {json_file_path}")
                return

            with open(json_file_path, "r", encoding="utf-8") as f:
                test_data = json.load(f)

            # Find the test case and update it with retrieved sources
            updated = False
            for test in test_data:
                if test.get("test_id") == test_case.test_id:
                    # Add retrieved_sources to metadata
                    if "metadata" not in test:
                        test["metadata"] = {}

                    # Format retrieved sources as list of dictionaries with URL, title, and timestamp
                    formatted_sources = []
                    for source in retrieved_sources:
                        parsed_sources = self._parse_source_text(source)
                        for parsed_source in parsed_sources:
                            formatted_sources.append(
                                {
                                    "url": parsed_source["url"],
                                    "title": parsed_source["title"],
                                    "retrieved_at": self._get_current_timestamp(),
                                }
                            )

                    test["metadata"]["retrieved_sources"] = formatted_sources
                    updated = True
                    logger.info(f"Updated test {test_case.test_id} with {len(retrieved_sources)} retrieved sources")
                    break

            if not updated:
                logger.warning(f"Test case {test_case.test_id} not found in {json_file_path}")
                return

            # Write updated JSON back to file
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully saved retrieved sources to {json_file_path}")

        except Exception as e:
            logger.error(f"Error saving retrieved sources to JSON: {e}")

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime

        return datetime.now().isoformat()

    def _parse_source_text(self, source: str) -> List[Dict[str, str]]:
        """
        Parse source text to extract URLs and titles from Isschat's formatted response

        Args:
            source: Source text from Isschat (may contain markdown links)

        Returns:
            List of dictionaries with 'url' and 'title' keys
        """
        import re

        sources = []

        # Pattern to match markdown links: [title](url)
        markdown_pattern = r"\[([^\]]+)\]\(([^)]+)\)"

        # Find all markdown links in the source text
        matches = re.findall(markdown_pattern, source)

        if matches:
            # Process all markdown links found
            for title, url in matches:
                sources.append({"url": url.strip(), "title": title.strip()})
            return sources

        # Pattern to match plain URLs
        url_pattern = r"https?://[^\s\)]+"
        url_matches = re.findall(url_pattern, source)

        if url_matches:
            # Process all URLs found
            for url in url_matches:
                url = url.strip()
                title = self._extract_title_from_url(url)
                sources.append({"url": url, "title": title})
            return sources

        # If no URLs found, treat the entire source as title and try to extract URL
        sources.append({"url": source.strip(), "title": self._extract_title_from_source_fallback(source)})
        return sources

    def _extract_title_from_url(self, url: str) -> str:
        """
        Extract title from a clean URL

        Args:
            url: Clean URL

        Returns:
            Extracted title
        """
        try:
            parsed_url = urlparse(url)

            # For Confluence URLs, try to extract page title from URL
            if "atlassian.net/wiki" in url:
                # Extract page ID and try to create a meaningful title
                path_parts = parsed_url.path.split("/")
                if "pages" in path_parts:
                    page_index = path_parts.index("pages")
                    if page_index + 1 < len(path_parts):
                        page_id = path_parts[page_index + 1]
                        # If there's a title after the page ID, use it
                        if page_index + 2 < len(path_parts):
                            title_part = path_parts[page_index + 2]
                            # Replace URL encoding and make it readable
                            title = title_part.replace("+", " ").replace("%20", " ")
                            return title
                        else:
                            return f"Confluence Page {page_id}"

            # For other URLs, use the last part of the path or domain
            if parsed_url.path and parsed_url.path != "/":
                path_parts = [part for part in parsed_url.path.split("/") if part]
                if path_parts:
                    title = path_parts[-1].replace("-", " ").replace("_", " ")
                    return title.title()

            # Fallback to domain name
            return parsed_url.netloc.replace("www.", "")

        except Exception:
            # Fallback: use domain or first part of URL
            return url.split("/")[2] if "/" in url else url

    def _extract_title_from_source_fallback(self, source: str) -> str:
        """
        Fallback method to extract title from source text

        Args:
            source: Source text

        Returns:
            Extracted title
        """
        try:
            # If the source contains title information, extract it
            if ":" in source:
                parts = source.split(":", 1)
                return parts[0].strip()

            # Look for text before newlines or markdown
            lines = source.split("\n")
            if lines:
                first_line = lines[0].strip()
                if first_line and len(first_line) < 100:
                    return first_line

            # Fallback: use first 50 characters as title
            return source[:50] + ("..." if len(source) > 50 else "")

        except Exception:
            # Final fallback: use first 30 characters of source
            return source[:30] + ("..." if len(source) > 30 else "")
