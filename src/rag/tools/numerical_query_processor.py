"""
Numerical query processor for handling queries that require aggregation across documents.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ...core.documents import RetrievalDocument


@dataclass
class NumericalQueryResult:
    """Result of numerical query processing."""

    query_type: str
    aggregated_value: Optional[float]
    source_chunks: List[Dict[str, Any]]
    confidence: float
    explanation: str


class NumericalQueryProcessor:
    """Processes numerical queries and aggregates results across document chunks."""

    def __init__(self):
        self.numerical_patterns = [
            (r"how many\s+(\w+)", "count"),
            (r"number of\s+(\w+)", "count"),
            (r"total\s+(\w+)", "sum"),
            (r"sum of\s+(\w+)", "sum"),
            (r"count\s+(\w+)", "count"),
            (r"(\d+)\s+(\w+)", "exact_count"),
        ]

        self.target_entities = [
            "interviews?",
            "participants?",
            "users?",
            "people",
            "surveys?",
            "responses?",
            "counts?",
            "totals?",
        ]

    def is_numerical_query(self, query: str) -> bool:
        """Detect if a query is asking for numerical information."""
        query_lower = query.lower()

        # Check for numerical query patterns
        for pattern, _ in self.numerical_patterns:
            if re.search(pattern, query_lower):
                return True

        # Check for numerical keywords
        numerical_keywords = [
            "how many",
            "number of",
            "total",
            "sum",
            "count",
            "amount",
            "quantity",
            "statistics",
            "stats",
            "figures",
            "data",
        ]

        return any(keyword in query_lower for keyword in numerical_keywords)

    def extract_query_intent(self, query: str) -> Dict[str, Any]:
        """Extract the intent and target entity from a numerical query."""
        query_lower = query.lower()

        for pattern, query_type in self.numerical_patterns:
            match = re.search(pattern, query_lower)
            if match:
                entity = match.group(1) if match.lastindex >= 1 else None
                return {"type": query_type, "entity": entity, "original_query": query, "pattern_match": match.group(0)}

        return {"type": "general_numerical", "entity": None, "original_query": query, "pattern_match": None}

    def process_numerical_query(self, query: str, retrieved_chunks: List[RetrievalDocument]) -> NumericalQueryResult:
        """Process a numerical query and aggregate results from chunks."""

        # Extract query intent
        intent = self.extract_query_intent(query)

        # Extract numerical information from chunks
        numerical_data = self._extract_numerical_data_from_chunks(retrieved_chunks, intent)

        # Aggregate the numerical data
        aggregated_result = self._aggregate_numerical_data(numerical_data, intent)

        return aggregated_result

    def _extract_numerical_data_from_chunks(
        self, chunks: List[RetrievalDocument], intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract numerical data from document chunks."""
        numerical_data = []

        for chunk in chunks:
            chunk_data = self._extract_numbers_from_chunk(chunk, intent)
            if chunk_data:
                numerical_data.extend(chunk_data)

        return numerical_data

    def _extract_numbers_from_chunk(self, chunk: RetrievalDocument, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract numerical values from a single chunk."""
        content = chunk.content
        metadata = chunk.metadata

        # Use pre-extracted numbers if available (from hierarchy-aware chunking)
        if metadata.get("extracted_numbers"):
            numbers = metadata["extracted_numbers"]
            # Filter numbers based on query intent
            filtered_numbers = self._filter_numbers_by_intent(numbers, intent)
            return [
                {
                    "value": num["value"],
                    "type": num["type"],
                    "context": num["context"],
                    "chunk_metadata": {
                        "title": metadata.get("title", ""),
                        "section_path": metadata.get("section_path", []),
                        "section_breadcrumb": metadata.get("section_breadcrumb", ""),
                        "url": metadata.get("url", ""),
                        "chunk_index": metadata.get("chunk_index", 0),
                    },
                }
                for num in filtered_numbers
            ]

        # Fallback: extract numbers directly from content
        numbers = self._extract_numbers_from_content(content, intent)
        return [
            {
                "value": num["value"],
                "type": num["type"],
                "context": num["context"],
                "chunk_metadata": {
                    "title": metadata.get("title", ""),
                    "section_path": metadata.get("section_path", []),
                    "section_breadcrumb": metadata.get("section_breadcrumb", ""),
                    "url": metadata.get("url", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                },
            }
            for num in numbers
        ]

    def _filter_numbers_by_intent(self, numbers: List[Dict[str, Any]], intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter numbers based on query intent."""
        if not intent.get("entity"):
            return numbers

        entity = intent["entity"].lower()
        relevant_numbers = []

        for num in numbers:
            context = num.get("context", "").lower()
            num_type = num.get("type", "").lower()

            # Check if the number is related to the query entity
            if (
                entity in context
                or entity in num_type
                or any(re.search(f"{entity}", context) for entity in self.target_entities)
            ):
                relevant_numbers.append(num)

        return relevant_numbers if relevant_numbers else numbers

    def _extract_numbers_from_content(self, content: str, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract numbers directly from content text."""
        numbers = []

        # Enhanced patterns for different types of numerical information
        patterns = [
            (r"(\d+(?:[.,]\d+)*)\s+(interviews?)", "interview_count"),
            (r"(\d+(?:[.,]\d+)*)\s+(total|sum|count)", "total_count"),
            (r"(\d+(?:[.,]\d+)*)\s*%", "percentage"),
            (r"(\d+(?:[.,]\d+)*)\s+(participants?|users?|people)", "participant_count"),
            (r"(\d+(?:[.,]\d+)*)\s+(responses?|surveys?)", "response_count"),
            (r"(\d+(?:[.,]\d+)*)\s+(on\s+)?teora", "teora_count"),
            (r"(\d+(?:[.,]\d+)*)", "number"),
        ]

        for pattern, number_type in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                value_str = match.group(1).replace(",", ".")
                try:
                    value = float(value_str)
                    numbers.append(
                        {"value": value, "type": number_type, "context": match.group(0), "position": match.start()}
                    )
                except ValueError:
                    continue

        return numbers

    def _aggregate_numerical_data(
        self, numerical_data: List[Dict[str, Any]], intent: Dict[str, Any]
    ) -> NumericalQueryResult:
        """Aggregate numerical data based on query intent."""

        if not numerical_data:
            return NumericalQueryResult(
                query_type=intent["type"],
                aggregated_value=None,
                source_chunks=[],
                confidence=0.0,
                explanation="No numerical data found in the retrieved documents.",
            )

        query_type = intent["type"]

        # Group numbers by type and source
        numbers_by_type = {}
        for item in numerical_data:
            num_type = item["type"]
            if num_type not in numbers_by_type:
                numbers_by_type[num_type] = []
            numbers_by_type[num_type].append(item)

        # Choose the most relevant number type
        relevant_type = self._choose_relevant_type(numbers_by_type, intent)
        relevant_numbers = numbers_by_type.get(relevant_type, [])

        # Aggregate based on query type
        if query_type in ["count", "sum"]:
            aggregated_value = self._aggregate_sum(relevant_numbers)
        elif query_type == "exact_count":
            aggregated_value = self._find_exact_count(relevant_numbers, intent)
        else:
            aggregated_value = self._aggregate_sum(relevant_numbers)

        # Calculate confidence
        confidence = self._calculate_confidence(relevant_numbers, intent)

        # Generate explanation
        explanation = self._generate_explanation(relevant_numbers, aggregated_value, intent, relevant_type)

        return NumericalQueryResult(
            query_type=query_type,
            aggregated_value=aggregated_value,
            source_chunks=relevant_numbers,
            confidence=confidence,
            explanation=explanation,
        )

    def _choose_relevant_type(self, numbers_by_type: Dict[str, List[Dict[str, Any]]], intent: Dict[str, Any]) -> str:
        """Choose the most relevant number type based on query intent."""
        entity = intent.get("entity", "").lower()

        # Priority mapping based on entity
        priority_map = {
            "interview": ["interview_count", "teora_count", "total_count", "participant_count"],
            "participant": ["participant_count", "total_count", "interview_count"],
            "user": ["participant_count", "total_count", "interview_count"],
            "response": ["response_count", "total_count", "interview_count"],
            "survey": ["response_count", "total_count", "interview_count"],
        }

        # Get priority list for entity
        priorities = priority_map.get(entity, ["interview_count", "total_count", "participant_count"])

        # Find the highest priority type that has data
        for priority_type in priorities:
            if priority_type in numbers_by_type:
                return priority_type

        # Fallback to the type with the most numbers
        return max(numbers_by_type.keys(), key=lambda k: len(numbers_by_type[k]))

    def _aggregate_sum(self, numbers: List[Dict[str, Any]]) -> float:
        """Sum numerical values, handling duplicates by section and context."""
        if not numbers:
            return 0.0

        # For interview counts, we need to be smart about aggregation
        # If we have a total (25) and breakdown (5+8+12), use the total
        # If we only have breakdown numbers, sum them

        # Separate total numbers from breakdown numbers
        total_numbers = []
        breakdown_numbers = []

        for num in numbers:
            context = num.get("context", "").lower()
            section = num["chunk_metadata"].get("section_breadcrumb", "")

            # Check if this is a total/summary number
            if (
                "total" in context
                or "completed" in context
                or "current numbers" in section.lower()
                or "statistics" in section.lower()
                and "breakdown" not in section.lower()
            ):
                total_numbers.append(num)
            else:
                breakdown_numbers.append(num)

        # Prefer total numbers if available
        if total_numbers:
            # Use the highest total number (most comprehensive)
            return max(num["value"] for num in total_numbers)

        # Otherwise, sum breakdown numbers (avoiding duplicates by section)
        sections_seen = set()
        unique_numbers = []

        for num in breakdown_numbers:
            section_key = num["chunk_metadata"].get("section_breadcrumb", "")
            context_key = f"{section_key}:{num.get('context', '')}"

            if context_key not in sections_seen:
                sections_seen.add(context_key)
                unique_numbers.append(num)

        return sum(num["value"] for num in unique_numbers)

    def _find_exact_count(self, numbers: List[Dict[str, Any]], intent: Dict[str, Any]) -> Optional[float]:
        """Find exact count from numbers."""
        if not numbers:
            return None

        # Return the first number found (assuming exact match)
        return numbers[0]["value"]

    def _calculate_confidence(self, numbers: List[Dict[str, Any]], intent: Dict[str, Any]) -> float:
        """Calculate confidence score for the aggregated result."""
        if not numbers:
            return 0.0

        # Base confidence on number of sources and context relevance
        base_confidence = min(len(numbers) * 0.3, 1.0)

        # Boost confidence if entity matches query intent
        entity = intent.get("entity", "").lower()
        entity_match_bonus = 0.0

        for num in numbers:
            context = num.get("context", "").lower()
            if entity and entity in context:
                entity_match_bonus += 0.2

        # Boost confidence for interview-specific queries
        interview_bonus = 0.0
        if "interview" in intent.get("original_query", "").lower():
            for num in numbers:
                if "interview" in num.get("type", "").lower():
                    interview_bonus += 0.3

        return min(base_confidence + entity_match_bonus + interview_bonus, 1.0)

    def _generate_explanation(
        self,
        numbers: List[Dict[str, Any]],
        aggregated_value: Optional[float],
        intent: Dict[str, Any],
        relevant_type: str,
    ) -> str:
        """Generate explanation for the aggregated result."""
        if aggregated_value is None:
            return "No numerical data found in the retrieved documents."

        if not numbers:
            return f"Found value: {aggregated_value}"

        # Generate source information
        sources = []
        for num in numbers:
            metadata = num["chunk_metadata"]
            section = metadata.get("section_breadcrumb", "Unknown Section")
            sources.append(f"{section} ({num['value']})")

        query_type = intent["type"]
        entity = intent.get("entity", "items")

        if query_type in ["count", "sum"]:
            explanation = f"Found {aggregated_value} {entity} across {len(numbers)} source(s):\n"
        else:
            explanation = f"Found {aggregated_value} {entity}:\n"

        explanation += "\n".join(f"• {source}" for source in sources[:5])  # Limit to 5 sources

        if len(sources) > 5:
            explanation += f"\n... and {len(sources) - 5} more sources"

        return explanation
