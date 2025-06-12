"""
Response formatting for RAG system outputs.
"""

from typing import Dict, Any, List, Optional


class ResponseFormatter:
    """Formats RAG system responses for different output formats."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the response formatter.

        Args:
            config: Formatting configuration
        """
        self.config = config or {}
        self.include_sources = self.config.get("include_sources", True)
        self.include_confidence = self.config.get("include_confidence", False)
        self.max_source_length = self.config.get("max_source_length", 200)

    def format_response(
        self, response: str, context_docs: List[Dict[str, Any]], confidence_scores: Optional[List[float]] = None
    ) -> str:
        """
        Format a RAG response with optional source information.

        Args:
            response: Generated response text
            context_docs: Documents used as context
            confidence_scores: Confidence scores for retrieved documents

        Returns:
            str: Formatted response
        """
        formatted_response = response

        if self.include_sources and context_docs:
            sources_section = self._format_sources(context_docs, confidence_scores)
            formatted_response += "\n\n" + sources_section

        return formatted_response

    def _format_sources(
        self, context_docs: List[Dict[str, Any]], confidence_scores: Optional[List[float]] = None
    ) -> str:
        """
        Format source information with clickable links.

        Args:
            context_docs: Documents used as context
            confidence_scores: Confidence scores for documents

        Returns:
            str: Formatted sources section with clickable links
        """
        if not context_docs:
            return ""

        sources_lines = ["**Sources:**"]

        for i, doc in enumerate(context_docs):
            source_line = f"{i + 1}. "

            # Add document title with clickable link if URL is available
            metadata = doc.get("metadata", {})
            title = metadata.get("title", metadata.get("source", f"Document {i + 1}"))
            source = metadata.get("source", "")
            url = metadata.get("url", "")

            if url and url != "#":
                # Create a clickable link for Streamlit
                source_line += f"**[{title}]({url})**"
            else:
                source_line += f"**{title}**"

            # Add source information if available and different from title
            if source and source != title:
                source_line += f" ({source})"

            # Add confidence score if available
            if confidence_scores and i < len(confidence_scores):
                confidence = confidence_scores[i]
                source_line += f" (confidence: {confidence:.2f})"

            # Add excerpt
            content = doc.get("content", "")
            if len(content) > self.max_source_length:
                content = content[: self.max_source_length] + "..."
            source_line += f"\n   _{content}_"

            sources_lines.append(source_line)

        return "\n\n".join(sources_lines)

    def format_for_api(
        self, response: str, context_docs: List[Dict[str, Any]], confidence_scores: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Format response for API consumption.

        Args:
            response: Generated response text
            context_docs: Documents used as context
            confidence_scores: Confidence scores for retrieved documents

        Returns:
            Dict: Structured API response
        """
        api_response = {"response": response, "sources": []}

        if context_docs:
            for i, doc in enumerate(context_docs):
                source_info = {"index": i, "content": doc.get("content", ""), "metadata": doc.get("metadata", {})}

                if confidence_scores and i < len(confidence_scores):
                    source_info["confidence"] = confidence_scores[i]

                api_response["sources"].append(source_info)

        return api_response

    def format_for_evaluation(
        self,
        response: str,
        context_docs: List[Dict[str, Any]],
        query: str,
        confidence_scores: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Format response for evaluation purposes.

        Args:
            response: Generated response text
            context_docs: Documents used as context
            query: Original query
            confidence_scores: Confidence scores for retrieved documents

        Returns:
            Dict: Structured evaluation response
        """
        eval_response = {
            "query": query,
            "response": response,
            "num_sources": len(context_docs),
            "sources": context_docs,
            "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else None,
            "response_length": len(response),
            "has_sources": len(context_docs) > 0,
        }

        return eval_response
