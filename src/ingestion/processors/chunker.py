import re
from typing import List, Dict, Any, Optional, Literal
from ...core.interfaces import Document


class DocumentChunker:
    """Simple document chunking."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []
        for doc in documents:
            chunks = self._split_document(doc)
            chunked_docs.extend(chunks)
        return chunked_docs

    def _split_document(self, document: Document) -> List[Document]:
        content = document.page_content

        if len(content) <= self.chunk_size:
            return [document]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + self.chunk_size

            # Try to break at word boundary
            if end < len(content):
                while end > start and content[end] not in " \n\t.,;!?":
                    end -= 1
                if end == start:
                    end = start + self.chunk_size

            chunk_text = content[start:end].strip()
            if chunk_text:
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({"chunk_index": chunk_index, "chunk_size": len(chunk_text), "is_chunk": True})

                chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
                chunk_index += 1

            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def get_chunking_stats(self, original_documents: List[Document], chunks: List[Document]) -> Dict[str, Any]:
        """Calculate statistics about the chunking process."""
        original_count = len(original_documents)
        chunk_count = len(chunks)
        avg_chunks_per_doc = chunk_count / original_count if original_count > 0 else 0

        return {"original_count": original_count, "chunk_count": chunk_count, "avg_chunks_per_doc": avg_chunks_per_doc}


class ConfluenceChunker(DocumentChunker):
    """Confluence-specific chunking strategies."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        strategy: Literal["confluence_sections", "hierarchical"] = "confluence_sections",
    ):
        super().__init__(config)
        self.strategy = strategy

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        if self.strategy == "confluence_sections":
            return self._chunk_confluence_sections(documents)
        elif self.strategy == "hierarchical":
            return self._chunk_by_headers(documents)
        else:
            return super().chunk_documents(documents)

    def _chunk_confluence_sections(self, documents: List[Document]) -> List[Document]:
        """Confluence-specific chunking based on macros and structure."""
        chunked_docs = []

        for doc in documents:
            text = doc.page_content
            sections = []

            # Split by Confluence macros and common patterns
            macro_patterns = [
                r"<ac:structured-macro[^>]*>.*?</ac:structured-macro>",  # Confluence macros
                r"<table[^>]*>.*?</table>",  # Tables
                r"<ac:layout[^>]*>.*?</ac:layout>",  # Layouts
                r"```[\s\S]*?```",  # Code blocks
                r"\|[^|]*\|.*?\n",  # Table rows
            ]

            # Find all macro boundaries
            boundaries = [0]
            for pattern in macro_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                    boundaries.extend([match.start(), match.end()])

            boundaries.append(len(text))
            boundaries = sorted(set(boundaries))

            # Create sections based on boundaries
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                section = text[start:end].strip()

                if section and len(section) > 50:  # Minimum viable section
                    sections.append(section)

            # If no meaningful sections found, fall back to header splitting
            if not sections:
                sections = self._split_by_confluence_headers(text)

            for i, section in enumerate(sections):
                if section.strip():
                    metadata = doc.metadata.copy()
                    metadata.update(
                        {
                            "chunk_index": i,
                            "chunk_strategy": "confluence_sections",
                            "content_length": len(section),
                        }
                    )

                    chunked_docs.append(Document(page_content=section, metadata=metadata))

        return chunked_docs

    def _chunk_by_headers(self, documents: List[Document]) -> List[Document]:
        """Hierarchical chunking based on headers."""
        chunked_docs = []

        for doc in documents:
            sections = self._split_by_confluence_headers(doc.page_content)

            for i, section in enumerate(sections):
                if section.strip():
                    metadata = doc.metadata.copy()
                    metadata.update(
                        {
                            "chunk_index": i,
                            "chunk_strategy": "hierarchical",
                            "content_length": len(section),
                        }
                    )

                    chunked_docs.append(Document(page_content=section, metadata=metadata))

        return chunked_docs

    def _split_by_confluence_headers(self, text: str) -> List[str]:
        """Split text by Confluence headers (h1-h6)."""
        # Confluence headers in both formats
        header_pattern = r"^(#{1,6}\s+.+|<h[1-6][^>]*>.*?</h[1-6]>)"

        sections = []
        current_section = []

        for line in text.split("\n"):
            if re.match(header_pattern, line.strip(), re.IGNORECASE):
                if current_section:
                    sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append("\n".join(current_section))

        # Filter out empty or too small sections
        return [s for s in sections if s.strip() and len(s.strip()) > 100]
