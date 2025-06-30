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

    def chunk_document(self, document: Document) -> List[Document]:
        """Convenience method to chunk a single document."""
        return self.chunk_documents([document])

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

                # Add contextual information to chunk content
                enriched_content = self._add_context_to_chunk(chunk_text, chunk_metadata)
                chunks.append(Document(page_content=enriched_content, metadata=chunk_metadata))
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

            # First, extract and process tables separately
            table_sections = self._extract_table_sections(text, doc.metadata)

            # Remove tables from text for regular processing
            text_without_tables = self._remove_tables_from_text(text)

            # Split by Confluence macros and common patterns (excluding tables)
            macro_patterns = [
                r"<ac:structured-macro[^>]*>.*?</ac:structured-macro>",  # Confluence macros
                r"<ac:layout[^>]*>.*?</ac:layout>",  # Layouts
                r"```[\s\S]*?```",  # Code blocks
            ]

            # Find all macro boundaries
            boundaries = [0]
            for pattern in macro_patterns:
                for match in re.finditer(pattern, text_without_tables, re.IGNORECASE | re.DOTALL):
                    boundaries.extend([match.start(), match.end()])

            boundaries.append(len(text_without_tables))
            boundaries = sorted(set(boundaries))

            # Create sections based on boundaries
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                section = text_without_tables[start:end].strip()

                if section and len(section) > 50:  # Minimum viable section
                    sections.append(section)

            # Add table sections
            sections.extend(table_sections)

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

                    # Add contextual information to chunk content
                    enriched_content = self._add_context_to_chunk(section, metadata)
                    chunked_docs.append(Document(page_content=enriched_content, metadata=metadata))

        return chunked_docs

    def _chunk_by_headers(self, documents: List[Document]) -> List[Document]:
        """Hierarchical chunking based on headers."""
        chunked_docs = []

        for doc in documents:
            hierarchical_chunks = self._create_hierarchical_chunks(doc.page_content)

            for chunk_data in hierarchical_chunks:
                metadata = doc.metadata.copy()
                metadata.update(
                    {
                        "chunk_index": chunk_data["index"],
                        "chunk_strategy": "hierarchical",
                        "content_length": len(chunk_data["content"]),
                        "hierarchy": chunk_data["hierarchy"],
                        "section_path": " > ".join(chunk_data["hierarchy"]),
                        "header_level": chunk_data["level"],
                    }
                )

                # Add contextual information to chunk content
                enriched_content = self._add_context_to_chunk(chunk_data["content"], metadata)
                chunked_docs.append(Document(page_content=enriched_content, metadata=metadata))

        return chunked_docs

    def _create_hierarchical_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create hierarchical chunks with proper section tracking."""
        # Patterns for different header formats
        markdown_header = r"^(#{1,6})\s+(.+)$"
        html_header = r"<h([1-6])[^>]*>(.*?)</h[1-6]>"

        chunks = []
        current_hierarchy = [""] * 6  # Track headers at each level
        current_content = []
        current_level = 0
        chunk_index = 0

        lines = text.split("\n")

        for line in lines:
            # Check for markdown headers
            md_match = re.match(markdown_header, line.strip())
            if md_match:
                level = len(md_match.group(1))
                header_text = md_match.group(2).strip()

                # Save current chunk if it has content
                if current_content and any(line.strip() for line in current_content):
                    hierarchy = [h for h in current_hierarchy[:current_level] if h]
                    if hierarchy:
                        chunks.append(
                            {
                                "index": chunk_index,
                                "content": "\n".join(current_content).strip(),
                                "hierarchy": hierarchy.copy(),
                                "level": current_level,
                            }
                        )
                        chunk_index += 1
                # Update hierarchy
                current_hierarchy[level - 1] = header_text
                # Clear lower levels
                for i in range(level, 6):
                    current_hierarchy[i] = ""
                current_level = level
                current_content = [line]
                continue
            # Check for HTML headers
            html_match = re.search(html_header, line, re.IGNORECASE)
            if html_match:
                level = int(html_match.group(1))
                header_text = re.sub(r"<[^>]+>", "", html_match.group(2)).strip()
                # Save current chunk if it has content
                if current_content and any(line.strip() for line in current_content):
                    hierarchy = [h for h in current_hierarchy[:current_level] if h]
                    if hierarchy:
                        chunks.append(
                            {
                                "index": chunk_index,
                                "content": "\n".join(current_content).strip(),
                                "hierarchy": hierarchy.copy(),
                                "level": current_level,
                            }
                        )
                        chunk_index += 1
                # Update hierarchy
                current_hierarchy[level - 1] = header_text
                # Clear lower levels
                for i in range(level, 6):
                    current_hierarchy[i] = ""
                current_level = level
                current_content = [line]
                continue
            # Regular content line
            current_content.append(line)
        # Don't forget the last chunk
        if current_content and any(line.strip() for line in current_content):
            hierarchy = [h for h in current_hierarchy[:current_level] if h]
            if hierarchy:
                chunks.append(
                    {
                        "index": chunk_index,
                        "content": "\n".join(current_content).strip(),
                        "hierarchy": hierarchy.copy(),
                        "level": current_level,
                    }
                )
        # If no headers found, create a single chunk with document title if available
        if not chunks and text.strip():
            chunks.append({"index": 0, "content": text.strip(), "hierarchy": ["Document"], "level": 1})
        return chunks

    def _split_by_confluence_headers(self, text: str) -> List[str]:
        """Split text by Confluence headers (h1-h6) - legacy method for backwards compatibility."""
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

    def _extract_table_sections(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """Extract and process table sections with proper structure and context."""
        table_sections = []

        # Patterns for different table formats
        html_table_pattern = r"<table[^>]*>(.*?)</table>"
        markdown_table_pattern = r"(\|[^|\n]*\|(?:\n\|[^|\n]*\|)*)"

        # Extract HTML tables
        for match in re.finditer(html_table_pattern, text, re.IGNORECASE | re.DOTALL):
            table_html = match.group(0)
            parsed_table = self._parse_html_table(table_html, metadata)
            if parsed_table:
                table_sections.append(parsed_table)

        # Extract markdown tables
        for match in re.finditer(markdown_table_pattern, text, re.MULTILINE):
            table_md = match.group(0)
            parsed_table = self._parse_markdown_table(table_md, metadata)
            if parsed_table:
                table_sections.append(parsed_table)

        return table_sections

    def _remove_tables_from_text(self, text: str) -> str:
        """Remove tables from text to avoid double processing."""
        # Remove HTML tables
        text = re.sub(r"<table[^>]*>.*?</table>", "", text, flags=re.IGNORECASE | re.DOTALL)

        # Remove markdown tables
        text = re.sub(r"\|[^|\n]*\|(?:\n\|[^|\n]*\|)*", "", text, flags=re.MULTILINE)

        return text

    def _parse_html_table(self, table_html: str, metadata: Dict[str, Any]) -> str:
        """Parse HTML table and create structured content with context."""
        # Extract table rows
        row_pattern = r"<tr[^>]*>(.*?)</tr>"
        rows = re.findall(row_pattern, table_html, re.IGNORECASE | re.DOTALL)

        if not rows:
            return ""

        # Parse header and data rows
        parsed_rows = []
        for i, row in enumerate(rows):
            # Extract cells (th or td)
            cell_pattern = r"<t[hd][^>]*>(.*?)</t[hd]>"
            cells = re.findall(cell_pattern, row, re.IGNORECASE | re.DOTALL)

            # Clean cell content
            clean_cells = []
            for cell in cells:
                # Remove HTML tags and normalize whitespace
                clean_cell = re.sub(r"<[^>]+>", " ", cell).strip()
                clean_cell = re.sub(r"\s+", " ", clean_cell)
                clean_cells.append(clean_cell)

            if clean_cells:
                parsed_rows.append(clean_cells)

        if not parsed_rows:
            return ""

        # Build structured table content
        context_info = self._get_document_context(metadata)
        table_content = f"{context_info}\n\n[TABLEAU]\n"

        # Add header if first row looks like headers
        if parsed_rows:
            first_row = parsed_rows[0]
            table_content += f"En-têtes: {' | '.join(first_row)}\n\n"

            # Add data rows
            for i, row in enumerate(parsed_rows[1:], 1):
                table_content += f"Ligne {i}: {' | '.join(row)}\n"

        return table_content

    def _parse_markdown_table(self, table_md: str, metadata: Dict[str, Any]) -> str:
        """Parse markdown table and create structured content with context."""
        lines = table_md.strip().split("\n")

        if len(lines) < 2:
            return ""

        # Parse table rows
        parsed_rows = []
        for line in lines:
            if "|" in line and not re.match(r"^\s*\|[\s\-:]*\|\s*$", line):  # Skip separator lines
                cells = [cell.strip() for cell in line.split("|")[1:-1]]  # Remove empty first/last
                if cells:
                    parsed_rows.append(cells)

        if not parsed_rows:
            return ""

        # Build structured table content
        context_info = self._get_document_context(metadata)
        table_content = f"{context_info}\n\n[TABLEAU]\n"

        if parsed_rows:
            # First row is usually headers in markdown tables
            headers = parsed_rows[0]
            table_content += f"En-têtes: {' | '.join(headers)}\n\n"

            # Add data rows
            for i, row in enumerate(parsed_rows[1:], 1):
                table_content += f"Ligne {i}: {' | '.join(row)}\n"

        return table_content

    def _add_context_to_chunk(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add contextual information to chunk content."""
        context_info = self._get_document_context(metadata)
        return f"{context_info}\n\n{content}"

    def _get_document_context(self, metadata: Dict[str, Any]) -> str:
        """Generate contextual information for a chunk."""
        context_parts = []

        # Document title
        if metadata.get("title"):
            context_parts.append(f"Document: {metadata['title']}")

        # Space information (for Confluence)
        if metadata.get("space_key"):
            context_parts.append(f"Espace: {metadata['space_key']}")

        # URL/Path information
        if metadata.get("url"):
            context_parts.append(f"URL: {metadata['url']}")

        # Hierarchy information (for hierarchical chunks)
        if metadata.get("section_path"):
            context_parts.append(f"Section: {metadata['section_path']}")

        # Source information
        if metadata.get("source"):
            context_parts.append(f"Source: {metadata['source']}")

        if context_parts:
            return f"[{' | '.join(context_parts)}]"

        return "[Contexte non disponible]"
