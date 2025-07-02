import re
from typing import List, Dict, Any, Optional, Literal
import tiktoken
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
    """Confluence-specific chunking strategies with multi-level semantic chunking."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        strategy: Literal["confluence_sections", "hierarchical", "semantic_hierarchical"] = "semantic_hierarchical",
        model_name: str = "gpt-4",
    ):
        super().__init__(config)
        self.strategy = strategy
        self.model_name = model_name

        # Initialize tokenizer for token-based sizing
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Dynamic chunk sizes based on content type
        # If chunk_size is provided, use it as the default for all content types
        default_token_limit = self.config.get("text_chunk_tokens", 1000)
        if "chunk_size" in self.config:
            # Convert character-based chunk_size to approximate token limit
            # Rough approximation: 1 token ≈ 4 characters
            default_token_limit = max(self.config["chunk_size"] // 4, 10)

        self.content_type_limits = {
            "text": self.config.get("text_chunk_tokens", default_token_limit),
            "table": self.config.get("table_chunk_tokens", int(default_token_limit * 2)),
            "list": self.config.get("list_chunk_tokens", int(default_token_limit * 1.5)),
            "code": self.config.get("code_chunk_tokens", int(default_token_limit * 1.5)),
        }

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        if self.strategy == "confluence_sections":
            return self._chunk_confluence_sections(documents)
        elif self.strategy == "hierarchical":
            return self._chunk_by_headers(documents)
        elif self.strategy == "semantic_hierarchical":
            return self._chunk_semantic_hierarchical(documents)
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

    def _chunk_semantic_hierarchical(self, documents: List[Document]) -> List[Document]:
        """Multi-level semantic chunking for Confluence with token-based sizing."""
        chunked_docs = []

        for doc in documents:
            # Extract document structure
            structure = self._analyze_document_structure(doc.page_content)

            # Create chunks based on semantic boundaries
            semantic_chunks = self._create_semantic_chunks(doc, structure)

            chunked_docs.extend(semantic_chunks)

        return chunked_docs

    def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure to identify semantic boundaries."""
        structure = {"headers": [], "tables": [], "lists": [], "code_blocks": [], "content_sections": []}

        lines = text.split("\n")
        current_position = 0

        for i, line in enumerate(lines):
            line_start = current_position
            current_position += len(line) + 1  # +1 for newline

            # Detect headers (both markdown and HTML)
            if self._is_header(line):
                level = self._get_header_level(line)
                title = self._extract_header_text(line)
                structure["headers"].append(
                    {"level": level, "title": title, "line_num": i, "position": line_start, "text": line}
                )

            # Detect table starts
            elif self._is_table_start(line):
                table_end = self._find_table_end(lines, i)
                if table_end > i:
                    structure["tables"].append(
                        {
                            "start_line": i,
                            "end_line": table_end,
                            "start_pos": line_start,
                            "content": "\n".join(lines[i : table_end + 1]),
                        }
                    )

            # Detect list items
            elif self._is_list_item(line):
                list_end = self._find_list_end(lines, i)
                if list_end > i:
                    structure["lists"].append(
                        {
                            "start_line": i,
                            "end_line": list_end,
                            "start_pos": line_start,
                            "content": "\n".join(lines[i : list_end + 1]),
                        }
                    )

            # Detect code blocks
            elif line.strip().startswith("```"):
                code_end = self._find_code_block_end(lines, i)
                if code_end > i:
                    structure["code_blocks"].append(
                        {
                            "start_line": i,
                            "end_line": code_end,
                            "start_pos": line_start,
                            "content": "\n".join(lines[i : code_end + 1]),
                        }
                    )

        return structure

    def _create_semantic_chunks(self, doc: Document, structure: Dict[str, Any]) -> List[Document]:
        """Create semantic chunks based on document structure and token limits."""
        chunks = []
        text = doc.page_content
        headers = structure["headers"]

        if not headers:
            # No headers found, use content-aware chunking
            return self._chunk_by_content_type(doc, structure)

        # Process sections between headers
        for i, header in enumerate(headers):
            section_start = header["position"]
            section_end = headers[i + 1]["position"] if i + 1 < len(headers) else len(text)

            section_text = text[section_start:section_end].strip()
            if not section_text:
                continue

            # Determine content type and appropriate token limit
            content_type = self._classify_content_type(section_text)
            token_limit = self.content_type_limits.get(content_type, self.content_type_limits["text"])

            # Check if section fits within token limit
            section_tokens = len(self.tokenizer.encode(section_text))

            if section_tokens <= token_limit:
                # Section fits, create single chunk
                chunk = self._create_chunk(doc, section_text, header, content_type, len(chunks))
                chunks.append(chunk)
            else:
                # Section too large, split semantically
                sub_chunks = self._split_large_section(
                    doc, section_text, header, content_type, token_limit, len(chunks)
                )
                chunks.extend(sub_chunks)

        return chunks

    def _chunk_by_content_type(self, doc: Document, structure: Dict[str, Any]) -> List[Document]:
        """Chunk document by content type when no headers are present."""
        chunks = []
        text = doc.page_content

        # Extract all structured content (tables, lists, code blocks)
        structured_content = []
        for content_type in ["tables", "lists", "code_blocks"]:
            for item in structure[content_type]:
                structured_content.append(
                    {
                        "start_pos": item["start_pos"],
                        "end_pos": item["start_pos"] + len(item["content"]),
                        "content": item["content"],
                        "type": content_type[:-1] if content_type.endswith("s") else content_type,
                    }
                )

        # Sort by position
        structured_content.sort(key=lambda x: x["start_pos"])

        # Create chunks, preserving structured content
        current_pos = 0
        chunk_index = 0

        for item in structured_content:
            # Add text before structured content
            if item["start_pos"] > current_pos:
                text_chunk = text[current_pos : item["start_pos"]].strip()
                if text_chunk:
                    chunks.extend(self._split_text_by_tokens(doc, text_chunk, "text", chunk_index))
                    chunk_index = len(chunks)

            # Add structured content as complete unit
            chunk = self._create_chunk(doc, item["content"], None, item["type"], chunk_index)
            chunks.append(chunk)
            chunk_index += 1
            current_pos = item["end_pos"]

        # Add remaining text
        if current_pos < len(text):
            remaining_text = text[current_pos:].strip()
            if remaining_text:
                chunks.extend(self._split_text_by_tokens(doc, remaining_text, "text", chunk_index))

        return chunks

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

        # Document title with hierarchy
        title_parts = []
        if metadata.get("parent_title"):
            title_parts.append(metadata["parent_title"])
        if metadata.get("title"):
            title_parts.append(metadata["title"])

        if title_parts:
            context_parts.append(f"Document: {' > '.join(title_parts)}")

        # Space information (for Confluence)
        if metadata.get("space_key"):
            context_parts.append(f"Espace: {metadata['space_key']}")

        # URL/Path information
        if metadata.get("url"):
            context_parts.append(f"URL: {metadata['url']}")

        # Hierarchy information (for hierarchical chunks)
        if metadata.get("hierarchy_path") and metadata.get("hierarchy_path") != "Root":
            context_parts.append(f"Section: {metadata['hierarchy_path']}")
        elif metadata.get("section_path"):
            context_parts.append(f"Section: {metadata['section_path']}")

        # Content type information
        if metadata.get("content_type") and metadata.get("content_type") != "text":
            context_parts.append(f"Type: {metadata['content_type']}")

        # Source information
        if metadata.get("source"):
            context_parts.append(f"Source: {metadata['source']}")

        if context_parts:
            return f"[{' | '.join(context_parts)}]"

        return "[Contexte non disponible]"

    # Helper methods for semantic chunking

    def _is_header(self, line: str) -> bool:
        """Check if line is a header (markdown or HTML)."""
        line = line.strip()
        # Markdown headers
        if re.match(r"^#{1,6}\s+", line):
            return True
        # HTML headers
        if re.match(r"<h[1-6][^>]*>.*</h[1-6]>", line, re.IGNORECASE):
            return True
        return False

    def _get_header_level(self, line: str) -> int:
        """Get header level from line."""
        line = line.strip()
        # Markdown headers
        md_match = re.match(r"^(#{1,6})\s+", line)
        if md_match:
            return len(md_match.group(1))
        # HTML headers
        html_match = re.match(r"<h([1-6])[^>]*>", line, re.IGNORECASE)
        if html_match:
            return int(html_match.group(1))
        return 1

    def _extract_header_text(self, line: str) -> str:
        """Extract text from header line."""
        line = line.strip()
        # Markdown headers
        md_match = re.match(r"^#{1,6}\s+(.+)$", line)
        if md_match:
            return md_match.group(1).strip()
        # HTML headers
        html_match = re.match(r"<h[1-6][^>]*>(.*?)</h[1-6]>", line, re.IGNORECASE)
        if html_match:
            return re.sub(r"<[^>]+>", "", html_match.group(1)).strip()
        return line

    def _is_table_start(self, line: str) -> bool:
        """Check if line starts a table."""
        line = line.strip()
        # HTML table
        if line.startswith("<table"):
            return True
        # Markdown table (line with pipes)
        if "|" in line and not line.startswith("|--"):
            return True
        return False

    def _find_table_end(self, lines: List[str], start: int) -> int:
        """Find end of table starting at start line."""
        if start >= len(lines):
            return start

        start_line = lines[start].strip()

        # HTML table
        if start_line.startswith("<table"):
            for i in range(start + 1, len(lines)):
                if "</table>" in lines[i]:
                    return i
            return len(lines) - 1

        # Markdown table
        if "|" in start_line:
            for i in range(start + 1, len(lines)):
                if "|" not in lines[i].strip() or not lines[i].strip():
                    return i - 1
            return len(lines) - 1

        return start

    def _is_list_item(self, line: str) -> bool:
        """Check if line is a list item."""
        line = line.strip()
        # Unordered list
        if re.match(r"^[-*+]\s+", line):
            return True
        # Ordered list
        if re.match(r"^\d+\.\s+", line):
            return True
        # HTML list
        if line.startswith("<li") or line.startswith("<ul") or line.startswith("<ol"):
            return True
        return False

    def _find_list_end(self, lines: List[str], start: int) -> int:
        """Find end of list starting at start line."""
        if start >= len(lines):
            return start

        # Look for consecutive list items or until different content
        for i in range(start + 1, len(lines)):
            line = lines[i].strip()
            if not line:  # Empty line might end list
                # Check next non-empty line
                next_content = False
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        if not self._is_list_item(lines[j]):
                            return i - 1
                        next_content = True
                        break
                if not next_content:
                    return i - 1
            elif not self._is_list_item(line) and not line.startswith("  "):  # Not indented continuation
                return i - 1

        return len(lines) - 1

    def _find_code_block_end(self, lines: List[str], start: int) -> int:
        """Find end of code block starting at start line."""
        if start >= len(lines):
            return start

        for i in range(start + 1, len(lines)):
            if lines[i].strip().startswith("```"):
                return i

        return len(lines) - 1

    def _classify_content_type(self, text: str) -> str:
        """Classify content type for appropriate token limits."""
        text_lower = text.lower()

        # Check for tables
        if "<table" in text_lower or "|" in text:
            return "table"

        # Check for code blocks
        if "```" in text or "<pre" in text_lower or "<code" in text_lower:
            return "code"

        # Check for lists
        if (
            re.search(r"^[-*+]\s+", text, re.MULTILINE)
            or re.search(r"^\d+\.\s+", text, re.MULTILINE)
            or "<li" in text_lower
        ):
            return "list"

        return "text"

    def _create_chunk(
        self, doc: Document, content: str, header: Optional[Dict], content_type: str, chunk_index: int
    ) -> Document:
        """Create a chunk document with enhanced metadata."""
        metadata = doc.metadata.copy()

        # Build hierarchy path
        hierarchy_path = []
        if header:
            hierarchy_path.append(header["title"])

        # Add parent page context for subpages
        if metadata.get("parent_title"):
            hierarchy_path.insert(0, metadata.get("parent_title"))

        # Extract cross-page references
        cross_references = self._extract_cross_references(content, doc.metadata)

        metadata.update(
            {
                "chunk_index": chunk_index,
                "chunk_strategy": "semantic_hierarchical",
                "content_type": content_type,
                "content_length": len(content),
                "token_count": len(self.tokenizer.encode(content)),
                "hierarchy_path": " > ".join(hierarchy_path) if hierarchy_path else "Root",
                "header_level": header["level"] if header else 0,
                "semantic_boundary": True,
                "cross_references": cross_references,
                "reference_count": len(cross_references),
                "chunk_size": len(content),
                "is_chunk": True,
            }
        )

        # Add contextual information
        enriched_content = self._add_context_to_chunk(content, metadata)

        return Document(page_content=enriched_content, metadata=metadata)

    def _split_large_section(
        self, doc: Document, text: str, header: Optional[Dict], content_type: str, token_limit: int, start_index: int
    ) -> List[Document]:
        """Split large section into smaller semantic chunks."""
        chunks = []

        # Try to split by paragraphs first
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_tokens = 0

        overlap_tokens = self.config.get("chunk_overlap_tokens", 50)

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            para_tokens = len(self.tokenizer.encode(paragraph))

            if current_tokens + para_tokens > token_limit and current_chunk:
                # Create chunk from accumulated paragraphs
                chunk_text = "\n\n".join(current_chunk)
                chunk = self._create_chunk(doc, chunk_text, header, content_type, start_index + len(chunks))
                chunks.append(chunk)

                # Start new chunk with overlap
                if overlap_tokens > 0 and current_chunk:
                    # Add last part of previous chunk for context
                    overlap_text = current_chunk[-1][-overlap_tokens * 4 :]  # Approximate char count
                    current_chunk = [overlap_text, paragraph]
                    current_tokens = len(self.tokenizer.encode(overlap_text)) + para_tokens
                else:
                    current_chunk = [paragraph]
                    current_tokens = para_tokens
            elif para_tokens > token_limit:
                # Single paragraph is too large, split it by sentences/words
                sub_chunks = self._split_large_paragraph(
                    doc, paragraph, header, content_type, token_limit, start_index + len(chunks)
                )
                chunks.extend(sub_chunks)
                current_chunk = []
                current_tokens = 0
            else:
                current_chunk.append(paragraph)
                current_tokens += para_tokens

        # Add remaining content
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk = self._create_chunk(doc, chunk_text, header, content_type, start_index + len(chunks))
            chunks.append(chunk)

        return chunks

    def _split_text_by_tokens(self, doc: Document, text: str, content_type: str, start_index: int) -> List[Document]:
        """Split text by token limits while preserving semantic boundaries."""
        token_limit = self.content_type_limits.get(content_type, self.content_type_limits["text"])
        text_tokens = len(self.tokenizer.encode(text))

        if text_tokens <= token_limit:
            chunk = self._create_chunk(doc, text, None, content_type, start_index)
            return [chunk]

        return self._split_large_section(doc, text, None, content_type, token_limit, start_index)

    def _extract_cross_references(self, content: str, doc_metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract cross-page references from content."""
        cross_refs = []

        # Confluence page links - ac:link macro
        link_pattern = r'<ac:link[^>]*>.*?<ri:page[^>]*ri:content-title="([^"]+)"[^>]*/>.*?</ac:link>'
        for match in re.finditer(link_pattern, content, re.IGNORECASE | re.DOTALL):
            linked_page = match.group(1)
            cross_refs.append(
                {
                    "type": "confluence_page_link",
                    "target_title": linked_page,
                    "context": match.group(0)[:100] + "..." if len(match.group(0)) > 100 else match.group(0),
                }
            )

        # Confluence space links
        space_link_pattern = r'<ac:link[^>]*>.*?<ri:space[^>]*ri:space-key="([^"]+)"[^>]*/>.*?</ac:link>'
        for match in re.finditer(space_link_pattern, content, re.IGNORECASE | re.DOTALL):
            space_key = match.group(1)
            cross_refs.append(
                {
                    "type": "confluence_space_link",
                    "target_space": space_key,
                    "context": match.group(0)[:100] + "..." if len(match.group(0)) > 100 else match.group(0),
                }
            )

        # Markdown-style links that might reference other pages
        md_link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        for match in re.finditer(md_link_pattern, content):
            link_text = match.group(1)
            link_url = match.group(2)

            # Check if it's an internal confluence link
            if "confluence" in link_url.lower() or link_url.startswith("/") or link_url.startswith("../"):
                cross_refs.append(
                    {
                        "type": "markdown_internal_link",
                        "link_text": link_text,
                        "link_url": link_url,
                        "context": match.group(0),
                    }
                )

        # @mention patterns (user references)
        mention_pattern = r'<ac:link[^>]*>.*?<ri:user[^>]*ri:username="([^"]+)"[^>]*/>.*?</ac:link>'
        for match in re.finditer(mention_pattern, content, re.IGNORECASE | re.DOTALL):
            username = match.group(1)
            cross_refs.append(
                {
                    "type": "user_mention",
                    "username": username,
                    "context": match.group(0)[:100] + "..." if len(match.group(0)) > 100 else match.group(0),
                }
            )

        # Include macro references (for content embedded from other pages)
        include_pattern = (
            r'<ac:structured-macro[^>]*ac:name="include"[^>]*>.*?'
            r'<ac:parameter[^>]*ac:name="[^"]*">([^<]+)</ac:parameter>.*?'
            r"</ac:structured-macro>"
        )
        for match in re.finditer(include_pattern, content, re.IGNORECASE | re.DOTALL):
            included_content = match.group(1).strip()
            cross_refs.append(
                {
                    "type": "include_macro",
                    "included_content": included_content,
                    "context": match.group(0)[:100] + "..." if len(match.group(0)) > 100 else match.group(0),
                }
            )

        return cross_refs

    def _split_large_paragraph(
        self,
        doc: Document,
        paragraph: str,
        header: Optional[Dict],
        content_type: str,
        token_limit: int,
        start_index: int,
    ) -> List[Document]:
        """Split a large paragraph that exceeds token limits."""
        chunks = []

        # Try to split by sentences first
        sentences = re.split(r"[.!?]+\s+", paragraph)
        current_chunk = []
        current_tokens = 0

        overlap_tokens = self.config.get("chunk_overlap_tokens", 50)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Add punctuation back if it was removed
            if not sentence.endswith((".", "!", "?")):
                sentence += "."

            sent_tokens = len(self.tokenizer.encode(sentence))

            if current_tokens + sent_tokens > token_limit and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = " ".join(current_chunk)
                chunk = self._create_chunk(doc, chunk_text, header, content_type, start_index + len(chunks))
                chunks.append(chunk)

                # Start new chunk with overlap
                if overlap_tokens > 0 and current_chunk:
                    overlap_text = current_chunk[-1][-overlap_tokens * 4 :]  # Approximate char count
                    current_chunk = [overlap_text, sentence]
                    current_tokens = len(self.tokenizer.encode(overlap_text)) + sent_tokens
                else:
                    current_chunk = [sentence]
                    current_tokens = sent_tokens
            elif sent_tokens > token_limit:
                # Single sentence is too large, split by words
                words = sentence.split()
                word_chunks = []
                current_word_chunk = []
                current_word_tokens = 0

                for word in words:
                    word_tokens = len(self.tokenizer.encode(word))
                    if current_word_tokens + word_tokens > token_limit and current_word_chunk:
                        word_chunks.append(" ".join(current_word_chunk))
                        current_word_chunk = [word]
                        current_word_tokens = word_tokens
                    else:
                        current_word_chunk.append(word)
                        current_word_tokens += word_tokens

                if current_word_chunk:
                    word_chunks.append(" ".join(current_word_chunk))

                # Create chunks from word splits
                for word_chunk in word_chunks:
                    chunk = self._create_chunk(doc, word_chunk, header, content_type, start_index + len(chunks))
                    chunks.append(chunk)

                current_chunk = []
                current_tokens = 0
            else:
                current_chunk.append(sentence)
                current_tokens += sent_tokens

        # Add remaining content
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk = self._create_chunk(doc, chunk_text, header, content_type, start_index + len(chunks))
            chunks.append(chunk)

        return chunks
