import re
from typing import List, Dict, Any, Optional
from ...core.interfaces import Document


class DocumentChunker:
    """Simple document chunking."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.chunk_size = self.config.get("chunk_size", 400)
        self.chunk_overlap = self.config.get("chunk_overlap", 80)

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
        content = document.content

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
                chunk_metadata.update({"chunk_index": chunk_index, "chunk_size": self.chunk_size, "is_chunk": True})

                # Add contextual information to chunk content
                enriched_content = self._add_context_to_chunk(chunk_text, chunk_metadata)
                chunks.append(Document(content=enriched_content, metadata=chunk_metadata))
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
    """Clean paragraph-based chunking for Confluence documents with hierarchy awareness."""

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, strategy: Optional[str] = None, model_name: Optional[str] = None
    ):
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 300)
        self.chunk_overlap = self.config.get("chunk_overlap", 60)
        self.strategy = strategy or "semantic_hierarchical"
        self.model_name = model_name

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents using the specified strategy."""
        chunked_docs = []
        for doc in documents:
            if self.strategy == "semantic_hierarchical":
                chunks = self._chunk_with_hierarchy(doc)
            elif self.strategy == "hierarchy_aware":
                chunks = self._chunk_with_hierarchy(doc)
            else:
                chunks = self._chunk_by_paragraphs(doc)
            chunked_docs.extend(chunks)
        return chunked_docs

    def _chunk_with_hierarchy(self, document: Document) -> List[Document]:
        """Chunk document while preserving hierarchical structure and section context."""
        content = document.content

        # Extract document hierarchy
        hierarchy = self._extract_document_hierarchy(content)

        # Split content into sections with hierarchy information
        sections = self._split_into_hierarchical_sections(content, hierarchy)

        chunks = []
        chunk_index = 0

        for section in sections:
            section_chunks = self._chunk_section_with_hierarchy(document, section, chunk_index)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return chunks

    def _extract_document_hierarchy(self, content: str) -> List[Dict[str, Any]]:
        """Extract hierarchical structure from document content."""
        hierarchy = []
        lines = content.split("\n")

        # Confluence/Markdown header patterns
        header_patterns = [
            (r"^# (.+)$", 1),  # H1
            (r"^## (.+)$", 2),  # H2
            (r"^### (.+)$", 3),  # H3
            (r"^#### (.+)$", 4),  # H4
            (r"^##### (.+)$", 5),  # H5
            (r"^###### (.+)$", 6),  # H6
            (r"^h1\. (.+)$", 1),  # Confluence H1
            (r"^h2\. (.+)$", 2),  # Confluence H2
            (r"^h3\. (.+)$", 3),  # Confluence H3
            (r"^h4\. (.+)$", 4),  # Confluence H4
            (r"^h5\. (.+)$", 5),  # Confluence H5
            (r"^h6\. (.+)$", 6),  # Confluence H6
        ]

        for line_num, line in enumerate(lines):
            line = line.strip()
            for pattern, level in header_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    hierarchy.append(
                        {
                            "level": level,
                            "title": match.group(1).strip(),
                            "line_num": line_num,
                            "parent_path": self._get_parent_path(hierarchy, level),
                        }
                    )
                    break

        return hierarchy

    def _get_parent_path(self, hierarchy: List[Dict[str, Any]], current_level: int) -> List[str]:
        """Get the path to the parent sections for the current level."""
        path = []
        for item in reversed(hierarchy):
            if item["level"] < current_level:
                path.insert(0, item["title"])
                current_level = item["level"]
        return path

    def _split_into_hierarchical_sections(self, content: str, hierarchy: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split content into sections with hierarchical metadata."""
        if not hierarchy:
            return [
                {
                    "content": content,
                    "level": 0,
                    "title": "Document",
                    "parent_path": [],
                    "section_path": ["Document"],
                    "start_line": 0,
                    "end_line": len(content.split("\n")),
                }
            ]

        lines = content.split("\n")
        sections = []

        for i, header in enumerate(hierarchy):
            start_line = header["line_num"]
            end_line = hierarchy[i + 1]["line_num"] if i + 1 < len(hierarchy) else len(lines)

            section_content = "\n".join(lines[start_line:end_line]).strip()
            if section_content:
                section_path = header["parent_path"] + [header["title"]]
                sections.append(
                    {
                        "content": section_content,
                        "level": header["level"],
                        "title": header["title"],
                        "parent_path": header["parent_path"],
                        "section_path": section_path,
                        "start_line": start_line,
                        "end_line": end_line,
                    }
                )

        return sections

    def _chunk_section_with_hierarchy(
        self, document: Document, section: Dict[str, Any], start_chunk_index: int
    ) -> List[Document]:
        """Chunk a section while preserving hierarchical context."""
        content = section["content"]

        # Split section content by paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = start_chunk_index

        for paragraph in paragraphs:
            paragraph_length = len(paragraph)

            # Check if adding this paragraph would exceed chunk size
            if current_length + paragraph_length > self.chunk_size and current_chunk:
                # Create chunk from accumulated paragraphs
                chunk_text = "\n\n".join(current_chunk)
                chunk = self._create_hierarchical_chunk(document, chunk_text, section, chunk_index)
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                if overlap_text:
                    current_chunk = [overlap_text, paragraph]
                    current_length = len(overlap_text) + paragraph_length + 2
                else:
                    current_chunk = [paragraph]
                    current_length = paragraph_length
                chunk_index += 1

            elif paragraph_length > self.chunk_size:
                # Handle oversized paragraphs
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunk = self._create_hierarchical_chunk(document, chunk_text, section, chunk_index)
                    chunks.append(chunk)
                    chunk_index += 1

                # Split large paragraph into smaller chunks
                sentence_chunks = self._split_large_paragraph(paragraph)
                for sentence_chunk in sentence_chunks:
                    chunk = self._create_hierarchical_chunk(document, sentence_chunk, section, chunk_index)
                    chunks.append(chunk)
                    chunk_index += 1

                current_chunk = []
                current_length = 0

            else:
                current_chunk.append(paragraph)
                current_length += paragraph_length + 2  # +2 for \n\n separator

        # Add remaining paragraphs as final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk = self._create_hierarchical_chunk(document, chunk_text, section, chunk_index)
            chunks.append(chunk)

        return chunks

    def _create_hierarchical_chunk(
        self, document: Document, content: str, section: Dict[str, Any], chunk_index: int
    ) -> Document:
        """Create a chunk with hierarchical metadata."""
        metadata = document.metadata.copy()

        # Add hierarchy-specific metadata
        metadata.update(
            {
                "chunk_index": chunk_index,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "content_length": len(content),
                "is_chunk": True,
                "chunk_strategy": "semantic_hierarchical",
                # Hierarchical metadata
                "section_title": section["title"],
                "section_level": section["level"],
                "section_path": section["section_path"],
                "parent_path": section["parent_path"],
                "section_breadcrumb": " > ".join(section["section_path"]),
                "hierarchy_depth": len(section["section_path"]),
                # Position metadata
                "section_start_line": section["start_line"],
                "section_end_line": section["end_line"],
                # Numerical content detection
                "has_numbers": self._detect_numerical_content(content),
                "number_count": self._count_numbers(content),
                "extracted_numbers": self._extract_numbers(content),
            }
        )

        # Add contextual information to chunk content
        enriched_content = self._add_hierarchical_context_to_chunk(content, metadata)

        return Document(content=enriched_content, metadata=metadata)

    def _detect_numerical_content(self, content: str) -> bool:
        """Detect if content contains numerical information."""
        # Pattern to match numbers (integers, floats, percentages, etc.)
        number_pattern = r"\b\d+(?:[.,]\d+)*\s*(?:%|percent|interviews?|count|total|sum|number|amount)\b"
        return bool(re.search(number_pattern, content, re.IGNORECASE))

    def _count_numbers(self, content: str) -> int:
        """Count numerical values in content."""
        number_pattern = r"\b\d+(?:[.,]\d+)*\b"
        return len(re.findall(number_pattern, content))

    def _extract_numbers(self, content: str) -> List[Dict[str, Any]]:
        """Extract numerical values with context."""
        numbers = []

        # Pattern to match numbers with context
        patterns = [
            (r"(\d+(?:[.,]\d+)*)\s+(interviews?)", "interview_count"),
            (r"(\d+(?:[.,]\d+)*)\s+(total|sum|count)", "total_count"),
            (r"(\d+(?:[.,]\d+)*)\s*%", "percentage"),
            (r"(\d+(?:[.,]\d+)*)\s+(participants?|users?|people)", "participant_count"),
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

    def _add_hierarchical_context_to_chunk(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add hierarchical contextual information to chunk content."""
        context_info = self._get_hierarchical_context(metadata)
        return f"{context_info}\n\n{content}"

    def _get_hierarchical_context(self, metadata: Dict[str, Any]) -> str:
        """Generate hierarchical contextual information for a chunk."""
        context_parts = []

        # Document title
        if metadata.get("title"):
            context_parts.append(f"Document: {metadata['title']}")

        # Section hierarchy
        if metadata.get("section_breadcrumb"):
            context_parts.append(f"Section: {metadata['section_breadcrumb']}")

        # Space
        if metadata.get("space_key"):
            context_parts.append(f"Space: {metadata['space_key']}")

        # Author
        if metadata.get("author_name"):
            context_parts.append(f"Author: {metadata['author_name']}")

        # Created date
        if metadata.get("created_date"):
            created = metadata["created_date"][:10]
            context_parts.append(f"Created: {created}")

        # Numerical content indicator
        if metadata.get("has_numbers"):
            context_parts.append(f"Numbers: {metadata.get('number_count', 0)} found")

        # URL
        if metadata.get("url"):
            context_parts.append(f"URL: {metadata['url']}")

        if context_parts:
            return f"[{' | '.join(context_parts)}]"

        return "[Document context]"

    def _chunk_by_paragraphs(self, document: Document) -> List[Document]:
        """Split document into paragraph-based chunks with metadata enrichment."""
        content = document.content

        # Split by double newlines to get paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        if not paragraphs:
            return [document]

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph_length = len(paragraph)

            # If adding this paragraph would exceed chunk size and we have content
            if current_length + paragraph_length > self.chunk_size and current_chunk:
                # Create chunk from accumulated paragraphs
                chunk_text = "\n\n".join(current_chunk)
                chunk = self._create_enriched_chunk(document, chunk_text, chunk_index)
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                if overlap_text:
                    current_chunk = [overlap_text, paragraph]
                    current_length = len(overlap_text) + paragraph_length + 2  # +2 for \n\n
                else:
                    current_chunk = [paragraph]
                    current_length = paragraph_length
                chunk_index += 1

            elif paragraph_length > self.chunk_size:
                # Single paragraph is too large, split it by sentences
                sentence_chunks = self._split_large_paragraph(paragraph)
                for sentence_chunk in sentence_chunks:
                    chunk = self._create_enriched_chunk(document, sentence_chunk, chunk_index)
                    chunks.append(chunk)
                    chunk_index += 1
                current_chunk = []
                current_length = 0

            else:
                current_chunk.append(paragraph)
                current_length += paragraph_length + 2  # +2 for \n\n separator

        # Add remaining paragraphs as final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk = self._create_enriched_chunk(document, chunk_text, chunk_index)
            chunks.append(chunk)

        return chunks

    def _get_overlap_text(self, paragraphs: List[str]) -> str:
        """Get overlap text from the end of current chunk."""
        if not paragraphs:
            return ""

        # Take the last paragraph for overlap
        last_paragraph = paragraphs[-1]

        # If last paragraph is longer than overlap size, take the end portion
        if len(last_paragraph) > self.chunk_overlap:
            return last_paragraph[-self.chunk_overlap :]

        return last_paragraph

    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Split a large paragraph into smaller chunks by sentences."""
        # Split by sentence boundaries
        sentences = re.split(r"[.!?]+\s+", paragraph)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Add punctuation back if it was removed
            if not sentence.endswith((".", "!", "?")):
                sentence += "."

            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                overlap_text = self._get_sentence_overlap(current_chunk)
                if overlap_text:
                    current_chunk = [overlap_text, sentence]
                    current_length = len(overlap_text) + sentence_length + 1  # +1 for space
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space

        # Add remaining sentences
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

        return chunks

    def _get_sentence_overlap(self, sentences: List[str]) -> str:
        """Get overlap text from sentences."""
        if not sentences:
            return ""

        last_sentence = sentences[-1]
        if len(last_sentence) > self.chunk_overlap:
            return last_sentence[-self.chunk_overlap :]

        return last_sentence

    def _create_enriched_chunk(self, document: Document, content: str, chunk_index: int) -> Document:
        """Create a chunk with enriched metadata."""
        metadata = document.metadata.copy()

        # Add basic chunk metadata
        metadata.update(
            {
                "chunk_index": chunk_index,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "content_length": len(content),
                "is_chunk": True,
                "chunk_strategy": "paragraph_based",
            }
        )

        # Add contextual information to chunk content
        enriched_content = self._add_context_to_chunk(content, metadata)

        return Document(content=enriched_content, metadata=metadata)

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

        # Space
        if metadata.get("space_key"):
            context_parts.append(f"Space: {metadata['space_key']}")

        # Author
        if metadata.get("author_name"):
            context_parts.append(f"Author: {metadata['author_name']}")

        # Created date
        if metadata.get("created_date"):
            created = metadata["created_date"][:10]  # YYYY-MM-DD format
            context_parts.append(f"Created: {created}")

        # Last modified date
        if metadata.get("last_modified_date"):
            modified = metadata["last_modified_date"][:10]  # YYYY-MM-DD format
            context_parts.append(f"Modified: {modified}")

        # URL
        if metadata.get("url"):
            context_parts.append(f"URL: {metadata['url']}")

        # Source
        if metadata.get("source"):
            context_parts.append(f"Source: {metadata['source']}")

        if context_parts:
            return f"[{' | '.join(context_parts)}]"

        return "[Document context]"
