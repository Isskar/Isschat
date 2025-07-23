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

        # Initialize metadata enricher for sibling documents retrieval
        self.metadata_enricher = self._initialize_metadata_enricher(config)

    def _initialize_metadata_enricher(self, config: Optional[Dict[str, Any]]):
        """Initialize the Confluence metadata enricher."""
        if not config:
            return None

        try:
            from ..connectors.metadata_enricher import ConfluenceMetadataEnricher

            # Extract configuration parameters
            if "confluence" in config:
                confluence_config = config["confluence"]
                base_url = confluence_config.get("base_url", "")
                username = confluence_config.get("username", "")
                api_token = confluence_config.get("api_token", "")
            else:
                base_url = config.get("confluence_url", "")
                username = config.get("confluence_email_address", "")
                api_token = config.get("confluence_private_api_key", "")

            if base_url and username and api_token:
                return ConfluenceMetadataEnricher(base_url=base_url, username=username, api_token=api_token)

        except (ImportError, KeyError):
            pass

        return None

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
        """Split content into sections with hierarchical metadata using look-ahead fusion strategy."""
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
        pending_titles = []  # Store titles of empty sections
        accumulated_content = []  # Store short sections to merge
        accumulated_metadata = None  # Store metadata for merged sections

        for i, header in enumerate(hierarchy):
            start_line = header["line_num"]
            end_line = hierarchy[i + 1]["line_num"] if i + 1 < len(hierarchy) else len(lines)

            # Extract raw section content
            section_content = "\n".join(lines[start_line:end_line]).strip()

            # Extract actual text content (excluding the header line itself)
            actual_content = self._extract_actual_content(section_content, header["title"])

            # Check if section has substantial content (more than 200 characters)
            if len(actual_content.strip()) > 200:
                # Process any accumulated short sections first
                if accumulated_content:
                    merged_section = self._create_merged_section(
                        accumulated_content, accumulated_metadata, pending_titles
                    )
                    if merged_section:
                        sections.append(merged_section)
                    accumulated_content.clear()
                    accumulated_metadata = None

                # This section has content, create chunk(s) with fused titles
                fused_section_path = self._create_fused_section_path(header, pending_titles)
                fused_breadcrumb = " > ".join(fused_section_path)

                sections.append(
                    {
                        "content": section_content,
                        "level": header["level"],
                        "title": header["title"],
                        "parent_path": header["parent_path"],
                        "section_path": fused_section_path,
                        "section_breadcrumb": fused_breadcrumb,
                        "start_line": start_line,
                        "end_line": end_line,
                        "fused_titles": pending_titles.copy(),  # Keep track of fused titles
                    }
                )

                # Clear pending titles as they've been consumed
                pending_titles.clear()
            else:
                # Section is short - decide whether to accumulate or mark as pending
                if len(actual_content.strip()) > 20:  # Has some content, just short
                    # Accumulate short sections for potential merging
                    accumulated_content.append(
                        {
                            "content": section_content,
                            "header": header,
                            "start_line": start_line,
                            "end_line": end_line,
                            "actual_content": actual_content,
                        }
                    )
                    if not accumulated_metadata:
                        accumulated_metadata = header
                else:
                    # Section is empty, add to pending titles
                    pending_titles.append(header["title"])

        # Process any remaining accumulated short sections
        if accumulated_content:
            merged_section = self._create_merged_section(accumulated_content, accumulated_metadata, pending_titles)
            if merged_section:
                sections.append(merged_section)

        return sections

    def _extract_actual_content(self, section_content: str, title: str) -> str:
        """Extract actual text content excluding the header line itself."""
        lines = section_content.split("\n")

        # Remove the header line (first line that contains the title)
        filtered_lines = []
        header_line_skipped = False

        for line in lines:
            line_stripped = line.strip()

            # Skip the header line containing the title (only the first occurrence)
            if not header_line_skipped and title in line_stripped:
                # Check if this line is actually a header (starts with # or h1-h6.)
                if (line_stripped.startswith("#") and title in line_stripped) or (
                    re.match(r"^h[1-6]\.\s+", line_stripped) and title in line_stripped
                ):
                    header_line_skipped = True
                    continue

            # Include all content after header
            if header_line_skipped:
                filtered_lines.append(line)

        actual_content = "\n".join(filtered_lines).strip()

        # Additional filtering: remove common "empty" patterns
        # Remove parenthetical notes like "(Cette section est vide...)"
        actual_content = re.sub(r"\([^)]*section[^)]*vide[^)]*\)", "", actual_content, flags=re.IGNORECASE)
        actual_content = re.sub(r"\([^)]*empty[^)]*section[^)]*\)", "", actual_content, flags=re.IGNORECASE)

        return actual_content.strip()

    def _create_fused_section_path(self, current_header: Dict[str, Any], pending_titles: List[str]) -> List[str]:
        """Create a fused section path combining pending titles with current path."""
        # Start with the parent path
        fused_path = current_header["parent_path"].copy()

        # Add pending titles (empty sections that came before)
        for title in pending_titles:
            if title not in fused_path:  # Avoid duplicates
                fused_path.append(title)

        # Add current section title
        if current_header["title"] not in fused_path:  # Avoid duplicates
            fused_path.append(current_header["title"])

        return fused_path

    def _create_merged_section(
        self, accumulated_content: List[Dict[str, Any]], base_metadata: Dict[str, Any], pending_titles: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Create a merged section from multiple short sections."""
        if not accumulated_content:
            return None

        # Combine all content
        merged_content_parts = []
        all_titles = []

        for section_data in accumulated_content:
            merged_content_parts.append(section_data["content"])
            all_titles.append(section_data["header"]["title"])

        merged_content = "\n\n".join(merged_content_parts)

        # Check if merged content meets minimum threshold
        if len(merged_content.strip()) < 200:
            # Still too short, add all titles to pending
            pending_titles.extend(all_titles)
            return None

        # Create merged section metadata
        first_section = accumulated_content[0]
        last_section = accumulated_content[-1]

        # Create a combined title representing all merged sections
        if len(all_titles) > 3:
            combined_title = f"{all_titles[0]} ... {all_titles[-1]} ({len(all_titles)} sections)"
        else:
            combined_title = " + ".join(all_titles)

        fused_section_path = self._create_fused_section_path(base_metadata, pending_titles)
        fused_section_path.append(combined_title)
        fused_breadcrumb = " > ".join(fused_section_path)

        return {
            "content": merged_content,
            "level": base_metadata["level"],
            "title": combined_title,
            "parent_path": base_metadata["parent_path"],
            "section_path": fused_section_path,
            "section_breadcrumb": fused_breadcrumb,
            "start_line": first_section["start_line"],
            "end_line": last_section["end_line"],
            "fused_titles": pending_titles.copy(),
            "merged_sections": all_titles,  # Track original section names
        }

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
                "section_breadcrumb": section.get("section_breadcrumb", " > ".join(section["section_path"])),
                "hierarchy_depth": len(section["section_path"]),
                "fused_titles": section.get("fused_titles", []),
                "merged_sections": section.get("merged_sections", []),
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
        enriched_content = self._add_enriched_context_to_chunk(content, metadata)

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

    def _add_enriched_context_to_chunk(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add enriched contextual information to chunk content."""
        context_header = self._build_structured_chunk_header(metadata)
        return f"{context_header}{content}"

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
        enriched_content = self._add_enriched_context_to_chunk(content, metadata)

        return Document(content=enriched_content, metadata=metadata)

    def _build_structured_chunk_header(self, metadata: Dict[str, Any]) -> str:
        """Build an LLM-friendly chunk header with contextual information."""
        header_parts = []

        # Document context line
        title = metadata.get("title", "Document")
        author = metadata.get("author_name")
        space_key = metadata.get("space_key")

        context_info = []
        if author:
            context_info.append(f"par {author}")
        if space_key:
            context_info.append(f"({space_key})")

        if context_info:
            header_parts.append(f'Document: "{title}" {" ".join(context_info)}')
        else:
            header_parts.append(f'Document: "{title}"')

        # Document path with emoji indicator
        hierarchy_breadcrumb = metadata.get("hierarchy_breadcrumb")
        if hierarchy_breadcrumb:
            header_parts.append(f"📍 {hierarchy_breadcrumb}")

        # Optional metadata for context (kept minimal)
        additional_context = []

        # Add creation date if available
        created = metadata.get("created_date")
        if created:
            additional_context.append(f"Créé: {created[:10]}")

        # Add URL for reference
        url = metadata.get("url")
        if url:
            additional_context.append(f"URL: {url}")

        # Add sibling documents context if available
        siblings = self._get_sibling_documents(metadata)
        if siblings:
            other_siblings = [s["title"] for s in siblings if not s["is_current"]]
            if other_siblings:
                # Limit to 3 siblings for conciseness
                siblings_list = ", ".join(other_siblings[:3])
                if len(other_siblings) > 3:
                    siblings_list += f" (+{len(other_siblings) - 3} autres)"
                additional_context.append(f"Documents voisins: {siblings_list}")

        # Add numerical content indicator
        if metadata.get("has_numbers"):
            number_count = metadata.get("number_count", 0)
            additional_context.append(f"Contient {number_count} valeurs numériques")

        # Include additional context as a single line if present
        if additional_context:
            header_parts.append(f"ℹ️ {' | '.join(additional_context)}")

        # Add separator before content
        header_parts.append("---")

        return "\n".join(header_parts) + "\n\n"

    def _build_full_hierarchy_path(self, metadata: Dict[str, Any]) -> str:
        """Build full path from root to current document."""
        if metadata.get("hierarchy_breadcrumb"):
            return metadata["hierarchy_breadcrumb"]
        return ""

    def _get_sibling_documents(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get sibling documents at the same hierarchical level."""
        current_title = metadata.get("title", "")
        current_page_id = metadata.get("page_id", "")

        # Initialize with current document
        siblings = [{"title": current_title, "is_current": True}] if current_title else []

        # Get parent information
        parent_page_id, parent_name = self._get_parent_info(metadata, current_page_id)

        if not parent_page_id:
            siblings.append({"title": f"[Autres documents de {parent_name}]", "is_current": False})
            return siblings

        # Retrieve actual sibling documents
        return self._fetch_sibling_documents(siblings, parent_page_id, parent_name, current_page_id)

    def _get_parent_info(self, metadata: Dict[str, Any], current_page_id: str) -> tuple[str, str]:
        """Extract parent page ID and name from metadata or API."""
        parent_pages = metadata.get("parent_pages", [])

        if parent_pages:
            parent = parent_pages[-1]
            return parent.get("id"), parent.get("title", "Dossier parent")

        if self.metadata_enricher and current_page_id:
            return self._fetch_parent_from_api(current_page_id)

        return "", "Dossier parent"

    def _fetch_parent_from_api(self, page_id: str) -> tuple[str, str]:
        """Fetch parent information from Confluence API."""
        try:
            url = f"{self.metadata_enricher.api_base}/content/{page_id}"
            response = self.metadata_enricher.session.get(url, params={"expand": "ancestors"})

            if response.status_code == 200:
                ancestors = response.json().get("ancestors", [])
                if ancestors:
                    parent = ancestors[-1]
                    return parent["id"], parent["title"]
        except Exception:
            pass

        return "", "Dossier parent"

    def _fetch_sibling_documents(
        self, siblings: List[Dict[str, Any]], parent_page_id: str, parent_name: str, current_page_id: str
    ) -> List[Dict[str, Any]]:
        """Fetch actual sibling documents from Confluence API."""
        if not self.metadata_enricher:
            siblings.append({"title": f"[Autres documents de {parent_name}]", "is_current": False})
            return siblings

        try:
            children_pages = self.metadata_enricher._get_page_children(parent_page_id)

            # Filter out current document and add siblings
            sibling_titles = []
            for page in children_pages:
                if page.get("id") != current_page_id:
                    sibling_titles.append(page["title"])
                    siblings.append({"title": page["title"], "is_current": False})

            # Add fallback message if no siblings found
            if not sibling_titles:
                siblings.append({"title": f"[Autres documents de {parent_name}]", "is_current": False})

        except Exception:
            siblings.append({"title": f"[Autres documents de {parent_name}]", "is_current": False})

        return siblings
