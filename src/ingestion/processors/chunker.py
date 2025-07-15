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
        try:
            from ..connectors.metadata_enricher import ConfluenceMetadataEnricher

            if config:
                # Check for direct confluence config or nested config
                if "confluence" in config:
                    confluence_config = config["confluence"]
                    base_url = confluence_config.get("base_url", "")
                    username = confluence_config.get("username", "")
                    api_token = confluence_config.get("api_token", "")
                else:
                    # Check for confluence config keys at root level
                    base_url = config.get("confluence_url", "")
                    username = config.get("confluence_email_address", "")
                    api_token = config.get("confluence_private_api_key", "")

                if base_url and username and api_token:
                    self.metadata_enricher = ConfluenceMetadataEnricher(
                        base_url=base_url, username=username, api_token=api_token
                    )
                else:
                    self.metadata_enricher = None
            else:
                self.metadata_enricher = None
        except (ImportError, KeyError):
            self.metadata_enricher = None

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
        return f"{context_info}{content}"

    def _get_hierarchical_context(self, metadata: Dict[str, Any]) -> str:
        """Generate hierarchical contextual information for a chunk."""
        return self._build_structured_chunk_header(metadata)

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
        return f"{context_info}{content}"

    def _get_document_context(self, metadata: Dict[str, Any]) -> str:
        """Generate contextual information for a chunk."""
        return self._build_structured_chunk_header(metadata)

    def _build_structured_chunk_header(self, metadata: Dict[str, Any]) -> str:
        """Build a compact, single-line compatible chunk header."""
        header_parts = []

        # Build metadata section
        metadata_parts = []
        title = metadata.get("title")
        if title:
            metadata_parts.append(f"Titre: {title}")

        author = metadata.get("author_name")
        if author:
            metadata_parts.append(f"Auteur: {author}")

        created = metadata.get("created_date")
        if created:
            metadata_parts.append(f"Créé: {created[:10]}")

        space_key = metadata.get("space_key")
        if space_key:
            metadata_parts.append(f"Espace: {space_key}")

        url = metadata.get("url")
        if url:
            metadata_parts.append(f"URL: {url}")

        # Build hierarchy section
        hierarchy_parts = []
        full_path = self._build_full_hierarchy_path(metadata)
        if full_path:
            hierarchy_parts.append(f"Chemin: {full_path}")

        siblings = self._get_sibling_documents(metadata)
        if siblings:
            current_doc = None
            other_siblings = []

            for sibling in siblings:
                if sibling["is_current"]:
                    current_doc = sibling["title"]
                else:
                    other_siblings.append(sibling["title"])

            if current_doc:
                hierarchy_parts.append(f"Document actuel: {current_doc}")

            if other_siblings:
                # Limit to 5 siblings for readability
                siblings_list = ", ".join(other_siblings[:5])
                if len(other_siblings) > 5:
                    siblings_list += f" (et {len(other_siblings) - 5} autres)"
                hierarchy_parts.append(f"Documents voisins: {siblings_list}")

        # Build content structure section
        content_parts = []
        section_breadcrumb = metadata.get("section_breadcrumb")
        if section_breadcrumb:
            content_parts.append(f"Section: {section_breadcrumb}")

        if metadata.get("has_numbers"):
            number_count = metadata.get("number_count", 0)
            content_parts.append(f"Numérique: {number_count} valeurs")

        # Combine all sections with clear separators
        if metadata_parts:
            header_parts.append(f"### METADATA ### {' | '.join(metadata_parts)}")
        if hierarchy_parts:
            header_parts.append(f"### HIÉRARCHIE ### {' | '.join(hierarchy_parts)}")
        if content_parts:
            header_parts.append(f"### STRUCTURE ### {' | '.join(content_parts)}")

        header_parts.append("### CONTENU ###")

        return " ".join(header_parts) + " "

    def _build_full_hierarchy_path(self, metadata: Dict[str, Any]) -> str:
        """Build full path from root to current document."""
        if metadata.get("hierarchy_breadcrumb"):
            return metadata["hierarchy_breadcrumb"]
        return ""

    def _get_sibling_documents(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get sibling documents at the same hierarchical level."""
        siblings = []
        current_title = metadata.get("title", "")
        current_page_id = metadata.get("page_id", "")

        # Add current document
        if current_title:
            siblings.append({"title": current_title, "is_current": True})

        # Try to get real sibling documents from parent pages
        parent_pages = metadata.get("parent_pages", [])
        parent_page_id = None
        parent_name = "Dossier parent"

        if parent_pages:
            # Get the immediate parent page ID and name
            parent_page_id = parent_pages[-1].get("id")
            parent_name = parent_pages[-1].get("title", "Dossier parent")
        elif self.metadata_enricher and current_page_id:
            # If parent_pages not available, try to get parent from API
            try:
                url = f"{self.metadata_enricher.api_base}/content/{current_page_id}"
                params = {"expand": "ancestors"}
                response = self.metadata_enricher.session.get(url, params=params)

                if response.status_code == 200:
                    page_data = response.json()
                    ancestors = page_data.get("ancestors", [])
                    if ancestors:
                        parent = ancestors[-1]  # Last ancestor is direct parent
                        parent_page_id = parent["id"]
                        parent_name = parent["title"]

            except Exception:
                pass  # Continue with fallback

        if not parent_page_id:
            siblings.append({"title": f"[Autres documents de {parent_name}]", "is_current": False})
            return siblings

        # Try to get actual sibling pages from Confluence API
        if self.metadata_enricher:
            try:
                children_pages = self.metadata_enricher._get_page_children(parent_page_id)

                # Add real sibling documents (excluding current document)
                sibling_titles = []
                for page in children_pages:
                    if page.get("id") != current_page_id:
                        sibling_titles.append(page["title"])
                        siblings.append({"title": page["title"], "is_current": False})

                # If no siblings found, add generic message
                if not sibling_titles:
                    siblings.append({"title": f"[Autres documents de {parent_name}]", "is_current": False})

            except Exception:
                # Fallback to generic suggestion if API call fails
                siblings.append({"title": f"[Autres documents de {parent_name}]", "is_current": False})
        else:
            # Fallback when metadata_enricher is not available
            siblings.append({"title": f"[Autres documents de {parent_name}]", "is_current": False})

        return siblings
