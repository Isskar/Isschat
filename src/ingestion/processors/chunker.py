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
    """Clean paragraph-based chunking for Confluence documents."""

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, strategy: Optional[str] = None, model_name: Optional[str] = None
    ):
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 300)
        self.chunk_overlap = self.config.get("chunk_overlap", 60)
        self.strategy = strategy or "paragraph_based"
        self.model_name = model_name

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents using paragraph-based approach."""
        chunked_docs = []
        for doc in documents:
            chunks = self._chunk_by_paragraphs(doc)
            chunked_docs.extend(chunks)
        return chunked_docs

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
