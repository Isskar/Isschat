"""
Document chunking for processing pipeline.
"""

from typing import List, Dict, Any, Optional

from ...core.interfaces import Document


class DocumentChunker:
    """Splits documents into appropriately sized chunks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config or {}
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)
        self.separator = self.config.get("separator", "\n\n")

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to chunk

        Returns:
            List[Document]: Chunked documents
        """
        chunked_docs = []

        for doc in documents:
            chunks = self._split_document(doc)
            chunked_docs.extend(chunks)

        return chunked_docs

    def _split_document(self, document: Document) -> List[Document]:
        """
        Split a single document into chunks.

        Args:
            document: Document to split

        Returns:
            List[Document]: Document chunks
        """
        content = document.page_content
        chunks = []

        # Simple text splitting by separator
        parts = content.split(self.separator)
        current_chunk = ""

        for part in parts:
            # If the part itself is larger than chunk_size, split it further
            if len(part) > self.chunk_size:
                # Add current chunk if it exists
                if current_chunk:
                    chunks.append(self._create_chunk(document, current_chunk, len(chunks)))
                    current_chunk = ""

                # Split the large part into smaller chunks
                large_part_chunks = self._split_large_text(part)
                for large_chunk in large_part_chunks:
                    chunks.append(self._create_chunk(document, large_chunk, len(chunks)))
            elif len(current_chunk) + len(part) + len(self.separator) <= self.chunk_size:
                if current_chunk:
                    current_chunk += self.separator + part
                else:
                    current_chunk = part
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(document, current_chunk, len(chunks)))
                current_chunk = part

        # Add the last chunk
        if current_chunk:
            chunks.append(self._create_chunk(document, current_chunk, len(chunks)))

        return chunks

    def _split_large_text(self, text: str) -> List[str]:
        """
        Split text that is larger than chunk_size into smaller pieces.

        Args:
            text: Text to split

        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If this is not the last chunk, try to find a good break point
            if end < len(text):
                # Look for space or punctuation to break on
                break_point = end
                for i in range(end - 1, max(start + self.chunk_size // 2, start), -1):
                    if text[i] in " \n\t.,;!?":
                        break_point = i + 1
                        break
                end = break_point

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Apply overlap for next chunk
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def _create_chunk(self, original_doc: Document, chunk_content: str, chunk_index: int) -> Document:
        """
        Create a chunk document from original document.

        Args:
            original_doc: Original document
            chunk_content: Content of the chunk
            chunk_index: Index of the chunk

        Returns:
            Document: Chunk document
        """
        chunk_metadata = original_doc.metadata.copy()
        chunk_metadata.update({"chunk_index": chunk_index, "chunk_size": len(chunk_content), "is_chunk": True})

        return Document(page_content=chunk_content, metadata=chunk_metadata)

    def get_chunking_stats(self, original_docs: List[Document], chunked_docs: List[Document]) -> Dict[str, Any]:
        """
        Get chunking statistics.

        Args:
            original_docs: Original documents
            chunked_docs: Chunked documents

        Returns:
            Dict: Chunking statistics
        """
        return {
            "original_count": len(original_docs),
            "chunk_count": len(chunked_docs),
            "avg_chunks_per_doc": len(chunked_docs) / len(original_docs) if original_docs else 0,
            "avg_chunk_size": sum(len(doc.page_content) for doc in chunked_docs) / len(chunked_docs)
            if chunked_docs
            else 0,
        }
