"""
Text processor for document chunking and preprocessing.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tiktoken
from langdetect import detect, LangDetectException

from ..models import RawDocument, TextChunk, ProcessedDocument


@dataclass
class TextProcessorConfig:
    """Configuration for text processor."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    encoding_name: str = "cl100k_base"
    clean_whitespace: bool = True
    detect_language: bool = True
    extract_entities: bool = False
    preserve_formatting: bool = True


class TextProcessor:
    """Process raw documents into chunks ready for embedding."""

    def __init__(self, config: Optional[TextProcessorConfig] = None):
        """Initialize text processor with configuration."""
        self.config = config or TextProcessorConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._encoding = tiktoken.get_encoding(self.config.encoding_name)

    def process(self, raw_document: RawDocument) -> ProcessedDocument:
        """
        Process a raw document into chunks.

        Args:
            raw_document: The raw document to process

        Returns:
            ProcessedDocument with chunks and metadata
        """
        try:
            cleaned_text = self._clean_text(raw_document.content)

            language = None
            if self.config.detect_language:
                language = self._detect_language(cleaned_text)

            entities = []
            if self.config.extract_entities:
                entities = self._extract_entities(cleaned_text)

            chunks = self.chunk_text(cleaned_text)
            enriched_chunks = self.enrich_metadata(
                chunks,
                {
                    **raw_document.metadata,
                    "language": language,
                    "entities": entities,
                    "original_length": len(raw_document.content),
                    "cleaned_length": len(cleaned_text),
                },
            )

            processed_doc = ProcessedDocument(
                raw_document=raw_document,
                chunks=enriched_chunks,
                processing_metadata={
                    "chunk_config": {"size": self.config.chunk_size, "overlap": self.config.chunk_overlap},
                    "language": language,
                    "entities_count": len(entities),
                    "total_chunks": len(enriched_chunks),
                },
            )

            self.logger.info(f"Processed document {raw_document.id}: {len(enriched_chunks)} chunks created")

            return processed_doc

        except Exception as e:
            self.logger.error(f"Error processing document {raw_document.id}: {e}")
            raise

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Chunk text into semantic units.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        paragraphs = self._split_into_paragraphs(text)

        chunks = []
        current_chunk = []
        current_size = 0
        start_char = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph_size = self._count_tokens(paragraph)

            if paragraph_size > self.config.chunk_size:
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, chunk_index, start_char))
                    chunk_index += 1
                    start_char += len(chunk_text) + 2
                    current_chunk = []
                    current_size = 0

                sub_chunks = self._split_large_paragraph(paragraph)
                for sub_chunk in sub_chunks:
                    chunks.append(self._create_chunk(sub_chunk, chunk_index, start_char))
                    chunk_index += 1
                    start_char += len(sub_chunk) + 2

            elif current_size + paragraph_size > self.config.chunk_size:
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, chunk_index, start_char))
                    chunk_index += 1

                    if self.config.chunk_overlap > 0 and len(current_chunk) > 1:
                        overlap_text = current_chunk[-1]
                        overlap_size = self._count_tokens(overlap_text)
                        current_chunk = [overlap_text, paragraph]
                        current_size = overlap_size + paragraph_size
                        start_char += len(chunk_text) - len(overlap_text)
                    else:
                        current_chunk = [paragraph]
                        current_size = paragraph_size
                        start_char += len(chunk_text) + 2
                else:
                    current_chunk = [paragraph]
                    current_size = paragraph_size

            else:
                current_chunk.append(paragraph)
                current_size += paragraph_size

        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, chunk_index, start_char))

        return chunks

    def enrich_metadata(self, chunks: List[TextChunk], document_metadata: Dict[str, Any]) -> List[TextChunk]:
        """
        Enrich chunks with additional metadata.

        Args:
            chunks: List of chunks to enrich
            document_metadata: Document-level metadata

        Returns:
            List of enriched chunks
        """
        enriched_chunks = []

        for chunk in chunks:
            chunk_metadata = document_metadata.copy()

            chunk_metadata.update(
                {
                    "chunk_index": chunk.chunk_index,
                    "chunk_hash": chunk.chunk_hash,
                    "token_count": self._count_tokens(chunk.content),
                    "char_count": len(chunk.content),
                    "has_code": self._detect_code(chunk.content),
                    "has_list": self._detect_list(chunk.content),
                    "has_table": self._detect_table(chunk.content),
                }
            )

            chunk.metadata = chunk_metadata
            enriched_chunks.append(chunk)

        return enriched_chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not self.config.clean_whitespace:
            return text

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\t+", " ", text)

        lines = text.split("\n")
        lines = [line.rstrip() for line in lines]
        text = "\n".join(lines)

        return text.strip()

    def _detect_language(self, text: str) -> Optional[str]:
        """Detect text language."""
        try:
            sample = text[:1000]
            return detect(sample)
        except LangDetectException:
            self.logger.warning("Could not detect language")
            return None

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text (placeholder for NER)."""
        entities = []

        emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
        for email in emails:
            entities.append({"type": "EMAIL", "value": email})

        urls = re.findall(r"https?://[^\s]+", text)
        for url in urls:
            entities.append({"type": "URL", "value": url})

        return entities

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r"\n\s*\n", text)

        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return paragraphs

    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Split a large paragraph into smaller chunks."""
        sentences = self._split_into_sentences(paragraph)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = self._count_tokens(sentence)

            if current_size + sentence_size > self.config.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sentence_size
                else:
                    chunks.extend(self._split_by_tokens(sentence))
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_by_tokens(self, text: str) -> List[str]:
        """Split text by token count."""
        tokens = self._encoding.encode(text)
        chunks = []

        for i in range(0, len(tokens), self.config.chunk_size):
            chunk_tokens = tokens[i : i + self.config.chunk_size]
            chunk_text = self._encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._encoding.encode(text))

    def _create_chunk(self, content: str, index: int, start_char: int) -> TextChunk:
        """Create a text chunk."""
        return TextChunk(
            content=content,
            metadata={},  # Will be enriched later
            chunk_index=index,
            start_char=start_char,
            end_char=start_char + len(content),
        )

    def _detect_code(self, text: str) -> bool:
        """Detect if text contains code blocks."""
        code_patterns = [
            r"```[\s\S]*?```",  # Markdown code blocks
            r"`[^`]+`",  # Inline code
            r"^\s*(def|class|function|import|from|const|let|var)\s+",  # Common keywords
        ]

        for pattern in code_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False

    def _detect_list(self, text: str) -> bool:
        """Detect if text contains lists."""
        list_patterns = [
            r"^\s*[-*+]\s+",  # Unordered lists
            r"^\s*\d+\.\s+",  # Ordered lists
        ]

        for pattern in list_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False

    def _detect_table(self, text: str) -> bool:
        """Detect if text contains tables."""
        return "|" in text and text.count("|") >= 3
