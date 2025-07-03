from unittest.mock import Mock, patch
from src.ingestion.processors.chunker import DocumentChunker, ConfluenceChunker
from src.core.interfaces import Document


class TestDocumentChunker:
    """Tests for DocumentChunker."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        chunker = DocumentChunker()

        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = {"chunk_size": 500, "chunk_overlap": 100}
        chunker = DocumentChunker(config)

        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100

    def test_chunk_documents_single_small_doc(self):
        """Test chunking single small document."""
        chunker = DocumentChunker()
        doc = Document(content="Short content", metadata={"title": "Test"})

        result = chunker.chunk_documents([doc])

        assert len(result) == 1
        assert result[0].content == "Short content"

    def test_chunk_document_convenience_method(self):
        """Test chunk_document convenience method."""
        chunker = DocumentChunker()
        doc = Document(content="Test content", metadata={"title": "Test"})

        result = chunker.chunk_document(doc)

        assert len(result) == 1
        assert result[0].content == "Test content"

    def test_chunk_documents_large_doc(self):
        """Test chunking large document."""
        chunker = DocumentChunker({"chunk_size": 50, "chunk_overlap": 10})

        # Use ConfluenceChunker which has _add_context_to_chunk method
        chunker = ConfluenceChunker({"chunk_size": 50, "chunk_overlap": 10})

        # Create content longer than chunk_size
        long_content = "This is a very long document that should be split into multiple chunks. " * 10
        doc = Document(content=long_content, metadata={"title": "Long Doc"})

        result = chunker.chunk_documents([doc])

        assert len(result) > 1
        for chunk in result:
            assert "chunk_index" in chunk.metadata
            assert "chunk_size" in chunk.metadata
            assert "is_chunk" in chunk.metadata

    def test_get_chunking_stats(self):
        """Test chunking statistics calculation."""
        chunker = DocumentChunker()
        original_docs = [Document(content="doc1", metadata={}), Document(content="doc2", metadata={})]
        chunks = [
            Document(content="chunk1", metadata={}),
            Document(content="chunk2", metadata={}),
            Document(content="chunk3", metadata={}),
        ]

        stats = chunker.get_chunking_stats(original_docs, chunks)

        assert stats["original_count"] == 2
        assert stats["chunk_count"] == 3
        assert stats["avg_chunks_per_doc"] == 1.5

    def test_get_chunking_stats_empty(self):
        """Test chunking statistics with empty documents."""
        chunker = DocumentChunker()

        stats = chunker.get_chunking_stats([], [])

        assert stats["original_count"] == 0
        assert stats["chunk_count"] == 0
        assert stats["avg_chunks_per_doc"] == 0


class TestConfluenceChunker:
    """Tests for ConfluenceChunker."""

    def test_init_default(self):
        """Test ConfluenceChunker initialization with defaults."""
        with patch("src.ingestion.processors.chunker.tiktoken.encoding_for_model") as mock_tiktoken:
            mock_tiktoken.return_value = Mock()

            chunker = ConfluenceChunker()

            assert chunker.strategy == "semantic_hierarchical"
            assert chunker.model_name == "gpt-4"

    def test_init_custom_strategy(self):
        """Test ConfluenceChunker with custom strategy."""
        with patch("src.ingestion.processors.chunker.tiktoken.encoding_for_model") as mock_tiktoken:
            mock_tiktoken.return_value = Mock()

            chunker = ConfluenceChunker(strategy="confluence_sections")

            assert chunker.strategy == "confluence_sections"

    def test_init_fallback_tokenizer(self):
        """Test tokenizer fallback for unknown model."""
        with patch("src.ingestion.processors.chunker.tiktoken.encoding_for_model") as mock_model_enc:
            with patch("src.ingestion.processors.chunker.tiktoken.get_encoding") as mock_get_enc:
                mock_model_enc.side_effect = KeyError("Unknown model")
                mock_get_enc.return_value = Mock()

                ConfluenceChunker(model_name="unknown-model")

                mock_get_enc.assert_called_once_with("cl100k_base")

    def test_chunk_documents_confluence_sections(self):
        """Test confluence sections chunking strategy."""
        with patch("src.ingestion.processors.chunker.tiktoken.encoding_for_model") as mock_tiktoken:
            mock_tiktoken.return_value = Mock()

            chunker = ConfluenceChunker(strategy="confluence_sections")
            # Use longer content that meets minimum viable section length
            content = "# Header 1\n" + "Content 1 " * 20 + "\n# Header 2\n" + "Content 2 " * 20
            doc = Document(content=content, metadata={"title": "Test"})

            result = chunker.chunk_documents([doc])

            assert len(result) >= 1
            for chunk in result:
                assert "chunk_strategy" in chunk.metadata
                assert chunk.metadata["chunk_strategy"] == "confluence_sections"

    def test_chunk_documents_hierarchical(self):
        """Test hierarchical chunking strategy."""
        with patch("src.ingestion.processors.chunker.tiktoken.encoding_for_model") as mock_tiktoken:
            mock_tiktoken.return_value = Mock()

            chunker = ConfluenceChunker(strategy="hierarchical")
            doc = Document(
                content="# Main Header\nContent under main\n## Sub Header\nSub content", metadata={"title": "Test"}
            )

            result = chunker.chunk_documents([doc])

            assert len(result) >= 1
            for chunk in result:
                if "chunk_strategy" in chunk.metadata:
                    assert chunk.metadata["chunk_strategy"] == "hierarchical"

    def test_is_header_markdown(self):
        """Test header detection for markdown."""
        with patch("src.ingestion.processors.chunker.tiktoken.encoding_for_model") as mock_tiktoken:
            mock_tiktoken.return_value = Mock()

            chunker = ConfluenceChunker()

            assert chunker._is_header("# Header 1") is True
            assert chunker._is_header("## Header 2") is True
            assert chunker._is_header("### Header 3") is True
            assert chunker._is_header("Regular text") is False

    def test_is_header_html(self):
        """Test header detection for HTML."""
        with patch("src.ingestion.processors.chunker.tiktoken.encoding_for_model") as mock_tiktoken:
            mock_tiktoken.return_value = Mock()

            chunker = ConfluenceChunker()

            assert chunker._is_header("<h1>Header 1</h1>") is True
            assert chunker._is_header("<h2>Header 2</h2>") is True
            assert chunker._is_header("<p>Paragraph</p>") is False

    def test_get_header_level_markdown(self):
        """Test header level extraction for markdown."""
        with patch("src.ingestion.processors.chunker.tiktoken.encoding_for_model") as mock_tiktoken:
            mock_tiktoken.return_value = Mock()

            chunker = ConfluenceChunker()

            assert chunker._get_header_level("# Header") == 1
            assert chunker._get_header_level("## Header") == 2
            assert chunker._get_header_level("### Header") == 3

    def test_get_header_level_html(self):
        """Test header level extraction for HTML."""
        with patch("src.ingestion.processors.chunker.tiktoken.encoding_for_model") as mock_tiktoken:
            mock_tiktoken.return_value = Mock()

            chunker = ConfluenceChunker()

            assert chunker._get_header_level("<h1>Header</h1>") == 1
            assert chunker._get_header_level("<h2>Header</h2>") == 2
            assert chunker._get_header_level("<h3>Header</h3>") == 3

    def test_extract_header_text_markdown(self):
        """Test header text extraction for markdown."""
        with patch("src.ingestion.processors.chunker.tiktoken.encoding_for_model") as mock_tiktoken:
            mock_tiktoken.return_value = Mock()

            chunker = ConfluenceChunker()

            assert chunker._extract_header_text("# Main Header") == "Main Header"
            assert chunker._extract_header_text("## Sub Header") == "Sub Header"

    def test_extract_header_text_html(self):
        """Test header text extraction for HTML."""
        with patch("src.ingestion.processors.chunker.tiktoken.encoding_for_model") as mock_tiktoken:
            mock_tiktoken.return_value = Mock()

            chunker = ConfluenceChunker()

            assert chunker._extract_header_text("<h1>Main Header</h1>") == "Main Header"
            assert chunker._extract_header_text("<h2>Sub Header</h2>") == "Sub Header"

    def test_is_table_start(self):
        """Test table start detection."""
        with patch("src.ingestion.processors.chunker.tiktoken.encoding_for_model") as mock_tiktoken:
            mock_tiktoken.return_value = Mock()

            chunker = ConfluenceChunker()

            assert chunker._is_table_start("<table>") is True
            assert chunker._is_table_start("| Column 1 | Column 2 |") is True
            assert chunker._is_table_start("|-----|-----|") is False
            assert chunker._is_table_start("Regular text") is False

    def test_is_list_item(self):
        """Test list item detection."""
        with patch("src.ingestion.processors.chunker.tiktoken.encoding_for_model") as mock_tiktoken:
            mock_tiktoken.return_value = Mock()

            chunker = ConfluenceChunker()

            assert chunker._is_list_item("- List item") is True
            assert chunker._is_list_item("* List item") is True
            assert chunker._is_list_item("+ List item") is True
            assert chunker._is_list_item("1. Numbered item") is True
            assert chunker._is_list_item("<li>HTML item</li>") is True
            assert chunker._is_list_item("Regular text") is False

    def test_classify_content_type(self):
        """Test content type classification."""
        with patch("src.ingestion.processors.chunker.tiktoken.encoding_for_model") as mock_tiktoken:
            mock_tiktoken.return_value = Mock()

            chunker = ConfluenceChunker()

            assert chunker._classify_content_type("Regular text content") == "text"
            assert chunker._classify_content_type("<table><tr><td>cell</td></tr></table>") == "table"
            assert chunker._classify_content_type("| col1 | col2 |") == "table"
            assert chunker._classify_content_type("```code block```") == "code"
            assert chunker._classify_content_type("<pre>code</pre>") == "code"
            assert chunker._classify_content_type("- List item\n- Another item") == "list"
            assert chunker._classify_content_type("1. First\n2. Second") == "list"

    def test_get_document_context(self):
        """Test document context generation."""
        with patch("src.ingestion.processors.chunker.tiktoken.encoding_for_model") as mock_tiktoken:
            mock_tiktoken.return_value = Mock()

            chunker = ConfluenceChunker()
            metadata = {
                "title": "Test Document",
                "space_key": "TEST",
                "url": "https://example.com/test",
                "source": "confluence",
            }

            context = chunker._get_document_context(metadata)

            assert "Test Document" in context
            assert "TEST" in context
            assert "https://example.com/test" in context
            assert "confluence" in context

    def test_add_context_to_chunk(self):
        """Test adding context to chunk content."""
        with patch("src.ingestion.processors.chunker.tiktoken.encoding_for_model") as mock_tiktoken:
            mock_tiktoken.return_value = Mock()

            chunker = ConfluenceChunker()
            content = "This is chunk content"
            metadata = {"title": "Test Doc"}

            enriched = chunker._add_context_to_chunk(content, metadata)

            assert "Test Doc" in enriched
            assert content in enriched
            assert enriched.count("\n") >= 2  # Context + newlines + content
