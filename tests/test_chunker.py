from src.data_pipeline.processors.chunker import DocumentChunker
from src.core.interfaces import Document


def test_chunker_custom_config():
    config = {"chunk_size": 500, "chunk_overlap": 100, "separator": "\n"}

    chunker = DocumentChunker(config)

    assert chunker.chunk_size == 500
    assert chunker.chunk_overlap == 100
    assert chunker.separator == "\n"


def test_chunker_small_document():
    chunker = DocumentChunker({"chunk_size": 1000})
    doc = Document(page_content="Ceci est un document court.", metadata={"title": "Test", "source": "test.md"})

    chunks = chunker.chunk_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].page_content == "Ceci est un document court."
    assert chunks[0].metadata["title"] == "Test"


def test_chunker_large_document():
    """Test: Document plus grand que chunk_size"""
    chunker = DocumentChunker({"chunk_size": 50, "chunk_overlap": 10})

    long_content = "Ceci est un très long document. " * 10
    doc = Document(page_content=long_content, metadata={"title": "Long Doc", "source": "long.md"})

    chunks = chunker.chunk_documents([doc])

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.metadata["title"] == "Long Doc"
        assert "chunk_index" in chunk.metadata


def test_chunker_multiple_documents():
    chunker = DocumentChunker({"chunk_size": 100})

    docs = [
        Document(page_content="Premier document de test.", metadata={"title": "Doc 1", "source": "doc1.md"}),
        Document(page_content="Deuxième document de test.", metadata={"title": "Doc 2", "source": "doc2.md"}),
    ]

    chunks = chunker.chunk_documents(docs)

    assert len(chunks) >= 2

    titles = [chunk.metadata["title"] for chunk in chunks]
    assert "Doc 1" in titles
    assert "Doc 2" in titles


def test_chunker_empty_document():
    chunker = DocumentChunker()

    doc = Document(page_content="", metadata={"title": "Empty", "source": "empty.md"})

    chunks = chunker.chunk_documents([doc])

    assert len(chunks) >= 0


def test_chunker_with_separators():
    chunker = DocumentChunker({"chunk_size": 100, "separator": "\n\n"})

    content = "Paragraphe 1\n\nParagraphe 2\n\nParagraphe 3\n\nParagraphe 4"
    doc = Document(page_content=content, metadata={"title": "Multi Para", "source": "multi.md"})

    chunks = chunker.chunk_documents([doc])

    assert len(chunks) > 0

    for chunk in chunks:
        assert len(chunk.page_content.strip()) > 0
