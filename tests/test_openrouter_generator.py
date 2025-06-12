from src.generation.openrouter_generator import OpenRouterGenerator
from src.core.interfaces import Document


def test_openrouter_generator_get_stats():
    generator = OpenRouterGenerator(model_name="test-model", temperature=0.1, max_tokens=512)

    stats = generator.get_stats()

    assert stats["model_name"] == "test-model"


def test_openrouter_generator_format_sources():
    generator = OpenRouterGenerator(model_name="test-model", temperature=0.1, max_tokens=512)

    documents = [
        Document(page_content="Contenu 1", metadata={"title": "Doc 1", "source": "doc1.md"}),
        Document(page_content="Contenu 2", metadata={"title": "Doc 2", "source": "doc2.md"}),
    ]

    sources = generator._format_sources(documents)

    assert "Doc 1" in sources
    assert "Doc 2" in sources
    assert "doc1.md" in sources
    assert "doc2.md" in sources
