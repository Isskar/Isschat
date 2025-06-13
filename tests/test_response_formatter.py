from src.rag_system.response_formatter import ResponseFormatter


def test_response_formatter_initialization():
    formatter = ResponseFormatter()

    assert formatter.include_sources is True
    assert formatter.include_confidence is False
    assert formatter.max_source_length == 200


def test_response_formatter_custom_config():
    config = {"include_sources": False, "include_confidence": True, "max_source_length": 100}

    formatter = ResponseFormatter(config)

    assert formatter.include_sources is False
    assert formatter.include_confidence is True
    assert formatter.max_source_length == 100


def test_format_response_without_sources():
    config = {"include_sources": False}
    formatter = ResponseFormatter(config)

    response = "Voici la réponse à votre question."
    context_docs = [{"title": "Doc 1", "source": "doc1.md", "content": "Contenu 1"}]

    formatted = formatter.format_response(response, context_docs)

    assert formatted == response
    assert "Sources:" not in formatted


def test_format_response_with_sources():
    formatter = ResponseFormatter({"include_sources": True})

    response = "Voici la réponse à votre question."
    context_docs = [
        {"content": "Contenu du document 1", "metadata": {"title": "Document 1", "source": "doc1.md"}},
        {"content": "Contenu du document 2", "metadata": {"title": "Document 2", "source": "doc2.md"}},
    ]

    formatted = formatter.format_response(response, context_docs)

    assert response in formatted
    assert "Document 1" in formatted
    assert "Document 2" in formatted
    assert "doc1.md" in formatted
    assert "doc2.md" in formatted


def test_format_response_with_confidence_scores():
    config = {"include_sources": True, "include_confidence": True}
    formatter = ResponseFormatter(config)

    response = "Réponse de test."
    context_docs = [
        {"title": "Doc High", "source": "high.md", "content": "Contenu pertinent"},
        {"title": "Doc Low", "source": "low.md", "content": "Contenu moins pertinent"},
    ]
    confidence_scores = [0.95, 0.65]

    formatted = formatter.format_response(response, context_docs, confidence_scores)

    assert "0.95" in formatted or "95%" in formatted
    assert "0.65" in formatted or "65%" in formatted


def test_format_response_empty_context():
    formatter = ResponseFormatter()

    response = "Réponse sans contexte."
    context_docs = []

    formatted = formatter.format_response(response, context_docs)

    assert formatted == response


def test_format_sources_basic():
    """Test: Formatage des sources basique"""
    formatter = ResponseFormatter()

    context_docs = [
        {"content": "Documentation API", "metadata": {"title": "Guide API", "source": "api.md"}},
        {"content": "Guide d'utilisation", "metadata": {"title": "Tutorial", "source": "tutorial.md"}},
    ]

    sources = formatter._format_sources(context_docs)

    assert "Guide API" in sources
    assert "Tutorial" in sources
    assert "api.md" in sources
    assert "tutorial.md" in sources


def test_format_sources_with_long_content():
    config = {"max_source_length": 50}
    formatter = ResponseFormatter(config)

    long_content = "Ceci est un très long contenu qui dépasse la limite. " * 10
    context_docs = [{"content": long_content, "metadata": {"title": "Long Doc", "source": "long.md"}}]

    sources = formatter._format_sources(context_docs)

    assert len(sources) < len(long_content)
    assert "Long Doc" in sources


def test_format_response_special_characters():
    formatter = ResponseFormatter()

    response = "Réponse avec accents: àéèùç"
    context_docs = [{"title": "Doc spécial", "source": "spécial.md", "content": "Contenu spécial"}]

    formatted = formatter.format_response(response, context_docs)

    assert "àéèùç" in formatted
    assert "spécial" in formatted
