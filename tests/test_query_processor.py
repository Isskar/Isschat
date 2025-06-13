from src.rag_system.query_processor import QueryProcessor


def test_query_processor_basic():
    processor = QueryProcessor()

    query = "Comment configurer Confluence ?"
    result = processor.process_query(query)

    assert hasattr(result, "original_query")
    assert hasattr(result, "processed_query")
    assert result.original_query == query


def test_query_processor_empty_query():
    processor = QueryProcessor()

    result = processor.process_query("")

    assert result.original_query == ""


def test_query_processor_long_query():
    processor = QueryProcessor()

    long_query = "Comment " * 100 + "configurer Confluence ?"
    result = processor.process_query(long_query)

    assert result.original_query == long_query
    assert len(result.processed_query) > 0


def test_query_processor_special_characters():
    processor = QueryProcessor()

    query = "Qu'est-ce que l'API REST ? @#$%"
    result = processor.process_query(query)

    assert result.original_query == query
    assert result.processed_query is not None
