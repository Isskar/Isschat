from unittest.mock import Mock, patch
from src.vector_store.faiss_store import FAISSVectorStore
from src.core.interfaces import Document


def test_faiss_store_add_documents(test_config, mock_embeddings):
    with patch("src.core.embeddings_manager.EmbeddingsManager.get_embeddings", return_value=mock_embeddings):
        with patch("langchain_community.vectorstores.FAISS.from_documents") as mock_faiss:
            mock_store = Mock()
            mock_faiss.return_value = mock_store

            config = {"persist_directory": "/tmp/test_faiss"}
            store = FAISSVectorStore(config)

            documents = [
                Document(page_content="Contenu de test 1", metadata={"title": "Doc 1", "source": "test1.md"}),
                Document(page_content="Contenu de test 2", metadata={"title": "Doc 2", "source": "test2.md"}),
            ]

            store.add_documents(documents)

            mock_faiss.assert_called_once()
            assert store._store == mock_store


def test_faiss_store_search_documents(test_config, mock_embeddings):
    with patch("src.core.embeddings_manager.EmbeddingsManager.get_embeddings", return_value=mock_embeddings):
        # Patch the ensure_loaded method to prevent loading real DB
        with patch.object(FAISSVectorStore, "ensure_loaded", return_value=True):
            config = {"persist_directory": "/tmp/test_faiss"}
            store = FAISSVectorStore(config)

            mock_store = Mock()
            mock_doc = Mock()
            mock_doc.page_content = "Résultat de test"
            mock_doc.metadata = {"title": "Test", "source": "test.md"}
            mock_store.similarity_search.return_value = [mock_doc]

            # Force the mock store and prevent real loading
            store._store = mock_store
            store._is_loaded = True

            results = store.search_documents("question de test", k=1)

            assert len(results) == 1
            assert results[0].page_content == "Résultat de test"
            assert results[0].metadata["title"] == "Test"


def test_faiss_store_get_stats(test_config, mock_embeddings):
    """Test: Récupération des statistiques"""
    with patch("src.core.embeddings_manager.EmbeddingsManager.get_embeddings", return_value=mock_embeddings):
        config = {"persist_directory": "/tmp/test_faiss"}
        store = FAISSVectorStore(config)

        mock_store = Mock()
        mock_store.index.ntotal = 100
        store._store = mock_store

        stats = store.get_stats()

        assert "document_count" in stats
        assert stats["document_count"] == 100
