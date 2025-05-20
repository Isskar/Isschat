"""
Mock vector database for CI tests.

This module provides a simplified implementation of a vector database
that can be used for CI tests without depending on a real database.
"""

from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field


class MockVectorStore:
    """Mock vector database for CI tests."""

    def __init__(self, mock_data_path: str = None, mock_data: Dict[str, Any] = None):
        """
        Initialize a mock vector database.

        Args:
            mock_data_path: Path to a JSON file containing mock data
            mock_data: Dictionary containing mock data
        """
        self.data = {}

        if mock_data:
            self.data = mock_data
        elif mock_data_path:
            path = Path(mock_data_path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)

        # Expected format for self.data:
        # {
        #   "question1": [
        #     {"content": "...", "metadata": {"title": "...", "source": "..."}},
        #     {"content": "...", "metadata": {"title": "...", "source": "..."}}
        #   ],
        #   "question2": [ ... ]
        # }

        # If no data is provided, use default data
        if not self.data:
            self.data = {
                "How to add a page?": [
                    {
                        "content": "To add a page in Confluence, click on the '+' button in the navigation bar...",
                        "metadata": {"title": "Page Creation", "source": "https://example.com/doc1"},
                    }
                ],
                "How to manage permissions?": [
                    {
                        "content": "Permission management is done in the administration settings...",
                        "metadata": {"title": "Permission Management", "source": "https://example.com/doc2"},
                    }
                ],
            }

    def as_retriever(self, search_kwargs=None):
        """
        Return a mock retriever.

        Args:
            search_kwargs: Search arguments (ignored in mock version)

        Returns:
            MockRetriever: Instance of mock retriever
        """
        return MockRetriever(mock_data=self.data)


class MockRetriever(BaseRetriever):
    """Mock retriever for CI tests that inherits from BaseRetriever to be compatible with LangChain."""

    # Properly declare mock_data as a Field to satisfy Pydantic validation
    mock_data: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict, description="Dictionary mapping questions to lists of documents"
    )

    def __init__(self, mock_data: Dict[str, List[Dict[str, Any]]]):
        """
        Initialize a mock retriever with predefined data.

        Args:
            mock_data: Dictionary mapping questions to lists of documents
        """
        # Initialize the BaseRetriever with the mock_data parameter
        super().__init__(mock_data=mock_data)

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a given query. This method is required by BaseRetriever.

        Args:
            query: User query
            run_manager: Callback manager for retriever run, not used here

        Returns:
            List[Document]: List of documents matching the query or default documents
        """
        # If mock_data is empty, return an empty list
        if not self.mock_data:
            return []

        # Exact search (for tests)
        if query in self.mock_data:
            docs_data = self.mock_data[query]
        else:
            # Partial search (keywords)
            found = False
            for key, docs in self.mock_data.items():
                if any(word in key.lower() for word in query.lower().split()):
                    docs_data = docs
                    found = True
                    break

            if not found:
                # Default document if no matches
                docs_data = list(self.mock_data.values())[0] if self.mock_data else []

        # Convert to LangChain Documents
        docs = []
        for doc_data in docs_data:
            doc = Document(page_content=doc_data["content"], metadata=doc_data["metadata"])
            docs.append(doc)

        return docs

    # NOTE: Do NOT implement get_relevant_documents as it's already provided by BaseRetriever
    # It would cause recursion issues as BaseRetriever.get_relevant_documents calls _get_relevant_documents
