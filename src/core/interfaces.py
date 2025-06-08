"""
Core interfaces for the RAG system.
These abstract base classes define the contracts for all components.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """Document representation"""

    page_content: str
    metadata: Dict[str, Any]


@dataclass
class RetrievalResult:
    """Result from retrieval operation"""

    documents: List[Document]
    scores: List[float]
    query: str
    retrieval_time: float


@dataclass
class GenerationResult:
    """Result from generation operation"""

    answer: str
    sources: str
    generation_time: float
    token_count: int


class BaseExtractor(ABC):
    """Interface for data extraction from various sources"""

    @abstractmethod
    def extract(self) -> List[Document]:
        """Extract documents from the source"""
        pass

    @abstractmethod
    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the data source"""
        pass


class BaseDocumentProcessor(ABC):
    """Interface for document processing (filtering, chunking, etc.)"""

    @abstractmethod
    def process(self, documents: List[Document]) -> List[Document]:
        """Process a list of documents"""
        pass


class BaseEmbedder(ABC):
    """Interface for embedding generation"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        pass


class BaseVectorStore(ABC):
    """Interface for vector storage and retrieval"""

    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """Add documents with their embeddings to the store"""
        pass

    @abstractmethod
    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the vector store to disk"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector store from disk"""
        pass

    @abstractmethod
    def get_document_count(self) -> int:
        """Get the number of documents in the store"""
        pass


class BaseRetriever(ABC):
    """Interface for document retrieval"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve relevant documents for a query"""
        pass

    @abstractmethod
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get current retrieval configuration"""
        pass


class BaseGenerator(ABC):
    """Interface for answer generation"""

    @abstractmethod
    def generate(self, query: str, context_documents: List[Document]) -> GenerationResult:
        """Generate an answer based on query and context"""
        pass

    @abstractmethod
    def get_generation_config(self) -> Dict[str, Any]:
        """Get current generation configuration"""
        pass


class BaseRAGPipeline(ABC):
    """Interface for the complete RAG pipeline"""

    @abstractmethod
    def process_query(self, query: str) -> Tuple[str, str]:
        """Process a query and return answer and sources"""
        pass

    @abstractmethod
    def get_last_retrieval_result(self) -> Optional[RetrievalResult]:
        """Get the last retrieval result for analysis"""
        pass

    @abstractmethod
    def get_last_generation_result(self) -> Optional[GenerationResult]:
        """Get the last generation result for analysis"""
        pass


class BaseEvaluator(ABC):
    """Interface for system evaluation"""

    @abstractmethod
    def evaluate_single_query(self, question: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate a single query"""
        pass

    @abstractmethod
    def evaluate_batch(self, dataset: List[Dict]) -> List[Dict]:
        """Evaluate a batch of queries"""
        pass
