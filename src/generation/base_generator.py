"""
Base generator interface and implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.utils import convert_to_secret_str

from src.core.interfaces import GenerationResult, RetrievalResult
from src.core.exceptions import GenerationError, ConfigurationError
from src.core.config import get_config


class BaseGenerator(ABC):
    """Abstract base class for all generators."""

    @abstractmethod
    def generate(self, query: str, retrieval_result: RetrievalResult) -> GenerationResult:
        """
        Generate answer based on query and retrieved documents.

        Args:
            query: User query
            retrieval_result: Result from retrieval step

        Returns:
            GenerationResult with answer and metadata
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        pass

