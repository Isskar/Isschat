"""
HelpDesk module for the Confluence Assistant application.

This module provides functionality for querying a Confluence knowledge base
using natural language and retrieving relevant information.
"""
import os
import sys
import logging
import load_db
import collections
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Third-party imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError

# Local imports
from utils import (
    setup_logging,
    ConfluenceConnectionError,
    OpenRouterError,
    VectorStoreError,
    handle_errors,
    retry_api_call,
    GracefulDegradation
)

class HelpDesk:
    """
    A class to handle the main functionality of the Confluence Assistant.
    
    This class is responsible for initializing the necessary components for
    querying a Confluence knowledge base, including the language model,
    embeddings, and retrieval chain.
    
    Args:
        new_db (bool): Whether to create a new database or load an existing one.
        verbose (bool): Whether to enable verbose logging.
    """
    
    def __init__(self, new_db: bool = True, verbose: bool = False):
        """Initialize the HelpDesk with the specified configuration."""
        self.new_db = new_db
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Set up logging level based on verbose flag
        log_level = logging.DEBUG if verbose else logging.INFO
        setup_logging(level=log_level)
        
        # Initialize components with error handling
        try:
            self.template = self.get_template()
            self.embeddings = self.get_embeddings()
            self.llm = self.get_llm()
            self.prompt = self.get_prompt()
            
            # Initialize database with graceful degradation
            with GracefulDegradation(
                fallback_func=self._handle_db_error,
                exceptions=(ConfluenceConnectionError, VectorStoreError, Exception),
                log_errors=True
            ):
                if self.new_db:
                    self.logger.info("Creating new database...")
                    self.db = load_db.DataLoader().set_db(self.embeddings)
                else:
                    self.logger.info("Loading existing database...")
                    self.db = load_db.DataLoader().get_db(self.embeddings)
            
            # Configure retriever with optimized settings
            self.retriever = self._initialize_retriever()
            self.retrieval_qa_chain = self.get_retrieval_qa()
            
            self.logger.info("HelpDesk initialized successfully")
            
        except Exception as e:
            self.logger.critical("Failed to initialize HelpDesk: %s", str(e), exc_info=True)
            raise
    
    def _initialize_retriever(self):
        """Initialize the document retriever with optimized settings."""
        try:
            retriever = self.db.as_retriever(
                search_kwargs={
                    "k": 3,  # Reduce the number of documents retrieved (default is 4)
                    "fetch_k": 5  # Reduce the number of documents to consider before selecting the top k
                }
            )
            self.logger.debug("Retriever initialized with k=3, fetch_k=5")
            return retriever
        except Exception as e:
            self.logger.error("Failed to initialize retriever: %s", str(e), exc_info=True)
            raise VectorStoreError(f"Failed to initialize document retriever: {str(e)}")
    
    def _handle_db_error(self, error: Exception) -> None:
        """Handle database-related errors with a fallback to a dummy database."""
        self.logger.warning("Falling back to dummy database due to error: %s", str(error))
        # Create a simple in-memory database with a helpful message
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        
        dummy_docs = [
            Document(
                page_content="The Confluence Assistant is currently experiencing issues connecting to the knowledge base. "
                           "Please check your internet connection and try again later.",
                metadata={"source": "error", "title": "Connection Error"}
            )
        ]
        
        self.db = FAISS.from_documents(dummy_docs, self.embeddings)
        self.logger.info("Successfully initialized with dummy database")

    def get_template(self) -> str:
        """
        Get the template for the LLM prompt.
        
        Returns:
            str: The template string with placeholders for context and question.
        """
        template = """
        Tu es un assistant virtuel professionnel et amical nommé "Confluence Assistant". 
        Ta mission est d'aider les utilisateurs à trouver des informations dans la documentation Confluence.
        
        À partir de ces extraits de texte :
        -----
        {context}
        -----
        
        Réponds à la question suivante en français de manière conversationnelle et professionnelle.
        Utilise un ton amical mais professionnel, comme si tu étais un collègue serviable.
        Sois concis mais complet. Utilise des formulations comme "je vous suggère de...", "vous pourriez...", etc.
        Si tu n'as pas l'information, dis-le clairement et propose des alternatives.
        
        Question : {question}
        Réponse :
        """
        self.logger.debug("Generated template for LLM prompt")
        return template

    def get_prompt(self) -> PromptTemplate:
        """
        Create a PromptTemplate instance from the template.
        
        Returns:
            PromptTemplate: The configured prompt template.
            
        Raises:
            ValueError: If the template is invalid or missing required variables.
        """
        try:
            prompt = PromptTemplate(
                template=self.template,
                input_variables=["context", "question"]
            )
            self.logger.debug("Created prompt template with input variables: %s", 
                           prompt.input_variables)
            return prompt
        except Exception as e:
            self.logger.error("Failed to create prompt template: %s", str(e), exc_info=True)
            raise ValueError("Failed to create prompt template. Please check the template format.") from e

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Initialize and return the HuggingFace embeddings model.
        
        Returns:
            HuggingFaceEmbeddings: The initialized embeddings model.
            
        Raises:
            VectorStoreError: If the embeddings model cannot be loaded.
        """
        try:
            self.logger.info("Loading HuggingFace embeddings model: all-MiniLM-L6-v2")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.logger.debug("Successfully loaded embeddings model")
            return embeddings
        except Exception as e:
            self.logger.error("Failed to load embeddings model: %s", str(e), exc_info=True)
            raise VectorStoreError(f"Failed to load embeddings model: {str(e)}") from e

    @retry_api_call(
        max_retries=3,
        initial_wait=1.0,
        max_wait=10.0,
        exceptions=[APIError, APITimeoutError, APIConnectionError]
    )
    def get_llm(self) -> ChatOpenAI:
        """
        Initialize and return the language model for generating responses.
        
        Returns:
            ChatOpenAI: The initialized language model.
            
        Raises:
            OpenRouterError: If the language model cannot be initialized.
        """
        try:
            # Get API key from environment
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                error_msg = "OPENROUTER_API_KEY not found in environment variables"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Log API key status (masked for security)
            masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "[INVALID]"
            self.logger.info("Initializing ChatOpenAI with OpenRouter API (key: %s)", masked_key)
            
            # Configure OpenAI client with OpenRouter API
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                max_retries=3,
                timeout=30.0
            )
            
            # Initialize ChatOpenAI with optimized parameters
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1,        # More deterministic responses
                max_tokens=512,          # Limit response length
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                max_retries=3,
                request_timeout=30.0
            )
            
            self.logger.debug("Successfully initialized ChatOpenAI with model: gpt-4o")
            return llm
            
        except Exception as e:
            self.logger.error("Failed to initialize language model: %s", str(e), exc_info=True)
            raise OpenRouterError(f"Failed to initialize language model: {str(e)}") from e

    def get_retrieval_qa(self):
        """
        Create a retrieval-based question-answering chain.
        
        Returns:
            A configured retrieval QA chain.
            
        Raises:
            ValueError: If the chain cannot be created due to invalid components.
        """
        try:
            self.logger.debug("Creating retrieval QA chain")
            
            # Define a simple retrieval chain using the LCEL (LangChain Expression Language) approach
            retrieval_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            self.logger.debug("Successfully created retrieval QA chain")
            return retrieval_chain
            
        except Exception as e:
            self.logger.error("Failed to create retrieval QA chain: %s", str(e), exc_info=True)
            raise ValueError(f"Failed to create retrieval QA chain: {str(e)}") from e

    @handle_errors(
        exceptions={
            ValueError: "Invalid question format",
            ConnectionError: "Failed to connect to the knowledge base",
            TimeoutError: "Request timed out"
        },
        default_message="An error occurred while processing your question"
    )
    def retrieval_qa_inference(self, question: str, verbose: Optional[bool] = None) -> Tuple[str, str]:
        """
        Generate an answer to a question using the retrieval-augmented QA chain.
        
        Args:
            question: The question to answer.
            verbose: Whether to log detailed information. If None, uses the instance's verbose setting.
            
        Returns:
            A tuple containing the answer and the sources.
            
        Raises:
            ValueError: If the question is empty or invalid.
            ConnectionError: If there's a connection issue with the knowledge base.
            TimeoutError: If the request times out.
            Exception: For any other unexpected errors.
        """
        verbose = self.verbose if verbose is None else verbose
        
        if not question or not isinstance(question, str) or not question.strip():
            error_msg = "Question cannot be empty"
            self.logger.warning(error_msg)
            raise ValueError(error_msg)
            
        self.logger.info("Processing question: %s", question)
        
        try:
            # Get relevant documents from the retriever
            try:
                docs = self.retriever.get_relevant_documents(question)
                self.logger.debug("Retrieved %d documents for question", len(docs))
                
                # Log retrieved documents if verbose
                if verbose:
                    self._log_retrieved_documents(question, docs)
                    
            except Exception as e:
                self.logger.error("Error retrieving documents: %s", str(e), exc_info=True)
                raise ConnectionError("Failed to retrieve relevant documents") from e
            
            # Get answer from the QA chain
            try:
                answer = self.retrieval_qa_chain.invoke(question)
                self.logger.debug("Generated answer: %.50s...", answer[:50])
            except Exception as e:
                self.logger.error("Error generating answer: %s", str(e), exc_info=True)
                raise ConnectionError("Failed to generate answer") from e
            
            # Process sources
            try:
                sources = self.list_top_k_sources({"source_documents": docs}, k=2)
                if verbose:
                    self.logger.debug("Sources: %s", sources)
            except Exception as e:
                self.logger.warning("Error processing sources: %s", str(e), exc_info=True)
                sources = "Source information not available"
            
            return answer, sources
            
        except Exception as e:
            self.logger.error("Error in retrieval_qa_inference: %s", str(e), exc_info=True)
            raise
    
    def _log_retrieved_documents(self, question: str, docs: list) -> None:
        """Log information about retrieved documents."""
        self.logger.info("=== Retrieved Documents for question: '%s' ===", question)
        for i, doc in enumerate(docs[:3]):  # Log first 3 documents
            self.logger.info(
                "Document %d:\nTitle: %s\nSource: %s\nContent (excerpt): %.100s...",
                i + 1,
                doc.metadata.get('title', 'Not available'),
                doc.metadata.get('source', 'Not available'),
                doc.page_content
            )

    def list_top_k_sources(self, answer: Dict[str, Any], k: int = 2) -> str:
        """
        Format the top k sources from the answer into a user-friendly string.
        
        Args:
            answer: The answer dictionary containing source documents.
            k: The maximum number of sources to include.
            
        Returns:
            A formatted string with the top k sources.
        """
        try:
            # Safely extract source documents with error handling
            source_docs = answer.get("source_documents", [])
            if not source_docs:
                self.logger.debug("No source documents found in answer")
                return "Désolé, je n'ai trouvé aucune ressource pour répondre à ta question."
            
            # Process sources with error handling for metadata
            sources = []
            for i, doc in enumerate(source_docs):
                try:
                    title = doc.metadata.get('title', 'Sans titre')
                    source = doc.metadata.get('source', '#')
                    sources.append(f'[{title}]({source})')
                except Exception as e:
                    self.logger.warning("Error processing source document %d: %s", i, str(e))
            
            # If no valid sources were found
            if not sources:
                return "Désolé, je n'ai pas pu récupérer les sources pour cette réponse."
            
            # Get top k distinct sources
            k = max(1, min(k, len(sources)))  # Ensure k is between 1 and len(sources)
            distinct_sources = list(zip(*collections.Counter(sources).most_common()))[0][:k]
            distinct_sources_str = "  \n- ".join(distinct_sources)
            
            # Format the response based on number of sources
            if len(distinct_sources) == 1:
                return f"Voici la source qui pourrait t'être utile :  \n- {distinct_sources_str}"
            else:
                return f"Voici {len(distinct_sources)} sources qui pourraient t'être utiles :  \n- {distinct_sources_str}"
                
        except Exception as e:
            self.logger.error("Error in list_top_k_sources: %s", str(e), exc_info=True)
            return "Désolé, une erreur est survenue lors de la récupération des sources."
