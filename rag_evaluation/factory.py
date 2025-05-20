"""
Factory for creating components of the RAG evaluation system.

This module provides a unified interface to create and configure the various
components of the system (LLM, prompts, database, etc.).
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

from config_evaluation import EvaluationConfig, DatabaseType
from mock_db import MockVectorStore
from src.help_desk import HelpDesk


class ComponentFactory:
    """Factory to create the different components of the system."""

    @staticmethod
    def create_llm(config: EvaluationConfig) -> ChatOpenAI:
        """
        Create an instance of the LLM model based on the configuration.

        Args:
            config: System configuration

        Returns:
            ChatOpenAI: Configured LLM instance
        """
        return ChatOpenAI(
            model=config.llm.model_name,
            temperature=config.llm.temperature,
            openai_api_key=config.llm.api_key,
            openai_api_base=config.llm.api_base,
            max_tokens=config.llm.max_tokens,
        )

    @staticmethod
    def create_rag_prompt(config: EvaluationConfig) -> PromptTemplate:
        """
        Create the prompt for the RAG system.

        Args:
            config: System configuration

        Returns:
            PromptTemplate: Configured prompt instance
        """
        return PromptTemplate(template=config.prompts.rag_template, input_variables=["context", "question"])

    @staticmethod
    def create_database(config: EvaluationConfig):
        """
        Create the database (real or mocked).

        Args:
            config: System configuration

        Returns:
            Object: Configured database instance
        """
        if config.database.type == DatabaseType.MOCK:
            return MockVectorStore(mock_data=config.database.mock_data)
        else:
            # Use the real database
            from src.load_db import DataLoader

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            # Load the database
            loader = DataLoader(persist_directory=config.database.persist_directory)
            return loader.get_db(embeddings)

    @staticmethod
    def create_help_desk(config: EvaluationConfig, new_db: bool = False) -> HelpDesk:
        """
        Create an instance of HelpDesk using the shared configuration.

        If config.database.type is MOCK, uses a mocked database
        Otherwise, uses the real database.

        Args:
            config: System configuration
            new_db: If True, creates a new database

        Returns:
            HelpDesk: Configured HelpDesk instance
        """
        # If we're using a mock database, we need to modify the behavior
        if config.database.type == DatabaseType.MOCK:
            # Create a derived class that uses our mock database
            class MockHelpDesk(HelpDesk):
                def __init__(self, config: EvaluationConfig):
                    # Initialize without calling the parent to avoid loading the real DB
                    self.new_db = False
                    self.config = config  # Pass the config to use our settings
                    self.template = self.get_template()
                    self.embeddings = self.get_embeddings()
                    self.llm = self.get_llm()
                    self.prompt = self.get_prompt()

                    # Use our mock database
                    self.db = MockVectorStore(mock_data=config.database.mock_data)
                    self.retriever = self.db.as_retriever()

                    # Create the processing chain
                    self.retrieval_qa_chain = (
                        {"context": self.retriever, "question": RunnablePassthrough()}
                        | self.prompt
                        | self.llm
                        | StrOutputParser()
                    )

            return MockHelpDesk(config)
        else:
            # Use the normal implementation but with our configuration
            # Pass the config to HelpDesk so it can use our settings
            return HelpDesk(new_db=new_db, config=config)
