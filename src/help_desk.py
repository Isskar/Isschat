import load_db
import collections
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import os
from typing import Optional

from config import LLMPrompt


class HelpDesk:
    """Create the necessary objects to create a QARetrieval chain"""

    def __init__(self, new_db: bool = True, config: Optional[object] = None) -> None:
        """
        Initialize the HelpDesk with optional configuration.

        Args:
            new_db: Whether to create a new database or use an existing one
            config: Optional configuration object (from rag_evaluation)
        """
        self.new_db = new_db
        self.config = config
        self.template = self.get_template()
        self.embeddings = self.get_embeddings()
        self.llm = self.get_llm()
        self.prompt = self.get_prompt()

        if self.new_db:
            self.db = load_db.DataLoader().set_db(self.embeddings)
        else:
            self.db = load_db.DataLoader().get_db(self.embeddings)

        # Optimize the retriever for faster responses
        self.retriever = self.db.as_retriever(
            search_kwargs={
                "k": 3,  # Reduce the number of retrieved documents (default 4)
                "fetch_k": 5,  # Reduce the number of documents to consider before selecting the best k
            }
        )
        self.retrieval_qa_chain = self.get_retrieval_qa()

    def get_template(self) -> str:
        """
        Get the template for the RAG prompt.
        If config is provided, use its prompt template, otherwise use default LLMPrompt.

        Returns:
            str: The prompt template
        """
        # If we have a config from rag_evaluation, use its rag_template
        if hasattr(self, "config") and self.config and hasattr(self.config, "prompts"):
            return self.config
        # Otherwise use the default template from config.py
        return LLMPrompt.PROMPT_TEMPLATE

    def get_prompt(self) -> PromptTemplate:
        """
        Create a PromptTemplate from the template.

        Returns:
            PromptTemplate: The configured prompt template
        """
        prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])
        return prompt

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get the embeddings model.

        Returns:
            HuggingFaceEmbeddings: The embeddings model
        """
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return embeddings

    def get_llm(self):
        """
        Get the LLM model, either from config or using default settings.

        Returns:
            ChatOpenAI: The configured LLM
        """
        # If config is provided, use its LLM settings
        if hasattr(self, "config") and self.config and hasattr(self.config, "llm"):
            llm_config = self.config.llm
            return ChatOpenAI(
                model=llm_config.model_name,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                openai_api_key=llm_config.api_key,
                openai_api_base=llm_config.api_base,
            )

        # Otherwise use default settings
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        # Use ChatOpenAI with the custom client
        llm = ChatOpenAI(
            model="deepseek/deepseek-chat",
            temperature=0.1,
            max_tokens=512,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )
        return llm

    def get_retrieval_qa(self):
        """
        Create the retrieval QA chain.

        Returns:
            Chain: The retrieval QA chain
        """
        # Define a simple retrieval chain using the LCEL (LangChain Expression Language) approach
        # This avoids the Pydantic validation issues
        retrieval_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()} | self.prompt | self.llm | StrOutputParser()
        )
        return retrieval_chain

    def retrieval_qa_inference(self, question: str, verbose: bool = True) -> tuple[str, str]:
        """
        Run inference with the retrieval QA chain.

        Args:
            question: The question to answer
            verbose: Whether to print detailed information

        Returns:
            tuple[str, str]: The answer and sources
        """
        # Get the source documents directly from the retriever
        docs = self.retriever.get_relevant_documents(question)

        # Add logs to verify the retrieved documents
        if verbose:
            print(f"\n=== Documents retrieved for the question: '{question}' ===\n")
            for i, doc in enumerate(docs[:3]):  # Display the first 3 documents
                print(f"Document {i + 1}:")
                print(f"Title: {doc.metadata.get('title', 'Not available')}")
                print(f"Source: {doc.metadata.get('source', 'Not available')}")
                print(f"Content (excerpt): {doc.page_content[:150]}...\n")

        # Get the answer from the chain
        answer = self.retrieval_qa_chain.invoke(question)

        sources = self.list_top_k_sources({"source_documents": docs}, k=2)

        if verbose:
            print(sources)

        return answer, sources

    def list_top_k_sources(self, answer: dict, k: int = 2) -> str:
        """
        List the top k sources from the answer.

        Args:
            answer: The answer dictionary with source_documents
            k: The maximum number of sources to return

        Returns:
            str: A formatted string with the sources
        """
        sources = [f"[{res.metadata['title']}]({res.metadata['source']})" for res in answer["source_documents"]]

        if sources:
            k = min(k, len(sources))
            distinct_sources = list(zip(*collections.Counter(sources).most_common()))[0][:k]
            distinct_sources_str = "  \n- ".join(distinct_sources)

            if len(distinct_sources) == 1:
                return f"Here is the source that might be useful for you:  \n- {distinct_sources_str}"
            elif len(distinct_sources) > 1:
                return f"Here are {len(distinct_sources)} sources that might be useful for you:  \n- {distinct_sources_str}"

        return "Sorry, I couldn't find any resources to answer your question"
