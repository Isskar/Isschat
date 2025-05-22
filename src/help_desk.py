import load_db
import collections
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import os


class HelpDesk:
    """Create the necessary objects to create a QARetrieval chain"""

    def __init__(self, new_db: bool = True) -> None:
        self.new_db = new_db
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
        template = """
        You are a professional and friendly virtual assistant named "ISSCHAT".
        Your mission is to help users find information in the Confluence documentation.

        Based on these text excerpts:
        -----
        {context}
        -----

        Answer the following question IN FRENCH in a conversational and professional manner.
        Use a friendly but professional tone, as if you were a helpful colleague.
        Be concise but complete. Use French phrases like "je vous suggère de..."
        (I suggest that you...), "vous pourriez..." (you could...), etc.
        If you don't have the information, clearly state so and suggest alternatives.
        IMPORTANT: Always respond in French regardless of the language of the question.

        Question: {question}
        Answer:
        """
        return template

    def get_prompt(self) -> PromptTemplate:
        prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])
        return prompt

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return embeddings

    def get_llm(self):
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
        # Define a simple retrieval chain using the LCEL (LangChain Expression Language) approach
        # This avoids the Pydantic validation issues
        retrieval_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()} | self.prompt | self.llm | StrOutputParser()
        )
        return retrieval_chain

    def retrieval_qa_inference(self, question: str, verbose: bool = True) -> tuple[str, str]:
        # Get the source documents directly from the retriever using the new invoke method
        docs = self.retriever.invoke(question)

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
        sources = [f"[{res.metadata['title']}]({res.metadata['source']})" for res in answer["source_documents"]]

        if sources:
            k = min(k, len(sources))
            distinct_sources = list(zip(*collections.Counter(sources).most_common()))[0][:k]
            distinct_sources_str = "  \n- ".join(distinct_sources)

        if len(distinct_sources) == 1:
            return f"Voici la source que j'ai utilisée pour répondre à votre question:  \n- {distinct_sources_str}"  # noqa

        elif len(distinct_sources) > 1:
            return f"Voici les {len(distinct_sources)} sources que j'ai utilisées pour répondre à votre question:  \n- {distinct_sources_str}"  # noqa

        else:
            return "Désolé, je n'ai pas trouvé de ressources utiles pour répondre à votre question"
