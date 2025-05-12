import sys
import load_db
import collections
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openai import OpenAI
import os

class HelpDesk():
    """Create the necessary objects to create a QARetrieval chain"""
    def __init__(self, new_db=True):
        self.new_db = new_db
        self.template = self.get_template()
        self.embeddings = self.get_embeddings()
        self.llm = self.get_llm()
        self.prompt = self.get_prompt()

        if self.new_db:
            self.db = load_db.DataLoader().set_db(self.embeddings)
        else:
            self.db = load_db.DataLoader().get_db(self.embeddings)

        # Optimiser le retriever pour des réponses plus rapides
        self.retriever = self.db.as_retriever(
            search_kwargs={
                "k": 3,  # Réduire le nombre de documents récupérés (par défaut 4)
                "fetch_k": 5  # Réduire le nombre de documents à considérer avant de sélectionner les k meilleurs
            }
        )
        self.retrieval_qa_chain = self.get_retrieval_qa()

    def get_template(self):
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
        return template

    def get_prompt(self):
        prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )
        return prompt

    def get_embeddings(self):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return embeddings

    def get_llm(self):
        # Utiliser OpenRouter avec l'API OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        # Configure OpenAI client with OpenRouter API
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
            
        # Use ChatOpenAI with the custom client
        llm = ChatOpenAI(
            model="gpt-4o",  # Vous pouvez spécifier le modèle désiré sur OpenRouter
            temperature=0.1,        # Réduire la température pour des réponses plus déterministes
            max_tokens=512,         # Limiter la longueur de la réponse
            openai_api_key=api_key, # Utiliser la clé API OpenRouter
            openai_api_base="https://openrouter.ai/api/v1"  # Préciser l'URL de base OpenRouter
        )
        return llm

    def get_retrieval_qa(self):
        # Define a simple retrieval chain using the LCEL (LangChain Expression Language) approach
        # This avoids the Pydantic validation issues
        retrieval_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return retrieval_chain

    def retrieval_qa_inference(self, question, verbose=True):
        # Get the source documents directly from the retriever
        docs = self.retriever.get_relevant_documents(question)
        
        # Ajouter des logs pour vérifier les documents récupérés
        if verbose:
            print(f"\n=== Documents récupérés pour la question: '{question}' ===\n")
            for i, doc in enumerate(docs[:3]):  # Afficher les 3 premiers documents
                print(f"Document {i+1}:")
                print(f"Titre: {doc.metadata.get('title', 'Non disponible')}")
                print(f"Source: {doc.metadata.get('source', 'Non disponible')}")
                print(f"Contenu (extrait): {doc.page_content[:150]}...\n")
        
        # Get the answer from the chain
        answer = self.retrieval_qa_chain.invoke(question)
        
        sources = self.list_top_k_sources({"source_documents": docs}, k=2)

        if verbose:
            print(sources)

        return answer, sources

    def list_top_k_sources(self, answer, k=2):
        sources = [
            f'[{res.metadata["title"]}]({res.metadata["source"]})'
            for res in answer["source_documents"]
        ]

        if sources:
            k = min(k, len(sources))
            distinct_sources = list(zip(*collections.Counter(sources).most_common()))[0][:k]
            distinct_sources_str = "  \n- ".join(distinct_sources)

        if len(distinct_sources) == 1:
            return f"Voici la source qui pourrait t'être utile :  \n- {distinct_sources_str}"

        elif len(distinct_sources) > 1:
            return f"Voici {len(distinct_sources)} sources qui pourraient t'être utiles :  \n- {distinct_sources_str}"

        else:
            return "Désolé je n'ai trouvé aucune ressource pour répondre à ta question"
