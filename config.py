import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

CONFLUENCE_PRIVATE_API_KEY = os.getenv("CONFLUENCE_PRIVATE_API_KEY", "")
CONFLUENCE_API_KEY = os.getenv("CONFLUENCE_PRIVATE_API_KEY", "")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY", "")
CONFLUENCE_SPACE_NAME = os.getenv("CONFLUENCE_SPACE_NAME", "")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_EMAIL_ADRESS", "")
CONFLUENCE_EMAIL_ADRESS = os.getenv("CONFLUENCE_EMAIL_ADRESS", "")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "data/users.db"))
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", os.path.join(BASE_DIR, "db"))
EVALUATION_DATASET = os.getenv("EVALUATION_DATASET", os.path.join(BASE_DIR, "data/evaluation_dataset.tsv"))


@dataclass
class LLMConfig:
    """Configuration du modèle LLM partagée par tous les composants."""

    model_name: str = "deepseek/deepseek-chat"
    temperature: float = 0.1
    api_key: str = ""
    api_base: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 512


@dataclass
class LLMPrompt:
    PROMPT_TEMPLATE: str = """
        You are a professional and friendly virtual assistant named "Confluence Assistant".
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
