import os
from dataclasses import dataclass
from enum import Enum
from typing import Union, Tuple
from langchain_openai import ChatOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
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


class LLMBackend(str, Enum):
    LANGCHAIN = "langchain"
    PYDANTIC_AI = "pydantic-ai"


@dataclass
class LLMConfig:
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


class LLMFactory:
    def __init__(self, config: LLMConfig):
        self.config = config

    def create(self, backend: LLMBackend) -> Union[ChatOpenAI, OpenAIModel]:
        if backend == LLMBackend.LANGCHAIN:
            return self._create_langchain()
        elif backend == LLMBackend.PYDANTIC_AI:
            return self._create_pydantic_ai()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _create_langchain(self) -> ChatOpenAI:
        return ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            openai_api_key=self.config.api_key,
            openai_api_base=self.config.api_base,  # Important pour OpenRouter
        )

    def _create_pydantic_ai(self) -> Tuple[OpenAIModel, ModelSettings]:
        model = OpenAIModel(
            model_name=self.config.model_name,
            provider=OpenAIProvider(base_url=self.config.api_base, api_key=self.config.api_key),
        )

        model_settings = ModelSettings(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return model, model_settings
