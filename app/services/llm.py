from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from openai import AsyncOpenAI

from app.config import settings


def get_chat_llm() -> BaseChatModel:
    if settings.provider == "claude":
        return ChatAnthropic(
            model=settings.claude_model,
            api_key=settings.anthropic_api_key,
            temperature=settings.temperature_chat,
        )
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature_chat,
    )


def get_image_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


def get_structured_llm() -> BaseChatModel:
    """JSON 구조화 출력용 LLM."""
    if settings.provider == "claude":
        return ChatAnthropic(
            model=settings.claude_model,
            api_key=settings.anthropic_api_key,
            temperature=settings.temperature_structured,
        )
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature_structured,
        format="json",
    )

