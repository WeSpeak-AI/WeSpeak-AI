from langchain_ollama import ChatOllama
from app.config import settings


def get_chat_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature_chat,
    )


def get_structured_llm() -> ChatOllama:
    """JSON 구조화 출력용 LLM (format=json 강제)."""
    return ChatOllama(
        model=settings.model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature_structured,
        format="json",
    )
