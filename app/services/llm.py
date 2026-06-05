import asyncio
import contextlib
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from openai import AsyncOpenAI

from app.config import settings

OLLAMA_MAX = int(os.getenv("OLLAMA_MAX_CONCURRENT", "3"))
CLAUDE_MAX = int(os.getenv("CLAUDE_MAX_CONCURRENT", "50"))

_ollama_active = 0
_claude_sem = asyncio.Semaphore(CLAUDE_MAX)

_ollama_chat = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    temperature=settings.temperature_chat,
)

_ollama_structured = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    temperature=settings.temperature_structured,
    format="json",
)

_claude_chat = ChatAnthropic(
    model=settings.claude_model,
    api_key=settings.anthropic_api_key,
    temperature=settings.temperature_chat,
)

_claude_structured = ChatAnthropic(
    model=settings.claude_model,
    api_key=settings.anthropic_api_key,
    temperature=settings.temperature_structured,
)

_image_client = AsyncOpenAI(api_key=settings.openai_api_key)


def get_image_client() -> AsyncOpenAI:
    return _image_client


def get_claude_structured_llm() -> BaseChatModel:
    """voca 생성 전용 - 항상 Claude."""
    return _claude_structured


@contextlib.asynccontextmanager
async def routed_chat_llm():
    """chat, feedback용 - Ollama 슬롯 여유 있으면 Ollama, 초과하면 Claude 대기."""
    global _ollama_active
    if settings.provider == "ollama" and _ollama_active < OLLAMA_MAX:
        _ollama_active += 1
        try:
            yield _ollama_chat
        finally:
            _ollama_active -= 1
    else:
        async with _claude_sem:
            yield _claude_chat


@contextlib.asynccontextmanager
async def routed_structured_llm():
    """search, correction용 - Ollama 슬롯 여유 있으면 Ollama, 초과하면 Claude 대기."""
    global _ollama_active
    if settings.provider == "ollama" and _ollama_active < OLLAMA_MAX:
        _ollama_active += 1
        try:
            yield _ollama_structured
        finally:
            _ollama_active -= 1
    else:
        async with _claude_sem:
            yield _claude_structured
