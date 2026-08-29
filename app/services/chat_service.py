import time
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.logger import get_logger
from app.services.llm import routed_chat_llm

logger = get_logger("wespeak.chat")

SYSTEM_PROMPT = """You are a friendly English conversation partner for Korean language learners.
Your role is to help users practice natural English conversation.
Guidelines:
- Keep responses concise and natural (2-4 sentences)
- If the user makes grammar mistakes, subtly use the correct form in your reply without explicitly pointing it out
- Be encouraging and engaging
- Always respond in English only
- Match the conversational tone of the user"""

CONVERSATION_PROMPT_TEMPLATE = ChatPromptTemplate([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
])


def _convert_history(history: list[dict]) -> list:
    messages = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


async def chat_stream(history: list[dict]) -> AsyncIterator[str]:
    """토큰 단위 스트리밍. gRPC ChatService(spec 004)에서 사용."""
    logger.info("chat request (stream) - history_length=%d", len(history))
    start = time.perf_counter()
    try:
        async with routed_chat_llm() as llm:
            chatChain = CONVERSATION_PROMPT_TEMPLATE | llm
            messages = _convert_history(history)
            async for chunk in chatChain.astream({"chat_history": messages}):
                if chunk.content:
                    yield chunk.content
        logger.info("chat completed (stream) - %.1fms", (time.perf_counter() - start) * 1000)
    except Exception as e:
        logger.error("chat failed (stream) - %.1fms error=%s", (time.perf_counter() - start) * 1000, e, exc_info=True)
        raise


async def chat(history: list[dict]) -> str:
    """기존 REST(/chat)용 — chat_stream()의 토큰을 모아 완성된 문자열로 반환한다.
    REST 응답 동작은 이전(ainvoke 기반)과 동일하게 유지된다."""
    return "".join([token async for token in chat_stream(history)])
