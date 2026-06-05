import time

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


async def chat(history: list[dict]) -> str:
    logger.info("chat request - history_length=%d", len(history))
    start = time.perf_counter()
    try:
        async with routed_chat_llm() as llm:
            chatChain = CONVERSATION_PROMPT_TEMPLATE | llm
            messages = _convert_history(history)
            response = await chatChain.ainvoke({"chat_history": messages})
        logger.info("chat completed - %.1fms", (time.perf_counter() - start) * 1000)
        return response.content
    except Exception as e:
        logger.error("chat failed - %.1fms error=%s", (time.perf_counter() - start) * 1000, e, exc_info=True)
        raise
