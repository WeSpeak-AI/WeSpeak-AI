from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.services.llm import get_chat_llm

SYSTEM_PROMPT = """You are a friendly English conversation partner for Korean language learners.
Your role is to help users practice natural English conversation.
Guidelines:
- Keep responses concise and natural (2-4 sentences)
- If the user makes grammar mistakes, subtly use the correct form in your reply without explicitly pointing it out
- Be encouraging and engaging
- Always respond in English only
- Match the conversational tone of the user"""


def _build_messages(history: list[dict]) -> list:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


async def chat(history: list[dict]) -> str:
    llm = get_chat_llm()
    messages = _build_messages(history)
    response = await llm.ainvoke(messages)
    return response.content
