from langchain_core.messages import HumanMessage, SystemMessage

from app.services.llm import get_chat_llm

SYSTEM_PROMPT = """You are an English reading comprehension coach for language learners.
You will receive:
1. A book passage (first user message)
2. A student's spoken summary of that passage (last user message)

Your job is to evaluate the student's summary and provide constructive feedback.
Guidelines:
- Give feedback in 3-4 sentences
- Mention what they got right (accuracy of key points)
- Point out any important missing details
- Comment on language quality if notable errors exist
- Be encouraging and specific
- Respond in English only"""


def _build_messages(raw_messages: list[dict]) -> list:
    """
    백엔드가 보내는 메시지 형식:
    [
      {"role": "user",      "content": "<book passage>"},
      {"role": "assistant", "content": "Give me a summary."},
      {"role": "user",      "content": "<user's summary>"}
    ]
    """
    if len(raw_messages) < 3:
        book_content = raw_messages[0]["content"] if raw_messages else ""
        user_summary = raw_messages[-1]["content"] if raw_messages else ""
    else:
        book_content = raw_messages[0]["content"]
        user_summary = raw_messages[2]["content"]

    prompt = (
        f"Book passage:\n{book_content}\n\n"
        f"Student's summary:\n{user_summary}\n\n"
        "Please provide feedback on the student's summary."
    )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]


async def get_feedback(messages: list[dict]) -> str:
    llm = get_chat_llm()
    langchain_messages = _build_messages(messages)
    response = await llm.ainvoke(langchain_messages)
    return response.content
