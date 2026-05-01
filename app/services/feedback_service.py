import time

from langchain_core.prompts import ChatPromptTemplate

from app.logger import get_logger
from app.services.llm import get_chat_llm

logger = get_logger("wespeak.feedback")

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

BOOK_PROMPT_TEMPLATE = ChatPromptTemplate([
    ("system", SYSTEM_PROMPT),
    ("human", """Book passage:
The Amazon rainforest, often called the 'lungs of the Earth', produces about 20% of the world's oxygen. It covers over 5.5 million square kilometers across nine countries, with Brazil containing the largest portion. Deforestation due to agriculture and logging has destroyed nearly 20% of the original forest over the past 50 years, threatening thousands of species.

Student's summary:
The Amazon is a very big forest. It makes oxygen. Many animals live there but some people cut trees so it is dangerous for animals."""),
    ("ai", "You got the main ideas right — the Amazon produces oxygen and deforestation threatens wildlife. However, you missed some key details: the forest spans nine countries, covers 5.5 million square kilometers, and about 20% has already been lost. Try to include specific figures when the passage gives them. Your sentences are clear, but using connectors like 'as a result' or 'however' would make your summary flow more naturally."),
    ("human", """Book passage:
{book_content}

Student's summary:
{user_summary}

Please provide feedback on the student's summary.""")
])


def _extract_contents(raw_messages: list[dict]) -> dict:
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

    return {"book_content": book_content,
            "user_summary": user_summary}


async def get_feedback(messages: list[dict]) -> str:
    logger.info("feedback request")
    start = time.perf_counter()
    try:
        llm = get_chat_llm()
        bookChain = BOOK_PROMPT_TEMPLATE | llm
        user_input = _extract_contents(messages)
        response = await bookChain.ainvoke(user_input)
        logger.info("feedback completed - %.1fms", (time.perf_counter() - start) * 1000)
        return response.content
    except Exception as e:
        logger.error("feedback failed - %.1fms error=%s", (time.perf_counter() - start) * 1000, e, exc_info=True)
        raise
