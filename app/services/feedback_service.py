import time

from langchain_core.prompts import ChatPromptTemplate

from app.logger import get_logger
from app.services.llm import routed_chat_llm

logger = get_logger("wespeak.feedback")

SYSTEM_PROMPT = """You are an English reading comprehension coach for language learners.
You will receive:
1. A book passage (first user message)
2. A student's spoken summary of that passage (last user message)

Your job is to evaluate the student's summary and provide constructive feedback.
Guidelines:
- Give feedback in 3-4 sentences
- Focus on whether the student understood the overall message and main theme of the passage
- Evaluate how well the student captured the cause-and-effect relationships and the flow of ideas
- Be encouraging and specific
- Respond in English only"""

BOOK_PROMPT_TEMPLATE = ChatPromptTemplate([
    ("system", SYSTEM_PROMPT),
    ("human", """Book passage:
The Amazon rainforest, often called the 'lungs of the Earth', produces about 20% of the world's oxygen. It covers over 5.5 million square kilometers across nine countries, with Brazil containing the largest portion. Deforestation due to agriculture and logging has destroyed nearly 20% of the original forest over the past 50 years, threatening thousands of species.

Student's summary:
The Amazon is a very big forest. It makes oxygen. Many animals live there but some people cut trees so it is dangerous for animals."""),
    ("ai", "Great job grasping the core message — you understood that the Amazon plays a vital role in producing oxygen and that human activity is putting it at risk. Your summary shows a solid understanding of the cause-and-effect relationship between deforestation and the threat to wildlife. To make your summary even stronger, try to connect these ideas more explicitly, for example by explaining why deforestation happens and what that means for the forest's role on Earth. Keep it up — you're capturing the big picture well!"),
    ("human", """Book passage:
The Industrial Revolution, which began in Britain in the late 18th century, transformed societies from agrarian economies to manufacturing-based ones. Steam-powered machines replaced manual labor, enabling mass production in factories. While this created economic growth and urbanization, it also led to poor working conditions, child labor, and significant environmental pollution.

Student's summary:
I think it was about some machines and people working. There were factories. I'm not sure what the main point was."""),
    ("ai", "It seems like this passage was a bit tricky — don't worry, let me help you get the big picture. The passage is about how the invention of steam-powered machines in the 1700s completely changed the way people lived and worked: instead of farming, people moved to cities and worked in factories, which made economies grow but also created serious problems like harsh working conditions and pollution. Next time, try to ask yourself 'why did this happen and what changed because of it?' — that will help you find the main idea. You're on the right track by picking up on the factories and machines!"),
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
        async with routed_chat_llm() as llm:
            bookChain = BOOK_PROMPT_TEMPLATE | llm
            user_input = _extract_contents(messages)
            response = await bookChain.ainvoke(user_input)
        logger.info("feedback completed - %.1fms", (time.perf_counter() - start) * 1000)
        return response.content
    except Exception as e:
        logger.error("feedback failed - %.1fms error=%s", (time.perf_counter() - start) * 1000, e, exc_info=True)
        raise
