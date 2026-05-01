import time

from langchain_core.prompts import ChatPromptTemplate

from app.logger import get_logger
from app.services.llm import get_chat_llm

logger = get_logger("wespeak.topic")

SYSTEM_PROMPT = """You are an expert English conversation curriculum designer for a language learning app.
Your task is to write the 'content' field of a conversation topic card.

You will receive:
- Title: the name of the conversation topic
- Difficulty: the learner level (beginner / intermediate / advanced)
- Content: a rough draft or notes written by the admin

Your job is to rewrite the content into a polished, learner-friendly topic description that:
1. Clearly introduces what the conversation is about
2. Provides 2–3 specific discussion questions or prompts that naturally fit the title
3. Includes 3–5 key vocabulary words or expressions relevant to the topic (with brief definitions in parentheses)
4. Adjusts the complexity of language and questions to match the difficulty level:
   - beginner: simple sentences, everyday vocabulary, yes/no or short-answer questions
   - intermediate: compound sentences, idiomatic expressions, opinion-based questions
   - advanced: nuanced vocabulary, abstract or hypothetical questions, debate-style prompts
5. Maintains an encouraging, conversational tone throughout

Output format (plain text, no markdown headers or bullet symbols):
- Start with one or two sentences introducing the topic
- Then list the discussion questions
- Then list the key vocabulary

Respond with ONLY the final content text. Do not include labels like "Discussion Questions:" or "Vocabulary:".
Do not explain your reasoning. Do not add anything outside the content itself."""

TOPIC_PROMPT_TEMPLATE = ChatPromptTemplate([
    ("system", SYSTEM_PROMPT),
    ("human", """Title: Ordering at a Restaurant
Difficulty: beginner
Content (admin draft): Talking about food and ordering at a restaurant"""),
    ("ai", """Going to a restaurant is a great way to practice everyday English. In this topic, you will learn how to read a menu, ask questions, and place your order politely.

What is your favorite food? Can you describe it in a few words?
Do you prefer eating at home or at a restaurant? Why?
What do you usually say when a server asks, "Are you ready to order?"

menu — a list of food and drinks available at a restaurant
order — to ask for a specific food or drink
recommend — to suggest something you think is good
bill — the paper showing how much you need to pay
tip — extra money given to thank someone for good service"""),
    ("human", """Title: {title}
Difficulty: {difficulty}
Content (admin draft): {content}""")
])


async def get_topic(title: str, content: str, difficulty: str) -> str:
    logger.info("topic request - title=%s difficulty=%s", title, difficulty)
    start = time.perf_counter()
    try:
        llm = get_chat_llm()
        topicChain = TOPIC_PROMPT_TEMPLATE | llm
        response = await topicChain.ainvoke({"title": title, "content": content, "difficulty": difficulty})
        logger.info("topic completed - title=%s %.1fms", title, (time.perf_counter() - start) * 1000)
        return response.content
    except Exception as e:
        logger.error("topic failed - title=%s %.1fms error=%s", title, (time.perf_counter() - start) * 1000, e, exc_info=True)
        raise
