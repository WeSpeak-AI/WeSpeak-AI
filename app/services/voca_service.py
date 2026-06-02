import asyncio
import os
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageEmbeddings
from pydantic import BaseModel, Field

from app.logger import get_logger
from app.services.llm import get_structured_llm, get_image_client

logger = get_logger("wespeak.voca")

PINECONE_INDEX_NAME = os.getenv("PINECONE_VOCA_INDEX", "voca-index")
EMBEDDING_MODEL = os.getenv("UPSTAGE_EMBEDDING_MODEL", "solar-embedding-1-large")
MAX_CONCURRENT_LLM = int(os.getenv("VOCA_MAX_CONCURRENT_LLM", "10"))
MAX_CONCURRENT_RAG = int(os.getenv("VOCA_MAX_CONCURRENT_RAG", "20"))

# Step 1: title/category/description → numberOfDays개의 dayTopic 확정
TOPIC_SYSTEM_PROMPT = """You are an English vocabulary curriculum designer.
Generate exactly {numberOfDays} specific, distinct vocabulary subtopics for a vocabulary book.
Return ONLY valid JSON matching the schema. No markdown, no explanations."""

TOPIC_PROMPT_TEMPLATE = ChatPromptTemplate([
    ("system", TOPIC_SYSTEM_PROMPT),
    ("human", """Generate {numberOfDays} day topics for:
Title: {title}
Category: {category}
Description: {description}

Rules:
- Each topic must be a specific, searchable phrase (e.g. "Information & Resource Management")
- Topics must be varied and cover the full scope of the vocabulary book
- No duplicate topics""")
])

# Step 3: 각 day마다 topic + retrieved docs → words 생성
DAY_SYSTEM_PROMPT = """You are an expert English vocabulary teacher.
Use the provided [Context] to generate vocabulary words for a specific day topic.
Return ONLY valid JSON matching the exact schema. No markdown, no explanations.

For each word provide:
- term: the English word or phrase
- meaning: clear concise definition in Korean (한국어로 작성)
- phonetic: IPA transcription (e.g. /ˈwɜːrd/)
- example: one natural example sentence using the word (max 100 characters)
- imageUrl: always empty string ""
"""

DAY_PROMPT_TEMPLATE = ChatPromptTemplate([
    ("system", DAY_SYSTEM_PROMPT),
    ("human", """[Context / Reference Materials]:
{context}

[Request]:
Vocabulary Book: {title} ({category})
Day {day} Topic: {dayTopic}

Generate 5-8 vocabulary words that are most relevant to the day topic, using the context as reference.""")
])


class TopicListResult(BaseModel):
    topics: list[str] = Field(..., description="List of day topics, one per day")


class WordResult(BaseModel):
    term: str = Field(..., description="The English word or phrase")
    meaning: str = Field(..., description="Clear concise definition in English")
    phonetic: str = Field(..., description="IPA phonetic transcription, e.g. /ˈwɜːrd/")
    example: str = Field(..., description="One natural example sentence using the word")
    imageUrl: str = Field(default="", description="Always leave as empty string")


class DayResult(BaseModel):
    day: int = Field(..., description="The day number (1, 2, 3...)")
    dayTopic: str = Field(..., description="A focused subtopic related to the main theme")
    words: list[WordResult] = Field(..., description="5-8 vocabulary words for that day's topic")


class VocaBookResult(BaseModel):
    days: list[DayResult] = Field(..., description="List of daily vocabulary content")


_voca_store = None
_voca_embedding = None


def get_voca_store() -> tuple[PineconeVectorStore, UpstageEmbeddings]:
    global _voca_store, _voca_embedding
    if _voca_store is None:
        _voca_embedding = UpstageEmbeddings(model=EMBEDDING_MODEL)
        _voca_store = PineconeVectorStore(
            embedding=_voca_embedding,
            index_name=PINECONE_INDEX_NAME,
        )
    return _voca_store, _voca_embedding


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


async def _generate_day(
    llm,
    sem: asyncio.Semaphore,
    day: int,
    day_topic: str,
    title: str,
    category: str,
    docs,
) -> DayResult:
    context = format_docs(docs)
    chain = DAY_PROMPT_TEMPLATE | llm.with_structured_output(DayResult)
    async with sem:
        result = await chain.ainvoke({
            "day": day,
            "dayTopic": day_topic,
            "title": title,
            "category": category,
            "context": context,
        })
    if isinstance(result, DayResult):
        result.day = day
        result.dayTopic = day_topic
        return result
    day_result = DayResult(**result)
    day_result.day = day
    day_result.dayTopic = day_topic
    return day_result


async def _query_by_vector(sem: asyncio.Semaphore, store: PineconeVectorStore, vector: list):
    async with sem:
        return await asyncio.to_thread(store.similarity_search_by_vector, vector, k=5)


async def generate_voca(title: str, category: str, description: str, numberOfDays: int) -> VocaBookResult:
    logger.info("voca request - title=%s numberOfDays=%d", title, numberOfDays)
    start = time.perf_counter()
    try:
        llm = get_structured_llm()

        # Step 1: numberOfDays개의 dayTopic 확정
        topic_chain = TOPIC_PROMPT_TEMPLATE | llm.with_structured_output(TopicListResult)
        topic_result = await topic_chain.ainvoke({
            "title": title,
            "category": category,
            "description": description,
            "numberOfDays": numberOfDays,
        })
        topics = topic_result.topics[:numberOfDays]
        logger.info("topics generated - title=%s count=%d %.1fms",
                    title, len(topics), (time.perf_counter() - start) * 1000)

        # Step 2: topic 전체를 한 번에 batch embedding → Pinecone 병렬 쿼리
        store, embedding = get_voca_store()
        topic_vectors = await asyncio.to_thread(embedding.embed_documents, topics)
        rag_sem = asyncio.Semaphore(MAX_CONCURRENT_RAG)
        doc_lists = await asyncio.gather(*[_query_by_vector(rag_sem, store, vec) for vec in topic_vectors])
        logger.info("rag done - title=%s %.1fms",
                    title, (time.perf_counter() - start) * 1000)

        # Step 3: 각 day 병렬 생성 (최대 MAX_CONCURRENT_LLM 동시)
        llm_sem = asyncio.Semaphore(MAX_CONCURRENT_LLM)
        day_results = await asyncio.gather(*[
            _generate_day(llm, llm_sem, day + 1, topic, title, category, docs)
            for day, (topic, docs) in enumerate(zip(topics, doc_lists))
        ])

        logger.info("voca completed - title=%s %.1fms", title, (time.perf_counter() - start) * 1000)
        return VocaBookResult(days=list(day_results))
    except Exception as e:
        logger.error("voca failed - title=%s %.1fms error=%s",
                     title, (time.perf_counter() - start) * 1000, e, exc_info=True)
        raise



IMAGE_CONCURRENT = int(os.getenv("IMAGE_MAX_CONCURRENT", "3"))

IMAGE_PROMPT_TEMPLATE = (
    "A simple, clear educational illustration representing the English word '{term}'. "
    "Clean minimalist style, white background, no text, suitable for a vocabulary learning app."
)


class ImageResult(BaseModel):
    wordId: int
    imageData: str


class VocaWordImageResult(BaseModel):
    results: list[ImageResult]


async def generate_word_images(words: list) -> VocaWordImageResult:
    client = get_image_client()
    sem = asyncio.Semaphore(IMAGE_CONCURRENT)
    start = time.perf_counter()
    logger.info("image request - count=%d", len(words))

    async def _generate_one(word_item) -> ImageResult:
        async with sem:
            try:
                response = await client.images.generate(
                    model="dall-e-3",
                    prompt=IMAGE_PROMPT_TEMPLATE.format(term=word_item.term),
                    size="256x256",
                    quality="standard",
                    n=1,
                    response_format="b64_json",
                )
                return ImageResult(wordId=word_item.wordId, imageData=response.data[0].b64_json)
            except Exception as e:
                logger.error("image failed - wordId=%d term=%s error=%s", word_item.wordId, word_item.term, e)
                return ImageResult(wordId=word_item.wordId, imageData="")

    results = await asyncio.gather(*[_generate_one(w) for w in words])
    success = sum(1 for r in results if r.imageData)
    logger.info("image done - %d/%d succeeded %.1fms", success, len(results), (time.perf_counter() - start) * 1000)
    return VocaWordImageResult(results=list(results))