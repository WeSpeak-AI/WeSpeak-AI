import json
import time

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.logger import get_logger
from app.services.llm import get_structured_llm

logger = get_logger("wespeak.search")

SYSTEM_PROMPT = """You are an English dictionary. Given a word or phrase, return ONLY a JSON object with these exact fields:
{{
  "term": "<the word or phrase as given>",
  "meaning": "<clear concise definition in Korean, e.g  수정하다>",
  "phonetic": "<IPA phonetic transcription, e.g. /ˈwɜːrd/>",
  "example": "<one natural example sentence using the word>"
}}
Respond with ONLY the JSON object. No explanation, no markdown, no extra text."""

SEARCH_PROMPT_TEMPLATE = ChatPromptTemplate([
    ("system", SYSTEM_PROMPT),
    ("human", "Define this word: {query}")
])


class WordInfo(BaseModel):
    term: str
    meaning: str
    phonetic: str
    example: str


async def search_word(query: str) -> WordInfo:
    logger.info("search request - query=%s", query)
    start = time.perf_counter()
    try:
        llm = get_structured_llm()
        structured_llm = llm.with_structured_output(WordInfo)
        searchChain = SEARCH_PROMPT_TEMPLATE | structured_llm
        result = await searchChain.ainvoke({"query": query})
        logger.info("search completed - query=%s %.1fms", query, (time.perf_counter() - start) * 1000)
        if isinstance(result, WordInfo):
            return result
        return WordInfo(**result)
    except Exception as e:
        logger.warning("search structured output failed, retrying with raw LLM - query=%s error=%s", query, e)
        try:
            raw_llm = get_structured_llm()
            response = await (SEARCH_PROMPT_TEMPLATE | raw_llm).ainvoke({"query": query})
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            logger.info("search fallback completed - query=%s %.1fms", query, (time.perf_counter() - start) * 1000)
            return WordInfo(**data)
        except Exception as e2:
            logger.error("search failed - query=%s %.1fms error=%s", query, (time.perf_counter() - start) * 1000, e2, exc_info=True)
            raise
