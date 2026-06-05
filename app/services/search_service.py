import json
import time

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.logger import get_logger
from app.services.llm import routed_structured_llm

logger = get_logger("wespeak.search")

SYSTEM_PROMPT = """You are an English-Korean dictionary. Given a word or phrase, return ONLY a JSON object with these exact fields:
{{
  "term": "<always the English word or phrase, e.g. modify>",
  "meaning": "<clear concise Korean definition, e.g. 수정하다>",
  "phonetic": "<IPA phonetic transcription of the English term, e.g. /ˈmɒdɪfaɪ/>",
  "example": "<one natural English example sentence using the word>"
}}

Rules:
- If the input is English: define it and provide the Korean meaning
- If the input is Korean (한국어): find the single best matching English word or phrase, then define it
- term MUST always be written in English (Latin alphabet only — never Chinese, Japanese, or other scripts)
- meaning MUST always be written in Korean (한국어)
- Do NOT use Chinese characters (漢字) anywhere in the response
Respond with ONLY the JSON object. No explanation, no markdown, no extra text."""

SEARCH_PROMPT_TEMPLATE = ChatPromptTemplate([
    ("system", SYSTEM_PROMPT),
    ("human", "Define this word: eloquent"),
    ("ai", '{{"term": "eloquent", "meaning": "말을 유창하게 잘 하는, 표현이 풍부한", "phonetic": "/ˈeləkwənt/", "example": "She gave an eloquent speech at the conference."}}'),
    ("human", "Define this word: 수정하다"),
    ("ai", '{{"term": "modify", "meaning": "수정하다, 변경하다", "phonetic": "/ˈmɒdɪfaɪ/", "example": "Please modify your report before submitting it."}}'),
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
        async with routed_structured_llm() as llm:
            try:
                result = await (SEARCH_PROMPT_TEMPLATE | llm.with_structured_output(WordInfo)).ainvoke({"query": query})
                logger.info("search completed - query=%s %.1fms", query, (time.perf_counter() - start) * 1000)
                if isinstance(result, WordInfo):
                    return result
                return WordInfo(**result)
            except Exception as e:
                logger.warning("search structured output failed, retrying with raw LLM - query=%s error=%s", query, e)
                response = await (SEARCH_PROMPT_TEMPLATE | llm).ainvoke({"query": query})
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
