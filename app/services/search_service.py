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
  "phonetic": "<IPA phonetic transcription of the English term, e.g. /ˈmɑːdɪfaɪ/>",
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
    ("ai", '{{"term": "modify", "meaning": "수정하다, 변경하다", "phonetic": "/ˈmɑːdɪfaɪ/", "example": "Please modify your report before submitting it."}}'),
    ("human", "Define this word: ambiguous"),
    ("ai", '{{"term": "ambiguous", "meaning": "애매한, 모호한", "phonetic": "/æmˈbɪɡjuəs/", "example": "The instructions were ambiguous, so we asked for clarification."}}'),
    ("human", "Define this word: resilient"),
    ("ai", '{{"term": "resilient", "meaning": "회복력이 있는, 탄력적인", "phonetic": "/rɪˈzɪliənt/", "example": "She is resilient enough to bounce back from any setback."}}'),
    ("human", "Define this word: diligent"),
    ("ai", '{{"term": "diligent", "meaning": "부지런한, 성실한", "phonetic": "/ˈdɪlɪdʒənt/", "example": "She is a diligent student who always finishes her homework on time."}}'),
    ("human", "Define this word: negotiate"),
    ("ai", '{{"term": "negotiate", "meaning": "협상하다, 교섭하다", "phonetic": "/nɪˈɡoʊʃieɪt/", "example": "The two companies agreed to negotiate the terms of the contract."}}'),
    ("human", "Define this word: persist"),
    ("ai", '{{"term": "persist", "meaning": "지속하다, 끈기 있게 계속하다", "phonetic": "/pərˈsɪst/", "example": "If you persist in your efforts, you will eventually succeed."}}'),
    ("human", "Define this word: 포기하다"),
    ("ai", '{{"term": "give up", "meaning": "포기하다, 그만두다", "phonetic": "/ɡɪv ʌp/", "example": "Don\'t give up even when things get difficult."}}'),
    ("human", "Define this word: 긴장하다"),
    ("ai", '{{"term": "nervous", "meaning": "긴장한, 불안한", "phonetic": "/ˈnɜːrvəs/", "example": "He felt nervous before his first job interview."}}'),
    ("human", "Define this word: 겸손한"),
    ("ai", '{{"term": "humble", "meaning": "겸손한, 자만하지 않는", "phonetic": "/ˈhʌmbəl/", "example": "Despite his success, he remained humble and kind."}}'),
    ("human", "Define this word: 설명하다"),
    ("ai", '{{"term": "explain", "meaning": "설명하다, 해설하다", "phonetic": "/ɪkˈspleɪn/", "example": "Can you explain how this machine works?"}}'),
    ("human", "Define this word: 해결하다"),
    ("ai", '{{"term": "resolve", "meaning": "해결하다, 결심하다", "phonetic": "/rɪˈzɑːlv/", "example": "They worked together to resolve the conflict peacefully."}}'),
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
