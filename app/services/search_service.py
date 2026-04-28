import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.services.llm import get_structured_llm

SYSTEM_PROMPT = """You are an English dictionary. Given a word or phrase, return ONLY a JSON object with these exact fields:
{
  "term": "<the word or phrase as given>",
  "meaning": "<clear concise definition in English>",
  "phonetic": "<IPA phonetic transcription, e.g. /ˈwɜːrd/>",
  "example": "<one natural example sentence using the word>"
}
Respond with ONLY the JSON object. No explanation, no markdown, no extra text."""


class WordInfo(BaseModel):
    term: str
    meaning: str
    phonetic: str
    example: str


async def search_word(query: str) -> WordInfo:
    llm = get_structured_llm()
    structured_llm = llm.with_structured_output(WordInfo)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Define this word: {query}"),
    ]

    try:
        result = await structured_llm.ainvoke(messages)
        if isinstance(result, WordInfo):
            return result
        # with_structured_output이 dict를 반환하는 경우 대비
        return WordInfo(**result)
    except Exception:
        # 파싱 실패 시 raw LLM으로 재시도
        raw_llm = get_structured_llm()
        response = await raw_llm.ainvoke(messages)
        raw = response.content.strip()
        # 마크다운 코드블록 제거
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return WordInfo(**data)
