import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.services.llm import get_structured_llm

SYSTEM_PROMPT = """You are an expert English grammar and writing teacher.
Analyze the given essay or paragraph and return ONLY a JSON object with this exact structure:
{
  "overallScore": <integer 0-100 reflecting grammar and writing quality>,
  "corrections": [
    {
      "error": "<exact wrong text from the essay>",
      "correct": "<corrected version>",
      "explanation": "<brief explanation of why it is wrong>"
    }
  ],
  "suggestions": [
    "<general writing improvement tip>"
  ]
}
Rules:
- overallScore: 90-100 excellent, 70-89 good, 50-69 fair, below 50 needs work
- corrections: list ONLY actual grammar/spelling errors found in the text. Empty array if no errors.
- suggestions: 2-3 general style and writing tips
- Respond with ONLY the JSON object. No explanation, no markdown, no extra text."""


class CorrectionItem(BaseModel):
    error: str
    correct: str
    explanation: str


class CorrectionResult(BaseModel):
    overallScore: int
    corrections: list[CorrectionItem]
    suggestions: list[str]


async def correct_essay(content: str) -> str:
    """
    에세이를 교정하고 CorrectionResult JSON 문자열을 반환한다.
    백엔드는 {"result": "<json_string>"} 형태로 수신하므로
    이 함수는 순수 JSON 문자열을 반환한다.
    """
    llm = get_structured_llm()
    structured_llm = llm.with_structured_output(CorrectionResult)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Please correct the following essay:\n\n{content}"),
    ]

    try:
        result = await structured_llm.ainvoke(messages)
        if isinstance(result, CorrectionResult):
            return result.model_dump_json()
        return CorrectionResult(**result).model_dump_json()
    except Exception:
        # 파싱 실패 시 raw LLM으로 재시도
        raw_llm = get_structured_llm()
        response = await raw_llm.ainvoke(messages)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return CorrectionResult(**data).model_dump_json()
