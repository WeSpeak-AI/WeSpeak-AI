import json
import time

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.logger import get_logger
from app.services.llm import routed_structured_llm

logger = get_logger("wespeak.correction")

SYSTEM_PROMPT = """You are an expert English grammar and writing teacher.
Analyze the given essay or paragraph and return ONLY a JSON object with this exact structure:
{{
  "overallScore": <integer 0-100 reflecting grammar and writing quality>,
  "corrections": [
    {{
      "error": "<exact wrong text from the essay>",
      "correct": "<corrected version>",
      "explanation": "<brief explanation of why it is wrong>"
    }}
  ],
  "suggestions": [
    "<general writing improvement tip>"
  ]
}}
Rules:
- overallScore: 90-100 excellent, 70-89 good, 50-69 fair, below 50 needs work
- corrections: list ONLY actual grammar/spelling errors found in the text. Empty array if no errors.
- suggestions: 2-3 general style and writing tips
- Respond with ONLY the JSON object. No explanation, no markdown, no extra text.

Example:
Input: "I goed to the store yesterday and buyed some apple. It was very cheaper than I think."
Output: {{"overallScore": 52, "corrections": [{{"error": "goed", "correct": "went", "explanation": "'go' is an irregular verb; the past tense is 'went', not 'goed'"}}, {{"error": "buyed", "correct": "bought", "explanation": "'buy' is an irregular verb; the past tense is 'bought'"}}, {{"error": "apple", "correct": "apples", "explanation": "countable nouns need a plural form when referring to more than one"}}, {{"error": "cheaper than I think", "correct": "cheaper than I thought", "explanation": "the main clause is in the past, so the subordinate clause should also be in the past tense"}}], "suggestions": ["Try to use a variety of sentence structures to make your writing more engaging.", "Double-check irregular verb forms before submitting.", "Adding transition words like 'also' or 'in addition' can improve the flow between sentences."]}}"""

WRITE_PROMPT_TEMPLATE = ChatPromptTemplate([
    ("system", SYSTEM_PROMPT),
    ("human", "Please correct the following essay:\n\n{content}")
])


class CorrectionItem(BaseModel):
    error: str
    correct: str
    explanation: str


class CorrectionResult(BaseModel):
    overallScore: int
    corrections: list[CorrectionItem]
    suggestions: list[str]


async def correct_essay(content: str) -> str:
    logger.info("correction request - content_length=%d", len(content))
    start = time.perf_counter()
    try:
        async with routed_structured_llm() as llm:
            try:
                result = await (WRITE_PROMPT_TEMPLATE | llm.with_structured_output(CorrectionResult)).ainvoke({"content": content})
                logger.info("correction completed - %.1fms", (time.perf_counter() - start) * 1000)
                if isinstance(result, CorrectionResult):
                    return result.model_dump_json()
                return CorrectionResult(**result).model_dump_json()
            except Exception as e:
                logger.warning("correction structured output failed, retrying with raw LLM - error=%s", e)
                response = await (WRITE_PROMPT_TEMPLATE | llm).ainvoke({"content": content})
                raw = response.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                data = json.loads(raw)
                logger.info("correction fallback completed - %.1fms", (time.perf_counter() - start) * 1000)
                return CorrectionResult(**data).model_dump_json()
    except Exception as e2:
        logger.error("correction failed - %.1fms error=%s", (time.perf_counter() - start) * 1000, e2, exc_info=True)
        raise
