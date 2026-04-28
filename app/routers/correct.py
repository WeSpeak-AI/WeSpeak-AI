from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import correction_service

router = APIRouter()


class CorrectionRequest(BaseModel):
    content: str


class CorrectionResponse(BaseModel):
    result: str  # CorrectionResult JSON 문자열 (프론트에서 JSON.parse)


@router.post("/correct", response_model=CorrectionResponse)
async def correct(request: CorrectionRequest) -> CorrectionResponse:
    try:
        result_json = await correction_service.correct_essay(request.content)
        return CorrectionResponse(result=result_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
