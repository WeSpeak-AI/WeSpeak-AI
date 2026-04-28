from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.services import feedback_service

router = APIRouter()


class FeedbackRequest(BaseModel):
    messages: list[dict]


@router.post("/feedback", response_class=PlainTextResponse)
async def feedback(request: FeedbackRequest) -> str:
    try:
        return await feedback_service.get_feedback(request.messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
