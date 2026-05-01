from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import PlainTextResponse

from app.services import topic_service

router = APIRouter()


class TopicRequest(BaseModel):
    title: str
    content: str
    difficulty: str


@router.post("/topic", response_class=PlainTextResponse)
async def topic(request: TopicRequest) -> str:
    try:
        return await topic_service.get_topic(request.title, request.content, request.difficulty)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))