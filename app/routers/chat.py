from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.services import chat_service

router = APIRouter()


class ChatRequest(BaseModel):
    messages: list[dict]


@router.post("/chat", response_class=PlainTextResponse)
async def chat(request: ChatRequest) -> str:
    try:
        return await chat_service.chat(request.messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
