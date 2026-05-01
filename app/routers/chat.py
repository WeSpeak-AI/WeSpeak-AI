import base64
import json

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from app.logger import logger
from app.services import stt_service, chat_service, tts_service

router = APIRouter()


class ChatResponse(BaseModel):
    user_text: str
    ai_text: str
    audio_data: str


@router.post("/chat", response_model=ChatResponse)
async def handle_chat(file: UploadFile = File(...), history: str = Form(...)) -> ChatResponse:
    try:
        history_list = json.loads(history)

        requested_bytes = await file.read()
        user_text = await stt_service.transcribe(requested_bytes)

        history_list.append({"role": "user", "content": user_text})
        ai_response = await chat_service.chat(history_list)

        response_bytes = await tts_service.text_to_speech(ai_response)
        encoded_audio = base64.b64encode(response_bytes).decode('utf-8')

        return ChatResponse(
            user_text=user_text,
            ai_text=ai_response,
            audio_data=encoded_audio
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid history JSON format")
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
