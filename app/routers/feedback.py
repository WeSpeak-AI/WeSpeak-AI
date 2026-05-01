import base64

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from app.services import stt_service, feedback_service, tts_service

router = APIRouter()


class FeedbackResponse(BaseModel):
    user_text: str
    feedback_text: str
    audio_data: str


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(file: UploadFile = File(...), book_content: str = Form(...)) -> FeedbackResponse:
    try:
        audio_bytes = await file.read()
        user_text = await stt_service.transcribe(audio_bytes)

        messages = [
            {"role": "user", "content": book_content},
            {"role": "assistant", "content": "Give me a summary."},
            {"role": "user", "content": user_text},
        ]
        feedback_text = await feedback_service.get_feedback(messages)

        audio_bytes = await tts_service.text_to_speech(feedback_text)
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

        return FeedbackResponse(
            user_text=user_text,
            feedback_text=feedback_text,
            audio_data=encoded_audio,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
