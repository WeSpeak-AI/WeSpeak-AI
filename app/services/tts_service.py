from typing import AsyncIterator

import edge_tts
from app.logger import get_logger

logger = get_logger("AI.tts")

# 한국어 비서 느낌을 원하시면 ko-KR-SunHiNeural,
# 영어 학습용이므로 영어 목소리 추천 (예: en-US-EmmaNeural)
VOICE = "en-US-EmmaNeural"


async def text_to_speech_stream(text: str) -> AsyncIterator[bytes]:
    """텍스트를 음성 청크로 스트리밍 변환. gRPC ChatService(spec 004)에서 사용."""
    communicate = edge_tts.Communicate(text, VOICE)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]


async def text_to_speech(text: str) -> bytes:
    """기존 REST(/chat)용 — text_to_speech_stream()의 청크를 모아 완성된 바이트열로 반환한다.
    REST 응답 동작은 이전과 동일하게 유지된다."""
    audio_data = b""
    async for chunk in text_to_speech_stream(text):
        audio_data += chunk
    return audio_data