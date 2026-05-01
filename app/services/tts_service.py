import edge_tts
import io
from app.logger import get_logger

logger = get_logger("AI.tts")

async def text_to_speech(text: str) -> bytes:
    """텍스트를 음성 바이너리로 변환"""
    # 한국어 비서 느낌을 원하시면 ko-KR-SunHiNeural,
    # 영어 학습용이므로 영어 목소리 추천 (예: en-US-EmmaNeural)
    voice = "en-US-EmmaNeural"
    communicate = edge_tts.Communicate(text, voice)

    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    return audio_data