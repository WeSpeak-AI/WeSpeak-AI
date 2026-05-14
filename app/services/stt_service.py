import asyncio
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from app.config import settings
from app.logger import get_logger

logger = get_logger("AI.stt")

_model = None
_executor = ThreadPoolExecutor(max_workers=1)


def get_model():
    global _model
    if settings.stt_provider != "local":
        return None
    if _model is None:
        from faster_whisper import WhisperModel
        logger.info("loading whisper model=%s device=%s compute_type=%s",
                    settings.whisper_model, settings.whisper_device, settings.whisper_compute_type)
        _model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        logger.info("whisper model loaded")
    return _model


def _transcribe_local(audio_bytes: bytes) -> str:
    start = time.perf_counter()
    model = get_model()

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            language=settings.whisper_language,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as e:
        logger.error("transcribe failed - elapsed=%.1fms error=%s", (time.perf_counter() - start) * 1000, e, exc_info=True)
        raise
    finally:
        os.unlink(tmp_path)

    logger.info("transcribe done - lang=%s duration=%.1fs elapsed=%.1fms chars=%d",
                info.language, info.duration, (time.perf_counter() - start) * 1000, len(text))
    return text


async def _transcribe_openai(audio_bytes: bytes) -> str:
    import openai
    start = time.perf_counter()
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as f:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=settings.whisper_language,
            )
        text = response.text.strip()
    except Exception as e:
        logger.error("openai transcribe failed - elapsed=%.1fms error=%s", (time.perf_counter() - start) * 1000, e, exc_info=True)
        raise
    finally:
        os.unlink(tmp_path)

    logger.info("openai transcribe done - elapsed=%.1fms chars=%d", (time.perf_counter() - start) * 1000, len(text))
    return text


async def transcribe(audio_bytes: bytes) -> str:
    if settings.stt_provider == "openai":
        return await _transcribe_openai(audio_bytes)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _transcribe_local, audio_bytes)
