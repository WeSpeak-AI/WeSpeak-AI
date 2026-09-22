import asyncio
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from app.config import settings
from app.logger import get_logger

logger = get_logger("AI.stt")

# 로컬 faster-whisper는 GPU 한 장에서 한 번에 1건만 처리한다. 사용 중이면 OpenAI STT로 넘긴다(llm.routed_chat_llm과 같은 방식).
LOCAL_MAX = int(os.getenv("STT_LOCAL_MAX_CONCURRENT", "1"))
OPENAI_MAX = int(os.getenv("STT_OPENAI_MAX_CONCURRENT", "50"))

_model = None
_executor = ThreadPoolExecutor(max_workers=LOCAL_MAX)
_local_active = 0
_openai_sem = asyncio.Semaphore(OPENAI_MAX)
_openai_client = None


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


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _audio_filename(audio_bytes: bytes) -> str:
    """OpenAI는 파일 이름 확장자로 음성 형식을 판단한다(m4a를 .wav 이름으로 보내면 400). 헤더로 확장자를 고른다."""
    head = audio_bytes[:12]
    if head[4:8] == b"ftyp":
        return "audio.m4a"
    if head[:4] == b"RIFF":
        return "audio.wav"
    if head[:4] == b"OggS":
        return "audio.ogg"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "audio.webm"
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio.mp3"
    return "audio.m4a"  # 앱 녹음 기본 형식


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
    async with _openai_sem:
        start = time.perf_counter()
        try:
            response = await _get_openai_client().audio.transcriptions.create(
                model=settings.openai_stt_model,
                file=(_audio_filename(audio_bytes), audio_bytes),
                language=settings.whisper_language,
            )
            text = response.text.strip()
        except Exception as e:
            logger.error("openai transcribe failed - elapsed=%.1fms error=%s", (time.perf_counter() - start) * 1000, e, exc_info=True)
            raise

    logger.info("openai transcribe done - model=%s elapsed=%.1fms chars=%d",
                settings.openai_stt_model, (time.perf_counter() - start) * 1000, len(text))
    return text


async def transcribe(audio_bytes: bytes) -> str:
    """로컬 whisper 슬롯에 여유가 있으면 로컬, 사용 중이면 OpenAI STT. OpenAI 키가 없으면 로컬 대기열에서 기다린다."""
    global _local_active
    if settings.stt_provider == "openai":
        return await _transcribe_openai(audio_bytes)
    if _local_active >= LOCAL_MAX and settings.openai_api_key:
        logger.info("local stt busy (active=%d) - routing to openai", _local_active)
        return await _transcribe_openai(audio_bytes)

    _local_active += 1
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, _transcribe_local, audio_bytes)
    finally:
        _local_active -= 1
