import pytest

from app.grpc.chat_servicer import ChatServicer
from app.grpc.generated import wespeak_ai_pb2
from app.services import chat_service, stt_service


async def _async_iter(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_chat_emits_user_text_then_deltas_in_order(monkeypatch):
    async def fake_transcribe(audio_bytes: bytes) -> str:
        assert audio_bytes == b"\x01\x02\x03"
        return "hello world"

    async def fake_chat_stream(history):
        assert history[-1] == {"role": "user", "content": "hello world"}
        for token in ["Hi ", "there", "!"]:
            yield token

    monkeypatch.setattr(stt_service, "transcribe", fake_transcribe)
    monkeypatch.setattr(chat_service, "chat_stream", fake_chat_stream)

    upload = [
        wespeak_ai_pb2.ChatUploadChunk(metadata=wespeak_ai_pb2.ChatMetadata(history_json="[]")),
        wespeak_ai_pb2.ChatUploadChunk(audio_chunk=b"\x01\x02\x03"),
    ]

    servicer = ChatServicer()
    chunks = [chunk async for chunk in servicer.Chat(_async_iter(upload), context=None)]

    assert chunks[0].WhichOneof("payload") == "user_text_final"
    assert chunks[0].user_text_final == "hello world"

    deltas = [c.ai_text_delta for c in chunks[1:]]
    assert deltas == ["Hi ", "there", "!"]
    assert "".join(deltas) == "Hi there!"  # spec FR-006 동일성(나뉘어 온 조각을 이어붙인 결과)


@pytest.mark.asyncio
async def test_chat_emits_stream_error_on_exception(monkeypatch):
    async def failing_transcribe(audio_bytes: bytes) -> str:
        raise RuntimeError("stt boom")

    monkeypatch.setattr(stt_service, "transcribe", failing_transcribe)

    upload = [wespeak_ai_pb2.ChatUploadChunk(metadata=wespeak_ai_pb2.ChatMetadata(history_json="[]"))]

    servicer = ChatServicer()
    chunks = [chunk async for chunk in servicer.Chat(_async_iter(upload), context=None)]

    assert len(chunks) == 1
    assert chunks[0].WhichOneof("payload") == "stream_error"
    assert "stt boom" in chunks[0].stream_error.message
