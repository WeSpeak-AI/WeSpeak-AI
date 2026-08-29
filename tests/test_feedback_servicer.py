import pytest

from app.grpc.feedback_servicer import FeedbackServicer
from app.grpc.generated import wespeak_ai_pb2
from app.services import feedback_service, stt_service


async def _async_iter(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_feedback_emits_user_text_then_deltas_in_order(monkeypatch):
    async def fake_transcribe(audio_bytes: bytes) -> str:
        assert audio_bytes == b"\x01\x02\x03"
        return "my summary"

    async def fake_get_feedback_stream(messages):
        assert messages[0] == {"role": "user", "content": "book content"}
        assert messages[-1] == {"role": "user", "content": "my summary"}
        for token in ["Great ", "job", "!"]:
            yield token

    monkeypatch.setattr(stt_service, "transcribe", fake_transcribe)
    monkeypatch.setattr(feedback_service, "get_feedback_stream", fake_get_feedback_stream)

    upload = [
        wespeak_ai_pb2.FeedbackUploadChunk(metadata=wespeak_ai_pb2.FeedbackMetadata(book_content="book content")),
        wespeak_ai_pb2.FeedbackUploadChunk(audio_chunk=b"\x01\x02\x03"),
    ]

    servicer = FeedbackServicer()
    chunks = [chunk async for chunk in servicer.Feedback(_async_iter(upload), context=None)]

    assert chunks[0].WhichOneof("payload") == "user_text_final"
    assert chunks[0].user_text_final == "my summary"

    deltas = [c.feedback_text_delta for c in chunks[1:]]
    assert deltas == ["Great ", "job", "!"]
    assert "".join(deltas) == "Great job!"  # spec FR-006 동일성(나뉘어 온 조각을 이어붙인 결과)


@pytest.mark.asyncio
async def test_feedback_emits_stream_error_on_exception(monkeypatch):
    async def failing_transcribe(audio_bytes: bytes) -> str:
        raise RuntimeError("stt boom")

    monkeypatch.setattr(stt_service, "transcribe", failing_transcribe)

    upload = [wespeak_ai_pb2.FeedbackUploadChunk(metadata=wespeak_ai_pb2.FeedbackMetadata(book_content="book"))]

    servicer = FeedbackServicer()
    chunks = [chunk async for chunk in servicer.Feedback(_async_iter(upload), context=None)]

    assert len(chunks) == 1
    assert chunks[0].WhichOneof("payload") == "stream_error"
    assert "stt boom" in chunks[0].stream_error.message
