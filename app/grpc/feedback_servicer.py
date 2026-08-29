from app.grpc.generated import wespeak_ai_pb2, wespeak_ai_pb2_grpc
from app.logger import get_logger
from app.services import feedback_service, stt_service

logger = get_logger("AI.grpc.feedback")


class FeedbackServicer(wespeak_ai_pb2_grpc.FeedbackServiceServicer):
    """spec 004 User Story 1 — 리딩 피드백 gRPC 양방향 스트리밍 구현.

    수신: 첫 메시지는 FeedbackMetadata(book_content), 이후 반복되는 audio_chunk를 오디오로 조립.
    송신: user_text_final(1회) → feedback_text_delta(반복) 순으로 스트리밍. 오디오 응답은 원래
    없었다(REST /feedback도 TTS 미적용, chat_servicer.py와 달리 애초에 서버 TTS를 쓴 적이 없음).
    """

    async def Feedback(self, request_iterator, context):
        book_content = ""
        audio_buffer = bytearray()

        try:
            async for upload_chunk in request_iterator:
                which = upload_chunk.WhichOneof("payload")
                if which == "metadata":
                    book_content = upload_chunk.metadata.book_content
                elif which == "audio_chunk":
                    audio_buffer.extend(upload_chunk.audio_chunk)

            user_text = await stt_service.transcribe(bytes(audio_buffer))
            yield wespeak_ai_pb2.FeedbackChunk(user_text_final=user_text)

            messages = [
                {"role": "user", "content": book_content},
                {"role": "assistant", "content": "Give me a summary."},
                {"role": "user", "content": user_text},
            ]

            async for token in feedback_service.get_feedback_stream(messages):
                yield wespeak_ai_pb2.FeedbackChunk(feedback_text_delta=token)

        except Exception as e:
            logger.error("Feedback stream failed: %s", e, exc_info=True)
            yield wespeak_ai_pb2.FeedbackChunk(stream_error=wespeak_ai_pb2.StreamError(message=str(e)))
