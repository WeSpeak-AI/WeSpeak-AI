import json

from app.grpc.generated import wespeak_ai_pb2, wespeak_ai_pb2_grpc
from app.logger import get_logger
from app.services import chat_service, stt_service

logger = get_logger("AI.grpc.chat")


class ChatServicer(wespeak_ai_pb2_grpc.ChatServiceServicer):
    """spec 004 User Story 1 — 대화 음성 채팅 gRPC 양방향 스트리밍 구현.

    수신: 첫 메시지는 ChatMetadata(history_json), 이후 반복되는 audio_chunk를 오디오로 조립.
    송신: user_text_final(1회) → ai_text_delta(반복) 순으로 스트리밍.
    STT는 배치 처리라(research.md Decision 0) user_text_final은 한 번에 도착한다.
    음성 응답(TTS)은 AI 서버에서 생성하지 않는다 — 프론트엔드가 내장 TTS(expo-speech)로
    ai_text_delta를 직접 읽어준다(2026-08-24 결정, voca/reading 화면과 동일한 패턴).
    """

    async def Chat(self, request_iterator, context):
        history_json = None
        audio_buffer = bytearray()

        try:
            async for upload_chunk in request_iterator:
                which = upload_chunk.WhichOneof("payload")
                if which == "metadata":
                    history_json = upload_chunk.metadata.history_json
                elif which == "audio_chunk":
                    audio_buffer.extend(upload_chunk.audio_chunk)

            history_list = json.loads(history_json) if history_json else []

            user_text = await stt_service.transcribe(bytes(audio_buffer))
            yield wespeak_ai_pb2.ChatChunk(user_text_final=user_text)

            history_list.append({"role": "user", "content": user_text})

            async for token in chat_service.chat_stream(history_list):
                yield wespeak_ai_pb2.ChatChunk(ai_text_delta=token)

        except Exception as e:
            logger.error("Chat stream failed: %s", e, exc_info=True)
            yield wespeak_ai_pb2.ChatChunk(stream_error=wespeak_ai_pb2.StreamError(message=str(e)))
