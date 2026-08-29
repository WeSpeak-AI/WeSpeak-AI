from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ChatUploadChunk(_message.Message):
    __slots__ = ("metadata", "audio_chunk")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    AUDIO_CHUNK_FIELD_NUMBER: _ClassVar[int]
    metadata: ChatMetadata
    audio_chunk: bytes
    def __init__(self, metadata: _Optional[_Union[ChatMetadata, _Mapping]] = ..., audio_chunk: _Optional[bytes] = ...) -> None: ...

class ChatMetadata(_message.Message):
    __slots__ = ("history_json",)
    HISTORY_JSON_FIELD_NUMBER: _ClassVar[int]
    history_json: str
    def __init__(self, history_json: _Optional[str] = ...) -> None: ...

class ChatChunk(_message.Message):
    __slots__ = ("user_text_final", "ai_text_delta", "stream_error")
    USER_TEXT_FINAL_FIELD_NUMBER: _ClassVar[int]
    AI_TEXT_DELTA_FIELD_NUMBER: _ClassVar[int]
    STREAM_ERROR_FIELD_NUMBER: _ClassVar[int]
    user_text_final: str
    ai_text_delta: str
    stream_error: StreamError
    def __init__(self, user_text_final: _Optional[str] = ..., ai_text_delta: _Optional[str] = ..., stream_error: _Optional[_Union[StreamError, _Mapping]] = ...) -> None: ...

class FeedbackUploadChunk(_message.Message):
    __slots__ = ("metadata", "audio_chunk")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    AUDIO_CHUNK_FIELD_NUMBER: _ClassVar[int]
    metadata: FeedbackMetadata
    audio_chunk: bytes
    def __init__(self, metadata: _Optional[_Union[FeedbackMetadata, _Mapping]] = ..., audio_chunk: _Optional[bytes] = ...) -> None: ...

class FeedbackMetadata(_message.Message):
    __slots__ = ("book_content",)
    BOOK_CONTENT_FIELD_NUMBER: _ClassVar[int]
    book_content: str
    def __init__(self, book_content: _Optional[str] = ...) -> None: ...

class FeedbackChunk(_message.Message):
    __slots__ = ("user_text_final", "feedback_text_delta", "stream_error")
    USER_TEXT_FINAL_FIELD_NUMBER: _ClassVar[int]
    FEEDBACK_TEXT_DELTA_FIELD_NUMBER: _ClassVar[int]
    STREAM_ERROR_FIELD_NUMBER: _ClassVar[int]
    user_text_final: str
    feedback_text_delta: str
    stream_error: StreamError
    def __init__(self, user_text_final: _Optional[str] = ..., feedback_text_delta: _Optional[str] = ..., stream_error: _Optional[_Union[StreamError, _Mapping]] = ...) -> None: ...

class StreamError(_message.Message):
    __slots__ = ("message",)
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...
