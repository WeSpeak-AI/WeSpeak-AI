from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # "ollama" 또는 "claude"
    provider: str = "ollama"

    # Ollama 설정
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Claude 설정
    anthropic_api_key: str = ""
    claude_model: str = "claude-haiku-4-5-20251001"

    temperature_chat: float = 0.7
    temperature_structured: float = 0.1

    # STT 제공자: "local" (faster-whisper) 또는 "openai"
    stt_provider: str = "local"

    # faster-whisper 설정
    # 모델 크기: tiny / base / small / medium / large-v3
    whisper_model: str = "medium"
    # 실행 디바이스: cpu / cuda
    whisper_device: str = "cuda"
    # 연산 정밀도: int8 / float16 / float32
    whisper_compute_type: str = "int8"
    # 전사 언어 (영어 고정)
    whisper_language: str = "en"

    # OpenAI Whisper API 설정
    openai_api_key: str = ""



    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
