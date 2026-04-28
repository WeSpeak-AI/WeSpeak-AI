from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    temperature_chat: float = 0.7
    temperature_structured: float = 0.1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
