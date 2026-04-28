from fastapi import FastAPI

from app.routers import chat, correct, feedback, search

app = FastAPI(
    title="WeSpeak AI Server",
    description="LangChain + Ollama 기반 영어 학습 AI API",
    version="1.0.0",
)

app.include_router(chat.router)
app.include_router(feedback.router)
app.include_router(search.router)
app.include_router(correct.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
