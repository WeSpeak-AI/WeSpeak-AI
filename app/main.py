import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.logger import get_logger, setup_logging
from app.routers import chat, correct, feedback, ingest, search, topic, voca
from app.services.stt_service import get_model

setup_logging()
logger = get_logger("wespeak.http")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield


app = FastAPI(
    title="WeSpeak AI Server",
    description="LangChain + Ollama 기반 영어 학습 AI API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info("%s %s %d %.1fms", request.method, request.url.path, response.status_code, elapsed)
    return response


app.include_router(chat.router)
app.include_router(feedback.router)
app.include_router(search.router)
app.include_router(correct.router)
app.include_router(topic.router)
app.include_router(voca.router)
app.include_router(ingest.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
