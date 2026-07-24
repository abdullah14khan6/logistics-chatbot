import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.api.schemas import (
    ChatRequest,
    ChatResponseModel,
    ClearChatResponse,
    HealthResponse,
    ReadinessResponse,
)
from backend.chatbot.service import ChatbotService
from backend.config.settings import get_settings
from backend.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache
def get_chatbot_service() -> ChatbotService:
    return ChatbotService(settings)


def _log_warmup_result(future: Future[None]) -> None:
    try:
        future.result()
        logger.info("RAG dependencies are warm and ready.")
    except Exception:
        logger.exception("RAG dependency warmup failed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.warmup_future = None
    executor = None
    if settings.prewarm_on_startup:
        logger.info("Starting RAG dependency warmup.")
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-warmup")
        future = executor.submit(get_chatbot_service().warmup)
        future.add_done_callback(_log_warmup_result)
        app.state.warmup_future = future
    yield
    if executor:
        executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Logistics RAG Chatbot API",
    version="0.3.0",
    lifespan=lifespan,
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadinessResponse)
def ready() -> ReadinessResponse:
    future: Future[None] | None = getattr(app.state, "warmup_future", None)
    if future is None:
        return ReadinessResponse(
            status="ready",
            detail="Startup warmup is disabled; dependencies load on demand.",
        )
    if not future.done():
        raise HTTPException(status_code=503, detail="RAG dependencies are warming.")
    if future.exception() is not None:
        raise HTTPException(status_code=503, detail="RAG dependency warmup failed.")
    return ReadinessResponse(status="ready", detail="RAG dependencies are warm.")


@app.post("/chat", response_model=ChatResponseModel)
def chat(
    request: ChatRequest,
    service: ChatbotService = Depends(get_chatbot_service),
) -> ChatResponseModel:
    started = time.perf_counter()
    try:
        result = service.chat(
            message=request.message,
            session_id=request.session_id,
        )
    except Exception as exc:
        logger.exception("Chatbot request failed.")
        raise HTTPException(status_code=500, detail="Chatbot service failed.") from exc
    finally:
        logger.info("Chat request completed in %.3f seconds.", time.perf_counter() - started)

    return ChatResponseModel(
        response=result.response,
        session_id=result.session_id,
        intent=result.intent,
    )


@app.post("/chat/clear/{session_id}", response_model=ClearChatResponse)
def clear_chat(
    session_id: str,
    service: ChatbotService = Depends(get_chatbot_service),
) -> ClearChatResponse:
    service.clear_memory(session_id)
    return ClearChatResponse(cleared=True, session_id=session_id)
