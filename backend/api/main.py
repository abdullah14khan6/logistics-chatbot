from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.api.schemas import (
    ChatRequest,
    ChatResponseModel,
    ClearChatResponse,
    HealthResponse,
)
from backend.chatbot.service import ChatbotService
from backend.config.settings import get_settings
from backend.utils.logging import configure_logging

configure_logging()

app = FastAPI(title="Logistics RAG Chatbot API", version="0.2.0")
settings = get_settings()

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )


@lru_cache
def get_chatbot_service() -> ChatbotService:
    return ChatbotService(settings)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponseModel)
def chat(
    request: ChatRequest,
    service: ChatbotService = Depends(get_chatbot_service),
) -> ChatResponseModel:
    try:
        result = service.chat(
            message=request.message,
            session_id=request.session_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Chatbot service failed.") from exc

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
