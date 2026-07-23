from functools import lru_cache

from fastapi import FastAPI, HTTPException

from backend.api.schemas import ChatRequest, ChatResponseModel, HealthResponse
from backend.chatbot.service import ChatbotService
from backend.config.settings import get_settings
from backend.utils.logging import configure_logging

configure_logging()

app = FastAPI(title="Logistics RAG Chatbot API", version="0.2.0")


@lru_cache
def get_chatbot_service() -> ChatbotService:
    return ChatbotService(get_settings())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponseModel)
def chat(request: ChatRequest) -> ChatResponseModel:
    try:
        result = get_chatbot_service().chat(
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

