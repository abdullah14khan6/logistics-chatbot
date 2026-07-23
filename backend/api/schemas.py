from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


class ChatResponseModel(BaseModel):
    response: str
    session_id: str
    intent: str


class HealthResponse(BaseModel):
    status: str


class ClearChatResponse(BaseModel):
    cleared: bool
    session_id: str
