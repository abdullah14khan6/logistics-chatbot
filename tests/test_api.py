from fastapi.testclient import TestClient

from backend.api.main import app, get_chatbot_service
from backend.chatbot.service import ChatResponse


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class FakeChatbotService:
    def __init__(self) -> None:
        self.cleared_session_id = None

    def chat(self, message: str, session_id: str | None = None) -> ChatResponse:
        return ChatResponse(
            response=f"handled {message}",
            session_id=session_id or "generated-session",
            intent="company_services",
        )

    def clear_memory(self, session_id: str) -> None:
        self.cleared_session_id = session_id


def test_chat_endpoint_uses_service_dependency() -> None:
    fake_service = FakeChatbotService()
    app.dependency_overrides[get_chatbot_service] = lambda: fake_service
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"message": "hello", "session_id": "session-1"},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {
        "response": "handled hello",
        "session_id": "session-1",
        "intent": "company_services",
    }


def test_clear_chat_endpoint() -> None:
    fake_service = FakeChatbotService()
    app.dependency_overrides[get_chatbot_service] = lambda: fake_service
    client = TestClient(app)

    response = client.post("/chat/clear/session-1")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {"cleared": True, "session_id": "session-1"}
    assert fake_service.cleared_session_id == "session-1"

