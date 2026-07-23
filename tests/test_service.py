from backend.chatbot.intents import Intent
from backend.chatbot.service import ChatbotService
from backend.config.settings import Settings
from backend.rag.retriever import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.calls = 0
        self.chunks = chunks or []

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        self.calls += 1
        return self.chunks


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: str, context: str, history: str) -> str:
        self.calls += 1
        return f"answer for {question}"


def settings() -> Settings:
    return Settings(
        TRACKING_URL="https://track.example.com",
        HEAD_OF_SERVICES_NAME="Usama Shahid",
        HEAD_OF_SERVICES_EMAIL="hos@example.com",
        HEAD_OF_SERVICES_PHONE="+92 300 0000000",
    )


def test_special_intent_does_not_retrieve() -> None:
    retriever = FakeRetriever()
    generator = FakeGenerator()
    service = ChatbotService(settings(), retriever=retriever, generator=generator)

    result = service.chat("track my shipment", session_id="test-session")

    assert result.intent == Intent.SHIPMENT_TRACKING
    assert "https://track.example.com" in result.response
    assert retriever.calls == 0
    assert generator.calls == 0


def test_acknowledgement_does_not_retrieve() -> None:
    retriever = FakeRetriever()
    generator = FakeGenerator()
    service = ChatbotService(settings(), retriever=retriever, generator=generator)

    result = service.chat("yes", session_id="test-session")

    assert result.intent == Intent.ACKNOWLEDGEMENT
    assert "Please ask me" in result.response
    assert retriever.calls == 0
    assert generator.calls == 0


def test_rag_uses_retrieved_context() -> None:
    retriever = FakeRetriever(
        [
            RetrievedChunk(
                text="Paramount Logistics offers freight forwarding.",
                score=0.9,
                metadata={
                    "document_name": "company.pdf",
                    "page_number": 1,
                    "chunk_id": "company-p1-c0",
                },
            )
        ]
    )
    generator = FakeGenerator()
    service = ChatbotService(settings(), retriever=retriever, generator=generator)

    result = service.chat("What services are offered?", session_id="test-session")

    assert result.intent == Intent.RAG
    assert result.response == "answer for What services are offered?"
    assert retriever.calls == 1
    assert generator.calls == 1
