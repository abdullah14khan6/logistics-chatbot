from backend.chatbot.intents import Intent, IntentRouter
from backend.config.settings import Settings


def make_router() -> IntentRouter:
    return IntentRouter(
        Settings(
            TRACKING_URL="https://track.example.com",
            HEAD_OF_SERVICES_NAME="Usama Shahid",
            HEAD_OF_SERVICES_EMAIL="hos@example.com",
            HEAD_OF_SERVICES_PHONE="+92 300 0000000",
        )
    )


def test_tracking_intent_skips_rag() -> None:
    result = make_router().detect("Can I track my package with an AWB?")

    assert result.intent == Intent.SHIPMENT_TRACKING
    assert "https://track.example.com" in result.response


def test_pricing_intent_returns_service_head() -> None:
    result = make_router().detect("Please send freight cost rates")

    assert result.intent == Intent.PRICING
    assert "Usama Shahid" in result.response
    assert "hos@example.com" in result.response


def test_custom_solution_intent_returns_service_head() -> None:
    result = make_router().detect("We need a tailored solution for enterprise logistics")

    assert result.intent == Intent.CUSTOM_SOLUTION
    assert "Head of Services" in result.response


def test_unknown_intent_uses_rag() -> None:
    result = make_router().detect("What services does the company offer?")

    assert result.intent == Intent.RAG
    assert result.response is None


def test_acknowledgement_does_not_use_rag() -> None:
    result = make_router().detect("yes")

    assert result.intent == Intent.ACKNOWLEDGEMENT
    assert "Please ask me" in result.response
