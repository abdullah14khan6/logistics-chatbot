from backend.chatbot.intents import IntentAnalysis, _json_from_text


def test_intent_analysis_parses_json() -> None:
    analysis = IntentAnalysis.from_dict(
        {
            "intents": ["tracking", "pricing"],
            "company_specific": True,
            "needs_tracking": True,
            "needs_handoff": True,
            "confidence": 0.98,
        }
    )

    assert analysis.intents == ["tracking", "pricing"]
    assert analysis.needs_tracking
    assert analysis.needs_handoff
    assert analysis.primary_label() == "tracking,pricing"


def test_json_from_text_accepts_fenced_json() -> None:
    data = _json_from_text('```json\n{"intents":["warehousing"],"confidence":0.9}\n```')

    assert data["intents"] == ["warehousing"]


def test_head_of_services_flag_parses() -> None:
    analysis = IntentAnalysis.from_dict(
        {
            "intents": ["head_of_services"],
            "needs_head_of_services": True,
            "needs_handoff": True,
        }
    )

    assert analysis.needs_head_of_services
    assert analysis.primary_label() == "head_of_services"
