import pytest
from pydantic import ValidationError

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


def test_action_flags_are_normalized_deterministically() -> None:
    analysis = IntentAnalysis.from_dict(
        {
            "intents": ["pricing"],
            "actions": ["quote", "handoff"],
            "confidence": 0.9,
        }
    )

    assert analysis.needs_pricing
    assert analysis.needs_handoff

    company_plan = IntentAnalysis.from_dict(
        {
            "intents": ["warehousing"],
            "actions": ["company_lookup"],
        }
    )
    assert company_plan.company_specific


def test_string_boolean_is_rejected() -> None:
    with pytest.raises(ValidationError):
        IntentAnalysis.from_dict(
            {
                "intents": ["tracking"],
                "needs_tracking": "false",
            }
        )


def test_unknown_intent_is_rejected() -> None:
    with pytest.raises(ValidationError):
        IntentAnalysis.from_dict({"intents": ["made_up_route"]})
