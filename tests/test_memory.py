from pathlib import Path

from backend.chatbot.memory import (
    ConversationMemory,
    MemoryStore,
    SQLiteMemoryStore,
)


def test_social_turns_are_recorded_but_compacted_for_planner() -> None:
    memory = ConversationMemory()
    memory.add_user_message("hi", dialogue_act="greeting", meaningful=False)
    memory.add_ai_message("Hello!", dialogue_act="answer", meaningful=False)
    memory.add_user_message("sea freight", meaningful=True)

    history = memory.history_for_planner()

    assert "social-only message(s) omitted" in history
    assert "user: hi" not in history
    assert "user: sea freight" in history


def test_contact_age_tracks_assistant_turn_distance() -> None:
    memory = ConversationMemory()
    memory.add_ai_message("contact")
    memory.remember_contact("imports", ["email"])
    assert memory.contact_age("imports") == 0

    memory.add_ai_message("another answer")
    assert memory.contact_age("imports") == 1

    memory.add_ai_message("later answer")
    assert memory.contact_age("imports") == 2


def test_memory_store_prunes_to_maximum_sessions() -> None:
    store = MemoryStore(max_sessions=2, ttl_seconds=0)

    store.get("one")
    store.get("two")
    store.get("three")

    assert len(store._sessions) == 2
    assert "one" not in store._sessions


def test_sqlite_memory_store_survives_store_recreation(tmp_path: Path) -> None:
    path = tmp_path / "conversations.db"
    first_store = SQLiteMemoryStore(path, ttl_seconds=3600)
    with first_store.locked("session-1") as memory:
        memory.active_topic = "sea_freight"
        memory.shipment_context["destination"] = "Australia"
        memory.add_ai_message("Contact details")
        memory.remember_contact("head_of_services", ["email", "phone"])
        memory.remember_pending_question(
            "sea_freight",
            "What is the cargo weight?",
            ["weight"],
        )

    second_store = SQLiteMemoryStore(path, ttl_seconds=3600)
    with second_store.locked("session-1") as restored:
        assert restored.active_topic == "sea_freight"
        assert restored.shipment_context["destination"] == "Australia"
        assert restored.contact_age("head_of_services") == 0
        assert restored.pending_question_topic == "sea_freight"
