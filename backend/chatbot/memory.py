from dataclasses import dataclass, field


@dataclass
class ChatTurn:
    role: str
    content: str


@dataclass
class ConversationMemory:
    max_turns: int = 12
    _turns: list[ChatTurn] = field(default_factory=list)
    contact_card_shown: bool = False
    explained_topics: set[str] = field(default_factory=set)
    user_message_count: int = 0
    last_topic: str = ""
    last_handoff_suggestion_at: int = 0
    pending_question_topic: str = ""

    def add_user_message(self, message: str) -> None:
        self.user_message_count += 1
        self._append("user", message)

    def add_ai_message(self, message: str) -> None:
        self._append("assistant", message)

    def as_text(self) -> str:
        return "\n".join(f"{turn.role}: {turn.content}" for turn in self._turns)

    def clear(self) -> None:
        self._turns.clear()
        self.contact_card_shown = False
        self.explained_topics.clear()
        self.user_message_count = 0
        self.last_topic = ""
        self.last_handoff_suggestion_at = 0
        self.pending_question_topic = ""

    def remember_topics(self, topics: list[str]) -> None:
        for topic in topics:
            normalized = topic.strip().lower()
            if normalized:
                self.explained_topics.add(normalized)
                self.last_topic = normalized

    def recently_suggested_handoff(self, window: int = 4) -> bool:
        return self.user_message_count - self.last_handoff_suggestion_at < window

    def mark_handoff_suggested(self) -> None:
        self.last_handoff_suggestion_at = self.user_message_count

    def remember_pending_question(self, topic: str) -> None:
        self.pending_question_topic = topic.strip().lower()

    def clear_pending_question(self) -> None:
        self.pending_question_topic = ""

    def _append(self, role: str, content: str) -> None:
        self._turns.append(ChatTurn(role=role, content=content))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]


class MemoryStore:
    def __init__(self, max_turns: int = 12) -> None:
        self.max_turns = max_turns
        self._sessions: dict[str, ConversationMemory] = {}

    def get(self, session_id: str) -> ConversationMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(max_turns=self.max_turns)
        return self._sessions[session_id]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
