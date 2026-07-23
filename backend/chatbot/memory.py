from dataclasses import dataclass, field


@dataclass
class ChatTurn:
    role: str
    content: str


@dataclass
class ConversationMemory:
    max_turns: int = 12
    _turns: list[ChatTurn] = field(default_factory=list)

    def add_user_message(self, message: str) -> None:
        self._append("user", message)

    def add_ai_message(self, message: str) -> None:
        self._append("assistant", message)

    def as_text(self) -> str:
        return "\n".join(f"{turn.role}: {turn.content}" for turn in self._turns)

    def clear(self) -> None:
        self._turns.clear()

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

