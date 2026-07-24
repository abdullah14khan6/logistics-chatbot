import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, ContextManager, Protocol


@dataclass
class ChatTurn:
    role: str
    content: str
    dialogue_act: str = "request"
    meaningful: bool = True
    turn_index: int = 0


@dataclass
class PendingQuestion:
    topic: str
    question: str
    expected_fields: list[str] = field(default_factory=list)
    resume_query: str = ""
    kind: str = "clarification"
    created_at: int = 0


@dataclass
class ContactDisclosure:
    contact_id: str
    assistant_turn_index: int
    fields: tuple[str, ...] = ()


@dataclass
class ConversationMemory:
    max_turns: int = 16
    _turns: list[ChatTurn] = field(default_factory=list)
    contact_disclosures: dict[str, ContactDisclosure] = field(default_factory=dict)
    explained_topics: dict[str, int] = field(default_factory=dict)
    user_message_count: int = 0
    assistant_message_count: int = 0
    active_topic: str = ""
    shipment_context: dict[str, str] = field(default_factory=dict)
    pending_question: PendingQuestion | None = None

    @property
    def pending_question_topic(self) -> str:
        return self.pending_question.topic if self.pending_question else ""

    def add_user_message(
        self,
        message: str,
        dialogue_act: str = "request",
        meaningful: bool = True,
    ) -> None:
        self.user_message_count += 1
        self._append("user", message, dialogue_act, meaningful, self.user_message_count)

    def add_ai_message(
        self,
        message: str,
        dialogue_act: str = "answer",
        meaningful: bool = True,
    ) -> None:
        self.assistant_message_count += 1
        self._append(
            "assistant",
            message,
            dialogue_act,
            meaningful,
            self.assistant_message_count,
        )

    def as_text(self) -> str:
        return "\n".join(f"{turn.role}: {turn.content}" for turn in self._turns)

    def history_for_planner(self, max_meaningful_turns: int = 10) -> str:
        meaningful = [turn for turn in self._turns if turn.meaningful]
        selected = meaningful[-max_meaningful_turns:]
        omitted_social = len(self._turns) - len(meaningful)
        lines = [f"{turn.role}: {turn.content}" for turn in selected]
        if omitted_social:
            lines.insert(0, f"[{omitted_social} social-only message(s) omitted]")
        return "\n".join(lines)

    def state_for_planner(self) -> str:
        pending = asdict(self.pending_question) if self.pending_question else None
        state = {
            "active_topic": self.active_topic,
            "shipment_context": self.shipment_context,
            "pending_question": pending,
            "contacts_previously_shown": sorted(self.contact_disclosures),
        }
        return json.dumps(state, ensure_ascii=True)

    def clear(self) -> None:
        self._turns.clear()
        self.contact_disclosures.clear()
        self.explained_topics.clear()
        self.user_message_count = 0
        self.assistant_message_count = 0
        self.active_topic = ""
        self.shipment_context.clear()
        self.pending_question = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationMemory":
        memory = cls(max_turns=int(data.get("max_turns", 16)))
        memory._turns = [
            ChatTurn(**turn)
            for turn in data.get("_turns", [])
            if isinstance(turn, dict)
        ]
        memory.contact_disclosures = {
            contact_id: ContactDisclosure(
                contact_id=str(disclosure.get("contact_id", contact_id)),
                assistant_turn_index=int(
                    disclosure.get("assistant_turn_index", 0)
                ),
                fields=tuple(disclosure.get("fields", [])),
            )
            for contact_id, disclosure in data.get(
                "contact_disclosures", {}
            ).items()
            if isinstance(disclosure, dict)
        }
        memory.explained_topics = {
            str(topic): int(turn)
            for topic, turn in data.get("explained_topics", {}).items()
        }
        memory.user_message_count = int(data.get("user_message_count", 0))
        memory.assistant_message_count = int(
            data.get("assistant_message_count", 0)
        )
        memory.active_topic = str(data.get("active_topic", ""))
        memory.shipment_context = {
            str(key): str(value)
            for key, value in data.get("shipment_context", {}).items()
        }
        pending = data.get("pending_question")
        if isinstance(pending, dict):
            memory.pending_question = PendingQuestion(**pending)
        return memory

    def remember_topics(self, topics: list[str]) -> None:
        for topic in topics:
            normalized = topic.strip().lower()
            if normalized:
                self.explained_topics[normalized] = self.assistant_message_count
                self.active_topic = normalized

    def update_shipment_context(self, entities: dict[str, Any]) -> None:
        for key, value in entities.items():
            normalized = str(value or "").strip()
            if normalized:
                self.shipment_context[key] = normalized

    def topic_was_explained(self, topic: str) -> bool:
        return topic.strip().lower() in self.explained_topics

    def remember_pending_question(
        self,
        topic: str,
        question: str = "",
        expected_fields: list[str] | None = None,
        resume_query: str = "",
        kind: str = "clarification",
    ) -> None:
        self.pending_question = PendingQuestion(
            topic=topic.strip().lower(),
            question=question.strip(),
            expected_fields=expected_fields or [],
            resume_query=resume_query.strip(),
            kind=kind,
            created_at=self.assistant_message_count + 1,
        )

    def clear_pending_question(self) -> None:
        self.pending_question = None

    def remember_contact(self, contact_id: str, fields: list[str]) -> None:
        self.contact_disclosures[contact_id] = ContactDisclosure(
            contact_id=contact_id,
            assistant_turn_index=self.assistant_message_count,
            fields=tuple(fields),
        )

    def contact_age(self, contact_id: str) -> int | None:
        disclosure = self.contact_disclosures.get(contact_id)
        if disclosure is None:
            return None
        return self.assistant_message_count - disclosure.assistant_turn_index

    def _append(
        self,
        role: str,
        content: str,
        dialogue_act: str,
        meaningful: bool,
        turn_index: int,
    ) -> None:
        self._turns.append(
            ChatTurn(
                role=role,
                content=content,
                dialogue_act=dialogue_act,
                meaningful=meaningful,
                turn_index=turn_index,
            )
        )
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]


class MemoryStore:
    def __init__(
        self,
        max_turns: int = 16,
        ttl_seconds: int = 3600,
        max_sessions: int = 10000,
    ) -> None:
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: dict[str, ConversationMemory] = {}
        self._last_accessed: dict[str, float] = {}
        self._session_locks: dict[str, RLock] = {}
        self._store_lock = RLock()

    def get(self, session_id: str) -> ConversationMemory:
        with self._store_lock:
            self._prune()
            if session_id not in self._sessions:
                if len(self._sessions) >= self.max_sessions:
                    oldest = min(
                        self._last_accessed,
                        key=self._last_accessed.__getitem__,
                    )
                    self._remove(oldest)
                self._sessions[session_id] = ConversationMemory(
                    max_turns=self.max_turns
                )
                self._session_locks[session_id] = RLock()
            self._last_accessed[session_id] = time.monotonic()
            return self._sessions[session_id]

    @contextmanager
    def locked(self, session_id: str):
        memory = self.get(session_id)
        with self._store_lock:
            lock = self._session_locks[session_id]
        with lock:
            yield memory

    def clear(self, session_id: str) -> None:
        with self._store_lock:
            lock = self._session_locks.get(session_id)
        if lock is None:
            return
        with lock:
            with self._store_lock:
                self._remove(session_id)

    def _prune(self) -> None:
        if self.ttl_seconds <= 0:
            return
        cutoff = time.monotonic() - self.ttl_seconds
        expired = [
            session_id
            for session_id, accessed in self._last_accessed.items()
            if accessed < cutoff
        ]
        for session_id in expired:
            self._remove(session_id)

    def _remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._last_accessed.pop(session_id, None)
        self._session_locks.pop(session_id, None)


class ConversationStore(Protocol):
    def locked(self, session_id: str) -> ContextManager[ConversationMemory]:
        ...

    def clear(self, session_id: str) -> None:
        ...


class SQLiteMemoryStore:
    def __init__(
        self,
        path: Path,
        max_turns: int = 16,
        ttl_seconds: int = 3600,
        max_sessions: int = 10000,
    ) -> None:
        self.path = path
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._store_lock = RLock()
        self._session_locks: dict[str, RLock] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def locked(self, session_id: str):
        with self._store_lock:
            self._prune()
            lock = self._session_locks.setdefault(session_id, RLock())
        with lock:
            memory = self._load(session_id)
            try:
                yield memory
            finally:
                self._save(session_id, memory)

    def clear(self, session_id: str) -> None:
        with self._store_lock:
            lock = self._session_locks.setdefault(session_id, RLock())
        with lock:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM conversations WHERE session_id = ?",
                    (session_id,),
                )
            with self._store_lock:
                self._session_locks.pop(session_id, None)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _load(self, session_id: str) -> ConversationMemory:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM conversations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return ConversationMemory(max_turns=self.max_turns)
        return ConversationMemory.from_dict(json.loads(row[0]))

    def _save(self, session_id: str, memory: ConversationMemory) -> None:
        state_json = json.dumps(memory.to_dict(), ensure_ascii=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (session_id, state_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (session_id, state_json, time.time()),
            )
            count = connection.execute(
                "SELECT COUNT(*) FROM conversations"
            ).fetchone()[0]
            overflow = count - self.max_sessions
            if overflow > 0:
                connection.execute(
                    """
                    DELETE FROM conversations
                    WHERE session_id IN (
                        SELECT session_id
                        FROM conversations
                        ORDER BY updated_at ASC
                        LIMIT ?
                    )
                    """,
                    (overflow,),
                )

    def _prune(self) -> None:
        if self.ttl_seconds <= 0:
            return
        cutoff = time.time() - self.ttl_seconds
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM conversations WHERE updated_at < ?",
                (cutoff,),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection
