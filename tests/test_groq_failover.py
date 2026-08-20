from dataclasses import dataclass

import pytest

from backend.config.settings import Settings
from backend.llm.groq_failover import GroqFailoverClient


@dataclass
class FakeResponse:
    content: str


class FakeGroqError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class FakeClient:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return FakeResponse(self.result)


def settings() -> Settings:
    return Settings(
        GROQ_API_KEY="primary",
        GROQ_FALLBACK_API_KEY_1="fallback-1",
        GROQ_FALLBACK_API_KEY_2="fallback-2",
        groq_failover_cooldown_seconds=300,
    )


def pool_with(results: dict[str, object]):
    clients: dict[str, FakeClient] = {}

    def factory(**kwargs):
        key = kwargs["api_key"]
        clients[key] = FakeClient(results[key])
        return clients[key]

    pool = GroqFailoverClient(
        settings(),
        model="test-model",
        temperature=0,
        max_tokens=100,
        client_factory=factory,
    )
    return pool, clients


def test_rate_limit_uses_first_fallback_key() -> None:
    pool, clients = pool_with(
        {
            "primary": FakeGroqError("Rate limit reached", 429),
            "fallback-1": "fallback answer",
            "fallback-2": "unused",
        }
    )

    response = pool.invoke([("user", "hello")])

    assert response.content == "fallback answer"
    assert clients["primary"].calls == 1
    assert clients["fallback-1"].calls == 1
    assert "fallback-2" not in clients


def test_cooled_down_primary_is_skipped_on_next_request() -> None:
    pool, clients = pool_with(
        {
            "primary": FakeGroqError("tokens per day exceeded", 429),
            "fallback-1": "fallback answer",
            "fallback-2": "unused",
        }
    )

    pool.invoke([("user", "first")])
    pool.invoke([("user", "second")])

    assert clients["primary"].calls == 1
    assert clients["fallback-1"].calls == 2


def test_second_fallback_is_used_when_first_fallback_is_limited() -> None:
    pool, clients = pool_with(
        {
            "primary": FakeGroqError("Rate limit reached", 429),
            "fallback-1": FakeGroqError("Service unavailable", 503),
            "fallback-2": "second fallback answer",
        }
    )

    response = pool.invoke([("user", "hello")])

    assert response.content == "second fallback answer"
    assert clients["fallback-2"].calls == 1


def test_authentication_error_does_not_rotate_keys() -> None:
    pool, clients = pool_with(
        {
            "primary": FakeGroqError("Invalid API key", 401),
            "fallback-1": "must not be used",
            "fallback-2": "must not be used",
        }
    )

    with pytest.raises(FakeGroqError):
        pool.invoke([("user", "hello")])

    assert clients["primary"].calls == 1
    assert "fallback-1" not in clients
    assert "fallback-2" not in clients


def test_duplicate_keys_are_removed_from_pool() -> None:
    duplicate_settings = Settings(
        GROQ_API_KEY="same-key",
        GROQ_FALLBACK_API_KEY_1="same-key",
        GROQ_FALLBACK_API_KEY_2="other-key",
    )

    assert duplicate_settings.groq_api_keys() == ["same-key", "other-key"]


def test_qwen_client_disables_reasoning() -> None:
    captured_kwargs = {}

    def factory(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeClient("answer")

    pool = GroqFailoverClient(
        settings(),
        model="qwen/qwen3.6-27b",
        temperature=0,
        max_tokens=100,
        client_factory=factory,
    )

    pool.warmup()

    assert captured_kwargs["reasoning_effort"] == "none"


def test_non_qwen_client_does_not_receive_reasoning_effort() -> None:
    captured_kwargs = {}

    def factory(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeClient("answer")

    pool = GroqFailoverClient(
        settings(),
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=100,
        client_factory=factory,
    )

    pool.warmup()

    assert "reasoning_effort" not in captured_kwargs


def test_default_models_use_qwen(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_MODEL_NAME", raising=False)
    monkeypatch.delenv("INTENT_MODEL_NAME", raising=False)

    configured = Settings()

    assert configured.groq_model_name == "qwen/qwen3.6-27b"
    assert configured.intent_model_name == "qwen/qwen3.6-27b"
