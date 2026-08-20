import logging
import time
from threading import RLock
from typing import Any, Callable

from backend.config.settings import Settings

logger = logging.getLogger(__name__)


class GroqFailoverClient:
    def __init__(
        self,
        settings: Settings,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_keys = settings.groq_api_keys()
        self._clients: list[Any | None] = [None] * len(self._api_keys)
        self._unavailable_until: list[float] = [0.0] * len(self._api_keys)
        self._client_factory = client_factory
        self._lock = RLock()

    def invoke(self, messages: list[tuple[str, str]]):
        if not self._api_keys:
            raise RuntimeError("No Groq API key is configured.")

        now = time.monotonic()
        available = [
            index
            for index, blocked_until in enumerate(self._unavailable_until)
            if blocked_until <= now
        ]
        if not available:
            wait_seconds = max(1, int(min(self._unavailable_until) - now))
            raise RuntimeError(
                f"All configured Groq API keys are cooling down. Retry in {wait_seconds} seconds."
            )

        last_error: Exception | None = None
        for index in available:
            try:
                response = self._client(index).invoke(messages)
                if index:
                    logger.warning("Groq request succeeded with fallback key slot %s.", index)
                return response
            except Exception as exc:
                if not _is_retryable_groq_error(exc):
                    raise
                last_error = exc
                with self._lock:
                    self._unavailable_until[index] = (
                        time.monotonic() + self._cooldown_seconds(exc)
                    )
                logger.warning(
                    "Groq key slot %s is temporarily unavailable; trying the next slot.",
                    index,
                )

        if last_error is not None:
            raise last_error
        raise RuntimeError("No available Groq API key could serve the request.")

    def _cooldown_seconds(self, exc: Exception) -> int:
        status_code = _status_code(exc)
        if status_code == 429 or _looks_like_rate_limit(exc):
            return self.settings.groq_failover_cooldown_seconds
        return min(30, self.settings.groq_failover_cooldown_seconds)

    def warmup(self) -> None:
        if not self._api_keys:
            raise RuntimeError("No Groq API key is configured.")
        self._client(0)

    def _client(self, index: int):
        if self._clients[index] is None:
            with self._lock:
                if self._clients[index] is None:
                    factory = self._client_factory
                    if factory is None:
                        from langchain_groq import ChatGroq

                        factory = ChatGroq
                    client_kwargs = {
                        "api_key": self._api_keys[index],
                        "model": self.model,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "timeout": self.settings.llm_timeout_seconds,
                        "max_retries": self.settings.llm_max_retries,
                    }
                    if _is_qwen_model(self.model):
                        client_kwargs["reasoning_effort"] = "none"
                    self._clients[index] = factory(**client_kwargs)
        return self._clients[index]


def _is_qwen_model(model: str) -> bool:
    return "qwen" in model.strip().lower()


def _is_retryable_groq_error(exc: Exception) -> bool:
    status_code = _status_code(exc)

    if status_code in {401, 403, 404, 422}:
        return False
    if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
        return True

    name = type(exc).__name__.lower()
    if any(
        marker in name
        for marker in (
            "timeout",
            "ratelimit",
            "connection",
            "serviceunavailable",
            "internalserver",
        )
    ):
        return True

    message = str(exc).lower()
    return _looks_like_rate_limit(exc) or any(
        marker in message
        for marker in (
            "timed out",
            "timeout",
            "connection error",
            "service unavailable",
            "internal server error",
            "status code: 429",
            "status code: 500",
            "status code: 502",
            "status code: 503",
            "status code: 504",
        )
    )


def _status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def _looks_like_rate_limit(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "rate limit",
            "rate_limit",
            "tokens per day",
            "too many requests",
        )
    )
