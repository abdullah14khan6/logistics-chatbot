import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str] | None = None) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return default or []
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_path(name: str, default: str) -> Path:
    path = Path(_env(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True, init=False)
class Settings:
    groq_api_key: str
    groq_fallback_api_key_1: str
    groq_fallback_api_key_2: str
    groq_failover_cooldown_seconds: int
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_host: str
    tracking_url: str
    head_of_services_name: str
    head_of_services_email: str
    head_of_services_phone: str
    company_profile_path: Path
    data_dir: Path
    ingestion_manifest_path: Path
    embedding_model_name: str
    embedding_dimension: int
    embedding_local_files_only: bool
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str
    retrieval_top_k: int
    retrieval_candidate_k: int
    retrieval_min_score: float
    retrieval_cache_size: int
    intent_model_name: str
    groq_model_name: str
    groq_temperature: float
    llm_timeout_seconds: float
    llm_max_retries: int
    response_max_tokens: int
    response_max_words: int
    response_brief_max_words: int
    response_complex_max_words: int
    response_detailed_max_words: int
    memory_backend: str
    memory_db_path: Path
    memory_max_turns: int
    memory_ttl_seconds: int
    memory_max_sessions: int
    prewarm_on_startup: bool
    uvicorn_reload: bool
    cors_origins: list[str]
    chunk_size: int
    chunk_overlap: int
    ocr_dpi: int
    native_text_min_chars: int
    tesseract_cmd: str

    def __init__(self, **overrides: Any) -> None:
        _load_env_file()
        values = {
            "groq_api_key": _env("GROQ_API_KEY"),
            "groq_fallback_api_key_1": _env("GROQ_FALLBACK_API_KEY_1"),
            "groq_fallback_api_key_2": _env("GROQ_FALLBACK_API_KEY_2"),
            "groq_failover_cooldown_seconds": _env_int(
                "GROQ_FAILOVER_COOLDOWN_SECONDS", 14400
            ),
            "pinecone_api_key": _env("PINECONE_API_KEY"),
            "pinecone_index_name": _env("PINECONE_INDEX_NAME", "logistics-company-rag"),
            "pinecone_host": _env("PINECONE_HOST"),
            "tracking_url": _env("TRACKING_URL"),
            "head_of_services_name": _env("HEAD_OF_SERVICES_NAME"),
            "head_of_services_email": _env("HEAD_OF_SERVICES_EMAIL"),
            "head_of_services_phone": _env("HEAD_OF_SERVICES_PHONE"),
            "company_profile_path": _env_path(
                "COMPANY_PROFILE_PATH", "data/company_profile.json"
            ),
            "data_dir": _env_path("DATA_DIR", "data"),
            "ingestion_manifest_path": _env_path(
                "INGESTION_MANIFEST_PATH", "data/.ingestion_manifest.json"
            ),
            "embedding_model_name": _env("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5"),
            "embedding_dimension": _env_int("EMBEDDING_DIMENSION", 768),
            "embedding_local_files_only": _env_bool(
                "EMBEDDING_LOCAL_FILES_ONLY", False
            ),
            "pinecone_namespace": _env("PINECONE_NAMESPACE", "company-docs"),
            "pinecone_cloud": _env("PINECONE_CLOUD", "aws"),
            "pinecone_region": _env("PINECONE_REGION", "us-east-1"),
            "retrieval_top_k": _env_int("RETRIEVAL_TOP_K", 4),
            "retrieval_candidate_k": _env_int("RETRIEVAL_CANDIDATE_K", 8),
            "retrieval_min_score": _env_float("RETRIEVAL_MIN_SCORE", 0.45),
            "retrieval_cache_size": _env_int("RETRIEVAL_CACHE_SIZE", 128),
            "intent_model_name": _env("INTENT_MODEL_NAME", "qwen/qwen3.6-27b"),
            "groq_model_name": _env("GROQ_MODEL_NAME", "qwen/qwen3.6-27b"),
            "groq_temperature": _env_float("GROQ_TEMPERATURE", 0.2),
            "llm_timeout_seconds": _env_float("LLM_TIMEOUT_SECONDS", 20.0),
            "llm_max_retries": _env_int("LLM_MAX_RETRIES", 1),
            "response_max_tokens": _env_int("RESPONSE_MAX_TOKENS", 600),
            "response_max_words": _env_int("RESPONSE_MAX_WORDS", 140),
            "response_brief_max_words": _env_int(
                "RESPONSE_BRIEF_MAX_WORDS", 60
            ),
            "response_complex_max_words": _env_int(
                "RESPONSE_COMPLEX_MAX_WORDS", 250
            ),
            "response_detailed_max_words": _env_int(
                "RESPONSE_DETAILED_MAX_WORDS", 400
            ),
            "memory_backend": _env("MEMORY_BACKEND", "memory").strip().lower(),
            "memory_db_path": _env_path(
                "MEMORY_DB_PATH", "data/conversations.db"
            ),
            "memory_max_turns": _env_int("MEMORY_MAX_TURNS", 16),
            "memory_ttl_seconds": _env_int("MEMORY_TTL_SECONDS", 3600),
            "memory_max_sessions": _env_int("MEMORY_MAX_SESSIONS", 10000),
            "prewarm_on_startup": _env_bool("PREWARM_ON_STARTUP", False),
            "uvicorn_reload": _env_bool("UVICORN_RELOAD", False),
            "cors_origins": _env_list("CORS_ORIGINS", ["http://localhost:8501"]),
            "chunk_size": _env_int("CHUNK_SIZE", 900),
            "chunk_overlap": _env_int("CHUNK_OVERLAP", 150),
            "ocr_dpi": _env_int("OCR_DPI", 300),
            "native_text_min_chars": _env_int("NATIVE_TEXT_MIN_CHARS", 80),
            "tesseract_cmd": _env("TESSERACT_CMD"),
        }
        alias_map = {
            "GROQ_API_KEY": "groq_api_key",
            "GROQ_FALLBACK_API_KEY_1": "groq_fallback_api_key_1",
            "GROQ_FALLBACK_API_KEY_2": "groq_fallback_api_key_2",
            "PINECONE_API_KEY": "pinecone_api_key",
            "PINECONE_INDEX_NAME": "pinecone_index_name",
            "PINECONE_HOST": "pinecone_host",
            "TRACKING_URL": "tracking_url",
            "HEAD_OF_SERVICES_NAME": "head_of_services_name",
            "HEAD_OF_SERVICES_EMAIL": "head_of_services_email",
            "HEAD_OF_SERVICES_PHONE": "head_of_services_phone",
        }
        for key, value in overrides.items():
            values[alias_map.get(key, key)] = value
        for key, value in values.items():
            object.__setattr__(self, key, value)

    def groq_api_keys(self) -> list[str]:
        keys = [
            self.groq_api_key,
            self.groq_fallback_api_key_1,
            self.groq_fallback_api_key_2,
        ]
        return list(dict.fromkeys(key.strip() for key in keys if key.strip()))


@lru_cache
def get_settings() -> Settings:
    return Settings()
