import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


def _load_env_file(path: Path = Path(".env")) -> None:
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


def _env_list(name: str, default: list[str] | None = None) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return default or []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True, init=False)
class Settings:
    groq_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_host: str
    tracking_url: str
    head_of_services_name: str
    head_of_services_email: str
    head_of_services_phone: str
    data_dir: Path
    ingestion_manifest_path: Path
    embedding_model_name: str
    embedding_dimension: int
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str
    retrieval_top_k: int
    retrieval_min_score: float
    intent_model_name: str
    groq_model_name: str
    groq_temperature: float
    cors_origins: list[str]
    chunk_size: int
    chunk_overlap: int
    ocr_dpi: int
    tesseract_cmd: str

    def __init__(self, **overrides: Any) -> None:
        _load_env_file()
        values = {
            "groq_api_key": _env("GROQ_API_KEY"),
            "pinecone_api_key": _env("PINECONE_API_KEY"),
            "pinecone_index_name": _env("PINECONE_INDEX_NAME", "logistics-company-rag"),
            "pinecone_host": _env("PINECONE_HOST"),
            "tracking_url": _env("TRACKING_URL"),
            "head_of_services_name": _env("HEAD_OF_SERVICES_NAME"),
            "head_of_services_email": _env("HEAD_OF_SERVICES_EMAIL"),
            "head_of_services_phone": _env("HEAD_OF_SERVICES_PHONE"),
            "data_dir": Path(_env("DATA_DIR", "data")),
            "ingestion_manifest_path": Path(
                _env("INGESTION_MANIFEST_PATH", "data/.ingestion_manifest.json")
            ),
            "embedding_model_name": _env("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5"),
            "embedding_dimension": _env_int("EMBEDDING_DIMENSION", 768),
            "pinecone_namespace": _env("PINECONE_NAMESPACE", "company-docs"),
            "pinecone_cloud": _env("PINECONE_CLOUD", "aws"),
            "pinecone_region": _env("PINECONE_REGION", "us-east-1"),
            "retrieval_top_k": _env_int("RETRIEVAL_TOP_K", 4),
            "retrieval_min_score": _env_float("RETRIEVAL_MIN_SCORE", 0.45),
            "intent_model_name": _env("INTENT_MODEL_NAME", "llama-3.1-8b-instant"),
            "groq_model_name": _env("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
            "groq_temperature": _env_float("GROQ_TEMPERATURE", 0.2),
            "cors_origins": _env_list("CORS_ORIGINS", ["http://localhost:8501"]),
            "chunk_size": _env_int("CHUNK_SIZE", 900),
            "chunk_overlap": _env_int("CHUNK_OVERLAP", 150),
            "ocr_dpi": _env_int("OCR_DPI", 300),
            "tesseract_cmd": _env("TESSERACT_CMD"),
        }
        alias_map = {
            "GROQ_API_KEY": "groq_api_key",
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
