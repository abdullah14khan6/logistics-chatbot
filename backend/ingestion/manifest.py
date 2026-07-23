import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IngestionManifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, Any] = self._load()

    def has_processed(self, pdf_path: Path, content_hash: str) -> bool:
        record = self._data.get(str(pdf_path.resolve()))
        return bool(record and record.get("sha256") == content_hash)

    def mark_processed(self, pdf_path: Path, content_hash: str, vector_count: int) -> None:
        self._data[str(pdf_path.resolve())] = {
            "sha256": content_hash,
            "vector_count": vector_count,
        }
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

