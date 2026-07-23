from pathlib import Path

from backend.ingestion.manifest import IngestionManifest, file_sha256


def test_file_sha256_changes_with_content(tmp_path: Path) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_text("first", encoding="utf-8")
    first_hash = file_sha256(sample)

    sample.write_text("second", encoding="utf-8")

    assert file_sha256(sample) != first_hash


def test_manifest_tracks_processed_pdf(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    pdf_path = tmp_path / "company.pdf"
    pdf_path.write_text("content", encoding="utf-8")
    content_hash = file_sha256(pdf_path)

    manifest = IngestionManifest(manifest_path)
    assert not manifest.has_processed(pdf_path, content_hash)

    manifest.mark_processed(pdf_path, content_hash, vector_count=3)

    reloaded = IngestionManifest(manifest_path)
    assert reloaded.has_processed(pdf_path, content_hash)

