from pathlib import Path

from backend.config.settings import Settings
from backend.utils.preflight import has_failures, run_preflight


def test_preflight_reports_missing_pdf_directory(tmp_path: Path) -> None:
    settings = Settings(
        GROQ_API_KEY="groq",
        PINECONE_API_KEY="pinecone",
        TRACKING_URL="https://track.example.com",
        data_dir=tmp_path / "missing",
        tesseract_cmd=str(tmp_path / "tesseract.exe"),
    )

    results = run_preflight(settings)

    assert has_failures(results)
    assert any(result.name == "data PDFs" and not result.ok for result in results)

