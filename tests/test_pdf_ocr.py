import pytest

from backend.ingestion.pdf_ocr import PdfOcrExtractor


def test_pdf_ocr_reports_missing_tesseract() -> None:
    extractor = PdfOcrExtractor(tesseract_cmd="definitely-missing-tesseract.exe")

    with pytest.raises(RuntimeError, match="Tesseract OCR is not available"):
        extractor._ensure_tesseract_available()


def test_native_text_quality_threshold_avoids_unnecessary_ocr() -> None:
    extractor = PdfOcrExtractor(native_text_min_chars=30)

    assert extractor._native_text_is_usable(
        "Air Freight services include pickup, customs support, and final delivery."
    )
    assert not extractor._native_text_is_usable("ADDRESS")
