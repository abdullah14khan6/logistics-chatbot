import pytest

from backend.ingestion.pdf_ocr import PdfOcrExtractor


def test_pdf_ocr_reports_missing_tesseract() -> None:
    extractor = PdfOcrExtractor(tesseract_cmd="definitely-missing-tesseract.exe")

    with pytest.raises(RuntimeError, match="Tesseract OCR is not available"):
        extractor._ensure_tesseract_available()

