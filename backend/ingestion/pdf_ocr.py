from dataclasses import dataclass
import shutil
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

from backend.ingestion.cleaning import clean_text


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


class PdfOcrExtractor:
    def __init__(self, dpi: int = 300, tesseract_cmd: str = "") -> None:
        self.dpi = dpi
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract(self, pdf_path: Path) -> list[PageText]:
        self._ensure_tesseract_available()
        pages: list[PageText] = []
        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                pixmap = page.get_pixmap(dpi=self.dpi, alpha=False)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                text = clean_text(pytesseract.image_to_string(image))
                if text:
                    pages.append(PageText(page_number=page_index, text=text))
        return pages

    def _ensure_tesseract_available(self) -> None:
        configured = pytesseract.pytesseract.tesseract_cmd
        if shutil.which(configured) or Path(configured).exists():
            return
        raise RuntimeError(
            "Tesseract OCR is not available. Install Tesseract and ensure it is on PATH, "
            "or set TESSERACT_CMD in .env to the full tesseract executable path."
        )
