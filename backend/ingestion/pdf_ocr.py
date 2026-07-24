from dataclasses import dataclass
import re
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
    extraction_method: str = "ocr"


class PdfOcrExtractor:
    def __init__(
        self,
        dpi: int = 300,
        tesseract_cmd: str = "",
        native_text_min_chars: int = 80,
    ) -> None:
        self.dpi = dpi
        self.native_text_min_chars = native_text_min_chars
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract(self, pdf_path: Path) -> list[PageText]:
        pages: list[PageText] = []
        tesseract_checked = False
        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                native_text = clean_text(page.get_text("text"))
                if self._native_text_is_usable(native_text):
                    pages.append(
                        PageText(
                            page_number=page_index,
                            text=native_text,
                            extraction_method="native",
                        )
                    )
                    continue

                if not tesseract_checked:
                    self._ensure_tesseract_available()
                    tesseract_checked = True
                pixmap = page.get_pixmap(dpi=self.dpi, alpha=False)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                text = clean_text(pytesseract.image_to_string(image))
                if text:
                    pages.append(
                        PageText(
                            page_number=page_index,
                            text=text,
                            extraction_method="ocr",
                        )
                    )
        return pages

    def _native_text_is_usable(self, text: str) -> bool:
        alphanumeric = re.sub(r"[^A-Za-z0-9]", "", text)
        words = re.findall(r"[A-Za-z0-9]+", text)
        return (
            len(alphanumeric) >= self.native_text_min_chars
            and len(words) >= 8
        )

    def _ensure_tesseract_available(self) -> None:
        configured = pytesseract.pytesseract.tesseract_cmd
        if shutil.which(configured) or Path(configured).exists():
            return
        raise RuntimeError(
            "Tesseract OCR is not available. Install Tesseract and ensure it is on PATH, "
            "or set TESSERACT_CMD in .env to the full tesseract executable path."
        )
