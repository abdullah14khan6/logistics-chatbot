import argparse
import logging
from pathlib import Path

from backend.config.settings import get_settings
from backend.ingestion.pipeline import IngestionPipeline
from backend.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR and ingest company PDFs into Pinecone.")
    parser.add_argument(
        "--pdf",
        action="append",
        type=Path,
        help="Specific PDF path to ingest. Can be provided multiple times.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest PDFs even if the content hash is unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()
    pipeline = IngestionPipeline(settings)
    vector_count = pipeline.run(pdf_paths=args.pdf, force=args.force)
    logging.getLogger(__name__).info("Ingestion complete. Upserted %s vectors.", vector_count)


if __name__ == "__main__":
    main()

