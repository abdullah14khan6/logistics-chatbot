import logging
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from backend.config.settings import Settings
from backend.ingestion.manifest import IngestionManifest, file_sha256
from backend.ingestion.pdf_ocr import PageText, PdfOcrExtractor

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.extractor = PdfOcrExtractor(
            dpi=settings.ocr_dpi,
            tesseract_cmd=settings.tesseract_cmd,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

    def run(self, pdf_paths: list[Path] | None = None, force: bool = False) -> int:
        self._validate_configuration()
        manifest = IngestionManifest(self.settings.ingestion_manifest_path)
        pdfs = pdf_paths or sorted(self.settings.data_dir.glob("*.pdf"))
        if not pdfs:
            logger.warning("No PDF files found in %s", self.settings.data_dir)
            return 0

        index = self._pinecone_index()
        total_vectors = 0
        for pdf_path in pdfs:
            content_hash = file_sha256(pdf_path)
            if not force and manifest.has_processed(pdf_path, content_hash):
                logger.info("Skipping unchanged PDF: %s", pdf_path)
                continue

            logger.info("OCR extracting PDF: %s", pdf_path)
            pages = self.extractor.extract(pdf_path)
            documents = self._chunk_pages(pdf_path, pages)
            if not documents:
                logger.warning("No OCR text found in %s", pdf_path)
                continue

            vector_count = self._upsert_documents(index, documents)
            manifest.mark_processed(pdf_path, content_hash, vector_count)
            total_vectors += vector_count
            logger.info("Upserted %s vectors for %s", vector_count, pdf_path.name)

        return total_vectors

    def _chunk_pages(self, pdf_path: Path, pages: list[PageText]) -> list[Document]:
        documents: list[Document] = []
        for page in pages:
            page_document = Document(
                page_content=page.text,
                metadata={
                    "page_number": page.page_number,
                    "document_name": pdf_path.name,
                    "source": str(pdf_path),
                },
            )
            chunks = self.splitter.split_documents([page_document])
            for chunk_index, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = f"{pdf_path.stem}-p{page.page_number}-c{chunk_index}"
                documents.append(chunk)
        return documents

    def _upsert_documents(self, index, documents: list[Document], batch_size: int = 100) -> int:
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            texts = [document.page_content for document in batch]
            vectors = self.embeddings.embed_documents(texts)
            records = []
            for document, vector in zip(batch, vectors, strict=True):
                chunk_id = document.metadata["chunk_id"]
                vector_id = str(uuid5(NAMESPACE_URL, f"{document.metadata['source']}#{chunk_id}"))
                metadata = {
                    **document.metadata,
                    "text": document.page_content,
                }
                records.append(
                    {
                        "id": vector_id,
                        "values": vector,
                        "metadata": metadata,
                    }
                )
            index.upsert(vectors=records, namespace=self.settings.pinecone_namespace)
        return len(documents)

    def _pinecone_index(self):
        client = Pinecone(api_key=self.settings.pinecone_api_key)
        if self.settings.pinecone_host:
            return client.Index(host=self.settings.pinecone_host)
        return client.Index(self.settings.pinecone_index_name)

    def _validate_configuration(self) -> None:
        missing = []
        if not self.settings.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        if not self.settings.pinecone_index_name and not self.settings.pinecone_host:
            missing.append("PINECONE_INDEX_NAME or PINECONE_HOST")
        if missing:
            raise ValueError(f"Missing required ingestion setting(s): {', '.join(missing)}")

