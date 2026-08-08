import logging
import re
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException

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
            native_text_min_chars=settings.native_text_min_chars,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={
                "local_files_only": settings.embedding_local_files_only
            },
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

            logger.info("Extracting PDF text with OCR fallback: %s", pdf_path)
            pages = self.extractor.extract(pdf_path)
            documents = self._chunk_pages(pdf_path, pages, content_hash)
            if not documents:
                logger.warning("No OCR text found in %s", pdf_path)
                continue

            records = self._build_records(documents)
            self._delete_existing_document(index, pdf_path.name)
            vector_count = self._upsert_records(index, records)
            manifest.mark_processed(pdf_path, content_hash, vector_count)
            total_vectors += vector_count
            logger.info("Upserted %s vectors for %s", vector_count, pdf_path.name)

        return total_vectors

    def _chunk_pages(
        self,
        pdf_path: Path,
        pages: list[PageText],
        content_hash: str,
    ) -> list[Document]:
        documents: list[Document] = []
        for page in pages:
            section = self._section_title(page.text)
            content_type = self._content_type(page.text, section)
            page_document = Document(
                page_content=page.text,
                metadata={
                    "page_number": page.page_number,
                    "document_name": pdf_path.name,
                    "source": pdf_path.name,
                    "document_hash": content_hash,
                    "section": section,
                    "content_type": content_type,
                    "extraction_method": page.extraction_method,
                },
            )
            chunks = self.splitter.split_documents([page_document])
            for chunk_index, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = f"{pdf_path.stem}-p{page.page_number}-c{chunk_index}"
                documents.append(chunk)
        return documents

    def _delete_existing_document(self, index, document_name: str) -> None:
        try:
            index.delete(
                namespace=self.settings.pinecone_namespace,
                filter={"document_name": {"$eq": document_name}},
            )
        except NotFoundException as exc:
            if "Namespace not found" not in str(exc):
                raise
            logger.info(
                "Namespace %s is empty; no existing vectors to delete",
                self.settings.pinecone_namespace,
            )

    @staticmethod
    def _section_title(text: str) -> str:
        for line in text.splitlines():
            candidate = line.strip(" -•\t")
            if candidate and len(candidate) <= 100:
                return candidate
        return "Company information"

    @staticmethod
    def _content_type(text: str, section: str) -> str:
        normalized = f"{section}\n{text}".lower()
        email_count = len(re.findall(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", normalized))
        if email_count >= 2 or any(
            marker in normalized
            for marker in (
                "team leaders",
                "head office- management",
                "branch manager",
                "head of department",
            )
        ):
            return "staff_directory"
        if "address" in normalized and any(
            city in normalized for city in ("sialkot", "karachi")
        ):
            return "office"
        if any(
            marker in normalized
            for marker in (
                "our services",
                "service portfolio",
                "air freight",
                "ocean freight",
                "warehousing",
                "customs brokerage",
            )
        ):
            return "service"
        return "company"

    def _build_records(
        self,
        documents: list[Document],
        batch_size: int = 100,
    ) -> list[dict]:
        all_records: list[dict] = []
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            texts = [document.page_content for document in batch]
            vectors = self.embeddings.embed_documents(texts)
            records = []
            for document, vector in zip(batch, vectors, strict=True):
                chunk_id = document.metadata["chunk_id"]
                vector_id = str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{document.metadata['document_name']}#{chunk_id}",
                    )
                )
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
            all_records.extend(records)
        return all_records

    def _upsert_records(
        self,
        index,
        records: list[dict],
        batch_size: int = 100,
    ) -> int:
        for start in range(0, len(records), batch_size):
            index.upsert(
                vectors=records[start : start + batch_size],
                namespace=self.settings.pinecone_namespace,
            )
        return len(records)

    def _pinecone_index(self):
        client = Pinecone(api_key=self.settings.pinecone_api_key)
        if self.settings.pinecone_host:
            return client.Index(host=self.settings.pinecone_host)
        if not client.has_index(self.settings.pinecone_index_name):
            logger.info(
                "Creating Pinecone index %s (%s dimensions, cosine, %s/%s)",
                self.settings.pinecone_index_name,
                self.settings.embedding_dimension,
                self.settings.pinecone_cloud,
                self.settings.pinecone_region,
            )
            client.create_index(
                name=self.settings.pinecone_index_name,
                dimension=self.settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                ),
            )
        return client.Index(self.settings.pinecone_index_name)

    def _validate_configuration(self) -> None:
        missing = []
        if not self.settings.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        if not self.settings.pinecone_index_name and not self.settings.pinecone_host:
            missing.append("PINECONE_INDEX_NAME or PINECONE_HOST")
        if missing:
            raise ValueError(f"Missing required ingestion setting(s): {', '.join(missing)}")
