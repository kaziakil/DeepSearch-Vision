import json
import uuid
from pathlib import Path
from backend.services.parser import PDFParser
from backend.services.ocr import OCRPipeline
from backend.services.chunker import Chunker
from backend.models.document import IngestedDocument
from backend.core.config import settings
from backend.utils.logger import logger


class IngestionService:
    """
    Orchestrates ingestion pipeline:
    PDF/Image → Parser or OCR → Chunker → Persist JSON
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

    def __init__(self):
        self.parser = PDFParser()
        self.ocr = OCRPipeline()
        self.chunker = Chunker()

    def ingest(self, file_path: Path) -> IngestedDocument:
        suffix = file_path.suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        document_id = str(uuid.uuid4())

        logger.info(
            "ingestion_start",
            file=file_path.name,
            document_id=document_id,
        )

        # Extract
        if suffix == ".pdf":
            if self.parser.is_scanned(file_path):
                logger.info("scanned_pdf_detected", file=file_path.name)
                blocks = self.ocr.extract_from_pdf(file_path)
            else:
                blocks, _ = self.parser.parse(file_path)
        else:
            blocks = self.ocr.extract_from_image(file_path)

        if not blocks:
            raise ValueError(f"No blocks extracted from {file_path}")

        # Ensure deterministic ordering (CRITICAL for chunk stability)
        def _y_sort_key(b):
            bbox = getattr(b, "bbox", None)
            y = getattr(bbox, "y0", 0) if bbox else 0
            return (b.page or 0, y)

        blocks.sort(key=_y_sort_key)

        page_numbers = [b.page for b in blocks if isinstance(b.page, int)]
        page_count = max(page_numbers) if page_numbers else 1

        # Chunk 
        try:
            chunks = self.chunker.chunk(blocks, document_id)
        except Exception as e:
            logger.error("chunking_failed", error=str(e), document_id=document_id)
            raise

        document = IngestedDocument(
            document_id=document_id,
            filename=file_path.name,
            page_count=page_count,
            chunks=chunks,
            total_chunks=len(chunks),
            source_path=str(file_path),
        )

        self._save_chunks(document)

        logger.info(
            "ingestion_complete",
            document_id=document_id,
            chunks=len(chunks),
            pages=page_count,
        )

        return document

    def _save_chunks(self, document: IngestedDocument):
        output_path = settings.STORAGE_CHUNKS / f"{document.document_id}.json"
        tmp_path = output_path.with_suffix(".tmp")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(document.model_dump(), f, indent=2)

        tmp_path.replace(output_path)

        logger.info("chunks_saved", path=str(output_path))

