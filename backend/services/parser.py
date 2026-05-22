import fitz    # PyMuPDF
from pathlib import Path 
from backend.models.document import TextBlock, BoundingBox
from backend.utils.logger import logger


class PDFParser:
    """
    Extracts text blocks from native (non-scanned) PDFs.
    Preserves spatial layout via bbox coordinates.
    Falls back to PaddleOCR signal when text extraction is empty.
    """

    # Region classification thresholds (relative to page height)
    HEADER_THRESHOLD = 0.10
    FOOTER_THRESHOLD = 0.90
    MIN_TEXT_LENGTH = 10     # skip noise blocks

    def parse(self, pdf_path: Path) -> tuple[list[TextBlock], int]:
        """
        Returns (blocks, page_count).
        Each block carries page number + bbox + classified region type.
        """

        doc = fitz.open(str(pdf_path))
        blocks: list[TextBlock] = []

        logger.info("parsing_pdf", 
                    path=str(pdf_path),
                    pages=doc.page_count)
        
        for page_num, page in enumerate(doc, start=1):
            page_height = page.rect.height
            page_width = page.rect.width

            # Extract dict gives us blocks with full spatial metadata
            raw_blocks = page.get_text("dict")["blocks"]

            for block in raw_blocks:
                # Skip image blocks — handled by OCR pipeline
                if block["type"] != 0:
                    continue

                # Concatenate all spans in block into clean text
                text = " ".join(
                    span["text"].strip()
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                ).strip()

                if len(text) < self.MIN_TEXT_LENGTH:
                    continue

                x0, y0, x1, y1 = block["bbox"]
                region = self._classify_region(y0, y1, page_height)

                blocks.append(TextBlock(
                    text=text,
                    page=page_num,
                    bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
                    region_type=region,
                    source="pymupdf"
                ))

        # BUG FIX 1: save page_count before closing the document
        page_count = doc.page_count
        doc.close()

        logger.info("pdf_parsed",
                    path=str(pdf_path),
                    blocks_extracted=len(blocks))

        # BUG FIX 2: return saved page_count variable, not doc.page_count
        return blocks, page_count
    
    def _classify_region(self, y0: float, y1: float,
                         page_height: float) -> str:
        """
        Classifies block as header/footer/body based on 
        vertical position relative to page height.
        """
        relative_top = y0 / page_height
        relative_bottom = y1 / page_height

        if relative_top < self.HEADER_THRESHOLD:
            return "header"
        if relative_bottom > self.FOOTER_THRESHOLD:
            return "footer"
        return "body"
    
    def is_scanned(self, pdf_path: Path) -> bool: 
        """
        Detects if PDF is scanned (no extractable text).
        If True, route to PaddleOCR instead.
        """
        doc = fitz.open(str(pdf_path))
        total_text = ""
        # Sample first 3 pages only for speed
        for page in list(doc)[:3]:
            total_text += page.get_text("text")
        doc.close()
        return len(total_text.strip()) < 50
    
