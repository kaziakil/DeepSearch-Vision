import os
import numpy as np
from pathlib import Path
from PIL import Image
import fitz
from backend.models.document import TextBlock, BoundingBox
from backend.utils.logger import logger


class OCRPipeline:
    """
    Handles scanned PDFs and images using EasyOCR.
    Converts PDF pages to images then runs OCR per page.
    """

    def __init__(self):
        import easyocr
        logger.info("loading_ocr")
        self.ocr = easyocr.Reader(
            ["en"],
            gpu=True        # uses MPS on Apple Silicon
        )
        logger.info("ocr_loaded")

    def extract_from_pdf(self, pdf_path: Path) -> list[TextBlock]:
        """
        Converts each PDF page to image, runs OCR, returns blocks.
        Use this for scanned PDFs where PyMuPDF yields no text.
        """
        doc = fitz.open(str(pdf_path))
        blocks: list[TextBlock] = []

        logger.info(
            "ocr_pdf_start",
            path=str(pdf_path),
            pages=doc.page_count
        )

        try:
            for page_num, page in enumerate(doc, start=1):
                pix = None
                img_array = None

                try:
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat)

                    img_array = np.frombuffer(
                        pix.samples,
                        dtype=np.uint8
                    ).reshape(
                        pix.height,
                        pix.width,
                        pix.n
                    )

                    if pix.n == 4:
                        img_array = img_array[:, :, :3]

                    page_blocks = self._run_ocr(img_array, page_num)
                    blocks.extend(page_blocks)

                except Exception as e:
                    logger.exception(
                        "ocr_page_failed",
                        page=page_num,
                        error=str(e)
                    )
                finally:
                    del pix
                    del img_array

        finally:
            doc.close()

        logger.info(
            "ocr_pdf_complete",
            path=str(pdf_path),
            blocks=len(blocks)
        )

        return blocks

    def extract_from_image(self, image_path: Path) -> list[TextBlock]:
        """Runs OCR directly on a single image file."""
        try:
            with Image.open(image_path) as img:
                img_array = np.array(img.convert("RGB"))
        except Exception as e:
            logger.exception(
                "image_load_failed",
                path=str(image_path),
                error=str(e)
            )
            return []

        blocks = self._run_ocr(img_array, page_num=1)
        logger.info(
            "ocr_image_complete",
            path=str(image_path),
            blocks=len(blocks),
            pages=1
        )
        return blocks

    def _run_ocr(self, img_array: np.ndarray,
                 page_num: int) -> list[TextBlock]:
        """
        Core OCR inference.
        Returns TextBlock list with bbox normalized to absolute pixel coordinates.
        """
        result = self.ocr.readtext(img_array)
        blocks: list[TextBlock] = []

        if not result:
            return blocks

        for (bbox_points, text, confidence) in result:
            if confidence < 0.7:
                continue

            clean_text = text.strip()
            if len(clean_text) < 3:
                continue

            pts = np.asarray(bbox_points)
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]

            blocks.append(TextBlock(
                text=clean_text,
                page=page_num,
                bbox=BoundingBox(
                    x0=float(min(x_coords)),
                    y0=float(min(y_coords)),
                    x1=float(max(x_coords)),
                    y1=float(max(y_coords))
                ),
                region_type="body",
                source="easyocr",
                confidence=float(confidence)
            ))

        return blocks
    

    