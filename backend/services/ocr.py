from pathlib import Path 
from PIL import Image
import fitz 
import numpy as np
from paddleocr import PaddleOCR
from backend.models.document import TextBlock, BoundingBox
from backend.utils.logger import logger




class OCRPipeline:
    """
    Handles scanned PDFs and images using PaddleOCR.
    Converts PDF pages to images then runs OCR per page.
    """

    def __init__(self):
        logger.info("loading_paddleocr")
        # use_angle_cls: handles rotated text
        # lang: english, swap to 'ch' for Chinese etc.
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,  # PaddleOCR currently runs on CPU on Apple Silicon
            show_log=False
        )
        logger.info(
            "ocr_device",
            gpu=False,
            engine="paddleocr",
            lang="en")
        logger.info("paddleocr_loaded")

    def extract_from_pdf(self, pdf_path: Path) -> list[TextBlock]:
        """
        Converts each PDF page to image, run OCR, returns blocks.
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

                    page_blocks = self._run_ocr(
                        img_array,
                        page_num
                    )

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
            logger.exception("image_load_failed", path=str(image_path), error=str(e))
            return []

        blocks = self._run_ocr(img_array, page_num=1)
        logger.info("ocr_image_complete",
                    path=str(image_path),
                    blocks=len(blocks),
                    pages=1)
        return blocks
    
    def _run_ocr(self, img_array: np.ndarray,
                 page_num: int) -> list[TextBlock]:
        """
        Core OCR inference. 
        Returns TextBlock list with bbox normalized to absolute pixel coordinates.
        """
        result = self.ocr.predict(img_array)
        blocks: list[TextBlock] = []

        if not result:
            return blocks
        
        res = result[0] if isinstance(result, list) else result

        if not isinstance(res, dict) or not res:
            return blocks

        rec_polys = res.get("rec_polys", [])
        rec_texts = res.get("rec_texts", [])
        rec_scores = res.get("rec_scores", [])

        min_len = min(len(rec_polys), len(rec_texts), len(rec_scores))
        for i in range(min_len):
            bbox_points = rec_polys[i]
            text = rec_texts[i]
            confidence = rec_scores[i]
            # Validate structure FIRST
            if not bbox_points or len(bbox_points) < 4:
                continue

            if confidence < 0.7: # skip low-confidence detections
                continue
            
            clean_text = text.strip()
            if len(clean_text) < 3: # skip noise
                continue

            # PaddleOCR returns 4 corner points – convert to x0y0x1y1
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
                region_type="body", # OCR doesn't classify regions
                source="paddleocr",
                confidence=float(confidence)
            ))

        return blocks



