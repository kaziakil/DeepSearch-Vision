import re
import unicodedata
from backend.models.document import TextBlock, DocumentChunk, BoundingBox
from backend.core.config import settings
from backend.utils.logger import logger


class Chunker:
    """
    Produces semantic chunks from extracted text blocks.

    Strategy:
    - Headers start a new chunk always
    - Body blocks are merged until MAX_CHUNK_CHARS is hit
    - Tables and figures are isolated as single chunks
    - Footer blocks are skipped 
    - Every chunk retains page + bbox of its constituent blocks
    """

    MAX_CHUNK_CHARS = settings.CHUNK_MAX_CHARS
    MIN_CHUNK_CHARS = settings.CHUNK_MIN_CHARS
    OVERLAP_CHARS = settings.CHUNK_OVERLAP_CHARS

    def _blocks_to_text(
        self,
        blocks: list[TextBlock]
        ) -> str:
        return "\n\n".join(
            b.text.strip()
            for b in blocks
        ).strip()
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize extracted PDF/OCR text for
        cleaner embeddings and retrieval.
        """
        text = unicodedata.normalize("NFKC", text)
    
        text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
    
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" *\n *", "\n", text)

        return text.strip()

    def chunk(self, blocks: list[TextBlock],
              document_id: str) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        buffer_text = ""
        buffer_blocks: list[TextBlock] = []

        active_header_block: TextBlock | None = None

        for block in blocks:
            # Skip footers – noise in retrieval
            if block.region_type == "footer":
                continue

            # Normalize ONCE at ingestion point
            block.text = self._normalize_text(block.text)

            # Tables and figures are always isolated chunks
            if block.region_type in ("table", "figure"):
                if buffer_text:
                    chunks.append(self._build_chunk(
                        buffer_text, buffer_blocks, document_id
                    ))
                    buffer_text = ""
                    buffer_blocks = []
                chunks.append(self._build_chunk(
                    block.text, [block], document_id
                ))
                continue
            
            # Header flushes the current buffer and start fresh
            if block.region_type == "header":
                # Flush existing body chunk first
                if buffer_text:
                    chunks.append(
                        self._build_chunk(
                            buffer_text,
                            buffer_blocks,
                            document_id
                        )
                    )
                    buffer_text = ""
                    buffer_blocks = []
                    active_header_block = None
                    
                # Store header context
                active_header_block = block
                continue

            # Force chunk boundary on page change
            if (buffer_blocks and 
                buffer_text.strip() and
                block.page != buffer_blocks[-1].page):
                active_header_block = None # reset header context on page change
                chunks.append(
                    self._build_chunk(
                        buffer_text,
                        buffer_blocks,
                        document_id
                    )
                )

                buffer_text = ""
                buffer_blocks = []

            # Body blocks: merge until size limit
            if len(buffer_text) + len(block.text) > self.MAX_CHUNK_CHARS:
                if buffer_text:
                    chunks.append(self._build_chunk(
                        buffer_text, buffer_blocks, document_id
                    ))
                    # Carry overlap into next chunk for context continuity 
                    overlap_blocks = []
                    current_overlap_chars = 0
                    for prev_block in reversed(buffer_blocks):
                        if prev_block.region_type == "header":
                            continue
                        overlap_blocks.insert(0, prev_block)
                        current_overlap_chars += len(prev_block.text)
                        if current_overlap_chars >= self.OVERLAP_CHARS:
                            break
                    buffer_blocks = overlap_blocks + [block]
                    buffer_text = self._blocks_to_text(buffer_blocks)
                else:
                    buffer_blocks = [block]
                    buffer_text = self._blocks_to_text(buffer_blocks)
            else:
                if not buffer_text:
                    buffer_blocks = [block]
                    buffer_text = self._blocks_to_text(buffer_blocks)
                else:
                    buffer_blocks.append(block)
                    buffer_text += "\n\n" + block.text.strip()
        
        # Flush remaining buffer
        if buffer_text and len(buffer_text) >= self.MIN_CHUNK_CHARS:
            chunks.append(self._build_chunk(
                buffer_text, buffer_blocks, document_id
            ))

        logger.info("chunking_complete",
                    document_id=document_id,
                    total_chunks=len(chunks))
        return chunks
    
    def _build_chunk(
        self,
        text: str,
        blocks: list[TextBlock],
        document_id: str
        ) -> DocumentChunk:

        if not blocks:
            raise ValueError(
                "Cannot build chunk with empty blocks"
            )

        merged_bbox = self._merge_bboxes(
            [b.bbox for b in blocks]
        )

        page = blocks[0].page
        source = blocks[0].source

        region_types = {b.region_type for b in blocks}

        region = (
            next(iter(region_types))
            if len(region_types) == 1
            else "mixed"
        )

        normalized_text=self._normalize_text(text)

        return DocumentChunk(
            document_id=document_id,
            text=normalized_text,
            page=page,
            bbox=merged_bbox,
            region_type=region,
            char_count=len(normalized_text),
            token_estimate=len(normalized_text) // 4,
            source=source
        )
    
    def _merge_bboxes(self, bboxes: list[BoundingBox]) -> BoundingBox:
        """Union of all bboxes – smallest enclosing rectangle."""
        if not bboxes:
            raise ValueError(
                "Cannot merge empty bbox list"
            )
        return BoundingBox(
            x0=min(b.x0 for b in bboxes),
            y0=min(b.y0 for b in bboxes),
            x1=max(b.x1 for b in bboxes),
            y1=max(b.y1 for b in bboxes)
        )
    
