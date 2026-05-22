"""
Day 2 validation script.
Tests: PDF → parse/OCR → chunk → JSON output
Run: python scripts/test_ingestion.py <path_to_pdf>
"""

import sys
import json
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.ingestion import IngestionService
from backend.utils.logger import setup_logging

setup_logging(debug=True)

BASE_DIR = Path(__file__).resolve().parent.parent
pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "datasets/raw/sample.pdf"

if not pdf_path.exists():
    print(f"No PDF found at {pdf_path}")
    sys.exit(1)

service = IngestionService()

try:
    document = service.ingest(pdf_path)

    assert document.chunks, "No chunks produced"
    assert document.total_chunks == len(document.chunks), "Chunk count mismatch"
except Exception as e:
    print(f"Ingestion failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print(f"Document ID:  {document.document_id}")
print(f"Filename:     {document.filename}")
print(f"Pages:        {document.page_count}")
print(f"Total Chunks: {document.total_chunks}")
print("=" * 60)

for i, chunk in enumerate(document.chunks[:3]):
    print(f"\nChunk {i+1}")
    print(f"  Region: {chunk.region_type}")
    print(f"  Page:   {chunk.page}")
    print(f"  BBox:   ({chunk.bbox.x0:.0f}, {chunk.bbox.y0:.0f}, "
          f"{chunk.bbox.x1:.0f}, {chunk.bbox.y1:.0f})")
    print(f"  Chars:  {chunk.char_count}")
    print(f"  Text:   {chunk.text[:120]}...")


chunk_dir = BASE_DIR / "storage" / "chunks"
chunk_dir.mkdir(parents=True, exist_ok=True)
chunk_files = list(chunk_dir.glob("*.json"))

if not chunk_files:
    raise RuntimeError("No chunk JSON files found")

latest = max(chunk_files, key=lambda p: p.stat().st_mtime)

print(f"\nChunk JSON saved: {latest}")

with open(latest, "r", encoding="utf-8") as f:
    data = json.load(f)

assert data["document_id"] == document.document_id
assert len(data["chunks"]) == document.total_chunks

