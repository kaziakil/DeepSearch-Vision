"""
Day 3 validation script.
Tests: ingest PDF → embed → store → query → ranked results
Run: python scripts/test_retrieval.py <path_to_pdf>
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.ingestion import IngestionService
from backend.services.retrieval import RetrievalService
from backend.utils.logger import logger, setup_logging

setup_logging(debug=False)

pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 \
    else Path("datasets/raw/sample.pdf")

if not pdf_path.exists():
    print(f"File not found: {pdf_path}")
    sys.exit(1)

# Clean slate — delete existing collection before re-ingesting
from backend.services.vector_store import VectorStore
_store = VectorStore()
if _store.client.collection_exists(_store.COLLECTION):
    _store.client.delete_collection(_store.COLLECTION)
    print("Flushed existing collection")

# Ingest
print(f"\nIngesting: {pdf_path.name}")
ingestion = IngestionService()
document = ingestion.ingest(pdf_path)
print(f"Indexed {document.total_chunks} chunks from {document.page_count} pages")

# Retrieval
retrieval = RetrievalService()

# Sequential consistency check — BEFORE warmup
print("\nWaiting for Qdrant indexing to stabilize...")
expected = document.total_chunks
timeout = 15
start = time.time()
count = 0   # initialized before the loop — prevents NameError if get_collection_info raises

while time.time() - start < timeout:
    try:
        count = retrieval.store.get_collection_info()["points_count"]
        if count >= expected:
            break
    except Exception as e:
        print(f"Qdrant not ready yet: {e}")
    time.sleep(0.5)

print(f"Qdrant ready: {count}/{expected} points indexed")

# Warmup query — AFTER index stabilization
_ = retrieval.search("warmup query", top_k=1)

# Query loop
test_queries = [
    # Questions from "Scientific Advertising by Claude C. Hopkins"
    "What is the role of headlines in advertising according to Hopkins?"
    
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print("="*60)

    t0 = time.time()

    try:
        results = retrieval.search(query, top_k=3) or []
        latency = (time.time() - t0) * 1000

        if not results:
            print("No results returned")
            continue

        for i, result in enumerate(results):
            reranker_score = result.reranker_score or 0.0
            rrf_score = result.rrf_score or 0.0
            page = result.page or -1
            region = result.region_type or "unknown"
            text = result.text or ""

            print(f"\nRank {i+1}")
            print(f"  Reranker Score: {reranker_score:.4f}")
            print(f"  RRF Score:      {rrf_score:.4f}")
            print(f"  Page:           {page}")
            print(f"  Region:         {region}")
            # print(f"  Text:           {text[:150]}...")
            # print(f"  Text:           {text}")
            print(f"  Text:           {text[:300]}...")



        # Observability
        query_tokens = set(query.lower().split())

        def safe_text(r):
            return (getattr(r, "text", "") or "").lower()

        def safe_score(r):
            return float(getattr(r, "reranker_score", 0.0) or 0.0)

        overlaps = []
        for r in results:
            chunk_text = safe_text(r)       # renamed — no longer shadows outer `text`
            overlap = sum(1 for t in query_tokens if t in chunk_text)
            overlaps.append(overlap / max(len(query_tokens), 1))

        avg_overlap = sum(overlaps) / max(len(overlaps), 1)

        scores = [safe_score(r) for r in results]
        score_spread = max(scores) - min(scores) if scores else 0.0

        unique_chunks = len(set(getattr(r, "chunk_id", None) for r in results))

        print("\n--- Observability ---")
        print(f"Relevance proxy: {avg_overlap:.3f}")
        print(f"Score spread:    {score_spread:.3f}")
        print(f"Unique chunks:   {unique_chunks}/{len(results)}")

        if avg_overlap < 0.2:
            print("Low retrieval relevance")
        if score_spread < 0.5:
            print("Reranker not differentiating results")
        if unique_chunks < len(results):
            print("Duplicate chunks detected")

        print(f"\nLatency: {latency:.0f}ms")

    except Exception as e:
        print(f"Query failed: {e}")
        continue

