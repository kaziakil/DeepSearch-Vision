"""
Day 1 validation script.
Tests: PDF load -> chunk -> embed -> store -> query -> retrieve
Run: python scripts/test_pipeline.py
"""

import sys
import json
import uuid
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVector, SparseVectorParams, SparseIndexParams,
    NamedVector, NamedSparseVector
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.embedding import embedding_service
from backend.core.config import settings
from backend.utils.logger import logger, setup_logging

setup_logging(debug=True)


# 1. Synthetic test chunks (replace with real PDF next) 
test_chunks = [
    {
        "chunk_id": str(uuid.uuid4()),
        "text": "The quarterly revenue increased by 23% driven by strong Saas growth.",
        "page": 1,
        "bbox": [50, 100, 500, 130]
    },
    {
        "chunk_id": str(uuid.uuid4()),
        "text": "Customer acquisition cost decreased from $420 to $310 this quarter.",
        "page": 2,
        "bbox": [50, 200, 500, 230]
    },
    {
        "chunk_id": str(uuid.uuid4()),
        "text": "Net retention rate reached 118%, indicating strong exapnsion revenue.",
        "page": 2,
        "bbox": [50, 300, 500, 330]
    },
]


# 2. Generate embeddings
logger.info("generating_embeddings", num_chunks=len(test_chunks))
texts = [c["text"] for c in test_chunks]
embeddings = embedding_service.embed(texts)
logger.info("embeddings_generated",
            dense_dim=len(embeddings["dense"][0]),
            sparse_count=len(embeddings["sparse"][0]))


# 3. Connect to Qdrant and create collection
client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
collection = settings.QDRANT_COLLECTION + "_test"

if client.collection_exists(collection):
    client.delete_collection(collection)

dense_dim = len(embeddings["dense"][0])

client.create_collection(
    collection_name=collection,
    vectors_config= {
        "dense": VectorParams(size=dense_dim, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(index=SparseIndexParams())
    }
)
logger.info("collection_created", name=collection, dense_dim=dense_dim)


# 4. Upsert points
points = []
for i, chunk in enumerate(test_chunks):
    sparse_weights = embeddings["sparse"][i]
    sparse_indices = [int(k) for k in sparse_weights.keys()]
    sparse_values = [float(v) for v in sparse_weights.values()]

    points.append(PointStruct(
        id=str(chunk["chunk_id"]),
        vector={
            "dense": embeddings["dense"][i],
            "sparse": SparseVector(
                indices=sparse_indices,
                values=sparse_values
            )
        },
        payload={
            "text": chunk["text"],
            "page": chunk["page"],
            "bbox": chunk["bbox"]
        }
    ))
 
client.upsert(collection_name=collection, points=points)
logger.info("chunks_stored", count=len(points))


# 5. Query
query = "What happended to revenue this quarter?"
logger.info("running_query", query=query)

query_embedding = embedding_service.embed_query(query)

# Dense search
dense_results = client.query_points(
    collection_name=collection,
    query=query_embedding["dense"][0],
    using="dense",
    limit=3,
    with_payload=True
).points

# Sparse search
sparse_weights = query_embedding["sparse"][0]
sparse_results = client.query_points(
    collection_name=collection,
    query=SparseVector(
        indices=[int(k) for k in sparse_weights.keys()],
        values=[float(v) for v in sparse_weights.values()]
    ),
    using="sparse",
    limit=3,
    with_payload=True
).points


# 6. RRF Fusion
def reciprocal_rank_fusion(results_list: list, k: int = 60) -> list:
    scores = {}
    payloads = {}
    for results in results_list:
        for rank, result in enumerate(results):
            doc_id = result.id
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            payloads[doc_id] = result.payload
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"id": doc_id, "score": score, "payload": payloads[doc_id]}
            for doc_id, score in ranked]

fused = reciprocal_rank_fusion([dense_results, sparse_results])


# Print results
print("\n" + "="*60)
print(f"Query: {query}")
print("="*60)
for i, result in enumerate(fused[:3]):
    print(f"\nRank {i+1} | RRF Score: {result['score']:.4f}")
    print(f"Text: {result['payload']['text']}")
    print(f"Page: {result['payload']['page']}")
    print(f"BBox: {result['payload']['bbox']}")


# cleanup
client.delete_collection(collection)
logger.info("pipeline_test_completed")

