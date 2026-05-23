import time
from tenacity import retry, stop_after_attempt, wait_fixed
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVector, SparseVectorParams, SparseIndexParams,
    HnswConfigDiff,
    Filter, FieldCondition, MatchValue
)
from backend.models.document import DocumentChunk
from backend.core.config import settings
from backend.utils.logger import logger


class VectorStore:
    """
    Manages Qdrant collection lifecycle and chunk upserts.
    Collection uses named vectors — one dense, one sparse.
    """

    COLLECTION = settings.QDRANT_COLLECTION

    def __init__(self):
        self.dense_dim = settings.DENSE_DIM  # BGE-M3 dense output dimension
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self._ensure_collection()
        logger.info(
            "qdrant_connected",
            collection=self.COLLECTION,
            info=self.get_collection_info()
        )

    def _ensure_collection(self):
        """Creates collection if it doesn't exist. Idempotent."""
        if self.client.collection_exists(self.COLLECTION):
            info = self.client.get_collection(self.COLLECTION)
            dense_config = info.config.params.vectors["dense"]
            if dense_config.size != self.dense_dim:
                raise ValueError(
                    f"Existing collection dense dimension mismatch: "
                    f"expected {self.dense_dim}, "
                    f"got {dense_config.size}"
                )
            logger.info(
                "collection_exists",
                name=self.COLLECTION,
                dense_dim=dense_config.size
            )
            return

        self.client.create_collection(
            collection_name=self.COLLECTION,
            vectors_config={
                "dense": VectorParams(
                    size=self.dense_dim,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=16,               # HNSW graph connections
                        ef_construct=100    # build-time accuracy
                    )
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False       # keep in RAM for speed
                    )
                )
            }
        )

        # Add payload index for fast filtering/deletion
        try:
            self.client.create_payload_index(
                collection_name=self.COLLECTION,
                field_name="document_id",
                field_schema="keyword"
            )
        except Exception as e:
                logger.debug("payload_index_skipped", error=str(e))
        
        logger.info("collection_created",
                    name=self.COLLECTION,
                    dense_dim=self.dense_dim)
        
    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: dict) -> int:
        """Public entry point. Tracks true end-to-end latency including retries."""
        start_time = time.time()
        result = self._upsert_with_retry(chunks, embeddings)
        logger.info(
            "upsert_total_latency",
            seconds=time.time() - start_time,
            chunks=result,
            collection=self.COLLECTION,
            attempts_possible=3
        )
        return result

    @retry(
            stop=stop_after_attempt(3),
            wait=wait_fixed(2),
            reraise=True
        )    
    def _upsert_with_retry(self, chunks: list[DocumentChunk], embeddings: dict) -> int:
        """Retried worker. Latency here reflects only the successful attempt."""
        if not isinstance(embeddings, dict):
            raise TypeError("Embeddings must be dict with 'dense' and 'sparse'")

        dense_embeddings = embeddings.get("dense")
        sparse_embeddings = embeddings.get("sparse")

        if not dense_embeddings or not sparse_embeddings:
            raise ValueError("Missing dense or sparse embeddings")

        if len(chunks) != len(dense_embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) "
                f"!= dense embedding count ({len(dense_embeddings)})"
            )

        if len(chunks) != len(sparse_embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) "
                f"!= sparse embedding count ({len(sparse_embeddings)})"
            )

        points = []
        MAX_SPARSE_FEATURES = 512

        for i, chunk in enumerate(chunks):
            if not isinstance(chunk.chunk_id, (str, int)):
                raise ValueError("invalid chunk_id type")

            items = [
                (int(k), float(v))
                for k, v in sparse_embeddings[i].items()
            ]
            items = sorted(items, key=lambda x: x[1], reverse=True)[:MAX_SPARSE_FEATURES]
            items = sorted(items, key=lambda x: x[0])

            sparse_indices = [k for k, _ in items]
            sparse_values  = [v for _, v in items]
            dense_vector   = dense_embeddings[i]

            if len(dense_vector) != self.dense_dim:
                raise ValueError(
                    f"Dense vector dimension mismatch: "
                    f"expected {self.dense_dim}, got {len(dense_vector)}"
                )

            points.append(PointStruct(
                id=str(chunk.chunk_id),
                vector={
                    "dense": dense_vector,
                    "sparse": SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    )
                },
                payload={
                    "chunk_id":     chunk.chunk_id,
                    "document_id":  chunk.document_id,
                    "text":         chunk.text,
                    "page":         chunk.page,
                    "bbox": {
                        "x0": chunk.bbox.x0,
                        "y0": chunk.bbox.y0,
                        "x1": chunk.bbox.x1,
                        "y1": chunk.bbox.y1
                    },
                    "region_type":  chunk.region_type,
                    "char_count":   chunk.char_count,
                    "source":       chunk.source
                }
            ))

        attempt_start = time.time()
        try:
            self.client.upsert(
                collection_name=self.COLLECTION,
                points=points,
                wait=True
            )
        except Exception as e:
            logger.exception(
                "qdrant_upsert_failed",
                error=str(e),
                collection=self.COLLECTION
            )
            raise

        logger.info(
            "upsert_attempt_latency",          # per-attempt latency, distinct from total
            seconds=time.time() - attempt_start,
            chunks=len(points)
        )
        return len(points)

    def dense_search(
        self,
        query_vector: list[float],
        top_k: int,
        score_threshold: float | None = None
    ) -> list:

        if len(query_vector) != self.dense_dim:
            raise ValueError(
                f"Query vector dimension mismatch: "
                f"expected {self.dense_dim}, got {len(query_vector)}"
            )

        response = self.client.query_points(
            collection_name=self.COLLECTION,
            query=query_vector,   # raw vector (stable API)
            using="dense",        # named vector selector
            limit=top_k,
            with_payload=True,
            score_threshold=score_threshold
        )

        return self._extract_points(response)

    def sparse_search(
        self,
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int,
        score_threshold: float | None = None
    ) -> list:
        sparse_indices = [int(i) for i in sparse_indices]
        sparse_values = [float(v) for v in sparse_values]

        if not sparse_indices or not sparse_values:
            raise ValueError("Sparse query is empty")

        if len(sparse_indices) != len(sparse_values):
            raise ValueError("Sparse query mismatch: indices != values")

        if len(sparse_indices) < 2:
            raise ValueError("Sparse query too weak (insufficient features)")

        sparse_vector = SparseVector(
            indices=sparse_indices,
            values=sparse_values
        )

        response = self.client.query_points(
            collection_name=self.COLLECTION,
            query=sparse_vector,  # raw sparse vector (stable API)
            using="sparse",       # named sparse vector selector
            limit=top_k,
            with_payload=True,
            score_threshold=score_threshold
        )

        return self._extract_points(response)

    def get_collection_info(self) -> dict:
        info = self.client.get_collection(self.COLLECTION)
        return {
            "points_count": info.points_count,
            "status": str(info.status)
        }

    def _extract_points(self, response) -> list:
        result = getattr(response, "result", None)
        if result is not None and hasattr(result, "points"):
            return result.points

        if hasattr(response, "points"):
            return response.points

        # Neither shape matched — this is a client/API contract violation.
        # Returning [] here would silently corrupt search results.
        logger.warning(
            "unexpected_qdrant_response_shape",
            response_type=type(response).__name__,
            available_attrs=[a for a in dir(response) if not a.startswith("_")]
        )
        return []

    def delete_document(self, document_id: str):
        """Removes all chunks belonging to a document."""
        if not document_id:
            raise ValueError("document_id cannot be empty")
        self.client.delete(
            collection_name=self.COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )]
            )
        )
        logger.info("document_deleted", document_id=document_id)



