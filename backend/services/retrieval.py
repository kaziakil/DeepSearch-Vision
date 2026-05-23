from FlagEmbedding import FlagReranker
from backend.services.vector_store import VectorStore
from backend.services.embedding import embedding_service
from backend.models.retrieval import RetrievalResult
from backend.core.config import settings
from backend.utils.logger import logger
from backend.models.document import BoundingBox
import time


class RetrievalService:
    """
    Hybrid retrieval pipeline:
    1. Dense search (BGE-M3 cosine similarity)
    2. Sparse search (BGE-M3 lexical weights)
    3. RRF fusion
    4. Cross-encoder reranking
    """

    def __init__(self, store: VectorStore = None):
        self.store = store or VectorStore()
        self._reranker = None           # lazy load — saves RAM at startup

    @property
    def reranker(self) -> FlagReranker:
        """Lazy-load reranker only when first needed."""
        if self._reranker is None:
            logger.info("loading_reranker", model=settings.RERANKER_MODEL)
            self._reranker = FlagReranker(
                settings.RERANKER_MODEL,
                use_fp16=settings.USE_FP16,
            )
            logger.info("reranker_loaded")
        return self._reranker

    def search(self, query: str, top_k: int = None) -> list[RetrievalResult]:
        """
        Full hybrid retrieval pipeline.
        Returns top_k results ranked by reranker score.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        top_k = top_k or settings.TOP_K_FINAL
        t_start = time.time()

        # Embed query 
        query_embeddings = embedding_service.embed_query(query)

        # validate embedding response 
        if "dense" not in query_embeddings or "sparse" not in query_embeddings:
            raise ValueError("Embedding service returned invalid structure")
        
        if not query_embeddings["dense"]:
            raise ValueError("Dense query embedding missing")
        
        if not query_embeddings["sparse"]:
            raise ValueError("Sparse query embedding missing")
                

        dense_vec = query_embeddings["dense"][0]
        sparse_weights = query_embeddings["sparse"][0]

        sparse_items = [
            (int(k), float(v))
            for k, v in sparse_weights.items()
        ]

        sparse_indices = [k for k, _ in sparse_items]
        sparse_values = [v for _, v in sparse_items]

        # Dual search 
        dense_results = self.store.dense_search(
            dense_vec, settings.TOP_K_DENSE)
        sparse_results = self.store.sparse_search(
            sparse_indices, sparse_values, settings.TOP_K_SPARSE)

        # RRF fusion 
        fused = self._reciprocal_rank_fusion(
            [dense_results, sparse_results])

        if not fused:
            return []

        # Rerank top candidates 
        # Only rerank top 20 — reranker is expensive
        candidates = fused[:20]
        pairs = [
            [
                query,
                c["payload"]["text"][:settings.MAX_RERANK_CHARS]
            ]
            for c in candidates
        ]
        scores = self.reranker.compute_score(pairs)
        
        if isinstance(scores, float):
            scores = [scores]

        # Attach reranker scores
        for candidate, score in zip(candidates, scores):
            candidate["reranker_score"] = float(score)

        # Sort by reranker score — this is the final ranking
        reranked = sorted(candidates,
                          key=lambda x: x["reranker_score"],
                          reverse=True)

        latency_ms = (time.time() - t_start) * 1000
        logger.info("retrieval_complete",
                    query=query[:60],
                    results=len(reranked[:top_k]),
                    latency_ms=round(latency_ms, 2))

        return [
            RetrievalResult(
                chunk_id=r["payload"]["chunk_id"],
                document_id=r["payload"]["document_id"],
                text=r["payload"]["text"],
                page=r["payload"]["page"],
                bbox=r["payload"] ["bbox"],
                region_type=r["payload"]["region_type"],
                rrf_score=r["rrf_score"],
                reranker_score=r["reranker_score"],
                source=r["payload"]["source"]
            )
            for r in reranked[:top_k]
        ]

    def _reciprocal_rank_fusion(self, results_list: list,
                                k: int = 60) -> list:
        """
        Merges multiple ranked lists into one via RRF.
        k=60 is the standard constant — dampens high ranks.
        """

        scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        for results in results_list:
            for rank, result in enumerate(results):
                if not result.payload:
                    continue
                doc_id = str(result.id)
                scores[doc_id] = scores.get(doc_id, 0.0) + \
                    1.0 / (k + rank + 1)
                payloads[doc_id] = result.payload

        ranked = sorted(scores.items(),
                        key=lambda x: x[1], reverse=True)
        return [
            {
                "id": doc_id,
                "rrf_score": score,
                "reranker_score": 0.0,
                "payload": payloads[doc_id]
            }
            for doc_id, score in ranked
        ]
    

