import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from backend.core.config import settings
from backend.utils.logger import logger


class EmbeddingService:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("loading_embedding_model",
                    model=settings.EMBEDDING_MODEL,
                    device=self.device)
        
        # BGE-M3: single model for dense + sparse + colbert
        self.model = BGEM3FlagModel(
            settings.EMBEDDING_MODEL,
            use_fp16=True,   # cuts RAM ~50%, minimal quality loss
            device=self.device 
        )
        logger.info("embedding_model_loaded")

    def embed(self, texts: list[str]) -> dict:
        """
        Returns dense vectors + sparse lexical weight.
        Both used for hybrid retrieval via RRF.
        """
        if not texts:
            return {"dense": [], "sparse": []}
        
        output = self.model.encode(
            texts,
            batch_size=4, # conservative for 8GM RAM
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False # skip colbert for V1, saves RAM
        )

        return {
            "dense": output["dense_vecs"].tolist(),
            "sparse": output["lexical_weights"] # list of {token: weight} dicts
        }
    
    def embed_query(self, query: str) -> dict:
        return self.embed([query])
        
# Singleton – load once, reuse across requests
embedding_service = EmbeddingService()


