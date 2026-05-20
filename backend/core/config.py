from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "DeepSearch-Vision"
    DEBUG: bool = True

    STORAGE_RAW: Path = Path("storage/raw")
    STORAGE_CHUNKS: Path = Path("storage/chunks")

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "deepsearch"

    EMBEDDING_MODEL: str = "BAAI-bge-m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    VLM_MODEL: str = "Qwen/Qwen2-VL-2B-Instruct"

    TOP_K_DENSE: int = 20
    TOP_K_SPARSE: int = 20
    TOP_K_FINAL: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore", # Ignore unknown env vars
    )

    def ensure_dirs(self):
        self.STORAGE_RAW.mkdir(parents=True, exist_ok=True)
        self.STORAGE_CHUNKS.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_dirs()

