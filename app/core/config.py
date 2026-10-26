
from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    PROJECT_NAME: str
    API_V1_PREFIX: str = "/api/v1"

    DEEPSEEK_API_KEY: str
    DEEPSEEK_BASE_URL: str = 'https://api.deepseek.com'
    DATABASE_URL: str
    DEBUG: bool = False
    ENVIRONMENT: Literal["development",
                         "staging", "production"] = "development"
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:3000", "http://localhost:8000"
    ]
    REDIS_URL: str
    SECRET_KEY: str

    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    ACCESS_COOKIE_NAME: str = "access_token"
    REFRESH_COOKIE_NAME: str = "refresh_token"

    DAILY_SUMMARY_LIMIT: int = 3

    # Character-based knobs for the legacy processor_service pipeline.
    SUMMARY_CHUNK_SIZE: int = 20000
    SUMMARY_CHUNK_OVERLAP: int = 400
    SUMMARY_MAX_CONCURRENCY: int = 15

    CHUNK_ENCODING: str = "cl100k_base"
    SUMMARY_CHUNK_TOKENS: int = 6000
    SUMMARY_CHUNK_OVERLAP_TOKENS: int = 200

    RAG_CHUNK_TOKENS: int = 800
    RAG_CHUNK_OVERLAP_TOKENS: int = 120

    # Embedding / pgvector tuning. Must match the embedding model's output dim.
    # Default 1536 matches OpenAI's text-embedding-3-small.
    EMBEDDING_DIMENSION: int = 1536
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    CONVERSATION_MODEL: str = "deepseek-v4-flash"

    DEEPSEEK_TIMEOUT_SECONDS: float = 120.0
    DEEPSEEK_REDUCE_TIMEOUT_SECONDS: float = 300.0
    DEEPSEEK_MAX_RETRIES: int = 3

    @property
    def rag_chunking_version(self) -> str:
        """Identifies how a stored generation was split.

        Stamped onto every EmbeddingGeneration, so changing the sizes above
        makes existing generations identifiable as stale. Bump the trailing
        version when the splitting strategy changes, not just its sizes.
        """
        return (
            f"rag-{self.CHUNK_ENCODING}-{self.RAG_CHUNK_TOKENS}"
            f"-{self.RAG_CHUNK_OVERLAP_TOKENS}-v1"
        )

    @property
    def cookie_secure(self) -> bool:
        """HTTPS-only cookies everywhere but local development over http."""
        return self.ENVIRONMENT != "development"

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def use_asyncpg_driver(cls, value: str) -> str:
        """Convert provider PostgreSQL URLs to SQLAlchemy's asyncpg URL."""
        if value.startswith("postgres://"):
            return value.replace("postgres://", "postgresql+asyncpg://", 1)
        if value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+asyncpg://", 1)
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore


settings = get_settings()
