
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
