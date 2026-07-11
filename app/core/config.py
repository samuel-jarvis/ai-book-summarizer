
from functools import lru_cache
from typing import Literal
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


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore


settings = get_settings()
