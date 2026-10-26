from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from app.core.config import settings


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    """Build the shared client on first use; no API calls occur here."""
    if not settings.OPENAI_API_KEY:
        raise ValueError("Set OPENAI_API_KEY before generating embeddings.")
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dimensions=settings.EMBEDDING_DIMENSION,
        api_key=SecretStr(settings.OPENAI_API_KEY)
    )


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed chunk contents asynchronously, preserving their order."""
    if not texts:
        return []
    return await get_embeddings().aembed_documents(texts)


async def embed_query(text: str) -> list[float]:
    """Embed a search query using the same model as stored chunks."""
    return await get_embeddings().aembed_query(text)
