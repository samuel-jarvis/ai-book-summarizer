"""Narrow over-retrieved chunks to the few that best answer the question."""

import asyncio

from langchain_core.documents import Document as LangChainDocument
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from app.core.config import settings
from app.services.llm.llm_service import LLMService
from app.services.llm.prompts import RERANK_PROMPT
from app.services.retrieval_service import (
    RetrievalResult,
    candidate_snapshots,
    to_source,
)


class RankingOutput(BaseModel):
    """Strict response contract so malformed output can never choose chunks."""

    model_config = ConfigDict(extra="forbid", strict=True)
    ranking: list[int] = Field(min_length=1, max_length=100)


def validate_ranking(output: str, count: int, keep: int) -> list[int] | None:
    """Return up to `keep` zero-based chunk indexes, best first, or None if unusable."""
    try:
        parsed = RankingOutput.model_validate_json(output)
    except (PydanticValidationError, TypeError):
        return None
    numbers = parsed.ranking
    if len(set(numbers)) != len(numbers) or any(not 1 <= n <= count for n in numbers):
        return None
    return [n - 1 for n in numbers[:keep]]


def format_passages(documents: list[LangChainDocument]) -> str:
    """Number each chunk with its book and pages so the model can compare them."""
    parts = []
    for number, doc in enumerate(documents, start=1):
        metadata = doc.metadata
        start, end = metadata.get("page_start"), metadata.get("page_end")
        where = f", pp. {start + 1}-{end + 1}" if start is not None and end is not None else ""
        parts.append(f"[{number}] ({metadata['document_title']}{where})\n{doc.page_content}")
    return "\n\n".join(parts)


class RerankService:
    """Use the existing LLM abstraction; one call over all candidates."""

    def __init__(self, llm: LLMService | None = None) -> None:
        """Allow a test LLM while defaulting to DeepSeek at temperature zero."""
        self.llm = llm if llm is not None else LLMService(temperature=0)

    async def rerank(
        self, query: str, retrieval: RetrievalResult, *, keep: int | None = None
    ) -> RetrievalResult:
        """Keep the `keep` best chunks, relabelled S1.. best first.

        The model's picks come first; if it names fewer than `keep`, or its output is
        unusable, or the provider fails, the nearest remaining chunks fill the rest
        (so a failure behaves like plain nearest-k). Every original hit stays in the
        result's candidates for audit. Provider text is never stored.
        """
        keep = settings.RAG_RERANK_KEEP if keep is None else keep
        documents = retrieval.documents
        if len(documents) <= keep:
            return retrieval
        try:
            output = await self.llm.generate(
                RERANK_PROMPT,
                {
                    "keep": keep,
                    "question": query,
                    "passages": format_passages(documents),
                },
            )
            picked = validate_ranking(output, len(documents), keep)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- any provider failure falls back to nearest-k.
            picked = None
        order = list(picked or [])
        order += [i for i in range(len(documents)) if i not in order][: keep - len(order)]
        chosen = [documents[i] for i in order]
        return RetrievalResult(
            documents=chosen,
            sources=[to_source(doc, f"S{n}") for n, doc in enumerate(chosen, start=1)],
            candidates=candidate_snapshots(retrieval.sources),
        )
