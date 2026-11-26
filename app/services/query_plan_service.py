"""Split a multi-part question into focused search queries before retrieval."""

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from app.core.config import settings
from app.services.llm.llm_service import LLMService
from app.services.llm.prompts import QUERY_SPLIT_PROMPT

MAX_QUERY_CHARS = 500


class PlanOutput(BaseModel):
    """Strict response contract so malformed output can never become a search."""

    model_config = ConfigDict(extra="forbid", strict=True)
    queries: list[str] = Field(min_length=1, max_length=10)


def validate_plan(output: str, max_queries: int) -> list[str] | None:
    """Return up to `max_queries` clean, distinct single-line queries, or None if unusable."""
    try:
        parsed = PlanOutput.model_validate_json(output)
    except (PydanticValidationError, TypeError):
        return None
    queries: list[str] = []
    for raw in parsed.queries:
        query = raw.strip()
        if (
            not query
            or len(query) > MAX_QUERY_CHARS
            or not any(char.isalnum() for char in query)
            or any(
                ord(char) < 32 or ord(char) == 127 or char in "  "
                for char in query
            )
        ):
            return None
        if query.casefold() not in {existing.casefold() for existing in queries}:
            queries.append(query)
    return queries[:max_queries]


@dataclass(frozen=True)
class QueryPlan:
    """The queries retrieval should run, plus audit details."""

    queries: list[str]
    model: str | None
    fallback: bool = False


class QueryPlanService:
    """Use the existing LLM abstraction; a single-topic question costs one short call."""

    def __init__(self, llm: LLMService | None = None) -> None:
        """Allow a test LLM while defaulting to DeepSeek at temperature zero."""
        self.llm = llm if llm is not None else LLMService(temperature=0)

    async def plan(self, query: str, *, max_queries: int | None = None) -> QueryPlan:
        """Split `query` into focused searches, falling back to the query itself.

        One returned query means a single topic, and the original text is kept
        rather than the model's paraphrase. Unusable output or a provider error
        also falls back to the original query; provider text is never stored.
        """
        limit = settings.RAG_MAX_SUBQUERIES if max_queries is None else max_queries
        if limit < 2 or not query.strip():
            return QueryPlan([query], None)
        try:
            output = await self.llm.generate(
                QUERY_SPLIT_PROMPT, {"question": query, "max_queries": limit}
            )
            queries = validate_plan(output, limit)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- any provider failure keeps the single query.
            queries = None
        if queries is None:
            return QueryPlan([query], self.llm.model, fallback=True)
        if len(queries) == 1:
            return QueryPlan([query], self.llm.model)
        return QueryPlan(queries, self.llm.model)
