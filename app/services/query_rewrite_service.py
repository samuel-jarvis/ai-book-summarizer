"""Rewrite retrieval queries and checkpoint them without changing user messages."""

import asyncio
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import AsyncSessionLocal
from app.core.exception import ConflictError, ValidationError
from app.models.conversation import (
    ConversationMessage,
    PromptMessageSnapshot,
    RunStatus,
)
from app.services.conversation_service import ConversationService
from app.services.history_service import ConversationHistory, HistoryService
from app.services.llm.llm_service import LLMService
from app.services.llm.prompts import QUERY_REWRITE_PROMPT


class RewriteOutput(BaseModel):
    """Small, strict response contract so ambiguous output cannot become a query."""

    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["ready", "ambiguous"]
    query: str = Field(max_length=1000)


def validate_rewrite(output: str) -> str | None:
    """Accept a ready, nonempty single-line query; reject unusable model output."""
    try:
        result = RewriteOutput.model_validate_json(output)
    except (PydanticValidationError, TypeError):
        return None
    query = result.query.strip()
    if (
        result.status != "ready"
        or not query
        or not any(char.isalnum() for char in query)
    ):
        return None
    if any(
        ord(char) < 32 or ord(char) == 127 or char in "\u2028\u2029"
        for char in result.query
    ):
        return None
    return query


@dataclass(frozen=True)
class RewriteResult:
    """Search input and audit details, separate from the original user message."""

    query: str
    history: ConversationHistory
    prompt_messages: list[PromptMessageSnapshot]
    model: str | None
    fallback: bool = False


class QueryRewriteService:
    """Use the existing LLM abstraction; no provider call is needed without history."""

    def __init__(self, llm: LLMService | None = None) -> None:
        """Allow a test LLM while defaulting to DeepSeek at temperature zero."""
        self.llm = llm if llm is not None else LLMService(temperature=0)

    async def rewrite(
        self, question: str, history: ConversationHistory
    ) -> RewriteResult:
        """Rewrite with context, falling back to the exact question on failure."""
        if not question.strip():
            raise ValidationError("Query must contain non-whitespace text.")
        if not history.messages:
            return RewriteResult(question, history, [], None)
        values = {
            "question": question,
            "history": [
                (message.role, message.content) for message in history.messages
            ],
        }
        prompt_messages = [
            PromptMessageSnapshot(role=message.type, content=message.content)
            for message in QUERY_REWRITE_PROMPT.format_messages(**values)
        ]
        try:
            output = await self.llm.generate(QUERY_REWRITE_PROMPT, values)
            query = validate_rewrite(output)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- any provider failure uses the raw question.
            # Retrieval can continue; provider details must never enter the run.
            query = None
        return RewriteResult(
            query if query is not None else question,
            history,
            prompt_messages,
            self.llm.model,
            fallback=query is None,
        )

    async def rewrite_run(
        self,
        run_id: UUID,
        user_id: UUID,
        *,
        sessions: async_sessionmaker[AsyncSession] = AsyncSessionLocal,
        token_budget: int = 2000,
        max_exchanges: int = 20,
    ) -> RewriteResult:
        """Load history, release the DB session, rewrite, and save on a running attempt.

        The later checkpoint rechecks status under a conversation lock, so
        cancellation or failure during inference rejects a stale write.
        """
        async with sessions() as db:
            service = ConversationService(db, user_id)
            run = await service.get_run(run_id)
            if run.status != RunStatus.RUNNING:
                raise ConflictError("Only a running attempt can rewrite its query.")
            message = await db.get(ConversationMessage, run.user_message_id)
            question = message.content
            history = await HistoryService(db, user_id).load(
                run.conversation_id,
                before_position=message.position,
                token_budget=token_budget,
                max_exchanges=max_exchanges,
            )
        result = await self.rewrite(question, history)
        async with sessions.begin() as db:
            await ConversationService(db, user_id).checkpoint(
                run_id, stage="rewritten", rewrite_result=result
            )
        return result
