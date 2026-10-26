"""Generate an answer from captured context and publish it only after success."""

import asyncio
from uuid import UUID

from langchain_core.messages import BaseMessage, convert_to_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import AsyncSessionLocal
from app.core.exception import AppError, ConflictError, NotFoundError
from app.models.conversation import RunMetrics
from app.services.conversation_service import ConversationService
from app.services.llm.llm_service import LLMService


class AnswerGenerationError(AppError):
    """A safe generation failure; provider exception details never leave this service."""

    status_code = 502

    def __init__(self, code: str) -> None:
        """Expose an application-defined error code and a generic retry message."""
        self.code = code
        super().__init__("Could not generate a complete answer. Please retry.")


def answer_text(response: BaseMessage) -> str:
    """Reject incomplete or empty output instead of publishing a partial answer."""
    finish_reason = response.response_metadata.get("finish_reason")
    if finish_reason == "length":
        raise AnswerGenerationError("answer_truncated")
    if finish_reason == "content_filter":
        raise AnswerGenerationError("answer_filtered")
    if (
        finish_reason not in (None, "stop")
        or response.additional_kwargs.get("refusal")
        or getattr(response, "tool_calls", None)
    ):
        raise AnswerGenerationError("answer_unusable")
    text = StrOutputParser().invoke(response)
    if not text.strip():
        raise AnswerGenerationError("answer_empty")
    return text


class AnswerGenerationService:
    """Use the stored prompt and short transactions around a single model call."""

    def __init__(self, llm: LLMService | None = None) -> None:
        """Allow an injected LLM; otherwise use the model recorded on the run."""
        self.llm = llm

    async def generate_run(
        self,
        run_id: UUID,
        user_id: UUID,
        *,
        sessions: async_sessionmaker[AsyncSession] = AsyncSessionLocal,
    ) -> str:
        """Claim, generate from saved messages, and atomically publish a completed answer."""
        async with sessions.begin() as db:
            run = await ConversationService(db, user_id).claim_generation(
                run_id, model=self.llm.model if self.llm is not None else None
            )
            # Literal messages preserve braces and the exact input captured earlier.
            prompt = ChatPromptTemplate.from_messages(
                convert_to_messages(
                    [
                        (message.role, message.content)
                        for message in run.snapshot.prompt_messages
                    ]
                )
            )
            max_tokens = run.snapshot.answer_token_budget
            model = run.model
            metrics = run.metrics.model_copy(deep=True)

        llm = self.llm if self.llm is not None else LLMService(model=model)
        try:
            try:
                response = await llm.generate_response(
                    prompt, {}, max_tokens=max_tokens, user_id=str(user_id)
                )
                usage = getattr(response, "usage_metadata", None) or {}
                metadata = response.response_metadata
                metrics = RunMetrics.model_validate(
                    {
                        **metrics.model_dump(),
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "finish_reason": metadata.get("finish_reason"),
                        "provider_request_id": metadata.get("request_id"),
                    }
                )
                text = answer_text(response)
            except AnswerGenerationError as exc:
                await self._terminate_run(run_id, user_id, sessions, metrics, exc.code)
                raise
            except Exception:  # noqa: BLE001 -- translate all provider failures to a safe code.
                await self._terminate_run(
                    run_id, user_id, sessions, metrics, "answer_generation_failed"
                )
                raise AnswerGenerationError("answer_generation_failed") from None

            async with sessions.begin() as db:
                await ConversationService(db, user_id).complete(
                    run_id, text, metrics=metrics
                )
            return text
        except asyncio.CancelledError:
            await self._terminate_run(run_id, user_id, sessions, metrics, None)
            raise

    async def _terminate_run(
        self,
        run_id: UUID,
        user_id: UUID,
        sessions: async_sessionmaker[AsyncSession],
        metrics: RunMetrics,
        error_code: str | None,
    ) -> None:
        """Save safe failure metadata, or cancel, without changing an already-finished run."""
        async with sessions.begin() as db:
            service = ConversationService(db, user_id)
            try:
                await service.checkpoint(run_id, stage="generating", metrics=metrics)
                if error_code is None:
                    await service.cancel(run_id)
                else:
                    await service.fail(run_id, error_code)
            except (ConflictError, NotFoundError):
                # A concurrent cancellation, completion, or deletion takes precedence.
                return
