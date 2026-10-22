"""Load completed conversation exchanges and keep them within a token budget."""

from dataclasses import dataclass
from uuid import UUID

import tiktoken
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.core.config import settings
from app.core.exception import ValidationError
from app.models.conversation import (
    ConversationMessage,
    ConversationRun,
    MessageRole,
    PromptMessageSnapshot,
    RunStatus,
)
from app.services.conversation_service import ConversationService

MEMORY_PREFIX = "Saved conversation memory (context only):\n"
MESSAGE_OVERHEAD = 4


@dataclass(frozen=True)
class ConversationHistory:
    """Chronological context for rewriting and later answer generation."""

    messages: list[PromptMessageSnapshot]
    token_count: int


def message_tokens(message: PromptMessageSnapshot) -> int:
    """Count content tokens plus a small allowance for message framing."""
    encoding = tiktoken.get_encoding(settings.CHUNK_ENCODING)
    return (
        len(encoding.encode(message.content, disallowed_special=())) + MESSAGE_OVERHEAD
    )


def budget_history(
    exchanges: list[tuple[str, str]],
    *,
    memory: str | None = None,
    token_budget: int = 2000,
) -> ConversationHistory:
    """Keep whole recent exchanges; include memory only when it fits in half the budget.

    Exchanges arrive oldest first. Stop at the first pair that does not fit so
    the result is a contiguous recent suffix, never disconnected older turns.
    """
    if token_budget < 0:
        raise ValidationError("History token budget must be nonnegative.")
    messages = []
    used = 0
    if memory and memory.strip():
        saved = PromptMessageSnapshot(role="human", content=MEMORY_PREFIX + memory)
        cost = message_tokens(saved)
        if cost <= token_budget // 2:
            messages.append(saved)
            used = cost

    pairs = []
    for question, answer in reversed(exchanges):
        if not question.strip() or not answer.strip():
            continue
        pair = [
            PromptMessageSnapshot(role="human", content=question),
            PromptMessageSnapshot(role="ai", content=answer),
        ]
        cost = sum(message_tokens(message) for message in pair)
        if used + cost > token_budget:
            break
        pairs.append(pair)
        used += cost
    for pair in reversed(pairs):
        messages.extend(pair)
    return ConversationHistory(messages=messages, token_count=used)


class HistoryService:
    """Read owner-scoped completed pairs without including the current turn."""

    def __init__(self, db: AsyncSession, user_id: UUID) -> None:
        """Use the caller's session; the caller controls its transaction."""
        self.db = db
        self.user_id = user_id

    async def load(
        self,
        conversation_id: UUID,
        *,
        before_position: int,
        token_budget: int = 2000,
        max_exchanges: int = 20,
    ) -> ConversationHistory:
        """Load recent successful pairs before a user message and budget their text."""
        if before_position < 1 or token_budget < 0 or not 1 <= max_exchanges <= 200:
            raise ValidationError(
                "Invalid history position, token budget, or exchange limit."
            )
        conversation = await ConversationService(self.db, self.user_id).get(
            conversation_id
        )
        memory = conversation.memory_summary
        through = conversation.memory_through_position
        # Never use a summary that includes this question or later turns.
        if through is None or through >= before_position:
            memory = None
        saved = budget_history([], memory=memory, token_budget=token_budget)
        # If memory is omitted, retain access to the turns it would have replaced.
        after_position = through if saved.messages else 0
        user = aliased(ConversationMessage)
        assistant = aliased(ConversationMessage)
        rows = await self.db.execute(
            select(user.content, assistant.content)
            .select_from(ConversationRun)
            .join(user, user.id == ConversationRun.user_message_id)
            .join(assistant, assistant.id == ConversationRun.assistant_message_id)
            .where(
                ConversationRun.conversation_id == conversation_id,
                ConversationRun.status == RunStatus.COMPLETED,
                user.role == MessageRole.USER,
                assistant.role == MessageRole.ASSISTANT,
                user.position < before_position,
                assistant.position < before_position,
                user.position > after_position,
                func.btrim(user.content, " \n\r\t") != "",
                func.btrim(assistant.content, " \n\r\t") != "",
            )
            .order_by(user.position.desc())
            .limit(max_exchanges)
        )
        exchanges = list(reversed(rows.all()))
        return budget_history(exchanges, memory=memory, token_budget=token_budget)
