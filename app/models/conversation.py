"""Project conversations, visible messages, and inspectable generation attempts.

Run snapshots are historical evidence: do not rebuild them from current documents.
Assign replacement Pydantic metadata objects when updating JSONB fields.
"""

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base, TimestampMixin
from app.core.types import PydanticJSONB


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SourceSnapshot(BaseModel):
    """A chunk selected into the answer context, with a display excerpt."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    generation_id: uuid.UUID
    document_title: str
    chunk_index: int = Field(ge=0)
    page_start: int | None = None
    page_end: int | None = None
    cosine_distance: float | None = None
    citation_label: str  # e.g. "S1"
    # Leading text of the chunk for the sources panel; the service truncates to fit.
    excerpt: str = Field(max_length=600)


class CandidateSnapshot(BaseModel):
    """A retrieval hit, kept as ids and a score; fetch its text from the chunk row."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    cosine_distance: float
    selected: bool = False


class PromptMessageSnapshot(BaseModel):
    role: str
    content: str


class RunSnapshot(BaseModel):
    """Inputs captured before generation; do not rebuild from current documents."""

    # Budgeted memory and completed exchanges, also available to the context builder.
    history_messages: list[PromptMessageSnapshot] = Field(default_factory=list)
    # Exact input to the query rewriter; empty when no history was available.
    rewrite_prompt_messages: list[PromptMessageSnapshot] = Field(default_factory=list)
    retrieval_candidates: list[CandidateSnapshot] = Field(default_factory=list)
    selected_sources: list[SourceSnapshot] = Field(default_factory=list)
    # Fully rendered, ordered model input: system instructions, history, context.
    prompt_messages: list[PromptMessageSnapshot] = Field(default_factory=list)


class RunMetrics(BaseModel):
    stage_durations_ms: dict[str, float] = Field(default_factory=dict)
    # True when the rewriter output was unusable and retrieval used the raw question.
    rewrite_fallback: bool = False
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    provider_request_id: str | None = None


class Conversation(TimestampMixin, Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    # Ownership is derived from Project.user_id, avoiding conflicting owner columns.
    title: Mapped[str | None] = mapped_column(String(255))
    memory_summary: Mapped[str | None] = mapped_column(Text)
    memory_through_position: Mapped[int | None] = mapped_column(Integer)
    __table_args__ = (
        CheckConstraint(
            "memory_through_position IS NULL OR memory_through_position > 0",
            name="memory_position_positive",
        ),
    )

    messages: Mapped[list["ConversationMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ConversationMessage.position",
    )


class ConversationMessage(TimestampMixin, Base):
    __tablename__ = "conversation_messages"
    __table_args__ = (
        # Named explicitly: the naming convention uses only the first column, so
        # these two would otherwise collide on uq_conversation_messages_conversation_id.
        UniqueConstraint(
            "conversation_id", "position", name="uq_conversation_messages_position"
        ),
        UniqueConstraint(
            "conversation_id",
            "client_request_id",
            name="uq_conversation_messages_client_request_id",
        ),
        UniqueConstraint("id", "conversation_id"),
        CheckConstraint("position > 0", name="position_positive"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole, native_enum=False, length=20, create_constraint=True)
    )
    content: Mapped[str] = mapped_column(Text, default="")
    # Supplied on user messages; null on assistant placeholders.
    client_request_id: Mapped[uuid.UUID | None] = mapped_column()

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class ConversationRun(TimestampMixin, Base):
    """One attempt to answer a user message; retries create new rows.

    response_text holds partial or completed output for this attempt. Only publish
    completed output to the assistant message. The service must validate message
    roles and reserve ordering under a conversation lock. Runs stuck in PENDING or
    RUNNING after a crash are failed by a periodic sweep.
    """

    __tablename__ = "conversation_runs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["user_message_id", "conversation_id"],
            ["conversation_messages.id", "conversation_messages.conversation_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["assistant_message_id", "conversation_id"],
            ["conversation_messages.id", "conversation_messages.conversation_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("assistant_message_id", "attempt_number"),
        CheckConstraint("attempt_number > 0", name="attempt_positive"),
        CheckConstraint(
            "user_message_id <> assistant_message_id", name="different_messages"
        ),
        Index(
            "uq_conversation_runs_active_conversation",
            "conversation_id",
            unique=True,
            postgresql_where=text("status IN ('PENDING', 'RUNNING')"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    user_message_id: Mapped[uuid.UUID] = mapped_column(index=True)
    assistant_message_id: Mapped[uuid.UUID] = mapped_column(index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, native_enum=False, length=20, create_constraint=True),
        default=RunStatus.PENDING,
        index=True,
    )
    stage: Mapped[str | None] = mapped_column(String(50))
    rewritten_query: Mapped[str | None] = mapped_column(Text)
    response_text: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(255))
    rewrite_model: Mapped[str | None] = mapped_column(String(255))
    # Identifies the prompts and retrieval settings used, e.g. "chat-v1".
    pipeline_version: Mapped[str | None] = mapped_column(String(100))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Safe application code only; provider exception details belong in correlated logs.
    error_code: Mapped[str | None] = mapped_column(String(100))
    snapshot: Mapped[RunSnapshot] = mapped_column(
        PydanticJSONB(RunSnapshot), default=RunSnapshot
    )
    metrics: Mapped[RunMetrics] = mapped_column(
        PydanticJSONB(RunMetrics), default=RunMetrics
    )
