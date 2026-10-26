"""Frontend-facing conversation requests, messages, run status, and source views."""

from datetime import datetime
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.conversation import (
    ConversationRun,
    MessageRole,
    RunMetrics,
    RunSnapshot,
    RunStatus,
    SourceSnapshot,
)
from app.schema.response import ApiResponse


class CreateConversationRequest(BaseModel):
    """Create a conversation inside an owned project."""

    model_config = ConfigDict(extra="forbid")
    project_id: UUID
    title: str | None = Field(default=None, min_length=1, max_length=255)


class RenameConversationRequest(BaseModel):
    """Replace a conversation's display title."""

    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=255)


class SendMessageRequest(BaseModel):
    """Preserve original user text and require an idempotency UUID from the client."""

    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1, max_length=20000)
    client_request_id: UUID

    @field_validator("content")
    @classmethod
    def nonempty_content(cls, value: str) -> str:
        """Reject whitespace-only text without stripping the stored message."""
        if not value.strip():
            raise ValueError("Message must contain non-whitespace text.")
        return value


class ConversationResponse(BaseModel):
    """Stable identifiers and timestamps for the conversation sidebar."""

    model_config = ConfigDict(from_attributes=True)
    id: UUID
    project_id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class RunResponse(BaseModel):
    """Polling state with captured sources for the matching assistant message."""

    model_config = ConfigDict(from_attributes=True)
    id: UUID
    conversation_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    attempt_number: int
    status: RunStatus
    stage: str | None
    model: str | None
    rewritten_query: str | None
    rewrite_model: str | None
    response_text: str | None
    error_code: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime
    sources: list[SourceSnapshot] = Field(default_factory=list)

    @classmethod
    def from_run(cls, run: ConversationRun) -> Self:
        """Expose captured source metadata without re-reading current documents."""
        view = cls.model_validate(run)
        view.sources = [
            source.model_copy(deep=True) for source in run.snapshot.selected_sources
        ]
        return view


class RunDetailResponse(RunResponse):
    """Owner-only diagnostics for an individual generation attempt."""

    snapshot: RunSnapshot
    metrics: RunMetrics


class MessageResponse(BaseModel):
    """Visible message text plus its latest attempt for rendering pending or failed turns."""

    model_config = ConfigDict(from_attributes=True)
    id: UUID
    conversation_id: UUID
    position: int
    role: MessageRole
    content: str
    client_request_id: UUID | None
    created_at: datetime
    updated_at: datetime
    latest_run: RunResponse | None = None


class MessagePage(BaseModel):
    """A chronological page; use its cursor to fetch the next page."""

    messages: list[MessageResponse]
    next_after_position: int | None


class SubmittedTurn(BaseModel):
    """The reserved messages and attempt returned before background generation."""

    user_message: MessageResponse
    assistant_message: MessageResponse
    run: RunResponse
    reused: bool


ConversationApiResponse = ApiResponse[ConversationResponse]
ConversationListApiResponse = ApiResponse[list[ConversationResponse]]
MessagePageApiResponse = ApiResponse[MessagePage]
SubmittedTurnApiResponse = ApiResponse[SubmittedTurn]
RunApiResponse = ApiResponse[RunResponse]
RunDetailApiResponse = ApiResponse[RunDetailResponse]
RunListApiResponse = ApiResponse[list[RunResponse]]
SourcesApiResponse = ApiResponse[list[SourceSnapshot]]
