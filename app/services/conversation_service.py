"""Owner-scoped persistence and serialized conversation turn transitions.

The caller owns the transaction: commit each reservation or checkpoint before
external work. No method calls a provider or keeps a transaction across inference.
All turn mutations lock the conversation first, including cancellation and retries.
"""

from typing import NamedTuple
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import utcnow
from app.core.exception import ConflictError, NotFoundError, ValidationError
from app.models.conversation import (
    Conversation,
    ConversationMessage,
    ConversationRun,
    MessageRole,
    RunMetrics,
    RunSnapshot,
    RunStatus,
)
from app.models.project import Project

ACTIVE_STATUSES = (RunStatus.PENDING, RunStatus.RUNNING)


class ReservedTurn(NamedTuple):
    user_message: ConversationMessage
    assistant_message: ConversationMessage
    run: ConversationRun
    reused: bool


class ConversationService:
    def __init__(self, db: AsyncSession, user_id: UUID) -> None:
        self.db = db
        self.user_id = user_id

    @staticmethod
    def _pagination(limit: int, offset: int = 0) -> None:
        if not 1 <= limit <= 200 or offset < 0:
            raise ValidationError(
                "Provide a limit from 1 to 200 and a nonnegative offset."
            )

    @staticmethod
    def _title(title: str | None) -> str | None:
        if title is None:
            return None
        if not title.strip() or len(title.strip()) > 255:
            raise ValidationError("Title must contain 1 to 255 characters.")
        return title.strip()

    async def _project(self, project_id: UUID) -> None:
        if (
            await self.db.scalar(
                select(Project.id).where(
                    Project.id == project_id, Project.user_id == self.user_id
                )
            )
            is None
        ):
            raise NotFoundError("Project not found.")

    async def get(self, conversation_id: UUID, *, lock: bool = False) -> Conversation:
        statement = (
            select(Conversation)
            .join(Project)
            .where(Conversation.id == conversation_id, Project.user_id == self.user_id)
        )
        if lock:
            statement = statement.with_for_update(of=Conversation).execution_options(
                populate_existing=True
            )
        conversation = await self.db.scalar(statement)
        if conversation is None:
            raise NotFoundError("Conversation not found.")
        return conversation

    async def create(self, project_id: UUID, title: str | None = None) -> Conversation:
        await self._project(project_id)
        conversation = Conversation(project_id=project_id, title=self._title(title))
        self.db.add(conversation)
        await self.db.flush()
        return conversation

    async def list_conversations(
        self, project_id: UUID, *, limit: int = 50, offset: int = 0
    ) -> list[Conversation]:
        self._pagination(limit, offset)
        await self._project(project_id)
        return list(
            await self.db.scalars(
                select(Conversation)
                .where(Conversation.project_id == project_id)
                .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
                .limit(limit)
                .offset(offset)
            )
        )

    async def rename(self, conversation_id: UUID, title: str) -> Conversation:
        conversation = await self.get(conversation_id, lock=True)
        conversation.title = self._title(title)
        await self.db.flush()
        return conversation

    async def delete(self, conversation_id: UUID) -> None:
        await self.get(conversation_id, lock=True)
        await self.db.execute(
            delete(Conversation).where(Conversation.id == conversation_id)
        )
        # Database cascades remove messages and runs, including any pending attempt.

    async def list_messages(
        self, conversation_id: UUID, *, after_position: int = 0, limit: int = 50
    ) -> list[ConversationMessage]:
        self._pagination(limit, after_position)
        await self.get(conversation_id)
        return list(
            await self.db.scalars(
                select(ConversationMessage)
                .where(
                    ConversationMessage.conversation_id == conversation_id,
                    ConversationMessage.position > after_position,
                )
                .order_by(ConversationMessage.position)
                .limit(limit)
            )
        )

    async def get_run(self, run_id: UUID) -> ConversationRun:
        run = await self.db.scalar(
            select(ConversationRun)
            .join(Conversation, Conversation.id == ConversationRun.conversation_id)
            .join(Project)
            .where(ConversationRun.id == run_id, Project.user_id == self.user_id)
        )
        if run is None:
            raise NotFoundError("Run not found.")
        return run

    async def _locked_run(self, run_id: UUID) -> ConversationRun:
        run = await self.get_run(run_id)
        await self.get(run.conversation_id, lock=True)
        await self.db.refresh(run)
        return run

    async def _ensure_idle(self, conversation_id: UUID) -> None:
        active = await self.db.scalar(
            select(ConversationRun.id)
            .where(
                ConversationRun.conversation_id == conversation_id,
                ConversationRun.status.in_(ACTIVE_STATUSES),
            )
            .limit(1)
        )
        if active is not None:
            raise ConflictError("This conversation already has an active run.")

    async def reserve_turn(
        self, conversation_id: UUID, *, content: str, client_request_id: UUID
    ) -> ReservedTurn:
        if not content.strip():
            raise ValidationError("Message must contain non-whitespace text.")
        conversation = await self.get(conversation_id, lock=True)
        existing = await self.db.scalar(
            select(ConversationMessage).where(
                ConversationMessage.conversation_id == conversation_id,
                ConversationMessage.client_request_id == client_request_id,
            )
        )
        if existing is not None:
            if existing.role != MessageRole.USER or existing.content != content:
                raise ConflictError(
                    "Request ID was already used for a different message."
                )
            run = await self.db.scalar(
                select(ConversationRun)
                .where(ConversationRun.user_message_id == existing.id)
                .order_by(ConversationRun.attempt_number.desc())
                .limit(1)
            )
            if run is None:
                raise ConflictError("Existing message has no reserved run.")
            assistant = await self.db.get(ConversationMessage, run.assistant_message_id)
            return ReservedTurn(existing, assistant, run, True)

        await self._ensure_idle(conversation_id)
        position = (
            await self.db.scalar(
                select(func.max(ConversationMessage.position)).where(
                    ConversationMessage.conversation_id == conversation_id
                )
            )
            or 0
        ) + 1
        user = ConversationMessage(
            conversation_id=conversation_id,
            position=position,
            role=MessageRole.USER,
            content=content,
            client_request_id=client_request_id,
        )
        assistant = ConversationMessage(
            conversation_id=conversation_id,
            position=position + 1,
            role=MessageRole.ASSISTANT,
            content="",
        )
        self.db.add_all([user, assistant])
        await self.db.flush()
        run = ConversationRun(
            conversation_id=conversation_id,
            user_message_id=user.id,
            assistant_message_id=assistant.id,
            attempt_number=1,
            status=RunStatus.PENDING,
            stage="reserved",
        )
        self.db.add(run)
        conversation.updated_at = utcnow()
        await self.db.flush()
        return ReservedTurn(user, assistant, run, False)

    async def start_run(
        self, run_id: UUID, *, model: str, pipeline_version: str
    ) -> ConversationRun:
        run = await self._locked_run(run_id)
        if run.status != RunStatus.PENDING:
            raise ConflictError("Only a pending run can start.")
        run.status = RunStatus.RUNNING
        run.stage = "starting"
        run.started_at = utcnow()
        run.model = model
        run.pipeline_version = pipeline_version
        await self.db.flush()
        return run

    async def checkpoint(
        self,
        run_id: UUID,
        *,
        stage: str,
        snapshot: RunSnapshot | None = None,
        metrics: RunMetrics | None = None,
        rewritten_query: str | None = None,
        rewrite_model: str | None = None,
        response_text: str | None = None,
    ) -> ConversationRun:
        run = await self._locked_run(run_id)
        if run.status != RunStatus.RUNNING:
            raise ConflictError("Only a running attempt can receive checkpoints.")
        if not stage.strip() or len(stage) > 50:
            raise ValidationError("Stage must contain 1 to 50 characters.")
        run.stage = stage
        for name, value in {
            "snapshot": snapshot,
            "metrics": metrics,
            "rewritten_query": rewritten_query,
            "rewrite_model": rewrite_model,
            "response_text": response_text,
        }.items():
            if value is not None:
                setattr(run, name, value)
        await self.db.flush()
        return run

    async def complete(
        self, run_id: UUID, response_text: str, *, metrics: RunMetrics | None = None
    ) -> ConversationRun:
        run = await self._locked_run(run_id)
        if run.status != RunStatus.RUNNING:
            raise ConflictError("Only a running attempt can complete.")
        if not response_text.strip():
            raise ValidationError("A completed response cannot be empty.")
        assistant = await self.db.get(ConversationMessage, run.assistant_message_id)
        user = await self.db.get(ConversationMessage, run.user_message_id)
        if assistant.role != MessageRole.ASSISTANT or user.role != MessageRole.USER:
            raise ConflictError("Run messages have invalid roles.")
        run.response_text = response_text
        assistant.content = response_text
        run.status = RunStatus.COMPLETED
        run.stage = "completed"
        run.finished_at = utcnow()
        if metrics is not None:
            run.metrics = metrics
        conversation = await self.db.get(Conversation, run.conversation_id)
        conversation.updated_at = utcnow()
        await self.db.flush()
        return run

    async def fail(self, run_id: UUID, error_code: str) -> ConversationRun:
        # This code is application-defined, never the provider's exception text.
        if (
            not error_code
            or len(error_code) > 100
            or not all(
                char.isascii() and (char.isalnum() or char == "_")
                for char in error_code
            )
        ):
            raise ValidationError("Provide a safe application error code.")
        return await self._terminate(run_id, RunStatus.FAILED, error_code)

    async def cancel(self, run_id: UUID) -> ConversationRun:
        return await self._terminate(run_id, RunStatus.CANCELLED)

    async def _terminate(
        self, run_id: UUID, status: RunStatus, error_code: str | None = None
    ) -> ConversationRun:
        run = await self._locked_run(run_id)
        if run.status == status:
            return run  # Repeated cancellation/failure notifications are harmless.
        if run.status not in ACTIVE_STATUSES:
            raise ConflictError("This attempt has already finished.")
        run.status = status
        run.stage = status.value
        run.error_code = error_code
        run.finished_at = utcnow()
        await self.db.flush()
        return run

    async def retry(self, run_id: UUID) -> ConversationRun:
        previous = await self._locked_run(run_id)
        if previous.status not in (RunStatus.FAILED, RunStatus.CANCELLED):
            raise ConflictError("Only failed or cancelled attempts can be retried.")
        await self._ensure_idle(previous.conversation_id)
        latest_attempt = await self.db.scalar(
            select(func.max(ConversationRun.attempt_number)).where(
                ConversationRun.assistant_message_id == previous.assistant_message_id
            )
        )
        assistant = await self.db.get(
            ConversationMessage, previous.assistant_message_id
        )
        last_position = await self.db.scalar(
            select(func.max(ConversationMessage.position)).where(
                ConversationMessage.conversation_id == previous.conversation_id
            )
        )
        if (
            latest_attempt != previous.attempt_number
            or assistant.position != last_position
        ):
            raise ConflictError(
                "Only the latest attempt of the latest turn can be retried."
            )
        run = ConversationRun(
            conversation_id=previous.conversation_id,
            user_message_id=previous.user_message_id,
            assistant_message_id=previous.assistant_message_id,
            attempt_number=previous.attempt_number + 1,
            status=RunStatus.PENDING,
            stage="reserved",
        )
        self.db.add(run)
        await self.db.flush()
        return run
