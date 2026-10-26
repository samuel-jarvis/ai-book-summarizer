"""Queue committed attempts without leaving unstarted runs active on send failure."""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exception import AppError, NotFoundError
from app.services.conversation_service import ConversationService
from app.workers.conversation import answer_conversation_task


class ConversationQueueError(AppError):
    """The answer could not be queued; retry after the task queue is available."""

    status_code = 503
    code = "queue_unavailable"


async def dispatch_conversation(db: AsyncSession, run_id: UUID, user_id: UUID) -> None:
    """Send a committed run ID, and safely fail it only if a worker has not claimed it."""
    try:
        await answer_conversation_task.kiq(str(run_id), str(user_id))
    except Exception:  # noqa: BLE001 -- Taskiq may wrap provider-specific broker exceptions.
        try:
            await ConversationService(db, user_id).fail_if_active(
                run_id, "queue_unavailable", pending_only=True
            )
            await db.commit()
        except NotFoundError:
            pass
        raise ConversationQueueError() from None
