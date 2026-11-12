import asyncio
import logging

from taskiq import TaskiqEvents, TaskiqState

from app.services.jobs.sweeper import run_sweeper_forever
from app.services.processor_service import process_summary
from app.taskiq_broker import broker
from app.workers.conversation import answer_conversation_task
from app.workers.document_embedding import embed_document_task
from app.workers.document_summary import summarize_document_task

__all__ = [
    "answer_conversation_task",
    "embed_document_task",
    "process_pdf_task",
    "summarize_document_task",
]

logger = logging.getLogger(__name__)


@broker.task
async def process_pdf_task(summary_id: str) -> None:
    await process_summary(summary_id)


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def start_job_sweeper(state: TaskiqState) -> None:
    """Run the stuck-job sweeper inside every worker; a Redis lock keeps it to one at a time.

    Starting it here means the first sweep happens the moment a worker comes
    back after a crash or deploy, which is exactly when jobs need rescuing.
    """
    state.job_sweeper = asyncio.create_task(
        run_sweeper_forever(broker.connection_pool), name="job-sweeper"
    )
    logger.info("Job sweeper started")


@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def stop_job_sweeper(state: TaskiqState) -> None:
    """Stop the sweeper loop with the worker."""
    task: asyncio.Task | None = getattr(state, "job_sweeper", None)
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
