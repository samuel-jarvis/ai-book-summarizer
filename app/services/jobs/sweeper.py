"""The sweeper: once a minute, find jobs whose worker went quiet and put them right.

Every worker renews its job's lease while it works (``lease.py``). A job
still marked in progress after its lease ran out belongs to a worker that
crashed, was killed by a deploy, or lost its connection. One sweeper pass:

- document summary / embedding jobs: back to PENDING and re-queued while
  automatic attempts remain, otherwise FAILED with ``worker_lost``;
- legacy ``/summary`` jobs: the same;
- chat runs: FAILED with ``worker_lost`` straight away (never re-run
  automatically; the user is watching and has a Retry button);
- anything still PENDING long after it was queued never reached a worker,
  so it is re-queued (documents, summaries) or failed with
  ``queue_unavailable`` (chat runs), which Retry re-dispatches.

Every change is made under ``FOR UPDATE SKIP LOCKED`` and re-checks the
status, so a worker that is merely slow and a sweeper that thinks it is dead
can never both win. The loop runs inside the worker process, with a Redis lock
so that only one worker sweeps per interval.
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.core.database import AsyncSessionLocal, utcnow
from app.models.conversation import ConversationRun, RunStatus
from app.models.document import Document, DocumentSummaryStatus, EmbeddingStatus
from app.models.job_error import JobErrorCode
from app.models.project_documents import ProjectDocument
from app.models.summary import Summary, SummaryStatus
from app.services.jobs.lease import lease_expired

logger = logging.getLogger(__name__)

SWEEPER_LOCK_KEY = "jobs:sweeper:lock"
SWEEPER_LAST_RUN_KEY = "jobs:sweeper:last_run"

Dispatch = Callable[..., Awaitable[None]]


class Action(StrEnum):
    """What one pass decides for one in-progress or stale row."""

    LEAVE = "leave"
    REQUEUE = "requeue"
    FAIL = "fail"


def plan_stuck_job(
    *,
    status: object,
    in_progress: object,
    pending: object,
    lease_expires_at: datetime | None,
    updated_at: datetime,
    attempts: int,
    now: datetime,
    max_attempts: int | None = None,
    stale_after: timedelta | None = None,
    requeue_on_loss: bool = True,
) -> Action:
    """Decide what to do with a job row, with no database in sight.

    `requeue_on_loss=False` is the chat-run policy: a lost worker fails the
    attempt instead of spending again automatically.
    """
    max_attempts = settings.JOB_MAX_ATTEMPTS if max_attempts is None else max_attempts
    stale_after = (
        timedelta(seconds=settings.JOB_STALE_PENDING_SECONDS)
        if stale_after is None
        else stale_after
    )
    if status == in_progress:
        if not lease_expired(lease_expires_at, now):
            return Action.LEAVE
        if requeue_on_loss and attempts < max_attempts:
            return Action.REQUEUE
        return Action.FAIL
    if status == pending and updated_at <= now - stale_after:
        return Action.REQUEUE if requeue_on_loss else Action.FAIL
    return Action.LEAVE


@dataclass
class SweepReport:
    """What one pass did, for logs and tests."""

    requeued: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.requeued or self.failed)


class JobSweeper:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession] = AsyncSessionLocal,
        *,
        dispatch_summary: Dispatch | None = None,
        dispatch_embedding: Dispatch | None = None,
        dispatch_legacy: Dispatch | None = None,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        """Inject session factory, dispatchers and clock so tests need no Redis."""
        self.sessions = sessions
        self._dispatch_summary = dispatch_summary
        self._dispatch_embedding = dispatch_embedding
        self._dispatch_legacy = dispatch_legacy
        self.now = now

    async def sweep(self) -> SweepReport:
        """Run one pass over every job table and re-queue after each commit."""
        report = SweepReport()
        now = self.now()
        await self._sweep_document_summaries(now, report)
        await self._sweep_document_embeddings(now, report)
        await self._sweep_conversation_runs(now, report)
        await self._sweep_legacy_summaries(now, report)
        if report:
            logger.info(
                "Job sweep: re-queued %d, failed %d", len(report.requeued), len(report.failed)
            )
        return report

    # -- documents ---------------------------------------------------------

    async def _candidate_documents(self, db: AsyncSession, status_column, in_progress, pending):
        """In-progress or pending documents with the project to re-queue them under."""
        result = await db.execute(
            select(Document, ProjectDocument.project_id)
            .join(ProjectDocument, ProjectDocument.document_id == Document.id)
            .where(status_column.in_([in_progress, pending]))
            .with_for_update(of=Document, skip_locked=True)
        )
        return result.all()

    async def _sweep_document_summaries(self, now: datetime, report: SweepReport) -> None:
        to_queue: list[tuple[uuid.UUID, uuid.UUID, uuid.UUID]] = []
        async with self.sessions.begin() as db:
            rows = await self._candidate_documents(
                db,
                Document.summary_status,
                DocumentSummaryStatus.STARTED,
                DocumentSummaryStatus.PENDING,
            )
            for document, project_id in rows:
                action = plan_stuck_job(
                    status=document.summary_status,
                    in_progress=DocumentSummaryStatus.STARTED,
                    pending=DocumentSummaryStatus.PENDING,
                    lease_expires_at=document.summary_lease_expires_at,
                    updated_at=document.updated_at,
                    attempts=document.summary_attempts,
                    now=now,
                )
                if action is Action.REQUEUE:
                    # A new job id fences the dead worker's late write.
                    document.summary_job_id = uuid.uuid4()
                    document.summary_status = DocumentSummaryStatus.PENDING
                    document.summary_lease_expires_at = None
                    document.updated_at = now
                    to_queue.append((document.id, project_id, document.summary_job_id))
                elif action is Action.FAIL:
                    document.summary_status = DocumentSummaryStatus.FAILED
                    document.summary_lease_expires_at = None
                    document.summary_error_code = JobErrorCode.WORKER_LOST.value
                    report.failed.append(f"document-summary:{document.id}")
        for document_id, project_id, job_id in to_queue:
            await self._send(
                self.dispatch_summary, str(document_id), str(project_id), str(job_id)
            )
            report.requeued.append(f"document-summary:{document_id}")

    async def _sweep_document_embeddings(self, now: datetime, report: SweepReport) -> None:
        to_queue: list[tuple[uuid.UUID, uuid.UUID, uuid.UUID]] = []
        async with self.sessions.begin() as db:
            rows = await self._candidate_documents(
                db,
                Document.embedding_status,
                EmbeddingStatus.STARTED,
                EmbeddingStatus.PENDING,
            )
            for document, project_id in rows:
                action = plan_stuck_job(
                    status=document.embedding_status,
                    in_progress=EmbeddingStatus.STARTED,
                    pending=EmbeddingStatus.PENDING,
                    lease_expires_at=document.embedding_lease_expires_at,
                    updated_at=document.updated_at,
                    attempts=document.embedding_attempts,
                    now=now,
                )
                if action is Action.REQUEUE:
                    document.embedding_job_id = uuid.uuid4()
                    document.embedding_status = EmbeddingStatus.PENDING
                    document.embedding_lease_expires_at = None
                    document.updated_at = now
                    to_queue.append((document.id, project_id, document.embedding_job_id))
                elif action is Action.FAIL:
                    document.embedding_status = EmbeddingStatus.FAILED
                    document.embedding_lease_expires_at = None
                    document.embedding_error_code = JobErrorCode.WORKER_LOST.value
                    report.failed.append(f"document-embedding:{document.id}")
        for document_id, project_id, job_id in to_queue:
            await self._send(
                self.dispatch_embedding, str(document_id), str(project_id), str(job_id)
            )
            report.requeued.append(f"document-embedding:{document_id}")

    # -- chat runs ---------------------------------------------------------

    async def _sweep_conversation_runs(self, now: datetime, report: SweepReport) -> None:
        async with self.sessions.begin() as db:
            runs = (
                await db.scalars(
                    select(ConversationRun)
                    .where(ConversationRun.status.in_([RunStatus.RUNNING, RunStatus.PENDING]))
                    .with_for_update(skip_locked=True)
                )
            ).all()
            for run in runs:
                action = plan_stuck_job(
                    status=run.status,
                    in_progress=RunStatus.RUNNING,
                    pending=RunStatus.PENDING,
                    lease_expires_at=run.lease_expires_at,
                    updated_at=run.updated_at,
                    attempts=run.attempt_number,
                    now=now,
                    requeue_on_loss=False,
                )
                if action is Action.FAIL:
                    # Mirrors ConversationService._terminate without a user scope.
                    run.error_code = (
                        "worker_lost"
                        if run.status == RunStatus.RUNNING
                        else "queue_unavailable"
                    )
                    run.status = RunStatus.FAILED
                    run.stage = RunStatus.FAILED.value
                    run.finished_at = now
                    run.lease_expires_at = None
                    report.failed.append(f"conversation-run:{run.id}")

    # -- legacy summaries --------------------------------------------------

    async def _sweep_legacy_summaries(self, now: datetime, report: SweepReport) -> None:
        to_queue: list[uuid.UUID] = []
        async with self.sessions.begin() as db:
            summaries = (
                await db.scalars(
                    select(Summary)
                    .where(
                        Summary.status.in_([SummaryStatus.PROCESSING, SummaryStatus.PENDING])
                    )
                    .with_for_update(skip_locked=True)
                )
            ).all()
            for summary in summaries:
                action = plan_stuck_job(
                    status=summary.status,
                    in_progress=SummaryStatus.PROCESSING,
                    pending=SummaryStatus.PENDING,
                    lease_expires_at=summary.lease_expires_at,
                    updated_at=summary.updated_at,
                    attempts=summary.attempts,
                    now=now,
                )
                if action is Action.REQUEUE:
                    # The worker mints a fresh job id on claim; clearing the old
                    # one fences the dead worker's late write meanwhile.
                    summary.job_id = None
                    summary.status = SummaryStatus.PENDING
                    summary.lease_expires_at = None
                    summary.updated_at = now
                    to_queue.append(summary.id)
                elif action is Action.FAIL:
                    summary.status = SummaryStatus.FAILED
                    summary.lease_expires_at = None
                    report.failed.append(f"summary:{summary.id}")
        for summary_id in to_queue:
            await self._send(self.dispatch_legacy, str(summary_id))
            report.requeued.append(f"summary:{summary_id}")

    # -- dispatch ----------------------------------------------------------

    @property
    def dispatch_summary(self) -> Dispatch:
        if self._dispatch_summary is None:
            from app.workers.document_summary import summarize_document_task

            self._dispatch_summary = summarize_document_task.kiq
        return self._dispatch_summary

    @property
    def dispatch_embedding(self) -> Dispatch:
        if self._dispatch_embedding is None:
            from app.workers.document_embedding import embed_document_task

            self._dispatch_embedding = embed_document_task.kiq
        return self._dispatch_embedding

    @property
    def dispatch_legacy(self) -> Dispatch:
        if self._dispatch_legacy is None:
            from app.tasks import process_pdf_task

            self._dispatch_legacy = process_pdf_task.kiq
        return self._dispatch_legacy

    @staticmethod
    async def _send(dispatch: Dispatch, *args: str) -> None:
        """Queue after commit; a failed send leaves a PENDING row the next pass re-queues."""
        try:
            await dispatch(*args)
        except Exception:  # noqa: BLE001 -- Redis trouble; the stale-pending rule covers it.
            logger.warning("Could not re-queue job %s; will retry next sweep", args)


async def run_sweeper_forever(
    redis_pool,
    *,
    sweeper: JobSweeper | None = None,
    interval: int | None = None,
) -> None:
    """Sweep on an interval, taking a short Redis lock so only one worker sweeps at a time.

    Runs until cancelled (worker shutdown). The lock expires on its own just
    before the next interval, so a sweeper that dies holding it blocks nobody.
    """
    sweeper = sweeper or JobSweeper()
    interval = interval or settings.JOB_SWEEP_INTERVAL_SECONDS
    while True:
        try:
            async with Redis(connection_pool=redis_pool) as redis:
                acquired = await redis.set(
                    SWEEPER_LOCK_KEY, "1", nx=True, ex=max(interval - 5, 1)
                )
                if acquired:
                    await sweeper.sweep()
                    await redis.set(SWEEPER_LAST_RUN_KEY, utcnow().isoformat())
        except RedisError:
            logger.warning("Job sweeper could not reach Redis; skipping this pass")
        except asyncio.CancelledError:
            raise
        except Exception:  # one bad pass must not stop future ones
            logger.exception("Job sweep failed")
        await asyncio.sleep(interval)
