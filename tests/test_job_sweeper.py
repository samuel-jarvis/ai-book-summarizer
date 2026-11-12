"""Sweeper tests: the pure decision rules, plus real-PostgreSQL passes when a URL is set."""

import os
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.schema import CreateSchema, DropSchema

import app.models  # noqa: F401
from app.core.database import Base
from app.models.conversation import (
    Conversation,
    ConversationMessage,
    ConversationRun,
    MessageRole,
    RunStatus,
)
from app.models.document import (
    ContentType,
    Document,
    DocumentSummaryStatus,
    EmbeddingStatus,
    StorageProvider,
)
from app.models.project import Project
from app.models.project_documents import ProjectDocument
from app.models.summary import Summary, SummaryStatus
from app.models.user import User
from app.services.jobs.sweeper import Action, JobSweeper, plan_stuck_job

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)
LIVE = NOW + timedelta(minutes=2)
DEAD = NOW - timedelta(minutes=2)
LONG_AGO = NOW - timedelta(hours=1)
RULES = {
    "in_progress": "started",
    "pending": "pending",
    "now": NOW,
    "max_attempts": 2,
    "stale_after": timedelta(minutes=10),
}


class PlanTests(unittest.TestCase):
    """What one pass decides for one row, with no database."""

    def test_in_progress_with_a_live_lease_is_left_alone(self):
        assert (
            plan_stuck_job(
                status="started", lease_expires_at=LIVE, updated_at=NOW, attempts=1, **RULES
            )
            is Action.LEAVE
        )

    def test_abandoned_job_is_requeued_while_attempts_remain(self):
        assert (
            plan_stuck_job(
                status="started", lease_expires_at=DEAD, updated_at=NOW, attempts=1, **RULES
            )
            is Action.REQUEUE
        )

    def test_abandoned_job_fails_once_attempts_are_used_up(self):
        assert (
            plan_stuck_job(
                status="started", lease_expires_at=DEAD, updated_at=NOW, attempts=2, **RULES
            )
            is Action.FAIL
        )

    def test_chat_policy_fails_instead_of_requeueing(self):
        assert (
            plan_stuck_job(
                status="started",
                lease_expires_at=DEAD,
                updated_at=NOW,
                attempts=1,
                requeue_on_loss=False,
                **RULES,
            )
            is Action.FAIL
        )

    def test_fresh_pending_waits_but_stale_pending_is_requeued(self):
        assert (
            plan_stuck_job(
                status="pending", lease_expires_at=None, updated_at=NOW, attempts=0, **RULES
            )
            is Action.LEAVE
        )
        assert (
            plan_stuck_job(
                status="pending", lease_expires_at=None, updated_at=LONG_AGO, attempts=0, **RULES
            )
            is Action.REQUEUE
        )

    def test_finished_rows_are_never_touched(self):
        for status in ("completed", "failed", "cancelled"):
            assert (
                plan_stuck_job(
                    status=status, lease_expires_at=DEAD, updated_at=LONG_AGO, attempts=0, **RULES
                )
                is Action.LEAVE
            )


@unittest.skipUnless(os.getenv("CHAT_TEST_DATABASE_URL"), "PostgreSQL URL not supplied")
class SweepTests(unittest.IsolatedAsyncioTestCase):
    """One real pass per job table, in a disposable schema, with fake dispatchers."""

    async def asyncSetUp(self):
        self.schema = "jobs_test_" + uuid4().hex
        self.engine = create_async_engine(
            os.environ["CHAT_TEST_DATABASE_URL"],
            connect_args={"server_settings": {"search_path": f"{self.schema},public"}},
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.addAsyncCleanup(self.cleanup)
        tables = [
            User,
            Project,
            Document,
            ProjectDocument,
            Summary,
            Conversation,
            ConversationMessage,
            ConversationRun,
        ]
        async with self.engine.begin() as connection:
            await connection.execute(CreateSchema(self.schema))
            await connection.run_sync(
                lambda conn: Base.metadata.create_all(
                    conn, tables=[model.__table__ for model in tables], checkfirst=False
                )
            )
        self.user_id, self.project_id = uuid4(), uuid4()
        async with self.sessions.begin() as db:
            db.add(User(id=self.user_id, email="owner@example.test"))
            await db.flush()
            db.add(Project(id=self.project_id, user_id=self.user_id, title="Books"))
        self.summary_dispatch = AsyncMock()
        self.embedding_dispatch = AsyncMock()
        self.legacy_dispatch = AsyncMock()
        self.sweeper = JobSweeper(
            self.sessions,
            dispatch_summary=self.summary_dispatch,
            dispatch_embedding=self.embedding_dispatch,
            dispatch_legacy=self.legacy_dispatch,
            now=lambda: NOW,
        )

    async def cleanup(self):
        try:
            async with self.engine.begin() as connection:
                await connection.execute(DropSchema(self.schema, cascade=True, if_exists=True))
        finally:
            await self.engine.dispose()

    async def add_document(self, **fields):
        document = Document(
            user_id=self.user_id,
            title="Book",
            file_size_bytes=10,
            content_hash="0" * 64,
            content_type=ContentType.PDF,
            storage_provider=StorageProvider.LOCAL,
            storage_key="missing.pdf",
            summary_job_id=uuid4(),
            embedding_job_id=uuid4(),
            **fields,
        )
        async with self.sessions.begin() as db:
            db.add(document)
            await db.flush()
            db.add(ProjectDocument(project_id=self.project_id, document_id=document.id))
        return document

    async def reload(self, model, row_id):
        async with self.sessions() as db:
            return await db.scalar(select(model).where(model.id == row_id))

    async def test_abandoned_document_jobs_are_requeued_with_a_new_fence(self):
        document = await self.add_document(
            summary_status=DocumentSummaryStatus.STARTED,
            summary_lease_expires_at=DEAD,
            summary_attempts=1,
            embedding_status=EmbeddingStatus.STARTED,
            embedding_lease_expires_at=DEAD,
            embedding_attempts=1,
        )
        old_summary_job, old_embedding_job = document.summary_job_id, document.embedding_job_id

        report = await self.sweeper.sweep()

        fresh = await self.reload(Document, document.id)
        assert fresh.summary_status == DocumentSummaryStatus.PENDING
        assert fresh.embedding_status == EmbeddingStatus.PENDING
        assert fresh.summary_job_id != old_summary_job
        assert fresh.embedding_job_id != old_embedding_job
        self.summary_dispatch.assert_awaited_once_with(
            str(document.id), str(self.project_id), str(fresh.summary_job_id)
        )
        self.embedding_dispatch.assert_awaited_once_with(
            str(document.id), str(self.project_id), str(fresh.embedding_job_id)
        )
        assert len(report.requeued) == 2 and not report.failed

    async def test_abandoned_document_job_fails_once_attempts_are_used_up(self):
        document = await self.add_document(
            summary_status=DocumentSummaryStatus.STARTED,
            summary_lease_expires_at=DEAD,
            summary_attempts=2,
        )
        await self.sweeper.sweep()
        fresh = await self.reload(Document, document.id)
        assert fresh.summary_status == DocumentSummaryStatus.FAILED
        assert fresh.summary_error_code == "worker_lost"
        self.summary_dispatch.assert_not_awaited()

    async def test_live_and_recent_jobs_are_left_alone(self):
        document = await self.add_document(
            summary_status=DocumentSummaryStatus.STARTED,
            summary_lease_expires_at=LIVE,
            summary_attempts=1,
            embedding_status=EmbeddingStatus.PENDING,
            updated_at=NOW,
        )
        report = await self.sweeper.sweep()
        fresh = await self.reload(Document, document.id)
        assert fresh.summary_status == DocumentSummaryStatus.STARTED
        assert fresh.embedding_status == EmbeddingStatus.PENDING
        assert not report

    async def test_stale_pending_document_is_requeued(self):
        document = await self.add_document(
            summary_status=DocumentSummaryStatus.COMPLETED,
            embedding_status=EmbeddingStatus.PENDING,
            created_at=LONG_AGO,
            updated_at=LONG_AGO,
        )
        await self.sweeper.sweep()
        self.embedding_dispatch.assert_awaited_once()
        self.summary_dispatch.assert_not_awaited()
        fresh = await self.reload(Document, document.id)
        assert fresh.updated_at == NOW

    async def add_run(self, status, lease, updated):
        """One conversation per run: only one active attempt may exist per conversation."""
        async with self.sessions.begin() as db:
            conversation = Conversation(project_id=self.project_id)
            db.add(conversation)
            await db.flush()
            user = ConversationMessage(
                conversation_id=conversation.id,
                position=1,
                role=MessageRole.USER,
                content="q",
                client_request_id=uuid4(),
            )
            assistant = ConversationMessage(
                conversation_id=conversation.id, position=2, role=MessageRole.ASSISTANT
            )
            db.add_all([user, assistant])
            await db.flush()
            run = ConversationRun(
                conversation_id=conversation.id,
                user_message_id=user.id,
                assistant_message_id=assistant.id,
                status=status,
                lease_expires_at=lease,
                updated_at=updated,
            )
            db.add(run)
            await db.flush()
            return run.id

    async def test_chat_runs_fail_instead_of_rerunning(self):
        lost = await self.add_run(RunStatus.RUNNING, DEAD, NOW)
        never_started = await self.add_run(RunStatus.PENDING, None, LONG_AGO)
        healthy = await self.add_run(RunStatus.RUNNING, LIVE, NOW)
        fresh_pending = await self.add_run(RunStatus.PENDING, None, NOW)

        await self.sweeper.sweep()

        lost_run = await self.reload(ConversationRun, lost)
        assert lost_run.status == RunStatus.FAILED
        assert lost_run.error_code == "worker_lost"
        assert lost_run.finished_at is not None
        stale_run = await self.reload(ConversationRun, never_started)
        assert stale_run.status == RunStatus.FAILED
        assert stale_run.error_code == "queue_unavailable"
        assert (await self.reload(ConversationRun, healthy)).status == RunStatus.RUNNING
        assert (await self.reload(ConversationRun, fresh_pending)).status == RunStatus.PENDING

    async def test_legacy_summary_abandoned_is_requeued_then_failed(self):
        async with self.sessions.begin() as db:
            first = Summary(
                user_id=self.user_id,
                file_path="x.pdf",
                source_text="text",
                status=SummaryStatus.PROCESSING,
                lease_expires_at=DEAD,
                attempts=1,
                job_id=uuid4(),
            )
            second = Summary(
                user_id=self.user_id,
                file_path="y.pdf",
                source_text="text",
                status=SummaryStatus.PROCESSING,
                lease_expires_at=DEAD,
                attempts=2,
                job_id=uuid4(),
            )
            db.add_all([first, second])
        await self.sweeper.sweep()
        fresh_first = await self.reload(Summary, first.id)
        fresh_second = await self.reload(Summary, second.id)
        assert fresh_first.status == SummaryStatus.PENDING
        assert fresh_first.job_id is None
        assert fresh_second.status == SummaryStatus.FAILED
        self.legacy_dispatch.assert_awaited_once_with(str(first.id))
