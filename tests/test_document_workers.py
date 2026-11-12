"""Document worker claim and outcome rules, with a fake session and no provider calls."""

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.models.document import DocumentSummaryStatus, EmbeddingStatus
from app.models.job_error import JobErrorCode
from app.services.jobs.lease import JobFailure, LeaseLost
from app.workers.document_embedding import DocumentEmbeddingWorker
from app.workers.document_summary import DocumentSummaryWorker

LIVE = datetime.now(UTC) + timedelta(minutes=2)
DEAD = datetime.now(UTC) - timedelta(minutes=2)


def fake_document(**overrides):
    """A document row as the workers see it after SELECT ... FOR UPDATE."""
    base = {
        "id": uuid4(),
        "user_id": uuid4(),
        "summary_job_id": uuid4(),
        "summary_status": DocumentSummaryStatus.PENDING,
        "summary_attempts": 0,
        "summary_lease_expires_at": None,
        "summary_error_code": None,
        "embedding_job_id": uuid4(),
        "embedding_status": EmbeddingStatus.PENDING,
        "embedding_attempts": 0,
        "embedding_lease_expires_at": None,
        "embedding_error_code": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class SummaryWorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = AsyncMock()
        self.sessions = MagicMock()
        self.document = fake_document()
        self.db.scalar.return_value = self.document
        self.worker = DocumentSummaryWorker(self.db, sessions=self.sessions)
        self.worker._fail = AsyncMock()
        self.worker._release = AsyncMock()
        self.kwargs = {
            "document_id": self.document.id,
            "project_id": uuid4(),
            "job_id": self.document.summary_job_id,
        }

    async def run_with(self, outcome):
        """Run the worker with the real work replaced by `outcome` (a value or exception)."""
        with patch.object(self.worker, "_summarize", AsyncMock(side_effect=outcome)):
            await self.worker.process(**self.kwargs)

    async def test_pending_job_is_claimed_and_counted(self):
        await self.run_with(None)
        assert self.document.summary_status == DocumentSummaryStatus.STARTED
        assert self.document.summary_attempts == 1
        assert self.document.summary_lease_expires_at is not None
        self.db.commit.assert_awaited()

    async def test_running_job_with_a_live_lease_is_left_alone(self):
        self.document.summary_status = DocumentSummaryStatus.STARTED
        self.document.summary_lease_expires_at = LIVE
        work = AsyncMock()
        with patch.object(self.worker, "_summarize", work):
            await self.worker.process(**self.kwargs)
        work.assert_not_awaited()
        self.db.rollback.assert_awaited()
        self.db.commit.assert_not_awaited()

    async def test_abandoned_job_with_an_expired_lease_is_claimed_again(self):
        self.document.summary_status = DocumentSummaryStatus.STARTED
        self.document.summary_lease_expires_at = DEAD
        self.document.summary_attempts = 1
        await self.run_with(None)
        assert self.document.summary_attempts == 2
        self.db.commit.assert_awaited()

    async def test_stale_job_id_is_fenced(self):
        self.kwargs["job_id"] = uuid4()
        work = AsyncMock()
        with patch.object(self.worker, "_summarize", work):
            await self.worker.process(**self.kwargs)
        work.assert_not_awaited()

    async def test_named_failure_keeps_its_code(self):
        with pytest.raises(JobFailure):
            await self.run_with(JobFailure(JobErrorCode.NO_TEXT))
        self.worker._fail.assert_awaited_once_with(
            self.document.id, self.document.summary_job_id, JobErrorCode.NO_TEXT
        )

    async def test_unknown_failure_becomes_provider_error(self):
        with pytest.raises(RuntimeError):
            await self.run_with(RuntimeError("402 insufficient balance"))
        self.worker._fail.assert_awaited_once_with(
            self.document.id, self.document.summary_job_id, JobErrorCode.PROVIDER_ERROR
        )

    async def test_shutdown_hands_the_job_back(self):
        with pytest.raises(asyncio.CancelledError):
            await self.run_with(asyncio.CancelledError())
        self.worker._release.assert_awaited_once()
        self.worker._fail.assert_not_awaited()

    async def test_lost_lease_writes_nothing(self):
        await self.run_with(LeaseLost())
        self.worker._fail.assert_not_awaited()
        self.worker._release.assert_not_awaited()


class EmbeddingWorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = AsyncMock()
        self.document = fake_document()
        self.db.scalar.return_value = self.document
        self.worker = DocumentEmbeddingWorker(self.db, sessions=MagicMock())
        self.worker._fail = AsyncMock()
        self.worker._release = AsyncMock()
        self.project_id = uuid4()

    async def test_claim_with_the_row_fence_token(self):
        with patch.object(self.worker, "_index", AsyncMock()):
            await self.worker.process(
                document_id=self.document.id,
                project_id=self.project_id,
                job_id=self.document.embedding_job_id,
            )
        assert self.document.embedding_status == EmbeddingStatus.STARTED
        assert self.document.embedding_attempts == 1

    async def test_legacy_message_without_job_id_adopts_the_rows_token(self):
        token = self.document.embedding_job_id
        with patch.object(self.worker, "_index", AsyncMock()):
            await self.worker.process(
                document_id=self.document.id, project_id=self.project_id
            )
        assert self.document.embedding_job_id == token
        assert self.document.embedding_status == EmbeddingStatus.STARTED

    async def test_row_without_a_token_mints_one(self):
        self.document.embedding_job_id = None
        with patch.object(self.worker, "_index", AsyncMock()):
            await self.worker.process(
                document_id=self.document.id, project_id=self.project_id
            )
        assert self.document.embedding_job_id is not None

    async def test_stale_token_is_fenced(self):
        work = AsyncMock()
        with patch.object(self.worker, "_index", work):
            await self.worker.process(
                document_id=self.document.id,
                project_id=self.project_id,
                job_id=uuid4(),
            )
        work.assert_not_awaited()

    async def test_live_lease_is_left_alone(self):
        self.document.embedding_status = EmbeddingStatus.STARTED
        self.document.embedding_lease_expires_at = LIVE
        work = AsyncMock()
        with patch.object(self.worker, "_index", work):
            await self.worker.process(
                document_id=self.document.id, project_id=self.project_id
            )
        work.assert_not_awaited()

    async def test_failure_records_a_safe_code(self):
        failing = AsyncMock(side_effect=JobFailure(JobErrorCode.FILE_MISSING))
        with patch.object(self.worker, "_index", failing), pytest.raises(JobFailure):
            await self.worker.process(
                document_id=self.document.id, project_id=self.project_id
            )
        self.worker._fail.assert_awaited_once_with(
            self.document.id, self.document.embedding_job_id, JobErrorCode.FILE_MISSING
        )
