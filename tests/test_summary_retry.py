"""Legacy summary retry rules, with a fake session and no database."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core.exception import ConflictError, NotFoundError, RateLimitError
from app.models.summary import SummaryStatus
from app.services.summary_service import SummaryService


class SummaryRetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = AsyncMock()
        self.user_id = uuid4()
        self.summary = SimpleNamespace(
            id=uuid4(),
            user_id=self.user_id,
            status=SummaryStatus.FAILED,
            job_id=uuid4(),
            attempts=2,
            lease_expires_at=None,
            content=None,
        )
        self.service = SummaryService(self.db)
        self.service.get_summary_by_id = AsyncMock(return_value=self.summary)
        self.service.ensure_daily_quota = AsyncMock()

    async def test_failed_summary_goes_back_to_pending_with_a_clean_slate(self):
        result = await self.service.retry(self.summary.id, self.user_id)
        assert result is self.summary
        assert self.summary.status is SummaryStatus.PENDING
        assert self.summary.job_id is None
        assert self.summary.attempts == 0
        self.db.commit.assert_awaited_once()

    async def test_only_failed_summaries_can_be_retried(self):
        for status in (SummaryStatus.PENDING, SummaryStatus.PROCESSING, SummaryStatus.COMPLETED):
            self.summary.status = status
            with self.subTest(status=status), pytest.raises(ConflictError):
                await self.service.retry(self.summary.id, self.user_id)
        self.db.commit.assert_not_awaited()

    async def test_retry_respects_the_daily_quota(self):
        self.service.ensure_daily_quota.side_effect = RateLimitError()
        with pytest.raises(RateLimitError):
            await self.service.retry(self.summary.id, self.user_id)
        assert self.summary.status is SummaryStatus.FAILED
        self.db.commit.assert_not_awaited()

    async def test_someone_elses_summary_is_not_found(self):
        self.service.get_summary_by_id.return_value = None
        with pytest.raises(NotFoundError):
            await self.service.retry(self.summary.id, uuid4())
