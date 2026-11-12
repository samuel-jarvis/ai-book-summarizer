"""Lease helper tests: claim rules, heartbeat renewal, and lost-lease handling."""

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from app.services.jobs.lease import (
    LeaseLost,
    claimable,
    job_lease,
    lease_deadline,
    lease_expired,
)

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)


class ClaimRuleTests(unittest.TestCase):
    """The pure functions every worker and the sweeper agree on."""

    def test_missing_or_past_lease_is_expired(self):
        assert lease_expired(None, NOW)
        assert lease_expired(NOW - timedelta(seconds=1), NOW)
        assert not lease_expired(NOW + timedelta(seconds=1), NOW)

    def test_pending_is_always_claimable(self):
        assert claimable("pending", None, pending="pending", started="started", now=NOW)

    def test_started_is_claimable_only_once_the_lease_ran_out(self):
        live = NOW + timedelta(minutes=1)
        dead = NOW - timedelta(minutes=1)
        assert not claimable("started", live, pending="pending", started="started", now=NOW)
        assert claimable("started", dead, pending="pending", started="started", now=NOW)
        assert claimable("started", None, pending="pending", started="started", now=NOW)

    def test_finished_jobs_are_never_claimable(self):
        for status in ("completed", "failed"):
            assert not claimable(status, None, pending="pending", started="started", now=NOW)

    def test_deadline_is_lease_length_ahead(self):
        assert lease_deadline(NOW) > NOW


class HeartbeatTests(unittest.IsolatedAsyncioTestCase):
    """The heartbeat renews while work runs and tells the worker when the job is gone."""

    async def test_renews_while_working_and_stops_afterwards(self):
        renew = AsyncMock(return_value=True)
        async with job_lease(renew, interval=0.01):
            await asyncio.sleep(0.06)
        beats = renew.await_count
        assert beats >= 2
        await asyncio.sleep(0.03)
        assert renew.await_count == beats

    async def test_lost_lease_cancels_the_work_and_raises_lease_lost(self):
        renew = AsyncMock(return_value=False)
        with pytest.raises(LeaseLost):
            async with job_lease(renew, interval=0.01):
                await asyncio.sleep(5)  # would be far too long; the heartbeat cuts it short
        # The owning task is left in a clean, uncancelled state.
        await asyncio.sleep(0)

    async def test_outside_cancellation_passes_through_unchanged(self):
        renew = AsyncMock(return_value=True)

        async def work():
            async with job_lease(renew, interval=0.01):
                await asyncio.sleep(5)

        task = asyncio.create_task(work())
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_renewal_errors_do_not_kill_a_healthy_job(self):
        renew = AsyncMock(side_effect=[RuntimeError("db blip"), True, True, True, True])
        async with job_lease(renew, interval=0.01):
            await asyncio.sleep(0.04)
        assert renew.await_count >= 2
