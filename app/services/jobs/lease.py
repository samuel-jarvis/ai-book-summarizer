"""Job leases: a due date that a live worker keeps pushing forward.

Think of a job as a library book on loan. Claiming it writes
``lease_expires_at = now + JOB_LEASE_SECONDS``; while the worker is busy, a
small heartbeat task renews that time every ``JOB_HEARTBEAT_SECONDS``. A job
whose lease has passed belongs to a worker that died, and the sweeper
(``services/jobs/sweeper.py``) may hand it to someone else.

The heartbeat also tells a worker when the job is no longer its own: if the
renewing UPDATE matches no row (the status moved on, or the job id rotated
because of a re-queue or a user retry), the work is cancelled and surfaces as
``LeaseLost`` so no money is spent finishing a result that would be fenced.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.core.database import AsyncSessionLocal, utcnow
from app.models.job_error import JobErrorCode

logger = logging.getLogger(__name__)

Renewer = Callable[[], Awaitable[bool]]


class LeaseLost(Exception):
    """The job was handed to another worker, or retried, while we were working."""


class JobFailure(Exception):
    """A job failed for a reason we can safely name to the user."""

    def __init__(self, code: JobErrorCode, message: str = "") -> None:
        super().__init__(message or code.value)
        self.code = code


def failure_code(exc: BaseException) -> JobErrorCode:
    """Map an exception to the safe code stored on the row; never provider text."""
    if isinstance(exc, JobFailure):
        return exc.code
    return JobErrorCode.PROVIDER_ERROR


def lease_deadline(now: datetime | None = None) -> datetime:
    """When a claim made now stops being trusted if nobody renews it."""
    return (now or utcnow()) + timedelta(seconds=settings.JOB_LEASE_SECONDS)


def lease_expired(lease_expires_at: datetime | None, now: datetime | None = None) -> bool:
    """A missing lease counts as expired: the row predates leases or was never renewed."""
    return lease_expires_at is None or lease_expires_at <= (now or utcnow())


def claimable(
    status: object,
    lease_expires_at: datetime | None,
    *,
    pending: object,
    started: object,
    now: datetime | None = None,
) -> bool:
    """A job is free to claim when queued, or when its last claimant stopped renewing."""
    return status == pending or (
        status == started and lease_expired(lease_expires_at, now)
    )


def row_renewer(
    model: type,
    lease_column: object,
    *conditions: object,
    sessions: async_sessionmaker[AsyncSession] = AsyncSessionLocal,
) -> Renewer:
    """Build a heartbeat that extends ``lease_column`` on the one row matching ``conditions``.

    The conditions must pin the row to *this* claim (id, job id/token, and the
    in-progress status), so that a renewal returning False means the job is
    no longer ours. Each renewal uses its own short session, because the
    worker's main session may be inside a transaction.
    """

    async def renew() -> bool:
        async with sessions.begin() as db:
            result = await db.execute(
                update(model).where(*conditions).values({lease_column: lease_deadline()})
            )
            return result.rowcount == 1

    return renew


@asynccontextmanager
async def job_lease(
    renew: Renewer, *, interval: float | None = None
) -> AsyncIterator[None]:
    """Keep a lease renewed around a block of work; raise ``LeaseLost`` if it is taken away.

    Cancellation from outside (taskiq shutting the worker down) passes through
    unchanged as ``CancelledError`` so the caller can hand the job back. Only
    a cancellation this helper itself triggered, because a renewal found the
    row no longer ours, is turned into ``LeaseLost``.
    """
    owner = asyncio.current_task()
    interval = settings.JOB_HEARTBEAT_SECONDS if interval is None else interval
    lost = False

    async def beat() -> None:
        nonlocal lost
        while True:
            await asyncio.sleep(interval)
            try:
                still_ours = await renew()
            except Exception:  # noqa: BLE001 -- a database blip must not kill a healthy job.
                logger.warning("Lease heartbeat failed; trying again next beat", exc_info=True)
                continue
            if not still_ours:
                lost = True
                if owner is not None:
                    owner.cancel()
                return

    heartbeat = asyncio.create_task(beat())
    try:
        yield
    except asyncio.CancelledError:
        if lost:
            if owner is not None:
                owner.uncancel()
            raise LeaseLost() from None
        raise
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
