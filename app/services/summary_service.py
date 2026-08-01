import uuid
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import utcnow
from app.core.exception import NotFoundError, RateLimitError, ValidationError
from app.schema.summary import SummarizeCreate, SummarizeCreateForm, SummarizeUpdate

from app.models.summary import Summary, SummaryStatus
from app.services.processor_service import extract_text_from_pdf


def _start_of_utc_day(moment: datetime) -> datetime:
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


class SummaryService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def count_started_today(self, user_id: uuid.UUID) -> int:
        """How many summaries this user has started since midnight UTC.

        Failed runs don't count — a job that died on our side shouldn't burn
        one of the user's three attempts.
        """
        query = (
            select(func.count())
            .select_from(Summary)
            .where(
                Summary.user_id == user_id,
                Summary.created_at >= _start_of_utc_day(utcnow()),
                Summary.status != SummaryStatus.FAILED,
            )
        )

        result = await self.db.execute(query)
        return result.scalar_one()

    async def ensure_daily_quota(self, user_id: uuid.UUID) -> None:
        """Raise `RateLimitError` once the user is out of daily summaries.

        Checked before the upload is streamed to disk, so a user over quota
        never pays for the transfer or the text extraction.
        """
        limit = settings.DAILY_SUMMARY_LIMIT

        if limit <= 0 or await self.count_started_today(user_id) < limit:
            return

        now = utcnow()
        reset_at = _start_of_utc_day(now + timedelta(days=1))
        hours = max(1, -(-int((reset_at - now).total_seconds()) // 3600))

        raise RateLimitError(
            f"You've used all {limit} of your summaries for today. "
            "Lumen AI is still early, and we're managing compute carefully "
            "so every summary gets finished properly — so uploads are capped "
            f"for now. Your next {limit} unlock in about {hours} "
            f"hour{'s' if hours != 1 else ''} (midnight UTC). "
            "Thanks for bearing with us."
        )

    async def start_summary(
        self, data: SummarizeCreateForm, user_id: uuid.UUID
    ) -> Summary:
        source_text = extract_text_from_pdf(data.file_path)

        payload = SummarizeCreate(title=data.title, source_text=source_text)

        new_summary = Summary(
            user_id=user_id,
            title=payload.title,
            file_path=data.file_path,
            source_text=payload.source_text,
        )

        self.db.add(new_summary)
        await self.db.commit()

        await self.db.refresh(new_summary)
        return new_summary

    async def get_summary_by_id(self, summary_id: uuid.UUID, user_id: uuid.UUID):
        query = select(Summary).where(
            Summary.id == summary_id, Summary.user_id == user_id)

        result = await self.db.execute(query)
        summary = result.scalar_one_or_none()
        return summary

    async def get_all(self, user_id: uuid.UUID):
        query = select(Summary).where(Summary.user_id == user_id)

        result = await self.db.execute(query)
        data = result.scalars().all()
        return data

    async def delete_book(self, summary_id: uuid.UUID, user_id: uuid.UUID):
        summary = await self._get_owned(summary_id, user_id)

        if summary.status is SummaryStatus.COMPLETED:
            raise ValidationError("You can't delete this")

        await self.db.delete(summary)
        await self.db.flush()
        await self.db.commit()

    async def update_book_title(
        self, summary_id: uuid.UUID, data: SummarizeUpdate, user_id: uuid.UUID
    ):
        summary = await self._get_owned(summary_id, user_id)

        summary.title = data.title
        await self.db.flush()
        await self.db.commit()

    async def _get_owned(self, summary_id: uuid.UUID, user_id: uuid.UUID) -> Summary:
        summary = await self.get_summary_by_id(summary_id, user_id)

        if summary is None:
            raise NotFoundError("Summary not found")

        return summary
