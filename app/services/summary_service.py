import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exception import NotFoundError, ValidationError
from app.schema.summary import SummarizeCreate, SummarizeCreateForm, SummarizeUpdate

from app.models.summary import Summary, SummaryStatus
from app.services.processor_service import extract_text_from_pdf


class SummaryService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

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
