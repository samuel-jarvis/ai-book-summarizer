import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.schema.summary import SummarizeCreate

from app.models.summary import Summary


class SummaryService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def start_summary(self, data: SummarizeCreate):
        new_summary = Summary(
            title=data.title,
            file_path=data.file_path,
        )

        self.db.add(new_summary)
        await self.db.commit()

        await self.db.refresh(new_summary)
        return new_summary

    async def get_summary_by_id(self, summary_id: uuid.UUID):
        query = select(Summary).where(Summary.id == summary_id)

        result = await self.db.execute(query)
        summary = result.scalar_one_or_none()
        return summary

    async def get_all(self):
        query = select(Summary)

        result = await self.db.execute(query)
        data = result.scalars().all()
        return data
