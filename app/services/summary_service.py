from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.schema.summary import SummarizeCreate

from app.models.summary import Summary


class SummaryService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def start_summary(self, data: SummarizeCreate):
        new_summary = Summary(
            title=data.title
        )

        self.db.add(new_summary)
        await self.db.flush()
        await self.db.refresh(new_summary)
        return new_summary
