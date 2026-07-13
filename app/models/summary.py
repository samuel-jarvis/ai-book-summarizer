import uuid
from typing import Optional
from sqlalchemy import Enum, ForeignKey, Index, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.core.database import Base


class SummaryStatus(enum.StrEnum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'


class Summary(Base):
    __tablename__ = "summaries"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)

    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    source_text: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[SummaryStatus] = mapped_column(
        Enum(SummaryStatus, native_enum=False, length=20),
        default=SummaryStatus.PENDING,
        nullable=False,
        index=True
    )
