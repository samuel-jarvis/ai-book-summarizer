import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.user import User


class SummaryStatus(enum.StrEnum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'


class Summary(TimestampMixin, Base):
    __tablename__ = "summaries"
    __table_args__ = (
        # The sweeper looks for PROCESSING rows whose worker stopped renewing.
        Index(
            "ix_summaries_lease",
            "lease_expires_at",
            postgresql_where=text("status = 'PROCESSING'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)

    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)

    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[SummaryStatus] = mapped_column(
        Enum(SummaryStatus, native_enum=False, length=20),
        default=SummaryStatus.PENDING,
        nullable=False,
        index=True
    )

    # Job lease bookkeeping (see services/jobs). `job_id` fences a slow worker
    # whose job was handed to someone else; `attempts` caps automatic re-runs;
    # a PROCESSING row past `lease_expires_at` belongs to a dead worker.
    job_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    attempts: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    user: Mapped[Optional["User"]] = relationship(back_populates="summaries")
