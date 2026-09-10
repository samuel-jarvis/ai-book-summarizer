import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.core.database import Base, TimestampMixin
from app.models.document import EmbeddingStatus

if TYPE_CHECKING:
    from app.models.document import Document


class EmbeddingGeneration(TimestampMixin, Base):
    """One indexing build. Activate only after every chunk has been persisted."""

    __tablename__ = "embedding_generations"
    __table_args__ = (
        UniqueConstraint("id", "document_id"),
        CheckConstraint("NOT is_active OR status = 'COMPLETED'", name="active_requires_completed"),
        CheckConstraint(f"dimensions = {settings.EMBEDDING_DIMENSION}", name="supported_dimensions"),
        Index("uq_embedding_generations_active_document", "document_id", unique=True,
              postgresql_where=text("is_active"), sqlite_where=text("is_active")),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    chunking_version: Mapped[str] = mapped_column(String(100), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[EmbeddingStatus] = mapped_column(
        Enum(EmbeddingStatus, native_enum=False, length=20),
        default=EmbeddingStatus.PENDING, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false"), nullable=False)

    document: Mapped["Document"] = relationship(back_populates="embedding_generations")
