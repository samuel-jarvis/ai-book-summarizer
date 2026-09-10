import enum
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Enum,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.document_embedding import DocumentEmbedding
    from app.models.document_summary import DocumentSummary
    from app.models.embedding_generation import EmbeddingGeneration
    from app.models.project_documents import ProjectDocument
    from app.models.user import User


class ContentType(enum.StrEnum):
    PDF = "pdf"
    TEXT = "text"
    IMAGE = "image"
    RAW_TEXT = "raw_text"

class StorageProvider(enum.StrEnum):
    LOCAL = "local"
    S3 = "s3"
    R2 = "r2"
    CLOUDINARY = "cloudinary"

class EmbeddingStatus(enum.StrEnum):
    PENDING = "pending"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"

class SummaryStatus(enum.StrEnum):
    PENDING = "pending"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"

class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("file_size_bytes >= 0", name="file_size_nonnegative"),
        Index("ix_documents_user_id_content_hash", "user_id", "content_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # SHA-256 hex digest of original uploaded bytes (UTF-8 for raw text).
    # Hashes are intentionally non-unique: users may keep duplicate uploads.
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    mime_type: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    content_type: Mapped[ContentType | None] = mapped_column(
        Enum(ContentType, native_enum=False, length=20),
        nullable=True,
        index=True,
    )
    original_filename: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # storage
    storage_provider: Mapped[StorageProvider | None] = mapped_column(
        Enum(StorageProvider, native_enum=False, length=20),
        nullable=True,
        index=True,
    )
    storage_key: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    storage_url: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    extracted_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # processing
    embedding_status: Mapped[EmbeddingStatus | None] = mapped_column(
        Enum(EmbeddingStatus, native_enum=False, length=20),
        default=EmbeddingStatus.PENDING,
        nullable=True,
        index=True,
    )

    # Current/latest generation; completed summaries remain separate history rows.
    summary_status: Mapped[SummaryStatus] = mapped_column(
        Enum(SummaryStatus, native_enum=False, length=20),
        default=SummaryStatus.PENDING,
        nullable=False,
        index=True,
    )

    # Identifies the reserved summary job and fences stale worker completions.
    summary_job_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)

    embedding_generations: Mapped[list["EmbeddingGeneration"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True)

    user: Mapped["User"] = relationship("User", back_populates="documents")
    embeddings: Mapped[list["DocumentEmbedding"]] = relationship(
        back_populates="document", cascade="all, delete-orphan")
    summaries: Mapped[list["DocumentSummary"]] = relationship(
        back_populates="document", cascade="all, delete-orphan")
    project_documents: Mapped[list["ProjectDocument"]] = relationship(
        back_populates="document", cascade="all, delete-orphan")
