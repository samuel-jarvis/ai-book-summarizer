import uuid
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.core.database import Base, TimestampMixin
from app.core.types import PydanticJSONB

if TYPE_CHECKING:
    from app.models.document import Document

EMBEDDING_DIMENSION = settings.EMBEDDING_DIMENSION

class DocumentEmbeddingMetadata(BaseModel):
    page_start: int | None = None
    page_end: int | None = None
    token_count: int


class DocumentEmbedding(TimestampMixin, Base):
    __tablename__ = "document_embeddings"
    __table_args__ = (
        ForeignKeyConstraint(
            ["generation_id", "document_id"],
            ["embedding_generations.id", "embedding_generations.document_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("document_id", "generation_id", "chunk_index"),
        CheckConstraint("chunk_index >= 0", name="chunk_index_nonnegative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)

    generation_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)

    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSION), nullable=False)

    embedding_metadata: Mapped[DocumentEmbeddingMetadata | None] = mapped_column(
        PydanticJSONB(DocumentEmbeddingMetadata), nullable=True)

    document: Mapped["Document"] = relationship(back_populates="embeddings")
