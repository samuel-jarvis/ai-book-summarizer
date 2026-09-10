
import enum
import uuid
from typing import TYPE_CHECKING

from pydantic import BaseModel
from sqlalchemy import Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base, TimestampMixin
from app.core.types import PydanticJSONB

if TYPE_CHECKING:
    from app.models.document import Document

class SummaryType(enum.Enum):
    SHORT = "short"
    DETAILED = "detailed"

class DocumentSummaryMetadata(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration: int | None = None
    extractor: str | None = None
    page_count: int | None = None

class DocumentSummary(TimestampMixin, Base):
    __tablename__ = "document_summaries"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    model: Mapped[str] = mapped_column(String(255))
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(nullable=False)
    summary_type: Mapped[SummaryType] = mapped_column(
        Enum(SummaryType),
        default=SummaryType.SHORT,
    )

    summary_metadata: Mapped[DocumentSummaryMetadata] = mapped_column(
        PydanticJSONB(DocumentSummaryMetadata), nullable=False,
        default=DocumentSummaryMetadata,
    )

    document: Mapped["Document"] = relationship(back_populates="summaries")
