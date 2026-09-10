import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.project import Project


class ProjectDocument(TimestampMixin, Base):
    __tablename__ = "project_documents"
    __table_args__ = (UniqueConstraint("project_id", "document_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)

    document: Mapped["Document"] = relationship("Document", back_populates="project_documents")
    project: Mapped["Project"] = relationship(back_populates="project_documents")
