import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exception import NotFoundError
from app.models.document import Document
from app.models.project import Project
from app.models.project_documents import ProjectDocument


class ProjectService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def attach_document(
        self, project_id: uuid.UUID, document_id: uuid.UUID, user_id: uuid.UUID
    ) -> ProjectDocument:
        """Attach an owned library document; the caller commits the transaction.

        Lock the project to serialize duplicate attachment requests. Both IDs
        are scoped to the authenticated user, including existing attachments.
        """
        project = await self.db.scalar(
            select(Project).where(
                Project.id == project_id, Project.user_id == user_id
            ).with_for_update()
        )
        if project is None:
            raise NotFoundError("Project not found.")

        document = await self.db.scalar(
            select(Document).where(
                Document.id == document_id, Document.user_id == user_id
            ).with_for_update()
        )
        if document is None:
            raise NotFoundError("Document not found.")

        existing = await self.db.scalar(select(ProjectDocument).where(
            ProjectDocument.project_id == project_id,
            ProjectDocument.document_id == document_id,
        ))
        if existing is not None:
            return existing

        attachment = ProjectDocument(project_id=project_id, document_id=document_id)
        self.db.add(attachment)
        await self.db.flush()
        return attachment
