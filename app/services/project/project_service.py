import uuid

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exception import NotFoundError
from app.models.document import Document
from app.models.document_summary import DocumentSummary
from app.models.project import Project
from app.models.project_documents import ProjectDocument
from app.schema.project import CreateProjectRequest
from app.services.document.document_service import DocumentService


class ProjectService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_project(
        self, data: CreateProjectRequest, user_id: uuid.UUID, file: UploadFile
    ) -> tuple[Project, Document]:
        """Create a project and its first document in one transaction.

        The document is stored first so an invalid upload never leaves a
        project behind, and the single commit at the end means a failure
        rolls back every row. The stored file is removed on failure because
        it lives outside the transaction.
        """
        document = await DocumentService(self.db).create_document(
            user_id=user_id, file=file
        )
        document.summary_job_id = uuid.uuid4()
        file_path = DocumentService.storage_path(document)

        try:
            project = Project(user_id=user_id, title=data.title)
            self.db.add(project)
            await self.db.flush()

            project_document = ProjectDocument(project_id=project.id, document_id=document.id)
            self.db.add(project_document)

            await self.db.commit()
        except Exception:
            if file_path is not None:
                file_path.unlink(missing_ok=True)
            raise

        await self.db.refresh(project)
        return project, document

    async def get_all(self, user_id: uuid.UUID) -> list[Project]:
        result = await self.db.scalars(select(Project).where(
            Project.user_id == user_id,
        ).order_by(Project.created_at.desc(), Project.id.desc()))
        return list(result.all())

    async def get_by_id(
        self, project_id: uuid.UUID, user_id: uuid.UUID
    ) -> Project:
        project = await self.db.scalar(
            select(Project).where(
                Project.id == project_id,
                Project.user_id == user_id,
            )
        )
        if project is None:
            raise NotFoundError("Project not found.")
        return project

    async def get_first_document_summary(
        self, project_id: uuid.UUID, user_id: uuid.UUID
    ) -> str | None:
        document_id = await self.db.scalar(
            select(ProjectDocument.document_id)
            .join(Project, Project.id == ProjectDocument.project_id)
            .where(
                ProjectDocument.project_id == project_id,
                Project.user_id == user_id,
            )
            .order_by(
                ProjectDocument.created_at.asc(),
                ProjectDocument.id.asc(),
            )
            .limit(1)
        )
        if document_id is None:
            return None

        return await self.db.scalar(
            select(DocumentSummary.summary)
            .where(DocumentSummary.document_id == document_id)
            .order_by(
                DocumentSummary.created_at.desc(),
                DocumentSummary.id.desc(),
            )
            .limit(1)
        )

    async def get_by_id_with_summary_join_unused(
        self, project_id: uuid.UUID, user_id: uuid.UUID
    ) -> tuple[Project, str | None]:
        """Previous single-query implementation, retained for reference."""
        result = await self.db.execute(
            select(Project, DocumentSummary.summary)
            .outerjoin(
                ProjectDocument,
                ProjectDocument.project_id == Project.id,
            )
            .outerjoin(
                DocumentSummary,
                DocumentSummary.document_id == ProjectDocument.document_id,
            )
            .where(
                Project.id == project_id,
                Project.user_id == user_id,
            )
            .order_by(
                ProjectDocument.created_at.asc().nulls_last(),
                ProjectDocument.id.asc().nulls_last(),
                DocumentSummary.created_at.desc().nulls_last(),
                DocumentSummary.id.desc().nulls_last(),
            )
            .limit(1)
        )
        row = result.one_or_none()
        if row is None:
            raise NotFoundError("Project not found.")
        project, summary = row
        return project, summary
