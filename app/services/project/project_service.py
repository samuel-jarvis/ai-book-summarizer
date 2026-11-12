import uuid
from typing import NamedTuple

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exception import ConflictError, NotFoundError
from app.models.document import Document, DocumentSummaryStatus, EmbeddingStatus
from app.models.document_summary import DocumentSummary
from app.models.project import Project
from app.models.project_documents import ProjectDocument
from app.schema.project import CreateProjectRequest
from app.services.document.document_service import DocumentService


class DocumentJobStatus(NamedTuple):
    """A document's two background jobs as the client sees them."""

    summary_status: DocumentSummaryStatus
    embedding_status: EmbeddingStatus
    summary_error_code: str | None = None
    embedding_error_code: str | None = None

    @classmethod
    def of(cls, document: Document) -> "DocumentJobStatus":
        return cls(
            document.summary_status,
            document.embedding_status,
            document.summary_error_code,
            document.embedding_error_code,
        )


class RetryPlan(NamedTuple):
    """Which of a document's jobs a retry reset, so the route can re-dispatch them."""

    document: Document
    summary: bool
    embedding: bool


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
        # One fence token per background job: a worker whose job was re-queued
        # or retried can no longer write its (stale) result.
        document.summary_job_id = uuid.uuid4()
        document.embedding_job_id = uuid.uuid4()
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

    async def retry_failed_jobs(
        self, project_id: uuid.UUID, user_id: uuid.UUID
    ) -> RetryPlan:
        """Put the first document's FAILED jobs back to PENDING, ready to be queued again.

        Each reset job gets a fresh job id (so a straggling old worker is
        fenced) and its attempt counter starts over. Jobs that did not fail
        are untouched; with nothing failed this is a 409. The route dispatches
        after this commits, and the sweeper covers a dispatch that is lost.
        """
        document = await self.db.scalar(
            select(Document)
            .join(ProjectDocument, ProjectDocument.document_id == Document.id)
            .join(Project, Project.id == ProjectDocument.project_id)
            .where(
                ProjectDocument.project_id == project_id,
                Project.user_id == user_id,
            )
            .order_by(ProjectDocument.created_at.asc(), ProjectDocument.id.asc())
            .limit(1)
            .with_for_update(of=Document)
        )
        if document is None:
            raise NotFoundError("Project not found.")

        retry_summary = document.summary_status == DocumentSummaryStatus.FAILED
        retry_embedding = document.embedding_status == EmbeddingStatus.FAILED
        if not (retry_summary or retry_embedding):
            await self.db.rollback()
            raise ConflictError("Nothing to retry: no job on this project has failed.")

        if retry_summary:
            document.summary_status = DocumentSummaryStatus.PENDING
            document.summary_job_id = uuid.uuid4()
            document.summary_attempts = 0
            document.summary_lease_expires_at = None
            document.summary_error_code = None
        if retry_embedding:
            document.embedding_status = EmbeddingStatus.PENDING
            document.embedding_job_id = uuid.uuid4()
            document.embedding_attempts = 0
            document.embedding_lease_expires_at = None
            document.embedding_error_code = None
        await self.db.commit()
        return RetryPlan(document, retry_summary, retry_embedding)

    async def get_document_statuses(
        self, project_ids: list[uuid.UUID], user_id: uuid.UUID
    ) -> dict[uuid.UUID, DocumentJobStatus]:
        """Map each owned project to its first document's summary and embedding status.

        One query for any number of projects: DISTINCT ON keeps the earliest
        linked document per project, matching `get_first_document_summary`.
        Projects with no documents are simply absent from the result.
        """
        if not project_ids:
            return {}

        result = await self.db.execute(
            select(
                ProjectDocument.project_id,
                Document.summary_status,
                Document.embedding_status,
                Document.summary_error_code,
                Document.embedding_error_code,
            )
            .join(Document, Document.id == ProjectDocument.document_id)
            .join(Project, Project.id == ProjectDocument.project_id)
            .where(
                ProjectDocument.project_id.in_(project_ids),
                Project.user_id == user_id,
            )
            .order_by(
                ProjectDocument.project_id,
                ProjectDocument.created_at.asc(),
                ProjectDocument.id.asc(),
            )
            .distinct(ProjectDocument.project_id)
        )
        return {
            project_id: DocumentJobStatus(*statuses)
            for project_id, *statuses in result.all()
        }

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
