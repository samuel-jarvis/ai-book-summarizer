"""Background document embedding and indexing."""

import asyncio
import logging
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import AsyncSessionLocal
from app.models.document import Document, EmbeddingStatus
from app.models.job_error import JobErrorCode
from app.models.project_documents import ProjectDocument
from app.services.document.document_service import DocumentService
from app.services.indexing.embedder import embed_texts
from app.services.indexing.loaders import load_document
from app.services.indexing.splitters import split_rag_documents
from app.services.indexing.vector_store import VectorStoreService
from app.services.jobs.lease import (
    JobFailure,
    LeaseLost,
    claimable,
    failure_code,
    job_lease,
    lease_deadline,
    row_renewer,
)
from app.taskiq_broker import broker

logger = logging.getLogger(__name__)


class DocumentEmbeddingWorker:
    def __init__(
        self,
        db: AsyncSession,
        sessions: async_sessionmaker[AsyncSession] = AsyncSessionLocal,
    ) -> None:
        self.db = db
        # Short side sessions for the heartbeat and for recording the outcome.
        self.sessions = sessions

    async def process(
        self, *, document_id: UUID, project_id: UUID, job_id: UUID | None = None
    ) -> None:
        """Claim the job, index under a renewed lease, and record the outcome once.

        `job_id` fences stale deliveries the way `summary_job_id` does for
        summaries. Messages queued before the fence existed carry none, so
        the claim then adopts (or mints) the row's `embedding_job_id`.
        """
        claimed = await self._claim(document_id, project_id, job_id)
        if claimed is None:
            return
        document, token = claimed
        renew = row_renewer(
            Document,
            Document.embedding_lease_expires_at,
            *self._owned(document_id, token),
            sessions=self.sessions,
        )
        try:
            async with job_lease(renew):
                await self._index(document, project_id=project_id, token=token)
        except LeaseLost:
            # Re-queued or retried behind our back; the new attempt owns the row.
            await self.db.rollback()
        except asyncio.CancelledError:
            # Worker shutdown, not a failure: hand the job back and re-queue it.
            await self.db.rollback()
            await self._release(document_id, project_id, token)
            raise
        except Exception as exc:
            logger.exception("Document %s embedding failed", document_id)
            await self.db.rollback()
            await self._fail(document_id, token, failure_code(exc))
            raise

    @staticmethod
    def _owned(document_id: UUID, token: UUID) -> tuple:
        """Conditions pinning the row to this claim; false once the job changed hands."""
        return (
            Document.id == document_id,
            Document.embedding_job_id == token,
            Document.embedding_status == EmbeddingStatus.STARTED,
        )

    async def _claim(
        self, document_id: UUID, project_id: UUID, job_id: UUID | None
    ) -> tuple[Document, UUID] | None:
        """Take the job under a row lock, or return None when it is not ours to run."""
        document = await self._get_document(document_id, project_id, for_update=True)
        if (
            document is None
            or (
                job_id is not None
                and document.embedding_job_id is not None
                and document.embedding_job_id != job_id
            )
            or not claimable(
                document.embedding_status,
                document.embedding_lease_expires_at,
                pending=EmbeddingStatus.PENDING,
                started=EmbeddingStatus.STARTED,
            )
        ):
            await self.db.rollback()
            return None

        token = job_id or document.embedding_job_id or uuid4()
        document.embedding_job_id = token
        document.embedding_status = EmbeddingStatus.STARTED
        document.embedding_attempts += 1
        document.embedding_lease_expires_at = lease_deadline()
        document.embedding_error_code = None
        await self.db.commit()
        return document, token

    async def _index(self, document: Document, *, project_id: UUID, token: UUID) -> None:
        """Split, embed, then swap in the new generation if the job is still ours."""
        document_id = document.id
        file_path = DocumentService.storage_path(document)
        if file_path is None or not file_path.exists():
            raise JobFailure(
                JobErrorCode.FILE_MISSING, "The document is not stored locally."
            )

        source_documents = await asyncio.to_thread(
            load_document,
            file_path,
            document_id=str(document_id),
            project_id=str(project_id),
        )
        chunks = split_rag_documents(source_documents)
        if not chunks:
            raise JobFailure(
                JobErrorCode.NO_TEXT, "The document contains no text to embed."
            )

        # Embed before taking any lock: a slow provider call must not hold the
        # document row, or the heartbeat's UPDATE would queue behind it.
        vectors = await embed_texts([chunk.page_content for chunk in chunks])

        locked = await self._get_document(document_id, project_id, for_update=True)
        if locked is None or locked.embedding_job_id != token:
            await self.db.rollback()
            return
        await VectorStoreService(self.db, document.user_id).add_document_chunks(
            chunks,
            document_id=document_id,
            project_id=project_id,
            vectors=vectors,
        )
        locked.embedding_lease_expires_at = None
        await self.db.commit()

    async def _fail(self, document_id: UUID, token: UUID, code: JobErrorCode) -> None:
        """Mark our attempt FAILED with a safe code, unless the job already changed hands."""
        async with self.sessions.begin() as db:
            await db.execute(
                update(Document)
                .where(*self._owned(document_id, token))
                .values(
                    embedding_status=EmbeddingStatus.FAILED,
                    embedding_lease_expires_at=None,
                    embedding_error_code=code.value,
                )
            )

    async def _release(self, document_id: UUID, project_id: UUID, token: UUID) -> None:
        """Hand a job back on shutdown: PENDING again, the attempt uncounted, re-queued."""
        async with self.sessions.begin() as db:
            result = await db.execute(
                update(Document)
                .where(*self._owned(document_id, token))
                .values(
                    embedding_status=EmbeddingStatus.PENDING,
                    embedding_lease_expires_at=None,
                    embedding_attempts=Document.embedding_attempts - 1,
                )
            )
        if result.rowcount != 1:
            return
        try:
            await embed_document_task.kiq(str(document_id), str(project_id), str(token))
        except Exception:  # noqa: BLE001 -- the sweeper covers a lost re-queue.
            logger.warning(
                "Could not re-queue document %s embedding on shutdown", document_id
            )

    async def _get_document(
        self, document_id: UUID, project_id: UUID, *, for_update: bool = False
    ) -> Document | None:
        query = (
            select(Document)
            .join(ProjectDocument, ProjectDocument.document_id == Document.id)
            .where(
                Document.id == document_id,
                ProjectDocument.project_id == project_id,
            )
        )
        if for_update:
            query = query.with_for_update(of=Document)
        return await self.db.scalar(query)


@broker.task
async def embed_document_task(
    document_id: str, project_id: str, job_id: str | None = None
) -> None:
    async with AsyncSessionLocal() as db:
        await DocumentEmbeddingWorker(db).process(
            document_id=UUID(document_id),
            project_id=UUID(project_id),
            job_id=UUID(job_id) if job_id else None,
        )
