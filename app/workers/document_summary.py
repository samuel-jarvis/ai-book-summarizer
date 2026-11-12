"""Background document summarization."""

import asyncio
import logging
import time
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.document import Document, DocumentSummaryStatus
from app.models.document_summary import (
    DocumentSummary,
    DocumentSummaryMetadata,
    SummaryType,
)
from app.models.job_error import JobErrorCode
from app.models.project_documents import ProjectDocument
from app.services.document.document_service import DocumentService
from app.services.indexing.loaders import load_document
from app.services.indexing.splitters import split_summary_documents
from app.services.jobs.lease import (
    JobFailure,
    LeaseLost,
    claimable,
    failure_code,
    job_lease,
    lease_deadline,
    row_renewer,
)
from app.services.llm.llm_service import LLMService
from app.services.llm.prompts import CHUNK_SUMMARY_PROMPT, COMBINED_SUMMARY_PROMPT
from app.taskiq_broker import broker

logger = logging.getLogger(__name__)


class DocumentSummaryWorker:
    def __init__(
        self,
        db: AsyncSession,
        sessions: async_sessionmaker[AsyncSession] = AsyncSessionLocal,
    ) -> None:
        self.db = db
        # Short side sessions for the heartbeat and for recording the outcome.
        self.sessions = sessions

    async def process(
        self, *, document_id: UUID, project_id: UUID, job_id: UUID
    ) -> None:
        """Claim the job, summarize under a renewed lease, and record the outcome once."""
        document = await self._claim(document_id, project_id, job_id)
        if document is None:
            return
        renew = row_renewer(
            Document,
            Document.summary_lease_expires_at,
            *self._owned(document_id, job_id),
            sessions=self.sessions,
        )
        try:
            async with job_lease(renew):
                await self._summarize(document, project_id=project_id, job_id=job_id)
        except LeaseLost:
            # The sweeper or a user retry handed this job to someone else; any
            # result we produced would be fenced by the new job id anyway.
            await self.db.rollback()
        except asyncio.CancelledError:
            # Worker shutdown, not a failure: hand the job back and re-queue it.
            await self.db.rollback()
            await self._release(document_id, project_id, job_id)
            raise
        except Exception as exc:
            logger.exception("Document %s summary failed", document_id)
            await self.db.rollback()
            await self._fail(document_id, job_id, failure_code(exc))
            raise

    @staticmethod
    def _owned(document_id: UUID, job_id: UUID) -> tuple:
        """Conditions pinning the row to this claim; false once the job changed hands."""
        return (
            Document.id == document_id,
            Document.summary_job_id == job_id,
            Document.summary_status == DocumentSummaryStatus.STARTED,
        )

    async def _claim(
        self, document_id: UUID, project_id: UUID, job_id: UUID
    ) -> Document | None:
        """Take the job under a row lock, or return None when it is not ours to run.

        Redeliveries of a job that is already running (live lease) or finished
        are skipped. A STARTED job whose lease ran out was abandoned by a dead
        worker and is claimed again, which is what unsticks it.
        """
        document = await self._get_document(document_id, project_id, for_update=True)
        if (
            document is None
            or document.summary_job_id != job_id
            or not claimable(
                document.summary_status,
                document.summary_lease_expires_at,
                pending=DocumentSummaryStatus.PENDING,
                started=DocumentSummaryStatus.STARTED,
            )
        ):
            await self.db.rollback()
            return None

        document.summary_status = DocumentSummaryStatus.STARTED
        document.summary_attempts += 1
        document.summary_lease_expires_at = lease_deadline()
        document.summary_error_code = None
        await self.db.commit()
        return document

    async def _summarize(
        self, document: Document, *, project_id: UUID, job_id: UUID
    ) -> None:
        """Map-reduce the PDF into a DocumentSummary and publish it if still ours."""
        document_id = document.id
        started = time.perf_counter()

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
        chunks = split_summary_documents(source_documents)
        if not chunks:
            raise JobFailure(
                JobErrorCode.NO_TEXT, "The document contains no text to summarize."
            )

        llm = LLMService()
        semaphore = asyncio.Semaphore(settings.SUMMARY_MAX_CONCURRENCY)

        async def summarize_chunk(text: str) -> str:
            async with semaphore:
                return await llm.generate(CHUNK_SUMMARY_PROMPT, {"text": text})

        chunk_summaries = await asyncio.gather(
            *(summarize_chunk(chunk.page_content) for chunk in chunks)
        )
        summary = await LLMService(
            timeout=settings.DEEPSEEK_REDUCE_TIMEOUT_SECONDS
        ).generate(
            COMBINED_SUMMARY_PROMPT,
            {"summaries": "\n\n".join(chunk_summaries)},
        )

        # Re-read under a lock: a job that changed hands while we worked must
        # not overwrite the newer attempt's result.
        document = await self._get_document(document_id, project_id, for_update=True)
        if document is None or document.summary_job_id != job_id:
            await self.db.rollback()
            return

        document.extracted_text = "\n\n".join(
            page.page_content for page in source_documents
        )
        document.summary_status = DocumentSummaryStatus.COMPLETED
        document.summary_lease_expires_at = None

        print(f"Document {document_id} summary completed")

        self.db.add(
            DocumentSummary(
                document_id=document_id,
                model=llm.model,
                prompt=(
                    CHUNK_SUMMARY_PROMPT.pretty_repr()
                    + "\n\n"
                    + COMBINED_SUMMARY_PROMPT.pretty_repr()
                ),
                summary=summary,
                summary_type=SummaryType.DETAILED,
                summary_metadata=DocumentSummaryMetadata(
                    duration=round(time.perf_counter() - started),
                    extractor="pypdf",
                    page_count=len(source_documents),
                ),
            )
        )
        await self.db.commit()

    async def _fail(self, document_id: UUID, job_id: UUID, code: JobErrorCode) -> None:
        """Mark our attempt FAILED with a safe code, unless the job already changed hands."""
        async with self.sessions.begin() as db:
            await db.execute(
                update(Document)
                .where(*self._owned(document_id, job_id))
                .values(
                    summary_status=DocumentSummaryStatus.FAILED,
                    summary_lease_expires_at=None,
                    summary_error_code=code.value,
                )
            )

    async def _release(self, document_id: UUID, project_id: UUID, job_id: UUID) -> None:
        """Hand a job back on shutdown: PENDING again, the attempt uncounted, re-queued.

        If re-queueing fails (Redis is going down with us), the row stays
        PENDING and the sweeper re-queues it as a stale pending job.
        """
        async with self.sessions.begin() as db:
            result = await db.execute(
                update(Document)
                .where(*self._owned(document_id, job_id))
                .values(
                    summary_status=DocumentSummaryStatus.PENDING,
                    summary_lease_expires_at=None,
                    summary_attempts=Document.summary_attempts - 1,
                )
            )
        if result.rowcount != 1:
            return
        try:
            await summarize_document_task.kiq(
                str(document_id), str(project_id), str(job_id)
            )
        except Exception:  # noqa: BLE001 -- the sweeper covers a lost re-queue.
            logger.warning(
                "Could not re-queue document %s summary on shutdown", document_id
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
async def summarize_document_task(
    document_id: str, project_id: str, job_id: str
) -> None:
    async with AsyncSessionLocal() as db:
        await DocumentSummaryWorker(db).process(
            document_id=UUID(document_id),
            project_id=UUID(project_id),
            job_id=UUID(job_id),
        )
