import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exception import ConflictError
from app.models.document import Document, EmbeddingStatus, SummaryStatus
from app.models.document_embedding import DocumentEmbedding
from app.models.document_summary import DocumentSummary
from app.models.embedding_generation import EmbeddingGeneration


class DocumentProcessingService:
    """Transaction helpers for the new pipeline. Callers commit before enqueueing.

    These methods flush but never commit, allowing result persistence and status
    transitions to share a transaction. They do not dispatch or execute jobs.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def reserve_summary(self, document_id: uuid.UUID, user_id: uuid.UUID) -> uuid.UUID:
        job_id = uuid.uuid4()
        claimed = await self.db.scalar(
            update(Document).where(
                Document.id == document_id,
                Document.user_id == user_id,
                Document.summary_job_id.is_(None),
                Document.summary_status != SummaryStatus.STARTED,
            ).values(summary_job_id=job_id, summary_status=SummaryStatus.PENDING)
            .returning(Document.id)
        )
        if claimed is None:
            raise ConflictError("Document unavailable or summary already scheduled.")
        return job_id

    async def start_summary(self, document_id: uuid.UUID, job_id: uuid.UUID) -> None:
        claimed = await self.db.scalar(
            update(Document).where(
                Document.id == document_id, Document.summary_job_id == job_id,
                Document.summary_status == SummaryStatus.PENDING,
            ).values(summary_status=SummaryStatus.STARTED).returning(Document.id)
        )
        if claimed is None:
            raise ConflictError("Summary job is no longer pending.")

    async def finish_summary(self, job_id: uuid.UUID, summary: DocumentSummary) -> None:
        document = await self.db.scalar(select(Document).where(
            Document.id == summary.document_id, Document.summary_job_id == job_id,
            Document.summary_status == SummaryStatus.STARTED,
        ).with_for_update())
        if document is None:
            raise ConflictError("Summary job is no longer active.")
        self.db.add(summary)
        document.summary_status = SummaryStatus.COMPLETED
        document.summary_job_id = None
        await self.db.flush()

    async def fail_summary(self, document_id: uuid.UUID, job_id: uuid.UUID) -> None:
        claimed = await self.db.scalar(
            update(Document).where(
                Document.id == document_id, Document.summary_job_id == job_id,
                Document.summary_status.in_([SummaryStatus.PENDING, SummaryStatus.STARTED]),
            ).values(summary_status=SummaryStatus.FAILED, summary_job_id=None)
            .returning(Document.id)
        )
        if claimed is None:
            raise ConflictError("Summary job is no longer active.")

    async def activate_embeddings(
        self, document_id: uuid.UUID, generation_id: uuid.UUID, expected_chunks: int
    ) -> None:
        """Publish a complete build atomically; caller commits the switch.

        The worker must finish writing chunks before calling this and must not
        modify published generations. Retrieval uses active_embeddings_query.
        """
        if expected_chunks <= 0:
            raise ValueError("expected_chunks must be positive")
        document = await self.db.scalar(select(Document).where(
            Document.id == document_id).with_for_update())
        generation = await self.db.scalar(select(EmbeddingGeneration).where(
            EmbeddingGeneration.id == generation_id,
            EmbeddingGeneration.document_id == document_id,
        ).with_for_update())
        if document is None or generation is None:
            raise ConflictError("Embedding generation unavailable.")
        await self.db.flush()
        result = await self.db.execute(select(
            func.count(), func.min(DocumentEmbedding.chunk_index),
            func.max(DocumentEmbedding.chunk_index),
        ).where(DocumentEmbedding.generation_id == generation_id))
        count, first, last = result.one()
        if (count, first, last) != (expected_chunks, 0, expected_chunks - 1):
            raise ConflictError("Embedding generation is incomplete.")
        await self.db.execute(update(EmbeddingGeneration).where(
            EmbeddingGeneration.document_id == document_id,
            EmbeddingGeneration.is_active.is_(True),
        ).values(is_active=False))
        generation.status = EmbeddingStatus.COMPLETED
        generation.is_active = True
        document.embedding_status = EmbeddingStatus.COMPLETED
        await self.db.flush()

    @staticmethod
    def active_embeddings_query(document_id: uuid.UUID, user_id: uuid.UUID):
        """Base retrieval query; add vector ordering and limits in the pipeline."""
        return select(DocumentEmbedding).join(
            EmbeddingGeneration,
            (DocumentEmbedding.generation_id == EmbeddingGeneration.id)
            & (DocumentEmbedding.document_id == EmbeddingGeneration.document_id),
        ).join(Document, Document.id == DocumentEmbedding.document_id).where(
            Document.id == document_id, Document.user_id == user_id,
            EmbeddingGeneration.is_active.is_(True),
            EmbeddingGeneration.status == EmbeddingStatus.COMPLETED,
        )
