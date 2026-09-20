"""Background document embedding and indexing."""

import asyncio
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models.document import Document, EmbeddingStatus
from app.models.project_documents import ProjectDocument
from app.services.document.document_service import DocumentService
from app.services.indexing.loaders import load_document
from app.services.indexing.splitters import split_rag_documents
from app.services.indexing.vector_store import VectorStoreService
from app.taskiq_broker import broker


class DocumentEmbeddingWorker:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def process(self, *, document_id: UUID, project_id: UUID) -> None:
        document = await self._get_document(
            document_id, project_id, for_update=True
        )
        if document is None:
            await self.db.rollback()
            return
        if document.embedding_status in {
            EmbeddingStatus.STARTED,
            EmbeddingStatus.COMPLETED,
        }:
            await self.db.rollback()
            return

        document.embedding_status = EmbeddingStatus.STARTED
        await self.db.commit()

        try:
            file_path = DocumentService.storage_path(document)
            if file_path is None:
                raise ValueError("The document is not stored locally.")

            source_documents = await asyncio.to_thread(
                load_document,
                file_path,
                document_id=str(document_id),
                project_id=str(project_id),
            )
            chunks = split_rag_documents(source_documents)
            if not chunks:
                raise ValueError("The document contains no text to embed.")

            await VectorStoreService(self.db, document.user_id).add_document_chunks(
                chunks,
                document_id=document_id,
                project_id=project_id,
            )
            await self.db.commit()
        except Exception:
            await self.db.rollback()
            document = await self._get_document(
                document_id, project_id, for_update=True
            )
            if document is not None:
                document.embedding_status = EmbeddingStatus.FAILED
                await self.db.commit()
            raise

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
async def embed_document_task(document_id: str, project_id: str) -> None:
    async with AsyncSessionLocal() as db:
        await DocumentEmbeddingWorker(db).process(
            document_id=UUID(document_id),
            project_id=UUID(project_id),
        )
