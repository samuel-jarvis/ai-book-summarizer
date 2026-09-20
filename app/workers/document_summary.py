"""Background document summarization."""

import asyncio
import time
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.document import Document, DocumentSummaryStatus
from app.models.document_summary import (
    DocumentSummary,
    DocumentSummaryMetadata,
    SummaryType,
)
from app.models.project_documents import ProjectDocument
from app.services.document.document_service import DocumentService
from app.services.indexing.loaders import load_document
from app.services.indexing.splitters import split_summary_documents
from app.services.llm.llm_service import LLMService
from app.services.llm.prompts import CHUNK_SUMMARY_PROMPT, COMBINED_SUMMARY_PROMPT
from app.taskiq_broker import broker


class DocumentSummaryWorker:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def process(
        self, *, document_id: UUID, project_id: UUID, job_id: UUID
    ) -> None:
        document = await self._get_document(
            document_id, project_id, for_update=True
        )
        if document is None or document.summary_job_id != job_id:
            await self.db.rollback()
            return
        if document.summary_status in {
            DocumentSummaryStatus.STARTED,
            DocumentSummaryStatus.COMPLETED,
        }:
            await self.db.rollback()
            return

        document.summary_status = DocumentSummaryStatus.STARTED
        await self.db.commit()

        started = time.perf_counter()
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
            chunks = split_summary_documents(source_documents)
            if not chunks:
                raise ValueError("The document contains no text to summarize.")

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

            document = await self._get_document(
                document_id, project_id, for_update=True
            )
            if document is None or document.summary_job_id != job_id:
                await self.db.rollback()
                return

            document.extracted_text = "\n\n".join(
                page.page_content for page in source_documents
            )
            document.summary_status = DocumentSummaryStatus.COMPLETED

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
        except Exception:
            print("Failed to summarize document")

            await self.db.rollback()
            document = await self._get_document(
                document_id, project_id, for_update=True
            )
            if document is not None and document.summary_job_id == job_id:
                document.summary_status = DocumentSummaryStatus.FAILED
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
async def summarize_document_task(
    document_id: str, project_id: str, job_id: str
) -> None:
    async with AsyncSessionLocal() as db:
        await DocumentSummaryWorker(db).process(
            document_id=UUID(document_id),
            project_id=UUID(project_id),
            job_id=UUID(job_id),
        )
