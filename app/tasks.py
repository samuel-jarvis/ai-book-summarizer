from app.services.processor_service import process_summary
from app.taskiq_broker import broker
from app.workers.document_embedding import embed_document_task
from app.workers.document_summary import summarize_document_task

__all__ = [
    "embed_document_task",
    "process_pdf_task",
    "summarize_document_task",
]


@broker.task
async def process_pdf_task(summary_id: str) -> None:
    await process_summary(summary_id)
