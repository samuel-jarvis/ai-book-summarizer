from app.services.processor_service import process_summary
from app.taskiq_broker import broker


@broker.task
async def process_pdf_task(summary_id: str) -> None:
    await process_summary(summary_id)
