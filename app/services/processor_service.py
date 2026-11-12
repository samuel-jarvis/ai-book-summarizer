import asyncio
import re
import time
import uuid
from pathlib import Path

import fitz  # pymupdf
import tqdm
from sqlalchemy import select, update

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.summary import Summary, SummaryStatus
from app.services.ai_service import DeepSeekAIService
from app.services.jobs.lease import (
    LeaseLost,
    claimable,
    job_lease,
    lease_deadline,
    row_renewer,
)
from app.utils.prompts import CHUNK_PROMPT, SUMMARY_PROMPT


def build_chunk_prompt(text: str) -> str:
    return CHUNK_PROMPT.format(text=text)


def build_summary_prompt(text: str) -> str:
    return SUMMARY_PROMPT.format(text=text)


def extract_text_from_pdf(file_path: str) -> str:
    text = []
    try:
        if not Path(file_path).exists():
            print(
                f"Error extracting text from PDF: no such file: '{file_path}'")
            return ""

        pdf_document = fitz.open(file_path)

        # Iterate through each page and extract text
        for page_num in tqdm.tqdm(range(pdf_document.page_count), desc="Extracting text from PDF"):
            page = pdf_document.load_page(page_num)
            text.append(page.get_text())

        pdf_document.close()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")

    # PostgreSQL text columns reject NUL bytes, which some PDFs emit.
    return "\n".join(text).replace("\x00", "")


def clean_text(text: str) -> str:
    # Remove extra whitespace
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'[^\S\r\n]+', ' ', text)

    text = text.strip()  # Remove leading and trailing

    return text


def extract_and_clean_text_from_pdf(file_path: str) -> str:
    raw_text = extract_text_from_pdf(file_path)
    cleaned_text = clean_text(raw_text)
    return cleaned_text


def chunk_text(
    text: str,
    chunk_size: int = settings.SUMMARY_CHUNK_SIZE,
    overlap: int = settings.SUMMARY_CHUNK_OVERLAP,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    text_length = len(text)
    chunks = []
    start = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap if end < text_length else text_length

    print(f"Created {len(chunks)} chunks")
    return chunks


async def summarize_chunks(
    chunks: list[str], prompt: str, user_id: str | None = None
) -> list[str]:
    ai_service = DeepSeekAIService()

    semaphore = asyncio.Semaphore(settings.SUMMARY_MAX_CONCURRENCY)
    total = len(chunks)
    completed = 0

    async def summarize_one(index: int, chunk: str) -> str:
        nonlocal completed
        async with semaphore:
            summary = await ai_service.summarize_text(
                build_chunk_prompt(chunk), user_id=user_id
            )
            completed += 1
            print(
                f"[chunk] {index + 1}/{total} returned ({completed}/{total} complete)")
            return summary

    return list(
        await asyncio.gather(
            *(summarize_one(index, chunk) for index, chunk in enumerate(chunks))
        )
    )


def combine_summaries(summaries: list[str]) -> str:
    combined_summary = "\n\n".join(summaries)
    return combined_summary


async def summarize_combined_summary(
    combined_summary: str, prompt: str, user_id: str | None = None
):
    ai_service = DeepSeekAIService(
        'deepseek-v4-flash',
        timeout=settings.DEEPSEEK_REDUCE_TIMEOUT_SECONDS,
    )
    final_summary = await ai_service.summarize_text(
        build_summary_prompt(combined_summary), user_id=user_id
    )
    return final_summary if final_summary else "No summary generated."


async def summarize_pdf(text: str, user_id: str | None = None) -> str:
    pipeline_started = time.perf_counter()

    # Chunk the cleaned text
    chunked_text = chunk_text(text)
    print(
        f"[timing] chunking: {len(text):,} chars -> {len(chunked_text)} chunks "
        f"(size {settings.SUMMARY_CHUNK_SIZE}, overlap {settings.SUMMARY_CHUNK_OVERLAP})"
    )

    # Summarize each chunk
    started = time.perf_counter()
    chunk_summaries = await summarize_chunks(chunked_text, CHUNK_PROMPT, user_id)
    map_elapsed = time.perf_counter() - started
    print(
        f"[timing] map stage: {map_elapsed:.1f}s for {len(chunked_text)} chunks "
        f"({map_elapsed / max(len(chunked_text), 1):.1f}s/chunk wall avg, "
        f"concurrency {settings.SUMMARY_MAX_CONCURRENCY})"
    )

    # Combine the chunk summaries
    combined_summary = combine_summaries(chunk_summaries)

    # Summarize the combined summary
    started = time.perf_counter()
    final_summary = await summarize_combined_summary(
        combined_summary, SUMMARY_PROMPT, user_id
    )
    print(
        f"[timing] reduce stage: {time.perf_counter() - started:.1f}s "
        f"({len(combined_summary):,} chars in -> {len(final_summary):,} chars out)"
    )

    print(f"[timing] total: {time.perf_counter() - pipeline_started:.1f}s")

    return final_summary


def _owned_summary(summary_id: uuid.UUID, job_id: uuid.UUID) -> tuple:
    """Conditions pinning the row to this claim; false once the job changed hands."""
    return (
        Summary.id == summary_id,
        Summary.job_id == job_id,
        Summary.status == SummaryStatus.PROCESSING,
    )


async def _finish_summary(
    summary_id: uuid.UUID, job_id: uuid.UUID, *, status: SummaryStatus, content: str | None = None
) -> None:
    """Record the outcome only if we still own the job, so a stale worker writes nothing."""
    values: dict = {"status": status, "lease_expires_at": None}
    if content is not None:
        values["content"] = content
    async with AsyncSessionLocal.begin() as db:
        await db.execute(
            update(Summary).where(*_owned_summary(summary_id, job_id)).values(**values)
        )


async def _release_summary(summary_id: uuid.UUID, job_id: uuid.UUID) -> None:
    """Hand the job back on shutdown: PENDING again, the attempt uncounted, re-queued."""
    async with AsyncSessionLocal.begin() as db:
        result = await db.execute(
            update(Summary)
            .where(*_owned_summary(summary_id, job_id))
            .values(
                status=SummaryStatus.PENDING,
                lease_expires_at=None,
                attempts=Summary.attempts - 1,
            )
        )
    if result.rowcount != 1:
        return
    from app.tasks import process_pdf_task  # noqa: PLC0415 -- app.tasks imports this module.

    try:
        await process_pdf_task.kiq(str(summary_id))
    except Exception:  # noqa: BLE001 -- the sweeper re-queues stale PENDING rows.
        print(f"Could not re-queue summary {summary_id} on shutdown")


async def process_summary(summary_id: str) -> None:
    """Claim the summary job under a lease, run the map-reduce, and record the outcome once.

    A PENDING row, or a PROCESSING row whose worker stopped renewing its lease,
    is claimed; anything else (already running, finished) is a harmless
    redelivery and is skipped, so a job can no longer run twice.
    """
    summary_uuid = uuid.UUID(summary_id)
    async with AsyncSessionLocal() as db:
        summary = await db.scalar(
            select(Summary).where(Summary.id == summary_uuid).with_for_update()
        )
        if summary is None:
            print(f"Summary {summary_id} not found; skipping.")
            await db.rollback()
            return
        if not claimable(
            summary.status,
            summary.lease_expires_at,
            pending=SummaryStatus.PENDING,
            started=SummaryStatus.PROCESSING,
        ):
            await db.rollback()
            return

        job_id = uuid.uuid4()
        summary.status = SummaryStatus.PROCESSING
        summary.job_id = job_id
        summary.attempts += 1
        summary.lease_expires_at = lease_deadline()
        await db.commit()
        source_text = summary.source_text
        user_id = str(summary.user_id) if summary.user_id else None

    renew = row_renewer(
        Summary, Summary.lease_expires_at, *_owned_summary(summary_uuid, job_id)
    )
    try:
        async with job_lease(renew):
            final_summary = await summarize_pdf(source_text, user_id=user_id)
    except LeaseLost:
        # Re-queued or retried behind our back; the new attempt owns the row.
        return
    except asyncio.CancelledError:
        # Worker shutdown, not a failure: hand the job back and re-queue it.
        await _release_summary(summary_uuid, job_id)
        raise
    except Exception as e:
        # Worker log only. The client sees status FAILED and nothing else — no
        # provider detail is persisted anywhere the API can reach.
        print(f"Failed to summarize summary {summary_id}: {e}")
        await _finish_summary(summary_uuid, job_id, status=SummaryStatus.FAILED)
        return

    await _finish_summary(
        summary_uuid, job_id, status=SummaryStatus.COMPLETED, content=final_summary
    )
