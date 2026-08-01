import asyncio
import time
import uuid
import fitz  # pymupdf
import re
from pathlib import Path
from app.utils.prompts import CHUNK_PROMPT, SUMMARY_PROMPT
import tqdm
from app.core.config import settings
from app.services.ai_service import DeepSeekAIService
from app.core.database import AsyncSessionLocal
from app.models.summary import Summary, SummaryStatus


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

    return "\n".join(text)


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
        'deepseek-v4-pro',
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


async def process_summary(summary_id: str) -> None:
    async with AsyncSessionLocal() as db:
        summary = await db.get(Summary, uuid.UUID(summary_id))
        if summary is None:
            print(f"Summary {summary_id} not found; skipping.")
            return

        summary.status = SummaryStatus.PROCESSING
        await db.commit()

        try:
            final_summary = await summarize_pdf(
                summary.source_text,
                user_id=str(summary.user_id) if summary.user_id else None,
            )
        except Exception as e:
            # Worker log only. The client sees status FAILED and nothing else — no
            # provider detail is persisted anywhere the API can reach.
            print(f"Failed to summarize summary {summary_id}: {e}")
            summary.status = SummaryStatus.FAILED
            await db.commit()
            return

        summary.content = final_summary
        summary.status = SummaryStatus.COMPLETED
        await db.commit()
