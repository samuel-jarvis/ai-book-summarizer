import asyncio
import uuid
import fitz  # pymupdf
import re
from pathlib import Path
from app.utils.prompts import CHUNK_PROMPT, SUMMARY_PROMPT
import tqdm
from app.services.ai_service import DeepSeekAIService
from app.core.database import AsyncSessionLocal
from app.models.summary import Summary, SummaryStatus

# if __package__ in (None, ""):
#     import sys

#     sys.path.append(str(Path(__file__).resolve().parents[2]))
#     from app.services.ai_service import DeepSeekAIService
# else:
#     from .ai_service import DeepSeekAIService

# to test the code without fastapi server
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# DEFAULT_BOOK_PATH = PROJECT_ROOT / "interview-questions.pdf"


# def resolve_pdf_path(file_path: str | None = None) -> str:
#     """Resolve PDF path robustly regardless of current working directory."""
#     path = Path(file_path).expanduser() if file_path else DEFAULT_BOOK_PATH
#     if not path.is_absolute():
#         path = PROJECT_ROOT / path

#     return str(path.resolve())


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


def chunk_text(text: str, chunk_size: int = 4000) -> list[str]:
    overlap = 400

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


async def summarize_chunks(chunks: list[str], prompt: str) -> list[str]:
    ai_service = DeepSeekAIService()
    summaries = []

    for chunk in tqdm.tqdm(chunks, total=len(chunks), desc="Summarizing chunks"):
        summary = await ai_service.summarize_text(build_chunk_prompt(chunk))
        summaries.append(summary)

    return summaries


def combine_summaries(summaries: list[str]) -> str:
    combined_summary = "\n\n".join(summaries)
    return combined_summary


async def summarize_combined_summary(combined_summary: str, prompt: str):
    ai_service = DeepSeekAIService()
    final_summary = await ai_service.summarize_text(build_summary_prompt(combined_summary))
    return final_summary if final_summary else "No summary generated."


async def summarize_pdf(text: str) -> str:
    # Chunk the cleaned text
    chunked_text = chunk_text(text)

    # Summarize each chunk
    chunk_summaries = await summarize_chunks(chunked_text, CHUNK_PROMPT)

    # Combine the chunk summaries
    combined_summary = combine_summaries(chunk_summaries)

    # Summarize the combined summary
    final_summary = await summarize_combined_summary(combined_summary, SUMMARY_PROMPT)

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
            final_summary = await summarize_pdf(summary.source_text)
        except Exception as e:
            print(f"Failed to summarize summary {summary_id}: {e}")
            summary.status = SummaryStatus.FAILED
            await db.commit()
            return

        summary.content = final_summary
        summary.status = SummaryStatus.COMPLETED
        await db.commit()


# if __name__ == "__main__":
#     pdf_path = resolve_pdf_path()
#     extracted_clean_text = extract_and_clean_text_from_pdf(pdf_path)
#     chunked_text = chunk_text(extracted_clean_text)
#     print(chunked_text)

#     # Example prompt
#     prompt = "Summarize the following text."
#     final_summary = asyncio.run(summarize_pdf(pdf_path))
#     print(final_summary)
