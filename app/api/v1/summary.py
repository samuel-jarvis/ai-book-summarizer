import uuid
from pathlib import Path
from typing_extensions import Annotated

from fastapi import APIRouter, Form, UploadFile, File, HTTPException, status
from redis.exceptions import RedisError

from app.schema.summary import SummarizeCreate, SummarizeCreateForm, SummarizeResponse, SummarizeDetailResponse, SummarizeApiResponse, SummarizeDetailApiResponse, SummarizeListApiResponse, SummarizeUpdate
from app.services.summary_service import SummaryService
from app.api.deps import DbSession
from app.tasks import process_pdf_task
from app.taskiq_broker import is_task_queue_available

UPLOAD_DIR = Path("temp_uploads").resolve()
MAX_UPLOAD_BYTES = 35 * 1024 * 1024  # 35 MB
CHUNK_SIZE = 1024 * 1024  # 1 MB

router = APIRouter()


@router.get("/summary", response_model=SummarizeListApiResponse)
async def get_summary(db: DbSession):
    summaries = await SummaryService(db).get_all()

    return SummarizeListApiResponse(
        success=True,
        message="Gotten all the books",
        data=[SummarizeResponse.model_validate(
            summary) for summary in summaries],
    )


@router.post("/summary", response_model=SummarizeApiResponse, status_code=status.HTTP_201_CREATED)
async def create_summary(
    title: Annotated[str, Form()],
    file: Annotated[UploadFile, File(description="Book file (pdf/epub)")],
    db: DbSession,
):
    if not await is_task_queue_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The background task queue is unavailable. Please try again later.",
        )

    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded"
        )

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_file_path = UPLOAD_DIR / f"{uuid.uuid4()}.pdf"

    total_bytes = 0
    file_too_large = False
    with open(temp_file_path, "wb") as f:
        while chunk := await file.read(CHUNK_SIZE):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                file_too_large = True
                break
            f.write(chunk)

    if file_too_large:
        temp_file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    payload = SummarizeCreateForm(title=title, file_path=str(temp_file_path))

    summary = await SummaryService(db).start_summary(data=payload)

    # Hand off the heavy PDF summarization to a background worker.
    try:
        await process_pdf_task.kiq(str(summary.id))
    except RedisError as exc:
        # Redis may go down between the availability check and task dispatch.
        await SummaryService(db).delete_book(summary.id)
        temp_file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The background task queue is unavailable. Please try again later.",
        ) from exc

    return SummarizeApiResponse(
        success=True,
        message="Summary created successfully",
        data=SummarizeResponse.model_validate(summary),
    )


@router.get("/summary/{summary_id}", response_model=SummarizeDetailApiResponse)
async def get_summary_by_id(summary_id: uuid.UUID, db: DbSession):
    summary = await SummaryService(db).get_summary_by_id(summary_id=summary_id)

    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Summary not found",
        )

    return SummarizeDetailApiResponse(
        success=True,
        message="Summary retrieved successfully",
        data=SummarizeDetailResponse.model_validate(summary),
    )


@router.delete("/summary/{summary_id}")
async def delete_summary_by_id(summary_id: uuid.UUID, db: DbSession):
    await SummaryService(db).delete_book(summary_id)

    return {
        "success": True,
        "message": "Summary deleted successfully"
    }


@router.put("/summary/{summary_id}", response_model=SummarizeApiResponse)
async def update_summary_title(summary_id: uuid.UUID, title: Annotated[str, Form()], db: DbSession):
    payload = SummarizeUpdate(title=title)
    await SummaryService(db).update_book_title(summary_id, payload)

    updated_summary = await SummaryService(db).get_summary_by_id(summary_id)

    return SummarizeApiResponse(
        success=True,
        message="Summary title updated successfully",
        data=SummarizeResponse.model_validate(updated_summary),
    )
