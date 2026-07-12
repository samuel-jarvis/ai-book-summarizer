import uuid
from pathlib import Path
from typing_extensions import Annotated

from fastapi import APIRouter, Form, UploadFile, File, HTTPException, status
from app.schema.summary import SummarizeCreate, SummarizeResponse, SummarizeDetailResponse, SummarizeApiResponse, SummarizeDetailApiResponse, SummarizeListApiResponse, SummarizeUpdate
from app.services.summary_service import SummaryService
from app.api.deps import DbSession
from app.tasks import process_pdf_task

UPLOAD_DIR = Path("temp_uploads").resolve()
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
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

    # Save the uploaded file to a safe temporary location using a UUID name
    # to prevent path traversal and filename-collision attacks.
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_file_path = UPLOAD_DIR / f"{uuid.uuid4()}.pdf"

    total_bytes = 0
    with open(temp_file_path, "wb") as f:
        while chunk := await file.read(CHUNK_SIZE):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                temp_file_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                )
            f.write(chunk)

    payload = SummarizeCreate(title=title, file_path=str(temp_file_path))

    summary = await SummaryService(db).start_summary(data=payload)

    # Hand off the heavy PDF summarization to a background worker.
    await process_pdf_task.kiq(str(summary.id))

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
