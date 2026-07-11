from typing_extensions import Annotated

from fastapi import APIRouter, Form, UploadFile, File, status
from app.schema.summary import SummarizeCreate, SummarizeResponse, SummarizeApiResponse
from app.services.summary_service import SummaryService
from app.api.deps import DbSession

router = APIRouter()


@router.get("/summary")
async def get_summary():
    return {"message": "This is the summary endpoint."}


@router.post("/summary", response_model=SummarizeApiResponse, status_code=status.HTTP_201_CREATED)
async def create_summary(
    title: Annotated[str, Form()],
    file: Annotated[UploadFile, File(description="Book file (pdf/epub)")],
    db: DbSession,
):
    payload = SummarizeCreate(title=title)

    summary = await SummaryService(db).start_summary(data=payload)

    return SummarizeApiResponse(
        success=True,
        message="Summary created successfully",
        data=SummarizeResponse.model_validate(summary),
    )
