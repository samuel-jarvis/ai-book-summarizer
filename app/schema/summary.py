import uuid

from pydantic import BaseModel, ConfigDict, Field

from app.models.summary import SummaryStatus
from app.schema.response import ApiResponse


class SummarizeCreate(BaseModel):
    title: str | None = Field(default=None, min_length=3, max_length=255)
    source_text: str = Field(..., min_length=3)


class SummarizeCreateForm(BaseModel):
    title: str = Field(..., min_length=3, max_length=255)
    file_path: str = Field(..., min_length=1, max_length=1024)


class SummarizeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    status: SummaryStatus


class SummarizeDetailResponse(SummarizeResponse):
    content: str | None = None


class SummarizeUpdate(BaseModel):
    title: str


SummarizeApiResponse = ApiResponse[SummarizeResponse]
SummarizeDetailApiResponse = ApiResponse[SummarizeDetailResponse]
SummarizeListApiResponse = ApiResponse[list[SummarizeResponse]]
