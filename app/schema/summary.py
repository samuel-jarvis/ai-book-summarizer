from typing import Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field
from app.models.summary import SummaryStatus
from app.schema.response import ApiResponse


class SummarizeCreate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=3, max_length=255)


class SummarizeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: Optional[str]
    status: SummaryStatus


SummarizeApiResponse = ApiResponse[SummarizeResponse]
