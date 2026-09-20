import uuid
from datetime import datetime

from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict, Field

from app.schema.response import ApiResponse


class CreateProjectRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    title: str = Field(min_length=3, max_length=255)


class CreateProjectForm(CreateProjectRequest):
    file: UploadFile


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    description: str | None
    created_at: datetime
    updated_at: datetime


class ProjectDetail(ProjectResponse):
    summary: str | None = None


CreateProjectResponse = ApiResponse[ProjectResponse]
ProjectDetailResponse = ApiResponse[ProjectDetail]
ProjectListResponse = ApiResponse[list[ProjectResponse]]
