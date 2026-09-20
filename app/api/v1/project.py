import uuid
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, status
from redis.exceptions import RedisError

from app.api.deps import CurrentUser, DbSession
from app.schema.project import (
    CreateProjectForm,
    CreateProjectRequest,
    CreateProjectResponse,
    ProjectDetail,
    ProjectDetailResponse,
    ProjectListResponse,
    ProjectResponse,
)
from app.services.project.project_service import ProjectService
from app.taskiq_broker import is_task_queue_available
from app.workers.document_embedding import embed_document_task
from app.workers.document_summary import summarize_document_task

router = APIRouter(prefix="/project")


@router.post("", response_model=CreateProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    data: Annotated[CreateProjectForm, Form(media_type="multipart/form-data")],
    db: DbSession,
    current_user: CurrentUser,
):
    if not await is_task_queue_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The task queue is unavailable. Please try again later.",
        )

    payload = CreateProjectRequest(title=data.title)
    project, document = await ProjectService(db).create_project(
        payload, current_user.id, data.file
    )

    try:
        await summarize_document_task.kiq(
            str(document.id), str(project.id), str(document.summary_job_id)
        )
        await embed_document_task.kiq(str(document.id), str(project.id))
    except RedisError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The project was created, but background processing could not "
                "be queued. Please try again later."
            ),
        ) from exc

    return CreateProjectResponse(
        message="Project created successfully",
        data=ProjectResponse.model_validate(project),
    )


@router.get("", response_model=ProjectListResponse)
async def get_all_projects(db: DbSession, current_user: CurrentUser):
    projects = await ProjectService(db).get_all(current_user.id)

    return ProjectListResponse(
        message="Projects retrieved successfully",
        data=[ProjectResponse.model_validate(project) for project in projects],
    )


@router.get("/{project_id}", response_model=ProjectDetailResponse)
async def get_project(project_id: uuid.UUID, db: DbSession, current_user: CurrentUser):
    service = ProjectService(db)
    project = await service.get_by_id(project_id, current_user.id)
    summary = await service.get_first_document_summary(
        project_id, current_user.id
    )

    return ProjectDetailResponse(
        message="Project retrieved successfully",
        data=ProjectDetail(
            **ProjectResponse.model_validate(project).model_dump(),
            summary=summary,
        ),
    )
