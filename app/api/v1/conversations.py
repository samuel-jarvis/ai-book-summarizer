"""Authenticated conversation management, asynchronous messages, and run polling."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, status

from app.api.deps import CurrentUser, DbSession
from app.schema.conversation import (
    ConversationApiResponse,
    ConversationListApiResponse,
    ConversationResponse,
    CreateConversationRequest,
    MessagePage,
    MessagePageApiResponse,
    MessageResponse,
    RenameConversationRequest,
    RunApiResponse,
    RunDetailApiResponse,
    RunDetailResponse,
    RunListApiResponse,
    RunResponse,
    SendMessageRequest,
    SourcesApiResponse,
    SubmittedTurn,
    SubmittedTurnApiResponse,
)
from app.schema.response import ApiResponse
from app.services.conversation_dispatcher import dispatch_conversation
from app.services.conversation_service import ConversationService

router = APIRouter()
Limit = Annotated[int, Query(ge=1, le=200)]
Offset = Annotated[int, Query(ge=0)]


@router.post(
    "/conversations",
    response_model=ConversationApiResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    payload: CreateConversationRequest, db: DbSession, current_user: CurrentUser
):
    """Create a conversation in an owned project."""
    conversation = await ConversationService(db, current_user.id).create(
        payload.project_id, payload.title
    )
    await db.commit()
    return ConversationApiResponse(
        message="Conversation created.",
        data=ConversationResponse.model_validate(conversation),
    )


@router.get("/conversations", response_model=ConversationListApiResponse)
async def list_conversations(
    project_id: UUID,
    db: DbSession,
    current_user: CurrentUser,
    limit: Limit = 50,
    offset: Offset = 0,
):
    """List owned project conversations in most-recently-updated order."""
    rows = await ConversationService(db, current_user.id).list_conversations(
        project_id, limit=limit, offset=offset
    )
    return ConversationListApiResponse(
        message="Conversations retrieved.",
        data=[ConversationResponse.model_validate(row) for row in rows],
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationApiResponse)
async def get_conversation(
    conversation_id: UUID, db: DbSession, current_user: CurrentUser
):
    """Read one conversation after checking its project's ownership."""
    row = await ConversationService(db, current_user.id).get(conversation_id)
    return ConversationApiResponse(
        message="Conversation retrieved.", data=ConversationResponse.model_validate(row)
    )


@router.patch(
    "/conversations/{conversation_id}", response_model=ConversationApiResponse
)
async def rename_conversation(
    conversation_id: UUID,
    payload: RenameConversationRequest,
    db: DbSession,
    current_user: CurrentUser,
):
    """Rename a conversation without changing its messages or attempts."""
    row = await ConversationService(db, current_user.id).rename(
        conversation_id, payload.title
    )
    await db.commit()
    return ConversationApiResponse(
        message="Conversation renamed.", data=ConversationResponse.model_validate(row)
    )


@router.delete("/conversations/{conversation_id}", response_model=ApiResponse[None])
async def delete_conversation(
    conversation_id: UUID, db: DbSession, current_user: CurrentUser
):
    """Delete the owned conversation and its cascaded messages and attempts."""
    await ConversationService(db, current_user.id).delete(conversation_id)
    await db.commit()
    return ApiResponse[None](message="Conversation deleted.")


@router.get(
    "/conversations/{conversation_id}/messages", response_model=MessagePageApiResponse
)
async def list_messages(
    conversation_id: UUID,
    db: DbSession,
    current_user: CurrentUser,
    after_position: Offset = 0,
    limit: Limit = 50,
):
    """Return chronological messages with their latest assistant attempts and a cursor."""
    service = ConversationService(db, current_user.id)
    messages = await service.list_messages(
        conversation_id, after_position=after_position, limit=limit
    )
    latest = await service.latest_message_runs(conversation_id, messages)
    views = []
    for message in messages:
        view = MessageResponse.model_validate(message)
        if message.id in latest:
            view.latest_run = RunResponse.from_run(latest[message.id])
        views.append(view)
    cursor = messages[-1].position if len(messages) == limit else None
    return MessagePageApiResponse(
        message="Messages retrieved.",
        data=MessagePage(messages=views, next_after_position=cursor),
    )


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=SubmittedTurnApiResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def send_message(
    conversation_id: UUID,
    payload: SendMessageRequest,
    db: DbSession,
    current_user: CurrentUser,
):
    """Reserve and commit an idempotent turn before dispatching its background attempt."""
    turn = await ConversationService(db, current_user.id).reserve_turn(
        conversation_id,
        content=payload.content,
        client_request_id=payload.client_request_id,
    )
    await db.commit()
    if not turn.reused:
        await dispatch_conversation(db, turn.run.id, current_user.id)
    return SubmittedTurnApiResponse(
        message="Message accepted.",
        data=SubmittedTurn(
            user_message=MessageResponse.model_validate(turn.user_message),
            assistant_message=MessageResponse.model_validate(turn.assistant_message),
            run=RunResponse.from_run(turn.run),
            reused=turn.reused,
        ),
    )


@router.get("/conversations/{conversation_id}/runs", response_model=RunListApiResponse)
async def list_runs(
    conversation_id: UUID,
    db: DbSession,
    current_user: CurrentUser,
    limit: Limit = 50,
    offset: Offset = 0,
):
    """List all owned attempts, including failed attempts retained after retries."""
    rows = await ConversationService(db, current_user.id).list_runs(
        conversation_id, limit=limit, offset=offset
    )
    return RunListApiResponse(
        message="Runs retrieved.", data=[RunResponse.from_run(row) for row in rows]
    )


@router.get("/conversation-runs/{run_id}", response_model=RunApiResponse)
async def get_run(run_id: UUID, db: DbSession, current_user: CurrentUser):
    """Poll status, answer text, and captured sources without transferring full prompts."""
    run = await ConversationService(db, current_user.id).get_run(run_id)
    return RunApiResponse(message="Run retrieved.", data=RunResponse.from_run(run))


@router.get("/conversation-runs/{run_id}/debug", response_model=RunDetailApiResponse)
async def get_run_debug(run_id: UUID, db: DbSession, current_user: CurrentUser):
    """Inspect the owned attempt's exact saved prompts, token budgets, and metrics."""
    run = await ConversationService(db, current_user.id).get_run(run_id)
    return RunDetailApiResponse(
        message="Run retrieved.", data=RunDetailResponse.from_run(run)
    )


@router.get("/conversation-runs/{run_id}/sources", response_model=SourcesApiResponse)
async def get_sources(run_id: UUID, db: DbSession, current_user: CurrentUser):
    """Return the run's captured sources without querying current vector contents."""
    run = await ConversationService(db, current_user.id).get_run(run_id)
    return SourcesApiResponse(
        message="Sources retrieved.", data=run.snapshot.selected_sources
    )


@router.post("/conversation-runs/{run_id}/cancel", response_model=RunApiResponse)
async def cancel_run(run_id: UUID, db: DbSession, current_user: CurrentUser):
    """Cancel an active attempt; repeating a cancellation is harmless."""
    run = await ConversationService(db, current_user.id).cancel(run_id)
    await db.commit()
    return RunApiResponse(message="Run cancelled.", data=RunResponse.from_run(run))


@router.post(
    "/conversation-runs/{run_id}/retry",
    response_model=RunApiResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_run(run_id: UUID, db: DbSession, current_user: CurrentUser):
    """Create and queue a new attempt for the latest failed or cancelled turn."""
    run = await ConversationService(db, current_user.id).retry(run_id)
    await db.commit()
    await dispatch_conversation(db, run.id, current_user.id)
    return RunApiResponse(message="Retry accepted.", data=RunResponse.from_run(run))
