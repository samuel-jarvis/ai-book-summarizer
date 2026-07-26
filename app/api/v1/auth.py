import uuid
from typing import Annotated

from fastapi import APIRouter, Body, Request, Response, status

from app.api.deps import CurrentAuth, CurrentUser, DbSession
from app.core.config import settings
from app.core.cookies import clear_refresh_cookie, set_refresh_cookie
from app.core.exception import AuthenticationError
from app.models.user import User
from app.schema.auth import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    SessionListApiResponse,
    SessionResponse,
    TokenApiResponse,
    TokenResponse,
    UserApiResponse,
    UserResponse,
)
from app.services.auth_service import AuthService, IssuedTokens

router = APIRouter()

# Native clients send this to manage tokens themselves; browsers omit it and
# get the refresh token as an httpOnly cookie they can never read.
AUTH_MODE_HEADER = "X-Auth-Mode"

RefreshBody = Annotated[RefreshRequest | None, Body()]


def _is_bearer_mode(request: Request) -> bool:
    return request.headers.get(AUTH_MODE_HEADER, "").lower() == "bearer"


def _read_refresh_token(request: Request, payload: RefreshRequest | None) -> str:
    """Cookie first, then the request body for clients in bearer mode."""
    token = request.cookies.get(settings.REFRESH_COOKIE_NAME)

    if not token and payload is not None:
        token = payload.refresh_token

    if not token:
        raise AuthenticationError("No refresh token provided")

    return token


def _issue(
    request: Request,
    response: Response,
    user: User,
    tokens: IssuedTokens,
) -> TokenResponse:
    """Deliver a token pair the way this particular client expects it."""
    bearer_mode = _is_bearer_mode(request)

    if not bearer_mode:
        set_refresh_cookie(response, tokens.refresh_token)

    return TokenResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token if bearer_mode else None,
        expires_at=tokens.expires_at,
        user=UserResponse.model_validate(user),
    )


@router.post(
    "/auth/register",
    response_model=UserApiResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(payload: RegisterRequest, db: DbSession):
    user = await AuthService(db).register(payload)

    return UserApiResponse(
        success=True,
        message="Account created successfully",
        data=UserResponse.model_validate(user),
    )


@router.post("/auth/login", response_model=TokenApiResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: DbSession,
):
    service = AuthService(db)
    user = await service.authenticate(payload.email, payload.password)

    # Housekeeping while we're already here: drop this user's expired and
    # revoked rows before adding another session.
    await service.prune_expired(user.id)

    tokens = await service.start_session(
        user,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )

    return TokenApiResponse(
        success=True,
        message="Signed in successfully",
        data=_issue(request, response, user, tokens),
    )


@router.post("/auth/refresh", response_model=TokenApiResponse)
async def refresh(
    request: Request,
    response: Response,
    db: DbSession,
    payload: RefreshBody = None,
):
    raw_token = _read_refresh_token(request, payload)
    user, tokens = await AuthService(db).refresh(raw_token)

    return TokenApiResponse(
        success=True,
        message="Session refreshed successfully",
        data=_issue(request, response, user, tokens),
    )


@router.post("/auth/logout")
async def logout(
    request: Request,
    response: Response,
    db: DbSession,
    payload: RefreshBody = None,
):
    token = request.cookies.get(settings.REFRESH_COOKIE_NAME)

    if not token and payload is not None:
        token = payload.refresh_token

    if token:
        await AuthService(db).logout(token)

    # Always clear the cookie, even if the token was already unknown — the
    # browser should end up signed out either way.
    clear_refresh_cookie(response)

    return {
        "success": True,
        "message": "Signed out successfully",
    }


@router.get("/auth/me", response_model=UserApiResponse)
async def read_current_user(current_user: CurrentUser):
    return UserApiResponse(
        success=True,
        message="Current user retrieved successfully",
        data=UserResponse.model_validate(current_user),
    )


@router.get("/auth/sessions", response_model=SessionListApiResponse)
async def list_sessions(auth: CurrentAuth, db: DbSession):
    sessions = await AuthService(db).list_sessions(auth.user.id)

    return SessionListApiResponse(
        success=True,
        message="Sessions retrieved successfully",
        data=[
            SessionResponse.model_validate(session).model_copy(
                update={"current": session.id == auth.session.id})
            for session in sessions
        ],
    )


@router.delete("/auth/sessions/{session_id}")
async def revoke_session(
    session_id: uuid.UUID,
    auth: CurrentAuth,
    db: DbSession,
    response: Response,
):
    await AuthService(db).revoke_session_for_user(session_id, auth.user.id)

    # Revoking your own session is a sign-out; don't leave a dead cookie behind.
    if session_id == auth.session.id:
        clear_refresh_cookie(response)

    return {
        "success": True,
        "message": "Session revoked successfully",
    }
