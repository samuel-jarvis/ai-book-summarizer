from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.exception import AuthenticationError
from app.models.user import User
from app.services.auth_service import AuthContext, AuthService


DbSession = Annotated[AsyncSession, Depends(get_db)]

# auto_error=False so a missing header raises our own AppError and comes back
# in the standard `{message, data, success}` envelope like every other error.
bearer_scheme = HTTPBearer(
    auto_error=False,
    description="Access token issued by /api/v1/auth/login",
)


async def get_auth_context(
    request: Request,
    db: DbSession,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> AuthContext:
    """Resolve the caller from an access-token cookie, falling back to Bearer.

    Cookie first, so a browser running in cookie mode wins. The React client
    keeps its access token in a JS variable and sends the header instead — as
    do native clients — which is the path actually exercised today.
    """
    token = request.cookies.get(settings.ACCESS_COOKIE_NAME)

    if not token and credentials is not None:
        token = credentials.credentials

    if not token:
        raise AuthenticationError("Not authenticated")

    return await AuthService(db).resolve_access_token(token)


# FastAPI caches a dependency's result per request, so routes taking both
# CurrentUser and CurrentAuth still resolve the token only once.
CurrentAuth = Annotated[AuthContext, Depends(get_auth_context)]


async def get_current_user(auth: CurrentAuth) -> User:
    return auth.user


CurrentUser = Annotated[User, Depends(get_current_user)]
