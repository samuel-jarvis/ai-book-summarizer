import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import NamedTuple

from sqlalchemy import delete, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.config import settings
from app.core.database import utcnow
from app.core.exception import AuthenticationError, ConflictError, NotFoundError
from app.models.auth_session import AuthSession
from app.models.refresh_token import RefreshToken
from app.models.user import User, UserStatus
from app.schema.auth import RegisterRequest
from app.utils.auth import (
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)


class IssuedTokens(NamedTuple):
    access_token: str
    refresh_token: str
    expires_at: datetime


class AuthContext(NamedTuple):
    user: User
    session: AuthSession


class AuthService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def register(self, data: RegisterRequest) -> User:
        user = User(
            email=data.email,
            hashed_password=hash_password(data.password),
            first_name=data.first_name,
            last_name=data.last_name,
            status=UserStatus.ACTIVE,
        )
        self.db.add(user)

        try:
            await self.db.commit()
        except IntegrityError as exc:
            await self.db.rollback()
            raise ConflictError(
                "An account with that email already exists") from exc

        await self.db.refresh(user)
        return user

    async def authenticate(self, email: str, password: str) -> User:
        result = await self.db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        password_ok = verify_password(
            password, user.hashed_password if user else None)

        if user is None or not password_ok:
            raise AuthenticationError("Incorrect email or password")

        if user.status is not UserStatus.ACTIVE:
            raise AuthenticationError(
                f"This account is {user.status.value}")

        return user

    async def start_session(
        self,
        user: User,
        *,
        user_agent: str | None,
        ip_address: str | None,
    ) -> IssuedTokens:
        """Open a session for a freshly authenticated user and issue tokens."""
        now = utcnow()
        session = AuthSession(
            user_id=user.id,
            user_agent=user_agent,
            ip_address=ip_address,
            last_used_at=now,
            expires_at=now + timedelta(
                days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        )
        self.db.add(session)
        # Flush so the session id exists for the token rows and the JWT claim.
        await self.db.flush()

        refresh_token = self._issue_refresh_token(session)
        access_token, expires_at = create_access_token(
            user_id=user.id, session_id=session.id)

        user.last_login = now
        await self.db.commit()

        return IssuedTokens(access_token, refresh_token, expires_at)

    async def refresh(self, raw_token: str) -> tuple[User, IssuedTokens]:
        """Exchange a refresh token for a new pair, rotating the old one out."""
        token = await self._load_refresh_token(raw_token)
        session = token.session
        user = token.user
        now = utcnow()

        token.revoked = True

        if session.revoked or session.expires_at <= now:
            await self.db.commit()
            raise AuthenticationError("Session expired, please sign in again")

        if user.status is not UserStatus.ACTIVE:
            await self.db.commit()
            raise AuthenticationError("This account is no longer active")

        session.last_used_at = now
        refresh_token = self._issue_refresh_token(session)
        access_token, expires_at = create_access_token(
            user_id=user.id, session_id=session.id)

        await self.db.commit()

        return user, IssuedTokens(access_token, refresh_token, expires_at)

    async def logout(self, raw_token: str) -> None:
        """Revoke the session behind a refresh token, and all of its tokens."""
        result = await self.db.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == hash_refresh_token(raw_token))
        )
        token = result.scalar_one_or_none()

        # An unknown or already-revoked token means there is nothing left to
        # end; report success rather than leaking whether it was ever valid.
        if token is not None:
            await self.revoke_session(token.session_id)

    async def revoke_session(self, session_id: uuid.UUID) -> None:
        await self.db.execute(
            update(AuthSession)
            .where(AuthSession.id == session_id)
            .values(revoked=True)
        )
        await self.db.execute(
            update(RefreshToken)
            .where(RefreshToken.session_id == session_id)
            .values(revoked=True)
        )
        await self.db.commit()

    async def resolve_access_token(self, token: str) -> AuthContext:
        """Turn an access token into the live user and session behind it.

        The session is re-read on every request. That costs a query, but it is
        what makes logout and revocation take effect immediately instead of
        waiting out the access token's lifetime.
        """
        payload = decode_access_token(token)

        try:
            user_id = uuid.UUID(payload["sub"])
            session_id = uuid.UUID(payload["sid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AuthenticationError("Invalid or expired token") from exc

        session = await self.db.get(AuthSession, session_id)

        if (
            session is None
            or session.revoked
            or session.user_id != user_id
            or session.expires_at <= utcnow()
        ):
            raise AuthenticationError("Session expired, please sign in again")

        user = await self.db.get(User, user_id)

        if user is None or user.status is not UserStatus.ACTIVE:
            raise AuthenticationError("This account is no longer active")

        return AuthContext(user, session)

    async def list_sessions(self, user_id: uuid.UUID) -> Sequence[AuthSession]:
        """A user's live sessions, most recently used first."""
        query = (
            select(AuthSession)
            .where(
                AuthSession.user_id == user_id,
                AuthSession.revoked.is_(False),
                AuthSession.expires_at > utcnow(),
            )
            .order_by(AuthSession.last_used_at.desc())
        )
        result = await self.db.execute(query)
        return result.scalars().all()

    async def revoke_session_for_user(
        self, session_id: uuid.UUID, user_id: uuid.UUID
    ) -> None:
        session = await self.db.get(AuthSession, session_id)

        if session is None or session.user_id != user_id:
            raise NotFoundError("Session not found")

        await self.revoke_session(session_id)

    async def prune_expired(self, user_id: uuid.UUID) -> None:
        """Delete this user's dead sessions and refresh tokens.

        Called at login rather than on a schedule: it piggybacks on a request
        that is already slow from password hashing, needs no extra process, and
        keeps the tables bounded for anyone actually using the app. Only rows
        that are already expired or revoked are touched, so nothing live is at
        risk — and dormant accounts simply keep harmless dead rows until they
        come back.
        """
        now = utcnow()

        await self.db.execute(
            delete(RefreshToken).where(
                RefreshToken.user_id == user_id,
                or_(
                    RefreshToken.expires_at <= now,
                    RefreshToken.revoked.is_(True),
                ),
            )
        )
        # Tokens on these sessions go too, via ON DELETE CASCADE.
        await self.db.execute(
            delete(AuthSession).where(
                AuthSession.user_id == user_id,
                or_(
                    AuthSession.expires_at <= now,
                    AuthSession.revoked.is_(True),
                ),
            )
        )
        await self.db.commit()

    def _issue_refresh_token(self, session: AuthSession) -> str:
        raw_token = generate_refresh_token()

        self.db.add(RefreshToken(
            user_id=session.user_id,
            session_id=session.id,
            token_hash=hash_refresh_token(raw_token),
            expires_at=session.expires_at,
        ))

        return raw_token

    async def _load_refresh_token(self, raw_token: str) -> RefreshToken:
        query = (
            select(RefreshToken)
            .where(RefreshToken.token_hash == hash_refresh_token(raw_token))
            .options(
                joinedload(RefreshToken.session),
                joinedload(RefreshToken.user),
            )
        )
        result = await self.db.execute(query)
        token = result.scalar_one_or_none()

        if token is None or token.revoked or token.expires_at <= utcnow():
            raise AuthenticationError("Invalid or expired refresh token")

        return token
