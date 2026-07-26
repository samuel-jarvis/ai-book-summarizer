
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

import jwt
from pwdlib import PasswordHash

from app.core.config import settings
from app.core.exception import AuthenticationError

_password_hash = PasswordHash.recommended()

ACCESS_TOKEN_TYPE = "access"


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    """A throwaway hash used to keep verification timing constant."""
    return hash_password(secrets.token_urlsafe(32))


def verify_password(password: str, hashed_password: str | None) -> bool:
    return _password_hash.verify(password, hashed_password or _dummy_hash())


def create_access_token(
    *, user_id: uuid.UUID, session_id: uuid.UUID
) -> tuple[str, datetime]:
    """Return a signed access token and the moment it expires."""
    issued_at = datetime.now(timezone.utc)
    expires_at = issued_at + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": str(user_id),
        "sid": str(session_id),
        "type": ACCESS_TOKEN_TYPE,
        "iat": issued_at,
        "exp": expires_at,
        "jti": secrets.token_urlsafe(16),
    }
    token = jwt.encode(
        payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    return token, expires_at


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate an access token, or raise `AuthenticationError`."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise AuthenticationError("Invalid or expired token") from exc

    if payload.get("type") != ACCESS_TOKEN_TYPE:
        raise AuthenticationError("Invalid or expired token")

    return payload


def generate_refresh_token() -> str:
    """A high-entropy opaque token; only its digest is ever stored."""
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
