from fastapi import Response
from app.core.config import settings

REFRESH_COOKIE_PATH = f"{settings.API_V1_PREFIX}/auth"

COOKIE_SAMESITE = "lax"


def set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=COOKIE_SAMESITE,
        path=REFRESH_COOKIE_PATH,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
    )


def clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=COOKIE_SAMESITE,
        path=REFRESH_COOKIE_PATH,
    )
