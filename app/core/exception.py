from fastapi import status


class AppError(Exception):
    """Base class for expected, translatable application errors."""

    status_code: int = status.HTTP_400_BAD_REQUEST
    code: str = "app_error"
    headers: dict[str, str] | None = None

    def __init__(self, detail: str | None = None):
        self.detail = detail or self.__class__.__doc__ or "Application error"
        super().__init__(self.detail)


class NotFoundError(AppError):
    """The requested resource was not found."""

    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ValidationError(AppError):
    """The submitted data is invalid."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "validation_error"


class AuthenticationError(AppError):
    """Not authenticated."""

    status_code = status.HTTP_401_UNAUTHORIZED
    code = "authentication_error"
    headers = {"WWW-Authenticate": "Bearer"}


class PermissionDeniedError(AppError):
    """You do not have permission to perform this action."""

    status_code = status.HTTP_403_FORBIDDEN
    code = "permission_denied"


class ConflictError(AppError):
    """That resource already exists."""

    status_code = status.HTTP_409_CONFLICT
    code = "conflict"
