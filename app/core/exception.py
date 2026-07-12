from fastapi import status


class AppError(Exception):
    """Base class for expected, translatable application errors."""

    status_code: int = status.HTTP_400_BAD_REQUEST
    code: str = "app_error"

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
