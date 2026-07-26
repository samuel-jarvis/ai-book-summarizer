import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.user import UserStatus
from app.schema.response import ApiResponse


class RegisterRequest(BaseModel):
    email: EmailStr = Field(..., max_length=100)
    password: str = Field(..., min_length=8, max_length=128)
    first_name: str | None = Field(default=None, max_length=100)
    last_name: str | None = Field(default=None, max_length=100)


class LoginRequest(BaseModel):
    email: EmailStr = Field(..., max_length=100)
    password: str = Field(..., min_length=1, max_length=128)


class RefreshRequest(BaseModel):
    refresh_token: str | None = None


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    first_name: str | None
    last_name: str | None
    status: UserStatus
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    # Only populated for clients in bearer mode; browsers get the httpOnly
    # cookie instead and must never see this value in JS.
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_at: datetime
    user: UserResponse


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_agent: str | None
    ip_address: str | None
    created_at: datetime
    last_used_at: datetime
    expires_at: datetime
    # True for the session making the request, so the UI can label it and warn
    # before revoking it.
    current: bool = False


UserApiResponse = ApiResponse[UserResponse]
TokenApiResponse = ApiResponse[TokenResponse]
SessionListApiResponse = ApiResponse[list[SessionResponse]]
