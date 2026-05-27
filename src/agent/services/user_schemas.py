from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class UserServiceModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ErrorResponse(UserServiceModel):
    code: Optional[str] = None
    message: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class UserResponse(UserServiceModel):
    id: str
    email: Optional[str] = None
    full_name: str
    avatar_url: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    address: Optional[str] = None
    phone_number: Optional[str] = None
    bio: Optional[str] = None
    provider: str
    role: str
    status: str
    assigned_roles: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str


class UserStatsResponse(UserServiceModel):
    total_users: int
    active_users: int
    blocked_users: int
    admin_count: int
    age_groups: list[dict[str, Any]]
    gender_mix: list[dict[str, Any]]
