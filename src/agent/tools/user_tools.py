from __future__ import annotations

from typing import Any, Literal, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from src.agent.services.user_service import UserServiceClient


class GetCurrentUserInput(BaseModel):
    pass


class ListUsersInput(BaseModel):
    search: Optional[str] = Field(
        default=None,
        description="Optional local search over user id, email, and full_name after GET /users returns.",
    )
    role: Optional[Literal["USER", "ADMIN"]] = Field(default=None, description="Optional local role filter.")
    status: Optional[Literal["ACTIVE", "BLOCKED"]] = Field(default=None, description="Optional local status filter.")
    max_results: int = Field(default=20, ge=1, le=100, description="Maximum users to return to the agent.")


class GetUserByIdInput(BaseModel):
    user_id: str = Field(..., min_length=1, description="User ID path parameter for GET /users/{user_id}.")


class GetUserStatsInput(BaseModel):
    pass


def create_user_service_tools(client: Optional[UserServiceClient] = None) -> list[BaseTool]:
    service_client = client or UserServiceClient()

    @tool(
        "get_current_user",
        args_schema=GetCurrentUserInput,
        description="Call User Service GET /users/me to retrieve the authenticated user's own profile.",
    )
    async def get_current_user() -> Any:
        """Get the authenticated user's profile."""
        return await service_client.get_current_user()

    @tool(
        "list_users",
        args_schema=ListUsersInput,
        description=(
            "Call User Service GET /users to list users. Admin only. "
            "Use for questions about user information, searching users by name/email/id, or summarizing user lists. "
            "Optional filters search, role, status, and max_results are applied locally after the service returns."
        ),
    )
    async def list_users(
        search: Optional[str] = None,
        role: Optional[Literal["USER", "ADMIN"]] = None,
        status: Optional[Literal["ACTIVE", "BLOCKED"]] = None,
        max_results: int = 20,
    ) -> Any:
        """List users from User Service."""
        users = await service_client.list_users()
        filtered = _filter_users(users, search=search, role=role, status=status)
        return {
            "data": filtered[:max_results],
            "total_matches": len(filtered),
            "returned": min(len(filtered), max_results),
            "filters": {
                "search": search,
                "role": role,
                "status": status,
                "max_results": max_results,
            },
        }

    @tool(
        "get_user_by_id",
        args_schema=GetUserByIdInput,
        description="Call User Service GET /users/{user_id} to retrieve one user by ID. Admin only.",
    )
    async def get_user_by_id(user_id: str) -> Any:
        """Get one user by ID."""
        return await service_client.get_user_by_id(user_id)

    @tool(
        "get_user_stats",
        args_schema=GetUserStatsInput,
        description=(
            "Call User Service GET /users/stats to retrieve user counts and audience breakdowns. Admin only. "
            "Use for questions like total users, active users, blocked users, admin count, age groups, and gender mix."
        ),
    )
    async def get_user_stats() -> Any:
        """Get user statistics."""
        return await service_client.get_user_stats()

    return [get_current_user, list_users, get_user_by_id, get_user_stats]


def _filter_users(
    users: list[dict[str, Any]],
    *,
    search: Optional[str] = None,
    role: Optional[str] = None,
    status: Optional[str] = None,
) -> list[dict[str, Any]]:
    filtered = users
    if search:
        needle = search.strip().lower()
        filtered = [
            user
            for user in filtered
            if needle in str(user.get("id", "")).lower()
            or needle in str(user.get("email", "")).lower()
            or needle in str(user.get("full_name", "")).lower()
        ]
    if role:
        normalized_role = role.strip().upper()
        filtered = [user for user in filtered if str(user.get("role", "")).upper() == normalized_role]
    if status:
        normalized_status = status.strip().upper()
        filtered = [user for user in filtered if str(user.get("status", "")).upper() == normalized_status]
    return filtered
