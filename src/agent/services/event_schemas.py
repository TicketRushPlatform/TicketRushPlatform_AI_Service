from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class EventServiceModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ErrorResponse(EventServiceModel):
    code: Optional[int] = None
    message: Optional[str] = None


class PaginatedResponse(EventServiceModel):
    data: Any
    page: int
    page_size: int
    total_items: int
    total_pages: int


class SuccessResponse(EventServiceModel):
    data: Any
    message: Optional[str] = None
