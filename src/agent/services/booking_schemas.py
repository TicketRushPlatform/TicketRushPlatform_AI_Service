from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class BookingServiceModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ErrorResponse(BookingServiceModel):
    code: Optional[int] = None
    message: Optional[str] = None


class PaginatedResponse(BookingServiceModel):
    data: Any
    page: int
    page_size: int
    total_items: int
    total_pages: int


class SuccessResponse(BookingServiceModel):
    data: Any
    message: Optional[str] = None
