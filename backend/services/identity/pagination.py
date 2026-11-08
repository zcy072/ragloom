"""Shared pagination helpers for identity administration queries."""

from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True)
class PageWindow:
    """Resolved database range and public metadata for one requested page."""

    page: int
    page_size: int
    total: int
    total_pages: int
    offset: int


class Page[PageItem](TypedDict):
    """The page of rows a store returns, plus the metadata the API exposes."""

    items: list[PageItem]
    total: int
    page: int
    page_size: int
    total_pages: int


def page_of[PageItem](items: list[PageItem], window: PageWindow) -> Page[PageItem]:
    """把一页已取回的行连同分页元数据组装成返回给接口层的结构。"""
    return {
        "items": items,
        "total": window.total,
        "page": window.page,
        "page_size": window.page_size,
        "total_pages": window.total_pages,
    }


def resolve_page_window(total: int, page: int, page_size: int) -> PageWindow:
    """Validate a requested range, clamp its page, and calculate its SQL offset."""
    if page < 1:
        raise ValueError("page 必须大于 0")
    if page_size < 1:
        raise ValueError("page_size 必须大于 0")
    total_pages = max(1, (total + page_size - 1) // page_size)
    resolved_page = min(page, total_pages)
    return PageWindow(
        page=resolved_page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        offset=(resolved_page - 1) * page_size,
    )


def like_pattern(query: str) -> str | None:
    """Return an escaped SQL substring pattern, or None when no search is requested."""
    normalized = " ".join(query.split())
    if not normalized:
        return None
    escaped = normalized.replace("!", "!!").replace("%", "!%").replace("_", "!_")
    return f"%{escaped}%"
