"""Shared pagination helpers for identity administration queries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageWindow:
    """Resolved database range and public metadata for one requested page."""

    page: int
    page_size: int
    total: int
    total_pages: int
    offset: int


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
