"""Typed records shared by identity persistence and HTTP adapters."""

from typing import TypedDict

from services.identity.pagination import Page


class AuthenticatedAccount(TypedDict):
    id: int
    name: str
    phone: str
    status: str
    role_id: int | None
    password: str
    username: str
    role: str
    all_knowledge_bases: bool
    knowledge_base_ids: list[str]
    menu_permissions: list[str]


class UserSummary(TypedDict):
    """User account fields exposed by the administration list endpoint."""

    id: int
    name: str
    phone: str
    status: str
    role: str
    time: str
    username: str | None


UserPage = Page[UserSummary]


class RoleRecord(TypedDict):
    """Role and permission fields exposed by identity administration."""

    id: int
    name: str
    description: str
    menu_permissions: list[str]
    all_knowledge_bases: bool
    knowledge_base_ids: list[str]


RolePage = Page[RoleRecord]
