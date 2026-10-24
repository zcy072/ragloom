"""Typed records shared by identity persistence and HTTP adapters."""

from typing import TypedDict


class AccessTokenPayload(TypedDict):
    """Validated claims carried by an authenticated access token."""

    v: int
    aid: int
    sub: str
    iat: int
    exp: int
    cred: str


AuthenticatedAccount = TypedDict(
    "AuthenticatedAccount",
    {
        "id": int,
        "name": str,
        "phone": str,
        "status": str,
        "role_id": int | None,
        "password": str,
        "username": str,
        "role": str,
        "knowledge_base_permissions": list[str],
        "menu_permissions": list[str],
    },
)


class UserSummary(TypedDict):
    """User account fields exposed by the administration list endpoint."""

    id: int
    name: str
    phone: str
    status: str
    role: str
    time: str
    username: str | None


class UserPage(TypedDict):
    """A server-resolved page of user administration records."""

    items: list[UserSummary]
    total: int
    page: int
    page_size: int
    total_pages: int


class RoleRecord(TypedDict):
    """Role and permission fields exposed by identity administration."""

    id: int
    name: str
    description: str
    menu_permissions: list[str]
    knowledge_base_permissions: list[str]


class RolePage(TypedDict):
    """A server-resolved page of role administration records."""

    items: list[RoleRecord]
    total: int
    page: int
    page_size: int
    total_pages: int
