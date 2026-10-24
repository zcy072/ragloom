"""Canonical permission identifiers used by roles and authorization."""

import json
from typing import cast

MENU_PERMISSIONS = (
    "仪表盘统计",
    "知识库",
    "知识库管理",
    "问答对话",
    "模型配置",
    "用户管理",
    "角色管理",
)


def decode_permission_list(raw_value: object) -> list[str]:
    """Decode a permission column whose database contract is a JSON string list."""
    return cast(list[str], json.loads(str(raw_value or "[]")))
