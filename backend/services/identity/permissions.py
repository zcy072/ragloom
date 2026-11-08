"""Canonical permission identifiers used by roles and authorization."""

import json
import logging

logger = logging.getLogger(__name__)

MENU_PERMISSIONS = (
    "仪表盘统计",
    "知识库",
    "知识库管理",
    "问答对话",
    "模型配置",
    "用户管理",
    "角色管理",
    "运行监控",
    "用户分析",
    # 不对应菜单：在用户分析页打开用户的完整对话，每次查看都写入审计记录。
    "查看用户对话",
    # 不对应菜单：允许提问时同时搜索网页（按次计费）。
    "联网搜索",
)


def decode_permission_list(raw_value: object) -> list[str]:
    """Decode a permission column whose database contract is a JSON string list.

    损坏的记录按"没有权限"处理并记录日志，不让一条坏数据使所有读取它的请求失败。
    """
    try:
        value = json.loads(str(raw_value or "[]"))
    except json.JSONDecodeError:
        logger.warning("Ignoring malformed permission list: %.80r", raw_value)
        return []
    if not isinstance(value, list):
        logger.warning("Ignoring non-list permission value: %.80r", raw_value)
        return []
    return [str(item) for item in value if isinstance(item, str)]
