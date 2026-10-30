"""Application constants, environment-backed paths, and shared validation."""

import datetime
import re
from pathlib import Path

import bcrypt

from services.core.contracts import KnowledgeBaseConfig
from services.core.settings import get_settings

INITIAL_ADMIN_USERNAME = "admin"
# 角色知识库权限中的特殊值，表示全部知识库，不能用作知识库名称。
ALL_KNOWLEDGE_BASES_PERMISSION = "全公司"
DOCUMENT_METADATA_MAX_TYPE_LENGTH = 80
DOCUMENT_METADATA_MAX_TAGS = 12
DOCUMENT_METADATA_MAX_TAG_LENGTH = 40
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_CHUNK_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", " ", "")
DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_SILICONFLOW_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SQLITE_VACUUM_MIN_FREE_PAGES = 128
SQLITE_VACUUM_MIN_FREE_RATIO = 0.20


def storage_paths(project_dir: Path) -> tuple[str, str, str]:
    """读取可由环境变量覆盖的存储目录和 SQLite 文件位置。"""
    settings = get_settings(project_dir)
    return settings.storage_dir, settings.data_dir, settings.db_file


def initial_admin_password() -> str:
    """返回部署配置的初始管理员密码；为空表示首次启动时随机生成。"""
    return get_settings().initial_admin_password


BCRYPT_MAX_PASSWORD_BYTES = 72


MIN_PASSWORD_LENGTH = 8


# \w 按 Unicode 匹配，包括汉字；排除空白和控制字符，避免肉眼无法区分的账号名。
_USERNAME_PATTERN = re.compile(r"[\w.@-]+")
_PHONE_PATTERN = re.compile(r"\+?[0-9][0-9 -]{4,30}")


def validate_registration_fields(username: str, phone: str) -> None:
    """校验注册时的账号名称和选填的手机号码格式。"""
    if not _USERNAME_PATTERN.fullmatch(username):
        raise ValueError("账号名称只能包含文字、数字和 _ . @ -，不能有空格")
    if phone and not _PHONE_PATTERN.fullmatch(phone):
        raise ValueError("手机号码格式不正确")


def validate_new_password(password: str) -> None:
    """校验用户新设置的密码长度；初始管理员密码来自部署配置，不经过这里。"""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"密码过短：至少需要 {MIN_PASSWORD_LENGTH} 个字符")


def hash_password(password: str) -> str:
    """使用带随机盐的 bcrypt 保存密码。

    bcrypt 只接受不超过 72 字节的输入，超出时直接报错而不是静默截断；中文每字 3 字节，
    所以限制约为 24 个汉字。
    """
    encoded = password.encode("utf-8")
    if len(encoded) > BCRYPT_MAX_PASSWORD_BYTES:
        raise ValueError(
            f"密码过长：不能超过 {BCRYPT_MAX_PASSWORD_BYTES} 字节"
            "（英文数字约 72 个字符，每个汉字占 3 字节）"
        )
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("ascii")


# 用户名不存在时用它校验一次，让响应时间与密码错误时一致，避免探测用户名。
_DUMMY_PASSWORD_HASH = "$2b$12$FPloppm6AdPlWviIsdEgWel8VU6nFqlKV1lWWD41RFm35Y67xpZGe"


def verify_password(password: str, stored_hash: str) -> bool:
    """以恒定时间校验明文密码和 bcrypt 凭证。"""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        return False


def burn_password_check(password: str) -> None:
    """对固定的假哈希执行一次 bcrypt 校验，只为消耗与真实校验相同的时间。"""
    verify_password(password, _DUMMY_PASSWORD_HASH)


def db_now() -> str:
    """生成带时区的 UTC 时间字符串，供 SQLite 时间字段保存。"""
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def default_chunk_config() -> KnowledgeBaseConfig:
    """按知识库配置使用的字段格式返回默认分块参数。"""
    return KnowledgeBaseConfig(DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)


def _validated_path_segment(value: str, max_length: int, error: str) -> str:
    """去除首尾空白，并拒绝空值、超长以及路径穿越等不能作为单级路径名的值。"""
    name = (value or "").strip()
    if (
        not name
        or len(name) > max_length
        or name in {".", ".."}
        or any(char in name for char in ("/", "\\", "\x00"))
    ):
        raise ValueError(error)
    return name


def validate_kb_name(value: str) -> str:
    """规范化知识库名称，拒绝路径穿越等不能用于文件系统的值和权限保留名。"""
    name = _validated_path_segment(value, 80, "知识库名称无效")
    if name == ALL_KNOWLEDGE_BASES_PERMISSION:
        raise ValueError(f"“{ALL_KNOWLEDGE_BASES_PERMISSION}”是保留名称，不能用作知识库名称")
    return name


def validate_filename(value: str) -> str:
    """规范化上传文件名，并拒绝路径穿越和非法路径片段。"""
    return _validated_path_segment(value, 255, "文件名无效")


# ---------------------------------------------------------------------------
# 检索调参常量（集中管理，便于评估实验时对比修改）
# ---------------------------------------------------------------------------

# 混合检索（稠密加 BM25）每条路线的候选数量上限
RETRIEVAL_DENSE_K = 20

# BM25 词法补充每条路线的候选数量
RETRIEVAL_LEXICAL_K = 6

# 进入重排序前的候选集总量上限
RETRIEVAL_RERANK_MAX_CANDIDATES = 30

# 重排相关度下限：低于它的候选不进入上下文，全部低于时按"未找到依据"回答。
# 2026-10-29 在医疗库上用 bge-reranker-v2-m3 实测：库外问题最高分不超过 0.032，
# 库内问题前几名都在 0.8 以上，被丢掉的只有末尾的噪声候选。
RETRIEVAL_RERANK_MIN_SCORE = 0.05

# 路由融合权重：固定父子块 / 结构化块 / 视觉块
RETRIEVAL_ROUTE_WEIGHTS = {
    "fixed": 0.60,
    "structured": 0.25,
    "visual": 0.15,
}

# 知识库融合权重：当前库主导召回，通用库补充；总和为 1
RETRIEVAL_KB_WEIGHT_CURRENT = 0.70
RETRIEVAL_KB_WEIGHT_COMMON = 0.30

# 上下文扩展：上下文段数量和字符数预算
RETRIEVAL_CONTEXT_MAX_SPANS = 8
RETRIEVAL_CONTEXT_MAX_CHARS = 12000

# 上下文扩展：每个检索块向前/向后读取的相邻块数
RETRIEVAL_CONTEXT_NEIGHBOR_CHUNKS = 1

# ---------------------------------------------------------------------------
# 嵌入模型常量（集中管理，避免在 pipeline 和 retriever 中重复硬编码）
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# 会话轮次按一问一答估算为两条消息；中断回答产生的单条消息仍能正常处理。
# 回答流静默超过该秒数就发送一条 SSE 心跳；前端的空闲超时必须明显大于它。
CHAT_SSE_HEARTBEAT_SECONDS = 15
CHAT_MEMORY_RECENT_TURNS = 6
CHAT_MEMORY_KEEP_RECENT_TURNS = 4
CHAT_MEMORY_SUMMARY_TRIGGER_TURNS = 6
CHAT_MEMORY_RECENT_MESSAGE_LIMIT = CHAT_MEMORY_RECENT_TURNS * 2
CHAT_MEMORY_KEEP_RECENT_MESSAGE_COUNT = CHAT_MEMORY_KEEP_RECENT_TURNS * 2
CHAT_MEMORY_SUMMARY_TRIGGER_MESSAGE_COUNT = CHAT_MEMORY_SUMMARY_TRIGGER_TURNS * 2
# 摘要服务短暂失败时允许多保留一些原文，字符预算仍会限制实际提示词长度。
CHAT_MEMORY_UNSUMMARIZED_MESSAGE_LIMIT = 24
CHAT_MEMORY_RECENT_MAX_CHARS = 6000
CHAT_MEMORY_RECENT_MESSAGE_MAX_CHARS = 1500
CHAT_MEMORY_SUMMARY_MAX_INPUT_CHARS = 10000
CHAT_MEMORY_SUMMARY_MAX_CHARS = 3000
CHAT_MEMORY_SUMMARY_TIMEOUT_SECONDS = 20
CHAT_MEMORY_SUMMARY_MAX_TOKENS = 500

# 工具化检索：由模型通过 function calling 决定是否检索、用什么查询检索。
# 兼容接口未必支持 tools 参数，因此默认关闭，由环境变量显式开启。

def chat_retrieval_planning_enabled() -> bool:
    """读取当前问题是否启用工具化检索规划。"""
    return get_settings().chat_retrieval_planning


# 书面化补充查询：在原问题之外，再用口语转书面语的改写检索一遍并合并候选。
# 会多一次改写模型调用，因此默认关闭，由环境变量显式开启。

def chat_query_variant_enabled() -> bool:
    """读取当前问题是否启用书面化补充查询。"""
    return get_settings().chat_query_variant


# 补充检索：首轮证据不足时，由模型判断并给出新检索词再检索，最多 CHAT_AGENTIC_MAX_ROUNDS 轮。
# 依赖 function calling 且会增加模型调用，因此默认关闭，由环境变量显式开启。

def chat_agentic_retrieval_enabled() -> bool:
    """读取当前问题是否启用补充检索。"""
    return get_settings().chat_agentic_retrieval


CHAT_AGENTIC_MAX_ROUNDS = 2


# 检索前的查询理解（追问改写与工具化规划）共用的限额。
CHAT_QUERY_MODEL_TIMEOUT_SECONDS = 15
CHAT_QUERY_MAX_LENGTH = 300
CHAT_QUERY_HISTORY_LIMIT = 4
CHAT_RETRIEVAL_PLANNING_MAX_TOKENS = 200
