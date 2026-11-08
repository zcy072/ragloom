"""网页搜索：调用 DeepSeek Search API，把结果转为可与知识库候选一起重排的文档。

服务端只请求固定的搜索服务地址，不抓取结果中的任意 URL，避免 SSRF。网页正文由搜索服务
返回的摘要提供，进入提示词时与知识库资料一样按不可信数据处理。
"""

import asyncio
import collections
import datetime
import logging
import threading
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from langchain_core.documents import Document as LCDocument

from services.core import model_api_keys
from services.core.container import ApplicationContainer, get_application_container
from services.core.settings import get_settings
from services.retrieval.context import ContextSpan

logger = logging.getLogger(__name__)

WEB_SEARCH_PERMISSION = "联网搜索"
WEB_CHUNK_ROUTE = "web"
DEEPSEEK_PROVIDER_NAME = "DeepSeek"
# 地址写死在代码里，不跟随模型配置中可修改的 base_url，密钥只会发往官方接口。
DEEPSEEK_SEARCH_URL = "https://api.deepseek.com/v1/tools/search"
WEB_QUERY_MAX_CHARS = 128
WEB_SEARCH_COUNT = 6
WEB_SEARCH_TIMEOUT_SECONDS = 10.0
WEB_RESULT_MAX_CHARS = 1000
# 进入提示词的网页段数上限；网页段单独计字数，不占知识库上下文的名额。
WEB_CONTEXT_MAX_SPANS = 4

_client: httpx.AsyncClient | None = None


class WebSearchError(Exception):
    """搜索不可用；`reason` 是写入追踪记录的简短原因。"""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class WebResult:
    """一条网页搜索结果。"""

    title: str
    url: str
    site_name: str
    date: str
    content: str


class WebSearchProvider(Protocol):
    """可替换的网页搜索服务。"""

    async def search(self, query: str, *, ori_question: str, count: int) -> list[WebResult]:
        """返回搜索结果；不可用时抛出 WebSearchError。"""
        ...


def _get_client() -> httpx.AsyncClient:
    """返回进程级搜索客户端，首次使用时才创建。"""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(WEB_SEARCH_TIMEOUT_SECONDS),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            follow_redirects=False,
        )
    return _client


async def close_web_search_client() -> None:
    """应用关闭时释放搜索连接池。"""
    global _client
    client, _client = _client, None
    if client is not None:
        await client.aclose()


def _is_http_url(value: str) -> bool:
    """只接受 http/https 绝对地址，前端会把它渲染成链接。"""
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    return parts.scheme in {"http", "https"} and bool(parts.netloc)


def _text(value: object, limit: int) -> str:
    """把接口字段规整为单行受限文本。"""
    return " ".join(str(value or "").split())[:limit] if isinstance(value, str) else ""


def _site_name(value: object) -> str:
    """接口把站点名编码为 UTF-8 的十六进制串（如 e5a4aee8a786e7bd91 即“央视网”），能解码就解码。"""
    text = _text(value, 200)
    try:
        decoded = bytes.fromhex(text).decode("utf-8")
    except ValueError:
        return text[:100]
    return _text(decoded, 100) or text[:100]


def _local_date(value: object) -> str:
    """接口日期是 UTC 时间，按服务器时区（TZ）换算后再取日期，否则北京时间的新闻会早一天。"""
    text = _text(value, 40)
    try:
        moment = datetime.datetime.fromisoformat(text)
    except ValueError:
        return text[:10]
    if moment.tzinfo is not None:
        moment = moment.astimezone()
    return moment.date().isoformat()


def parse_results(payload: object) -> list[WebResult]:
    """解析搜索响应，丢弃地址不合法或没有正文的条目。"""
    items = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise WebSearchError("响应格式不正确")
    results: list[WebResult] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        content = _text(item.get("summary"), WEB_RESULT_MAX_CHARS) or _text(
            item.get("snippet"), WEB_RESULT_MAX_CHARS
        )
        if not _is_http_url(url) or len(url) > 2048 or not content or url in seen:
            continue
        seen.add(url)
        results.append(
            WebResult(
                title=_text(item.get("title"), 200) or url,
                url=url,
                site_name=_site_name(item.get("site_name")),
                date=_local_date(item.get("date")),
                content=content,
            )
        )
    return results


class DeepSeekSearchProvider:
    """DeepSeek Search API：POST /v1/tools/search，按成功请求计费。"""

    def __init__(self, api_key: str):
        self._api_key = api_key

    async def search(self, query: str, *, ori_question: str, count: int) -> list[WebResult]:
        body: dict[str, object] = {"query": query[:WEB_QUERY_MAX_CHARS], "count": count}
        if ori_question and ori_question != query:
            body["ori_question"] = ori_question[:WEB_QUERY_MAX_CHARS]
        try:
            response = await _get_client().post(
                DEEPSEEK_SEARCH_URL,
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.TimeoutException as exc:
            raise WebSearchError("搜索超时") from exc
        except httpx.HTTPError as exc:
            raise WebSearchError(f"网络错误：{type(exc).__name__}") from exc
        if response.status_code != 200:
            logger.warning("Web search unavailable: HTTP %s", response.status_code)
            reason = {401: "密钥无效", 402: "账户余额不足", 429: "请求过于频繁"}.get(
                response.status_code, f"HTTP {response.status_code}"
            )
            raise WebSearchError(reason)
        try:
            return parse_results(response.json())
        except ValueError as exc:
            raise WebSearchError("响应格式不正确") from exc


def configured_provider(
    container: ApplicationContainer | None = None,
) -> WebSearchProvider | None:
    """复用模型配置中 DeepSeek 的密钥；未配置时返回 None。"""
    encrypted_key = (container or get_application_container()).model_providers.api_key(
        DEEPSEEK_PROVIDER_NAME
    )
    if not encrypted_key:
        return None
    return DeepSeekSearchProvider(model_api_keys.decrypt_api_key(encrypted_key))


class SearchBudget:
    """进程内的调用限额：全站每日次数和每个账号每分钟次数，0 表示不限。

    计数在内存里，进程重启后清零；部署为单个后端进程时即为准确的每日上限。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day = ""
        self._daily_count = 0
        self._recent: dict[int, collections.deque[float]] = collections.defaultdict(
            collections.deque
        )

    def acquire(self, account_id: int) -> str | None:
        """占用一次额度；超限时返回原因，不占用。"""
        settings = get_settings()
        now = time.monotonic()
        today = datetime.date.today().isoformat()
        with self._lock:
            if today != self._day:
                self._day, self._daily_count = today, 0
            if settings.web_search_daily_limit and self._daily_count >= settings.web_search_daily_limit:
                return "已达到今日联网搜索次数上限"
            recent = self._recent[account_id]
            while recent and now - recent[0] >= 60:
                recent.popleft()
            if settings.web_search_user_per_minute and len(recent) >= settings.web_search_user_per_minute:
                return "联网搜索过于频繁，请稍后再试"
            recent.append(now)
            self._daily_count += 1
            return None

    def refund(self, account_id: int) -> None:
        """退回该账号最近占用的一次额度，用于搜索失败、服务方没有计费的情况。"""
        with self._lock:
            recent = self._recent.get(account_id)
            if recent:
                recent.pop()
            if self._daily_count:
                self._daily_count -= 1

    def reset(self) -> None:
        """清空计数，供测试使用。"""
        with self._lock:
            self._day, self._daily_count = "", 0
            self._recent.clear()


budget = SearchBudget()


@dataclass(frozen=True)
class WebSearchOutcome:
    """一次网页搜索的结果和追踪所需的事实。"""

    documents: list[LCDocument]
    elapsed_ms: float
    error: str = ""


def to_documents(results: list[WebResult]) -> list[LCDocument]:
    """转为检索文档；`kb_name` 为空，`url` 标记它来自网页。"""
    return [
        LCDocument(
            page_content=result.content,
            metadata={
                "source": result.title,
                "url": result.url,
                "site_name": result.site_name,
                "date": result.date,
                "kb_name": "",
                "chunk_route": WEB_CHUNK_ROUTE,
            },
        )
        for result in results
    ]


async def search_web(
    query: str,
    question: str,
    account_id: int,
    container: ApplicationContainer | None = None,
    provider: WebSearchProvider | None = None,
) -> WebSearchOutcome:
    """执行一次网页搜索；任何失败都降级为没有网页结果，并给出原因。"""
    started = time.perf_counter()

    def outcome(documents: list[LCDocument], error: str = "") -> WebSearchOutcome:
        return WebSearchOutcome(documents, round((time.perf_counter() - started) * 1000, 2), error)

    try:
        provider = provider or await asyncio.to_thread(configured_provider, container)
    except Exception as exc:
        logger.warning("Web search key unavailable: %s", exc)
        return outcome([], "读取 DeepSeek 密钥失败")
    if provider is None:
        return outcome([], "未配置 DeepSeek 密钥")
    rejected = budget.acquire(account_id)
    if rejected:
        return outcome([], rejected)
    try:
        results = await provider.search(query, ori_question=question, count=WEB_SEARCH_COUNT)
    except WebSearchError as exc:
        budget.refund(account_id)
        return outcome([], exc.reason)
    except Exception as exc:
        budget.refund(account_id)
        logger.warning("Web search failed: %s", exc, exc_info=True)
        return outcome([], f"搜索失败：{type(exc).__name__}")
    return outcome(to_documents(results))


def web_context_spans(documents: list[LCDocument]) -> list[ContextSpan]:
    """把重排后保留的网页结果转为上下文段，最多 WEB_CONTEXT_MAX_SPANS 段，按重排顺序。"""
    spans: list[ContextSpan] = []
    for rank, document in enumerate(documents[:WEB_CONTEXT_MAX_SPANS]):
        metadata = document.metadata or {}
        labels = [str(metadata.get(key) or "") for key in ("site_name", "date")]
        spans.append(
            ContextSpan(
                source=str(metadata.get("source") or metadata["url"]),
                content=document.page_content[:WEB_RESULT_MAX_CHARS],
                pages=(),
                page_unit="页",
                rank=rank,
                contextual_titles=tuple(label for label in labels if label),
                chunk_routes=(WEB_CHUNK_ROUTE,),
                url=str(metadata["url"]),
            )
        )
    return spans
