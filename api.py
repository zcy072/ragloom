"""FastAPI application composition."""

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

import app_storage
from routers import (
    auth,
    chat,
    documents,
    knowledge_bases,
    models,
    ops,
    roles,
    user_analytics,
    users,
)
from services.chat import ranking as chat_ranking
from services.chat import web_search as chat_web_search
from services.core import readiness
from services.core.errors import (
    AccessDeniedError,
    ApplicationError,
    AuthenticationError,
    ConflictError,
    InvalidRequestError,
    InvalidStateError,
    OperationFailedError,
    PayloadTooLargeError,
    PreconditionFailedError,
    RateLimitedError,
    ResourceNotFoundError,
    ServiceUnavailableError,
    UnsupportedMediaError,
)
from services.core.logging_setup import configure_logging
from services.core.request_context import bind_request_id, new_request_id, reset_request_id
from services.core.settings import get_settings
from services.indexing import worker as index_worker
from services.retrieval import retriever
from services.retrieval.chunk_index import get_chunk_index

logger = logging.getLogger(__name__)


HISTORY_PRUNE_INTERVAL_SECONDS = 24 * 60 * 60


async def _prune_history_periodically(interval_seconds: float) -> None:
    """长期运行的进程不会重启，所以按固定间隔清理过期历史；单次失败不影响后续轮次。"""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await asyncio.to_thread(app_storage.prune_expired_history)
        except Exception:
            logger.exception("Periodic history pruning failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """初始化进程级共享服务，并在应用退出时按相反顺序释放资源。"""
    # 日志在启动时而不是导入时配置：测试会导入本模块，不能往部署环境的日志文件里写。
    configure_logging(get_settings())
    # Startup order matters: storage must be valid before workers or cache warmup read it.
    # The rerank client is process-scoped, while queued jobs remain durable in SQLite.
    app_storage.initialize_app_storage()
    # Milvus is an external service: wait for it before workers or warmup touch the index.
    get_chunk_index().ensure_collection()
    chat_ranking.start_rerank_client()
    index_worker.start()
    retriever.start_retriever_cache_warmup()
    prune_task = asyncio.create_task(
        _prune_history_periodically(HISTORY_PRUNE_INTERVAL_SECONDS)
    )
    try:
        yield
    finally:
        prune_task.cancel()
        # 正在执行的清理线程不会被取消，这里只等任务对象结束，不等线程。
        with contextlib.suppress(asyncio.CancelledError):
            await prune_task
        # Stop accepting background work before closing shared network resources.
        index_worker.stop()
        await chat_ranking.close_rerank_client()
        await chat_web_search.close_web_search_client()


class RequestIdMiddleware:
    """为每个 HTTP 请求生成请求 ID，绑定到日志上下文并通过 X-Request-ID 响应头返回。

    用纯 ASGI 中间件而不是 BaseHTTPMiddleware：SSE 响应体的生成和 uvicorn 的访问日志
    都在这次调用内执行，能拿到同一个请求 ID。不采用客户端传入的 ID，因为问答的追踪 ID
    取这个值，客户端不能指定。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = new_request_id()

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        token = bind_request_id(request_id)
        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            reset_request_id(token)


app = FastAPI(title="RAGLoom API", lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)


# 顺序即匹配优先级：先到先得，子类要放在父类之前。
_STATUS_BY_ERROR_TYPE = {
    AuthenticationError: 401,
    InvalidRequestError: 422,
    ResourceNotFoundError: 404,
    AccessDeniedError: 403,
    ConflictError: 409,
    PreconditionFailedError: 412,
    UnsupportedMediaError: 415,
    PayloadTooLargeError: 413,
    RateLimitedError: 429,
    InvalidStateError: 400,
    OperationFailedError: 500,
    ServiceUnavailableError: 503,
}


@app.exception_handler(ApplicationError)
async def application_error_response(
    _request: Request, exc: ApplicationError
) -> JSONResponse:
    """Translate framework-independent application errors at the HTTP boundary."""
    status_code = next(
        (status for error_type, status in _STATUS_BY_ERROR_TYPE.items() if isinstance(exc, error_type)),
        500,
    )
    headers = (
        {"Retry-After": str(exc.retry_after_seconds)}
        if isinstance(exc, RateLimitedError)
        else None
    )
    return JSONResponse(
        status_code=status_code,
        content={"detail": exc.public_message, "code": exc.code},
        headers=headers,
    )


@app.get("/api/health", include_in_schema=False)
def health() -> dict[str, str]:
    """进程存活探针：只证明应用已启动并能响应，不检查 Milvus 等外部依赖。"""
    return {"status": "ok"}


@app.get("/api/ready", include_in_schema=False)
async def ready() -> JSONResponse:
    """就绪探针：SQLite 和 Milvus 都能查询时返回 200，否则 503。

    不需要登录，所以只返回每项是否正常；错误详情在运行监控页查看。
    """
    statuses = await run_in_threadpool(
        readiness.check_dependencies, app_storage.DB_FILE, get_chunk_index()
    )
    all_ok = all(status.ok for status in statuses)
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "ready" if all_ok else "not_ready",
            "checks": {status.name: "ok" if status.ok else "error" for status in statuses},
        },
    )


for route_group in (
    auth.router,
    users.router,
    roles.router,
    chat.router,
    models.router,
    knowledge_bases.router,
    documents.router,
    ops.router,
    user_analytics.router,
):
    app.include_router(route_group)
