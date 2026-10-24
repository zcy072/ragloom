"""FastAPI application composition."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import app_storage
from routers import (
    auth,
    chat,
    documents,
    knowledge_bases,
    models,
    roles,
    users,
)
from services.chat import ranking as chat_ranking
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
from services.core.settings import get_settings
from services.indexing import worker as index_worker
from services.retrieval import retriever

LOG_LEVEL = get_settings().log_level
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for logger_name in ("services", "routers", "app_storage"):
    logging.getLogger(logger_name).setLevel(LOG_LEVEL)
logging.getLogger("jieba").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """初始化进程级共享服务，并在应用退出时按相反顺序释放资源。"""
    # Startup order matters: storage must be valid before workers or cache warmup read it.
    # The rerank client is process-scoped, while queued jobs remain durable in SQLite.
    app_storage.initialize_app_storage()
    chat_ranking.start_rerank_client()
    index_worker.start()
    retriever.start_retriever_cache_warmup()
    try:
        yield
    finally:
        # Stop accepting background work before closing shared network resources.
        index_worker.stop()
        await chat_ranking.close_rerank_client()


app = FastAPI(title="RAGLoom API", lifespan=lifespan)


@app.exception_handler(ApplicationError)
async def application_error_response(
    _request: Request, exc: ApplicationError
) -> JSONResponse:
    """Translate framework-independent application errors at the HTTP boundary."""
    status_by_type = {
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
    status_code = next(
        (status for error_type, status in status_by_type.items() if isinstance(exc, error_type)),
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


for route_group in (
    auth.router,
    users.router,
    roles.router,
    chat.router,
    models.router,
    knowledge_bases.router,
    documents.router,
):
    app.include_router(route_group)
