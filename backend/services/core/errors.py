"""Framework-independent application errors shared across service modules."""


class ApplicationError(Exception):
    """Base error carrying a stable machine code and safe public message."""

    default_code = "application_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        """保存可公开展示的错误信息，并生成稳定的机器错误码。"""
        super().__init__(message)
        self.code = code or self.default_code
        self.public_message = message


class AuthenticationError(ApplicationError):
    """The supplied credentials do not identify an active authenticated subject."""

    default_code = "authentication_failed"


class InvalidRequestError(ApplicationError):
    """The caller supplied a value that the application cannot accept."""

    default_code = "invalid_request"


class ResourceNotFoundError(ApplicationError):
    """A requested application resource does not exist."""

    default_code = "resource_not_found"


class AccessDeniedError(ApplicationError):
    """The current principal cannot access the requested resource."""

    default_code = "access_denied"


class PreconditionFailedError(ApplicationError):
    """A required application configuration or state is missing."""

    default_code = "precondition_failed"


class UnsupportedMediaError(ApplicationError):
    """The supplied media type is not supported by the application."""

    default_code = "unsupported_media_type"


class PayloadTooLargeError(ApplicationError):
    """The supplied payload exceeds an application limit."""

    default_code = "payload_too_large"


class InvalidStateError(ApplicationError):
    """The requested operation is not valid for the current resource state."""

    default_code = "invalid_state"


class ConflictError(ApplicationError):
    """The requested operation conflicts with the current resource state."""

    default_code = "conflict"


class OperationFailedError(ApplicationError):
    """An application operation failed after it started mutating resources."""

    default_code = "operation_failed"


class RateLimitedError(ApplicationError):
    """The caller exceeded an attempt limit and must wait before retrying."""

    default_code = "rate_limited"

    def __init__(
        self, message: str, *, retry_after_seconds: int, code: str | None = None
    ) -> None:
        """保存建议等待的秒数，供 HTTP 层写入 Retry-After。"""
        super().__init__(message, code=code)
        self.retry_after_seconds = max(1, int(retry_after_seconds))


class ServiceUnavailableError(ApplicationError):
    """A required storage or external service is temporarily unavailable."""

    default_code = "service_unavailable"
