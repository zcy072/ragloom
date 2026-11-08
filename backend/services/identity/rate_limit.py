"""登录、注册和问答的次数限制：进程内滑动窗口，重启后计数清零。

只适合当前的单进程部署；多进程或多实例部署需要把计数放到共享存储。
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable

from services.core.errors import RateLimitedError
from services.core.settings import get_settings

_MAX_KEY_CHARS = 128
_MAX_TRACKED_KEYS = 10_000


class EventWindow:
    """统计每个键在滑动窗口内的事件次数；次数达到上限后该键被拒绝，直到最早的事件过期。"""

    def __init__(
        self,
        max_events: int,
        window_seconds: int,
        *,
        max_keys: int = _MAX_TRACKED_KEYS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_events = max_events
        self._window = window_seconds
        self._max_keys = max_keys
        self._clock = clock
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """上限或窗口为 0 表示关闭该项限制。"""
        return self._max_events > 0 and self._window > 0

    def retry_after(self, key: str) -> int:
        """返回键还需等待的秒数；未达到上限时返回 0。"""
        if not self.enabled:
            return 0
        with self._lock:
            events = self._events.get(key)
            if events is None:
                return 0
            self._expire(events, self._clock())
            if len(events) < self._max_events:
                return 0
            return max(1, math.ceil(events[0] + self._window - self._clock()))

    def record(self, key: str) -> None:
        """记录一次事件。"""
        if not self.enabled:
            return
        with self._lock:
            now = self._clock()
            events = self._events.get(key)
            if events is None:
                self._make_room(now)
                events = self._events[key] = deque()
            self._expire(events, now)
            # 已达上限时忽略新事件：否则每次被拒绝的尝试都会把窗口往后推，锁定时间越拖越长。
            if len(events) < self._max_events:
                events.append(now)

    def clear(self, key: str) -> None:
        """清除一个键的全部记录。"""
        with self._lock:
            self._events.pop(key, None)

    def _expire(self, events: deque[float], now: float) -> None:
        while events and events[0] + self._window <= now:
            events.popleft()

    def _make_room(self, now: float) -> None:
        """键数量到达上限时先清掉已过期的键，仍然放不下就丢弃最早插入的一成。

        提交大量不同的用户名不能无限占用内存；代价是极端情况下受害者的记录可能被挤掉。
        """
        if len(self._events) < self._max_keys:
            return
        for key in [k for k, events in self._events.items() if not self._live(events, now)]:
            del self._events[key]
        if len(self._events) >= self._max_keys:
            for key in list(self._events)[: max(1, self._max_keys // 10)]:
                del self._events[key]

    def _live(self, events: deque[float], now: float) -> bool:
        return bool(events) and events[-1] + self._window > now


def _wait_text(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} 秒"
    return f"{math.ceil(seconds / 60)} 分钟"


def _username_key(username: str) -> str:
    """用户名统一为小写并限制长度；无论账号是否存在都使用同一套计数。"""
    return username.strip().lower()[:_MAX_KEY_CHARS]


class AuthRateLimiter:
    """登录按用户名和来源地址限制失败次数，注册按来源地址限制提交次数。"""

    def __init__(
        self,
        *,
        login_failures: int,
        login_window_seconds: int,
        client_failures: int,
        client_window_seconds: int,
        register_attempts: int,
        register_window_seconds: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._by_username = EventWindow(login_failures, login_window_seconds, clock=clock)
        self._by_client = EventWindow(client_failures, client_window_seconds, clock=clock)
        self._registrations = EventWindow(register_attempts, register_window_seconds, clock=clock)

    @classmethod
    def from_settings(cls) -> AuthRateLimiter:
        settings = get_settings()
        return cls(
            login_failures=settings.auth_login_max_failures,
            login_window_seconds=settings.auth_login_window_seconds,
            client_failures=settings.auth_client_max_failures,
            client_window_seconds=settings.auth_client_window_seconds,
            register_attempts=settings.auth_register_max_attempts,
            register_window_seconds=settings.auth_register_window_seconds,
        )

    def check_login(self, username: str, client: str) -> None:
        """已被限制时抛出 RateLimitedError；被限制的请求不再计数，也不会校验密码。"""
        wait = self._by_username.retry_after(_username_key(username))
        if client:
            wait = max(wait, self._by_client.retry_after(client))
        if wait:
            raise RateLimitedError(
                f"登录尝试次数过多，请 {_wait_text(wait)}后再试",
                retry_after_seconds=wait,
                code="login_rate_limited",
            )

    def record_login_failure(self, username: str, client: str) -> None:
        self._by_username.record(_username_key(username))
        if client:
            self._by_client.record(client)

    def record_login_success(self, username: str) -> None:
        """密码正确后清零该用户名的失败计数；来源地址的计数保留，以便识别撞库。"""
        self._by_username.clear(_username_key(username))

    def check_registration(self, client: str) -> None:
        """注册每次提交都计数，无论成功与否；超过上限时抛出 RateLimitedError。"""
        if not client:
            return
        wait = self._registrations.retry_after(client)
        if wait:
            raise RateLimitedError(
                f"注册提交过于频繁，请 {_wait_text(wait)}后再试",
                retry_after_seconds=wait,
                code="registration_rate_limited",
            )
        self._registrations.record(client)


class ChatRequestLimiter:
    """按账号限制同时进行的流式问答数和每分钟提问次数。

    每次提问都会调用 LLM、向量化和重排，单个账号刷请求就能产生不少费用。并发名额在流结束时
    释放；万一某条异常路径漏了释放，名额超过 ``stale_seconds`` 也会自动作废，账号不会被永久卡住。
    """

    def __init__(
        self,
        *,
        max_concurrent: int,
        per_minute: int,
        stale_seconds: int = 900,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._stale_seconds = stale_seconds
        self._clock = clock
        self._per_minute = EventWindow(per_minute, 60, clock=clock)
        self._active: dict[int, dict[object, float]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls) -> ChatRequestLimiter:
        settings = get_settings()
        return cls(
            max_concurrent=settings.chat_user_max_concurrent,
            per_minute=settings.chat_user_per_minute,
        )

    def acquire(self, account_id: int) -> Callable[[], None]:
        """占用一个问答名额并返回释放函数（可重复调用）；超限时抛出 RateLimitedError，不占用。"""
        key = str(account_id)
        with self._lock:
            now = self._clock()
            active = self._active.get(account_id, {})
            for token, started in list(active.items()):
                if started + self._stale_seconds <= now:
                    del active[token]
            if self._max_concurrent > 0 and len(active) >= self._max_concurrent:
                raise RateLimitedError(
                    "已有问答正在进行，请等待当前回答完成后再提问",
                    retry_after_seconds=5,
                    code="chat_concurrency_limited",
                )
            wait = self._per_minute.retry_after(key)
            if wait:
                raise RateLimitedError(
                    f"提问过于频繁，请 {_wait_text(wait)}后再试",
                    retry_after_seconds=wait,
                    code="chat_rate_limited",
                )
            self._per_minute.record(key)
            token = object()
            if self._max_concurrent > 0:
                active[token] = now
                self._active[account_id] = active

        def release() -> None:
            with self._lock:
                slots = self._active.get(account_id)
                if slots is not None:
                    slots.pop(token, None)
                    if not slots:
                        del self._active[account_id]

        return release


_limiter: AuthRateLimiter | None = None
_chat_limiter: ChatRequestLimiter | None = None
_limiter_lock = threading.Lock()


def get_auth_rate_limiter() -> AuthRateLimiter:
    """返回进程级限流器；首次使用时按当前环境变量创建。"""
    global _limiter
    with _limiter_lock:
        if _limiter is None:
            _limiter = AuthRateLimiter.from_settings()
        return _limiter


def get_chat_request_limiter() -> ChatRequestLimiter:
    """返回进程级问答限流器；首次使用时按当前环境变量创建。"""
    global _chat_limiter
    with _limiter_lock:
        if _chat_limiter is None:
            _chat_limiter = ChatRequestLimiter.from_settings()
        return _chat_limiter


def reset_auth_rate_limiter() -> None:
    """丢弃登录、注册和问答的全部计数，下次使用时按当前环境变量重建。供测试使用。"""
    global _limiter, _chat_limiter
    with _limiter_lock:
        _limiter = None
        _chat_limiter = None
