"""登录、注册、改密和问答的次数限制：进程内滑动窗口，重启后计数清零。

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

    def reserve(self, key: str) -> tuple[float | None, int]:
        """在锁内检查并占用一个名额，返回 (名额, 0)；已达上限时返回 (None, 需等待的秒数)。

        检查和计数在同一把锁里完成：并发请求不能都在计数前通过检查。名额在失败时保留，
        相当于一次 record；结果不应计数时用 release 退回。关闭限制时返回 (None, 0)。
        """
        if not self.enabled:
            return None, 0
        with self._lock:
            now = self._clock()
            events = self._events.get(key)
            if events is None:
                self._make_room(now)
                events = self._events[key] = deque()
            self._expire(events, now)
            if len(events) >= self._max_events:
                return None, max(1, math.ceil(events[0] + self._window - now))
            events.append(now)
            return now, 0

    def release(self, key: str, slot: float | None) -> None:
        """退回 reserve 占用的名额；名额已过期或已被清除时不做任何事。"""
        if slot is None:
            return
        with self._lock:
            events = self._events.get(key)
            if events is not None:
                try:
                    events.remove(slot)
                except ValueError:
                    pass

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


# 改密按账号统计原密码错误：15 分钟内 5 次即暂停，与登录失败窗口互不影响。
PASSWORD_CHANGE_MAX_FAILURES = 5
PASSWORD_CHANGE_WINDOW_SECONDS = 15 * 60


class AuthRateLimiter:
    """登录按用户名和来源地址限制失败次数，注册按来源地址限制提交次数，改密按账号限制原密码错误。"""

    def __init__(
        self,
        *,
        login_failures: int,
        login_window_seconds: int,
        client_failures: int,
        client_window_seconds: int,
        register_attempts: int,
        register_window_seconds: int,
        password_change_failures: int = PASSWORD_CHANGE_MAX_FAILURES,
        password_change_window_seconds: int = PASSWORD_CHANGE_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._by_username = EventWindow(login_failures, login_window_seconds, clock=clock)
        self._by_client = EventWindow(client_failures, client_window_seconds, clock=clock)
        self._registrations = EventWindow(register_attempts, register_window_seconds, clock=clock)
        self._password_changes = EventWindow(
            password_change_failures, password_change_window_seconds, clock=clock
        )

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

    def reserve_login(self, username: str, client: str) -> LoginAttempt:
        """校验密码前先按用户名和来源地址各占一个失败名额；已被限制时抛出 RateLimitedError。

        名额默认算作一次失败，并发请求不能都在失败计数前通过检查。被限制的请求不计数，
        也不会校验密码。密码正确时调用 LoginAttempt.succeeded 退回名额。
        """
        username_key = _username_key(username)
        username_slot, wait = self._by_username.reserve(username_key)
        client_slot = None
        if not wait and client:
            client_slot, wait = self._by_client.reserve(client)
            if wait:
                self._by_username.release(username_key, username_slot)
        if wait:
            raise RateLimitedError(
                f"登录尝试次数过多，请 {_wait_text(wait)}后再试",
                retry_after_seconds=wait,
                code="login_rate_limited",
            )
        return LoginAttempt(self, username_key, username_slot, client, client_slot)

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


    def reserve_password_change(self, account_id: int) -> Callable[[], None]:
        """校验原密码前先占一个失败名额，返回退回名额的函数；已被限制时抛出 RateLimitedError。

        原密码正确（无论改密是否成功）时调用返回的函数退回名额；原密码错误时不调用，名额即失败记录。
        """
        key = str(account_id)
        slot, wait = self._password_changes.reserve(key)
        if wait:
            raise RateLimitedError(
                f"原密码错误次数过多，请 {_wait_text(wait)}后再试",
                retry_after_seconds=wait,
                code="password_change_rate_limited",
            )
        return lambda: self._password_changes.release(key, slot)

    def record_password_change_success(self, account_id: int) -> None:
        self._password_changes.clear(str(account_id))


class LoginAttempt:
    """一次登录占用的失败名额；不调用 succeeded 时名额保留，即记为一次失败。"""

    def __init__(
        self,
        limiter: AuthRateLimiter,
        username_key: str,
        username_slot: float | None,
        client: str,
        client_slot: float | None,
    ) -> None:
        self._limiter = limiter
        self._username_key = username_key
        self._username_slot = username_slot
        self._client = client
        self._client_slot = client_slot

    def succeeded(self, *, clear_username: bool) -> None:
        """密码正确：退回两个名额。clear_username 时清零该用户名的失败计数（登录成功）；
        来源地址的计数保留，以便识别撞库。"""
        username_window = self._limiter._by_username
        if clear_username:
            username_window.clear(self._username_key)
        else:
            username_window.release(self._username_key, self._username_slot)
        if self._client:
            self._limiter._by_client.release(self._client, self._client_slot)


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
