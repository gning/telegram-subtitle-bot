"""Polling transport instrumentation and an out-of-event-loop watchdog."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from telegram.request import HTTPXRequest

WATCHDOG_EXIT_CODE = getattr(os, "EX_TEMPFAIL", 75)


class HeartbeatHTTPXRequest(HTTPXRequest):
    """HTTPX request that records completed successful HTTP responses.

    The application uses an instance of this class exclusively for
    ``getUpdates``, so a successful response is a polling heartbeat.
    """

    __slots__ = ("_on_success",)

    def __init__(self, *args: Any, on_success: Callable[[], None], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._on_success = on_success

    async def do_request(self, *args: Any, **kwargs: Any) -> tuple[int, bytes]:
        status_code, payload = await super().do_request(*args, **kwargs)
        try:
            response_ok = bool(json.loads(payload).get("ok"))
        except (AttributeError, json.JSONDecodeError, UnicodeDecodeError):
            response_ok = False
        if 200 <= status_code < 300 and response_ok:
            self._on_success()
        return status_code, payload


class PollingWatchdog:
    """Exit when polling stops completing, even if the asyncio loop is blocked.

    A daemon thread is intentional here. An asyncio task cannot detect the
    event-loop stall that this watchdog is meant to recover from.
    """

    def __init__(
        self,
        timeout_seconds: float,
        *,
        check_interval_seconds: float = 5.0,
        exit_code: int = WATCHDOG_EXIT_CODE,
        exit_func: Callable[[int], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be greater than zero")

        self.timeout_seconds = timeout_seconds
        self.exit_code = exit_code
        self._check_interval_seconds = min(check_interval_seconds, timeout_seconds)
        self._exit_func = exit_func or os._exit
        self._clock = clock
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_success = clock()

    def record_success(self) -> None:
        """Record a successfully completed getUpdates HTTP response."""
        with self._lock:
            self._last_success = self._clock()

    def start(self) -> None:
        """Start monitoring from now; safe to call more than once while running."""
        with self._lock:
            if self._thread is not None:
                if self._thread.is_alive():
                    return
                raise RuntimeError("PollingWatchdog instances cannot be restarted")
            self._last_success = self._clock()
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="telegram-polling-watchdog",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop monitoring without delaying application shutdown."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._check_interval_seconds + 0.1)

    def _run(self) -> None:
        while not self._stop_event.wait(self._check_interval_seconds):
            now = self._clock()
            with self._lock:
                elapsed = now - self._last_success
            if elapsed >= self.timeout_seconds:
                self._write_expiry_message(elapsed)
                self._exit_func(self.exit_code)
                return

    def _write_expiry_message(self, elapsed: float) -> None:
        """Best-effort diagnostic that cannot itself hang on a full log pipe."""
        message = (
            "Polling watchdog: no successful getUpdates response for "
            f"{elapsed:.1f}s (limit {self.timeout_seconds:.1f}s); "
            f"exiting with status {self.exit_code} for supervisor restart.\n"
        ).encode("utf-8", errors="replace")
        stderr_fd: int | None = None
        was_blocking: bool | None = None
        try:
            stderr_fd = sys.stderr.fileno()
            was_blocking = os.get_blocking(stderr_fd)
            os.set_blocking(stderr_fd, False)
            os.write(stderr_fd, message)
        except (AttributeError, OSError, ValueError):
            # The process exits immediately in production. A missing diagnostic
            # must never prevent the recovery action.
            pass
        finally:
            # A test callback may return instead of terminating the process.
            # Restore its stderr mode so the watchdog has no global side effect.
            if stderr_fd is not None and was_blocking is not None:
                try:
                    os.set_blocking(stderr_fd, was_blocking)
                except OSError:
                    pass
