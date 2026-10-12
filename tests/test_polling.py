import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from telegram.error import TimedOut
from telegram.request import HTTPXRequest

from bot.polling import HeartbeatHTTPXRequest, PollingWatchdog, WATCHDOG_EXIT_CODE


class HeartbeatHTTPXRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_response_records_heartbeat(self) -> None:
        heartbeat = Mock()
        request = HeartbeatHTTPXRequest(on_success=heartbeat)
        try:
            with patch.object(
                HTTPXRequest,
                "do_request",
                new=AsyncMock(return_value=(200, b'{"ok":true,"result":[]}')),
            ):
                result = await request.do_request("http://example.invalid", "POST")
        finally:
            await request.shutdown()

        self.assertEqual(result, (200, b'{"ok":true,"result":[]}'))
        heartbeat.assert_called_once_with()

    async def test_error_response_does_not_record_heartbeat(self) -> None:
        heartbeat = Mock()
        request = HeartbeatHTTPXRequest(on_success=heartbeat)
        try:
            with patch.object(
                HTTPXRequest,
                "do_request",
                new=AsyncMock(return_value=(500, b'{}')),
            ):
                await request.do_request("http://example.invalid", "POST")
        finally:
            await request.shutdown()

        heartbeat.assert_not_called()

    async def test_httpx_timeout_is_finite_and_propagated(self) -> None:
        async def time_out(_request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("simulated stuck response")

        request = HeartbeatHTTPXRequest(
            on_success=lambda: None,
            read_timeout=0.05,
            connect_timeout=0.2,
            write_timeout=0.2,
            pool_timeout=0.2,
            httpx_kwargs={"transport": httpx.MockTransport(time_out)},
        )
        try:
            self.assertEqual(request.read_timeout, 0.05)
            with self.assertRaises(TimedOut):
                await request.do_request("http://example.invalid/getUpdates", "POST")
        finally:
            await request.shutdown()


class PollingWatchdogSimulationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stalled_poll_request_triggers_nonzero_exit(self) -> None:
        """Simulate getUpdates never returning while the event loop stays alive."""
        expired = threading.Event()
        exit_codes: list[int] = []

        def fake_exit(code: int) -> None:
            exit_codes.append(code)
            expired.set()

        watchdog = PollingWatchdog(
            0.05,
            check_interval_seconds=0.005,
            exit_func=fake_exit,
        )
        request = HeartbeatHTTPXRequest(on_success=watchdog.record_success)

        async def never_returns(*args, **kwargs):
            await asyncio.Future()

        watchdog.start()
        try:
            with patch.object(HTTPXRequest, "do_request", new=never_returns):
                poll_task = asyncio.create_task(
                    request.do_request("http://example.invalid/getUpdates", "POST")
                )

                async def wait_for_expiry() -> bool:
                    for _ in range(100):
                        if expired.is_set():
                            return True
                        await asyncio.sleep(0.01)
                    return False

                fired = await wait_for_expiry()
                self.assertTrue(fired, "watchdog did not detect the stalled poll")
                self.assertEqual(exit_codes, [WATCHDOG_EXIT_CODE])
                self.assertFalse(poll_task.done())
                poll_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await poll_task
        finally:
            watchdog.stop()
            await request.shutdown()


if __name__ == "__main__":
    unittest.main()
