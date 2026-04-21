"""Streamlit(동기) → async 함수 호출용 헬퍼.

Streamlit 스크립트는 매 re-run마다 새 스레드에서 실행되지만 이벤트 루프는 없다.
asyncio.run()은 매 호출마다 새 루프를 만들어 안전하지만, 우리는 sources/http_client.py
싱글톤 AsyncClient를 한 루프 안에서만 안전하게 쓸 수 있다.

이 모듈은 프로세스 전역에 단일 백그라운드 이벤트 루프를 두고, 그 루프에 코루틴을
제출해 결과를 동기적으로 반환한다. 같은 루프를 재사용하므로 AsyncClient가 안전하다.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Awaitable, TypeVar

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop, _loop_thread
    with _lock:
        if _loop is not None and not _loop.is_closed():
            return _loop

        loop = asyncio.new_event_loop()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_runner, name="stock-async-loop", daemon=True)
        thread.start()

        _loop = loop
        _loop_thread = thread
        return loop


def run_sync(coro: Awaitable[T], timeout: float = 30.0) -> T:
    """코루틴을 백그라운드 루프에 제출하고 동기적으로 결과를 받는다."""
    loop = _ensure_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result(timeout=timeout)
