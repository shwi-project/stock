"""Streamlit(동기) → async 함수 호출용 헬퍼.

Streamlit 스크립트 스레드는 기본적으로 실행 중인 asyncio 이벤트 루프가 없으므로
`asyncio.run()`을 그대로 호출하면 된다. 만약 이미 루프가 돌고 있다면(테스트 환경
등) 별도 스레드에 위임한다.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Awaitable, TypeVar

T = TypeVar("T")


def run_sync(coro: Awaitable[T], timeout: float = 30.0) -> T:
    """코루틴을 동기 컨텍스트에서 실행해 결과를 반환."""
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False

    if running:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=timeout)

    return asyncio.run(coro)
