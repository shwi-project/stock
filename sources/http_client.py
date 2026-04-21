"""httpx 기반 HTTP 페치 헬퍼 + 지수 백오프 재시도.

Ported from shwi-project/stocklens-mcp (MIT License).
https://github.com/shwi-project/stocklens-mcp/blob/main/stock_mcp_server/_http.py

Streamlit 환경에서는 매 호출마다 새 asyncio 루프가 생성되므로 싱글톤 클라이언트를
유지하지 않는다. `fetch()` 호출 범위에서 클라이언트를 생성/종료한다. 단일
`run_sync(coro)` 내부의 `asyncio.gather(...)`는 각기 자체 클라이언트를 쓰지만
동시 접속 수는 `_MAX_CONCURRENT` 세마포어로 제한한다.
"""

from __future__ import annotations

import asyncio
import random

import httpx

_TIMEOUT = 8.0
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

_MAX_CONCURRENT = 15
_sem_cache: dict[int, asyncio.Semaphore] = {}


def _semaphore_for_current_loop() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    key = id(loop)
    sem = _sem_cache.get(key)
    if sem is None:
        sem = asyncio.Semaphore(_MAX_CONCURRENT)
        _sem_cache[key] = sem
    return sem


async def fetch(
    url: str,
    *,
    params: dict | None = None,
    max_retries: int = 2,
) -> httpx.Response:
    sem = _semaphore_for_current_loop()
    last_exc: Exception | None = None

    async with sem:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers=_HEADERS,
            follow_redirects=True,
        ) as client:
            for attempt in range(max_retries + 1):
                try:
                    resp = await client.get(url, params=params)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        if attempt < max_retries:
                            backoff = (2 ** attempt) * 0.5 + random.uniform(0, 0.3)
                            await asyncio.sleep(backoff)
                            continue
                    return resp
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    last_exc = e
                    if attempt < max_retries:
                        backoff = (2 ** attempt) * 0.5 + random.uniform(0, 0.3)
                        await asyncio.sleep(backoff)
                        continue
                    raise

    if last_exc:
        raise last_exc
    raise RuntimeError("fetch failed without exception")
