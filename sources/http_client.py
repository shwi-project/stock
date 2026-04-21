"""싱글톤 httpx.AsyncClient + Semaphore + 백오프 재시도.

Ported from shwi-project/stocklens-mcp (MIT License).
https://github.com/shwi-project/stocklens-mcp/blob/main/stock_mcp_server/_http.py
"""

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
_semaphore: asyncio.Semaphore | None = None
_client: httpx.AsyncClient | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
    return _semaphore


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers=_HEADERS,
            follow_redirects=True,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=30,
                keepalive_expiry=30.0,
            ),
        )
    return _client


async def fetch(
    url: str,
    *,
    params: dict | None = None,
    max_retries: int = 2,
) -> httpx.Response:
    client = get_client()
    sem = _get_semaphore()
    last_exc: Exception | None = None

    async with sem:
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


async def close_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
