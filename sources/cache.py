"""장중/장마감 차등 TTL 캐시.

Ported from shwi-project/stocklens-mcp (MIT License).
https://github.com/shwi-project/stocklens-mcp/blob/main/stock_mcp_server/_cache.py
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, time as dtime
from functools import wraps
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

_KST = ZoneInfo("Asia/Seoul")
_MARKET_OPEN = dtime(9, 0)
_MARKET_CLOSE = dtime(15, 30)

_cache: dict[str, tuple[float, Any]] = {}
_lock = asyncio.Lock()


def is_market_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(tz=_KST)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return _MARKET_OPEN <= t <= _MARKET_CLOSE


def _make_key(func_name: str, args: tuple, kwargs: dict) -> str:
    parts = [func_name]
    parts.extend(repr(a) for a in args)
    parts.extend(f"{k}={v!r}" for k, v in sorted(kwargs.items()))
    return "|".join(parts)


def cached(ttl_market: int, ttl_closed: int | None = None):
    if ttl_closed is None:
        ttl_closed = ttl_market * 60

    def decorator(func: Callable[..., Awaitable[Any]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = _make_key(func.__name__, args, kwargs)

            async with _lock:
                entry = _cache.get(key)
                if entry is not None:
                    expiry, value = entry
                    if time.time() < expiry:
                        return value
                    del _cache[key]

            result = await func(*args, **kwargs)

            ttl = ttl_market if is_market_open() else ttl_closed
            async with _lock:
                _cache[key] = (time.time() + ttl, result)

            return result

        return wrapper

    return decorator


def clear_cache() -> None:
    _cache.clear()
