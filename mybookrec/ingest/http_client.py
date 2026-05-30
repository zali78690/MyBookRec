"""Shared HTTP client for ingestion fetchers.

Wraps httpx with a token-bucket rate limiter so we don't accidentally DOS a free API. One
client per source — different bases, headers, and rate limits — but the limiter and retry
behaviour are shared.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import httpx


class TokenBucket:
    """Simple monotonic-clock token bucket.

    Not threadsafe — call from one event loop / thread per bucket. For ingestion CLIs that's
    fine: each source runs serially.
    """

    def __init__(self, rate_per_sec: float, burst: int = 1) -> None:
        """Create a bucket producing `rate_per_sec` tokens, with capacity `burst`.

        Args:
            rate_per_sec: Steady-state token generation rate.
            burst: Maximum tokens that can accumulate during idleness.
        """
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        self.rate = rate_per_sec
        self.capacity = max(1, burst)
        self.tokens = float(self.capacity)
        self.updated_at = time.monotonic()

    def acquire(self) -> None:
        """Block until a token is available, then consume one."""
        while True:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated_at) * self.rate)
            self.updated_at = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            time.sleep(max(0.005, (1.0 - self.tokens) / self.rate))


@contextmanager
def rate_limited_client(
    *,
    rate_per_sec: float,
    timeout_sec: float,
    headers: dict[str, str] | None = None,
) -> Iterator[tuple[httpx.Client, TokenBucket]]:
    """Yield an httpx client paired with a token bucket.

    Callers call `bucket.acquire()` before each request. Cleanup is automatic on context exit.

    Args:
        rate_per_sec: Max sustained requests per second.
        timeout_sec: Per-request timeout.
        headers: Default request headers (User-Agent, etc.).

    Yields:
        Tuple of (httpx.Client, TokenBucket) to use for the duration of the block.
    """
    bucket = TokenBucket(rate_per_sec=rate_per_sec, burst=max(1, int(rate_per_sec)))
    client = httpx.Client(timeout=timeout_sec, headers=headers or {})
    try:
        yield client, bucket
    finally:
        client.close()


def get_json_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict | None = None,
    max_retries: int = 3,
    backoff_base_sec: float = 1.0,
) -> dict | list:
    """GET a JSON response with bounded retries on transient failures.

    Retries on 5xx + 429 + httpx network errors. 4xx (except 429) raises immediately —
    they indicate bugs, not transient blips.

    Args:
        client: An httpx.Client.
        url: Absolute URL to fetch.
        params: Query string parameters.
        max_retries: Total attempts including the first.
        backoff_base_sec: Initial backoff; doubles per retry.

    Returns:
        Decoded JSON object or array.

    Raises:
        httpx.HTTPStatusError: For non-transient 4xx responses, or after retries exhausted.
        httpx.RequestError: After retries exhausted on network errors.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.get(url, params=params)
            if response.status_code in (429,) or 500 <= response.status_code < 600:
                last_exc = httpx.HTTPStatusError(
                    f"transient {response.status_code}", request=response.request, response=response
                )
                time.sleep(backoff_base_sec * (2**attempt))
                continue
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            last_exc = e
            time.sleep(backoff_base_sec * (2**attempt))
    assert last_exc is not None
    raise last_exc
