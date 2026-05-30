"""Tests for the shared HTTP client + rate limiter."""

from __future__ import annotations

import time

import httpx
import pytest
import respx

from mybookrec.ingest.http_client import TokenBucket, get_json_with_retry, rate_limited_client


def test_token_bucket_rejects_invalid_rate() -> None:
    """Failure case: non-positive rate raises ValueError at construction."""
    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=0)


def test_token_bucket_enforces_rate_under_burst() -> None:
    """Expected use: when burst is exhausted, second acquire waits."""
    bucket = TokenBucket(rate_per_sec=20, burst=1)
    start = time.monotonic()
    bucket.acquire()
    bucket.acquire()
    elapsed = time.monotonic() - start
    # Second acquire should have waited roughly 1/20 = 50 ms.
    assert elapsed >= 0.04


@respx.mock
def test_get_json_with_retry_returns_payload_on_200() -> None:
    """Expected use: 200 OK with JSON body returns the decoded dict."""
    respx.get("https://example.test/x").mock(return_value=httpx.Response(200, json={"ok": True}))
    with rate_limited_client(rate_per_sec=10, timeout_sec=5) as (client, _):
        result = get_json_with_retry(client, "https://example.test/x")
    assert result == {"ok": True}


@respx.mock
def test_get_json_with_retry_raises_after_exhaustion() -> None:
    """Failure case: persistent 500s exhaust retries and raise."""
    respx.get("https://example.test/y").mock(return_value=httpx.Response(500))
    with rate_limited_client(rate_per_sec=50, timeout_sec=2) as (client, _), pytest.raises(httpx.HTTPStatusError):
        get_json_with_retry(client, "https://example.test/y", max_retries=2, backoff_base_sec=0.01)


@respx.mock
def test_get_json_with_retry_raises_immediately_on_4xx() -> None:
    """Edge case: non-429 4xx (e.g. 404) is not retried — it's a bug, not a blip."""
    respx.get("https://example.test/z").mock(return_value=httpx.Response(404))
    with rate_limited_client(rate_per_sec=50, timeout_sec=2) as (client, _), pytest.raises(httpx.HTTPStatusError):
        get_json_with_retry(client, "https://example.test/z", max_retries=3)
