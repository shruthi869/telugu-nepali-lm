"""Rate-aware urllib HTTP client with retry classification."""

from __future__ import annotations

import time
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener

from .common import (
    TRANSIENT_HTTP,
    USER_AGENT,
    CollectorError,
    Response,
    TransientError,
    parse_retry_after,
)

class HttpClient:
    """Small urllib client with per-host spacing and explicit retry classification."""

    def __init__(self, timeout: float = 45.0):
        self.timeout = timeout
        self.opener = build_opener()
        self.last_request: dict[str, float] = {}

    def open_stream(self, url: str, headers: dict[str, str] | None = None, delay: float = 0.0) -> Any:
        """Open an HTTP response without buffering its body in memory."""
        parsed = urlsplit(url)
        previous = self.last_request.get(parsed.netloc, 0.0)
        wait = delay - (time.monotonic() - previous)
        if wait > 0:
            time.sleep(wait)
        request_headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        request_headers.update(headers or {})
        request = Request(url, headers=request_headers)
        try:
            self.last_request[parsed.netloc] = time.monotonic()
            return self.opener.open(request, timeout=self.timeout)
        except HTTPError as exc:
            headers_out = dict(exc.headers.items()) if exc.headers else {}
            exc.close()
            if exc.code in TRANSIENT_HTTP:
                raise TransientError(f"HTTP {exc.code} for {url}", parse_retry_after(headers_out.get("Retry-After")))
            raise CollectorError(f"HTTP {exc.code} for {url}: access will not be bypassed") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise TransientError(f"network error for {url}: {exc}") from exc

    def fetch(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        delay: float = 0.0,
        max_bytes: int = 50 * 1024 * 1024,
    ) -> Response:
        """Fetch a small response such as robots.txt or a sitemap into memory."""
        try:
            with self.open_stream(url, headers=headers, delay=delay) as reply:
                declared = reply.headers.get("Content-Length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise CollectorError(f"response exceeds configured {max_bytes:,}-byte limit for {url}")
                body = reply.read(max_bytes + 1)
                if len(body) > max_bytes:
                    raise CollectorError(f"response exceeds configured {max_bytes:,}-byte limit for {url}")
                return Response(reply.geturl(), reply.status, dict(reply.headers.items()), body)
        except (HTTPException, TimeoutError, OSError) as exc:
            raise TransientError(f"network error while reading {url}: {exc}") from exc
