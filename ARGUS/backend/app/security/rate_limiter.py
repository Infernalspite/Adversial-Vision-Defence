"""Lightweight in-memory rate limiter for expensive endpoints."""

from collections import defaultdict, deque
from time import monotonic

from starlette.requests import Request
from starlette.responses import JSONResponse

from app.security.security_config import security_settings


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def check(self, key: str, route: str) -> bool:
        now = monotonic(); window = security_settings.rate_limit_window_seconds
        hits = self._hits[(key, route)]
        while hits and hits[0] <= now - window: hits.popleft()
        if len(hits) >= security_settings.rate_limit_requests: return False
        hits.append(now); return True


limiter = InMemoryRateLimiter()


async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/v1/") and any(request.url.path.startswith(prefix) for prefix in ("/api/v1/analyze", "/api/v1/attacks", "/api/v1/detection", "/api/v1/defense", "/api/v1/learning")):
        client = request.client.host if request.client else "unknown"
        if not limiter.check(client, request.url.path):
            return JSONResponse(status_code=429, content={"error": {"code": "RATE_LIMITED", "message": "Too many requests for this endpoint.", "request_id": request.state.request_id}})
    return await call_next(request)
