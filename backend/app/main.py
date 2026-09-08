"""FastAPI application entry point."""

from fastapi import FastAPI, Request
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.exceptions import HTTPException

from app.api.routes import attacks, audit, defense, detection, evaluation, health, inference, learning, trust, verification
from app.core.config import settings
from app.security.rate_limiter import rate_limit_middleware
from app.security.request_validation import error_response, request_id
from app.security.security_headers import security_headers_middleware
from app.security.security_config import security_settings

app = FastAPI(
    title="ARGUS-AEGIS API",
    version="0.1.0",
    description="Typed API skeleton for autonomous adversarial defense and recovery.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=security_settings.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(security_headers_middleware)
app.middleware("http")(rate_limit_middleware)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID", request_id(request))
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > security_settings.max_image_bytes + 1024 * 1024:
        return error_response(request, 413, "REQUEST_TOO_LARGE", "Request exceeds the configured size limit.")
    try:
        response = await asyncio.wait_for(call_next(request), timeout=security_settings.request_timeout_seconds)
    except asyncio.TimeoutError:
        return error_response(request, 504, "REQUEST_TIMEOUT", "The request exceeded the configured processing time limit.")
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return error_response(request, 422, "VALIDATION_ERROR", "Request validation failed.")


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    code = "REQUEST_ERROR" if exc.status_code < 500 else "SERVICE_ERROR"
    return error_response(request, exc.status_code, code, str(exc.detail))


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    return error_response(request, 500, "INTERNAL_ERROR", "The request could not be completed safely.")

app.include_router(health.router, prefix="/api/v1")
app.include_router(inference.router, prefix="/api/v1")
app.include_router(inference.analysis_router, prefix="/api/v1")
app.include_router(attacks.router, prefix="/api/v1")
app.include_router(detection.router, prefix="/api/v1")
app.include_router(trust.router, prefix="/api/v1")
app.include_router(verification.router, prefix="/api/v1")
app.include_router(defense.router, prefix="/api/v1")
app.include_router(audit.router, prefix="/api/v1")
app.include_router(evaluation.router, prefix="/api/v1")
app.include_router(learning.router, prefix="/api/v1")


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    """Return a small service descriptor."""
    return {"service": settings.app_name, "status": "scaffold"}
