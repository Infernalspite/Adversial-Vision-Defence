"""Structured request and error helpers."""

from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse


def request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    if value is None:
        value = str(uuid4()); request.state.request_id = value
    return value


def error_response(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message, "request_id": request_id(request)}})
