"""Zero-trust image validation for untrusted upload bytes."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.security.security_config import security_settings


class InputValidationError(ValueError):
    """Raised when untrusted upload bytes fail validation."""


def validate_image_payload(payload: bytes, filename: str | None = None, content_type: str | None = None) -> np.ndarray:
    """Validate extension, bytes, decoded dimensions, and actual image decodability."""
    if not payload:
        raise InputValidationError("Image payload is empty")
    if len(payload) > security_settings.max_image_bytes:
        raise InputValidationError("Image exceeds the configured size limit")
    extension = Path(filename or "").suffix.lower().lstrip(".")
    if extension and extension not in security_settings.formats:
        raise InputValidationError("Image extension is not allowed")
    encoded = np.frombuffer(payload, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise InputValidationError("Uploaded bytes are not a supported image")
    height, width = decoded.shape[:2]
    if width > security_settings.max_image_width or height > security_settings.max_image_height:
        raise InputValidationError("Image dimensions exceed the configured limit")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
