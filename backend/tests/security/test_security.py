import json
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.audit.audit_logger import AuditEvent, AuditLogger
from app.main import app
from app.security.input_validator import InputValidationError, validate_image_payload

client = TestClient(app)


def png_bytes(width=4, height=4):
    success, encoded = cv2.imencode(".png", np.zeros((height, width, 3), dtype=np.uint8))
    assert success
    return encoded.tobytes()


def test_invalid_and_corrupt_uploads_are_rejected():
    response = client.post("/api/v1/inference", files={"image": ("bad.txt", b"nope", "image/png")})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "REQUEST_ERROR"
    with __import__("pytest").raises(InputValidationError): validate_image_payload(b"bad", "x.png", "image/png")


def test_dimensions_and_extension_are_bounded():
    with __import__("pytest").raises(InputValidationError): validate_image_payload(png_bytes(5000, 4), "x.png", "image/png")
    with __import__("pytest").raises(InputValidationError): validate_image_payload(png_bytes(), "x.svg", "image/png")


def test_security_headers_and_cors():
    response = client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_audit_hash_chain_detects_tampering(tmp_path: Path):
    path = tmp_path / "audit.json"
    logger = AuditLogger(path); logger.record(AuditEvent("r", "decision", "ok"))
    assert logger.verify_integrity() is True
    records = json.loads(path.read_text(encoding="utf-8")); records[0]["message"] = "tampered"; path.write_text(json.dumps(records), encoding="utf-8")
    assert AuditLogger(path).verify_integrity() is False