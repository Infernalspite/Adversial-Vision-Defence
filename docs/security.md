# Security and Production Hardening

Phase 11 hardens the MVP without claiming production certification.

## Zero-Trust Ingest

Uploads are validated as untrusted bytes. The validator checks payload size, extension, actual OpenCV decoding, and decoded width/height. Client MIME type is not treated as proof of format. Supported extensions are configurable through `ARGUS_ALLOWED_IMAGE_FORMATS`.

Raw upload bytes are decoded into an RGB representation only after validation. The backend does not persist arbitrary upload filenames.

## Request Validation and Limits

Security settings centralize image limits, attack iterations, patch size, evolution generations, population size, rate limits, origins, and demo mode. Pydantic contracts continue to enforce endpoint-specific numeric ranges.

## Rate Limiting

Expensive endpoint families use an in-memory per-client limiter. Distributed production deployments require a shared limiter and worker-aware state.

## Headers and CORS

The API emits `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, and a compatible Content Security Policy. CORS origins are environment-configured and are not wildcarded.

## Audit Integrity

Audit events include a SHA-256 canonical-record hash and previous-record hash. This is a **TAMPER-EVIDENT AUDIT CHAIN**, not blockchain or proof of complete system integrity. JSON persistence is suitable only for the MVP.

## Errors and Logging

Validation and HTTP failures use an `{error: {code, message, request_id}}` envelope. Unhandled errors do not expose stack traces or filesystem paths. Raw images, secrets, and credentials are not logged.

## Failure Safety

Final states remain exactly `TRUSTED`, `DEFENDED`, and `ABSTAIN`. Exceptions are returned as structured API errors; the decision engine independently requires evidence for `TRUSTED` and `DEFENDED`, and abstention remains the conservative fallback.

## Model Security

The existing model registry controls model construction. ResNet inference uses evaluation/inference mode. Attack gradients are limited to attack-generation code; normal inference does not update model parameters.

## Configuration

Copy `.env.example` to `.env` and review limits before running. Demo mode and offline learning remain explicitly separate from production inference.

## Limitations

This is a hackathon MVP. The rate limiter is process-local, JSON audit storage is not a concurrent database, and no authentication or distributed security control is included. `scripts/security_check.py` is a lightweight posture check, not a complete security audit.
