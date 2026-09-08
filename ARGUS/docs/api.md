# API Contract

Base URL: `/api/v1`

## Health

`GET /health` returns `{ "status": "ok", "service": "argus-aegis" }`.

## Inference

`POST /inference` accepts a multipart `image` upload and returns the full `InferenceResponse` contract. The current scaffold returns an `ABSTAIN` response with no model calculation.

The response includes request correlation, original and defended predictions, confidence values, attack summary, trust map, defense summary, semantic verification score, final state, explanation, processing time, and audit records.

## Attack Lab

- `GET /attacks/supported` lists `fgsm`, `pgd`, and `adversarial_patch`.
- `POST /attacks/analyze` is reserved for structured triage.

## Defense

- `GET /defense/strategies` lists planned strategies.
- `POST /defense/preview` is reserved for policy-driven orchestration.

## Audit

`GET /audit/{request_id}` is reserved for request-level audit retrieval.

## Detection

`POST /detection/analyze` accepts a multipart `image` upload. Optional form fields are `detectors`, a comma-separated list of registered detector names, and `threshold`, a value from `0` to `1`.

The response contains `request_id`, `attack_score`, `attack_detected`, `detection_threshold`, per-detector scores/evidence/timing/metadata, an explanation, and total processing time. Detection scores are evidence only; this endpoint does not perform defense or return a final trust state.

## Trust Map

`POST /trust-map/analyze` accepts an image plus optional `trust_weights` JSON, `localization_threshold`, `minimum_region_area`, and an optional `patch_mask` upload. It returns compact PNG data references for the trust map and overlay, global trust score, spatial evidence availability, and suspicious regions. Patch masks are optional evaluation metadata and are not inferred or required for ordinary analysis.

## Defense

`POST /defense/apply` accepts an image and internally runs Phase 3 detection, Phase 4 trust mapping, policy selection, one active MVP defense, and baseline re-inference. It returns original/defended predictions, attack score, global trust, suspicious regions, selected strategy, parameters, defended image reference, and defense trace. It deliberately does not return a final trust state.

## Verification

`POST /verification/analyze` accepts `original_image` and `defended_image` multipart fields. It compares existing or internally generated model predictions with object, geometry, and scene consistency evidence, then returns a verification score. It does not return `TRUSTED`, `DEFENDED`, or `ABSTAIN`.

## Complete Analysis

`POST /analyze` is the main Phase 7 endpoint. It accepts an image and optional `mode`/`attack_type` metadata, then runs detection, trust mapping, defense, re-inference, verification, and the final decision engine. The final state is exactly `TRUSTED`, `DEFENDED`, or `ABSTAIN`. Evaluation metadata does not influence the decision rules.

## Audit Retrieval

- `GET /audit?limit=20` returns recent persisted audit records.
- `GET /audit/{request_id}` returns the decision-chain record for one request.

Records contain metadata and evidence summaries only; image bytes are not persisted.

## Implementation Notes

Routes must remain adapters. Detection, defense selection, verification, and fusion belong in their respective domain packages.
