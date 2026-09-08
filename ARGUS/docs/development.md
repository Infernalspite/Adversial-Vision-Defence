# Development

## Local Loop

1. Create or update a typed contract.
2. Add a focused test describing the expected behavior.
3. Implement the owning domain module, not the route.
4. Add audit evidence for security decisions.
5. Run backend and frontend checks.

## Boundaries

Keep image processing and ML logic in `backend/app/`; keep HTTP adapters in `backend/app/api/`; keep fetch logic in `frontend/src/services/`; keep presentation in React components.

## Environment

Copy `.env.example` to `.env` for local values. Do not commit secrets or generated datasets.

## Next Phase

Implement model adapters first, then pipeline orchestration, detectors and attack scoring, trust and verification, defense strategies, and finally API/UI integration. Preserve `TRUSTED`, `DEFENDED`, and `ABSTAIN` as the stable external decision vocabulary.
