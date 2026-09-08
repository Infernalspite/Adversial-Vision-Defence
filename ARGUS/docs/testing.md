# Testing

The test layout mirrors the security domains:

- `backend/tests/unit/`: contract and component-level tests.
- `backend/tests/integration/`: API and end-to-end pipeline tests.
- `backend/tests/fixtures/`: controlled clean, attacked, and sample inputs.
- `frontend/src/App.test.tsx`: React shell test; add component and workflow tests as UI behavior grows.

Planned evaluation coverage includes clean images, FGSM, PGD, adversarial patches, detector accuracy, attack detection rate, defense recovery rate, false abstention, and end-to-end state transitions.

Run backend tests with `pytest` from `backend/`. Run frontend tests with `npm run test:run` from `frontend/`.

The current tests validate interfaces and the intentionally explicit unimplemented state. They do not claim ML correctness.
