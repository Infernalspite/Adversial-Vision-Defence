from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_learning_endpoints_are_bounded_and_read_only():
    mine = client.post("/api/v1/learning/mine")
    assert mine.status_code == 200
    evolve = client.post("/api/v1/learning/evolve", json={"attack": "pgd", "generations": 99, "population": 99})
    assert evolve.status_code == 422
    experiments = client.get("/api/v1/learning/experiments")
    assert experiments.status_code == 200
    negatives = client.get("/api/v1/learning/hard-negatives")
    assert negatives.status_code == 200
