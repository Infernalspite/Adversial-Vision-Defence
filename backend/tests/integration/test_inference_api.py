import io

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_test_image() -> bytes:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 1] = 255
    success, encoded = cv2.imencode('.png', image)
    assert success
    return encoded.tobytes()


def test_valid_image_upload_returns_baseline_response():
    response = client.post(
        '/api/v1/inference',
        files={'image': ('sample.png', io.BytesIO(make_test_image()), 'image/png')},
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        'request_id', 'model_name', 'predicted_class', 'predicted_class_id',
        'confidence', 'top_predictions', 'inference_time_ms',
    }
    assert payload['model_name'] == 'resnet18'
    assert 0 <= payload['confidence'] <= 1
    assert len(payload['top_predictions']) == 5


def test_invalid_file_is_rejected():
    response = client.post(
        '/api/v1/inference',
        files={'image': ('sample.txt', b'not an image', 'text/plain')},
    )
    assert response.status_code == 400