"""Image decoding and ResNet-18 preprocessing utilities."""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np
import torch

RESNET_IMAGE_SIZE: Final[int] = 224
RESNET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
RESNET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)


def decode_image(payload: bytes) -> np.ndarray:
    """Decode uploaded bytes into an RGB ``uint8`` image."""
    if not payload:
        raise ValueError("Image payload is empty")

    encoded = np.frombuffer(payload, dtype=np.uint8)
    bgr_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError("Uploaded file is not a supported image")
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def preprocess_image(image: np.ndarray, image_size: int = RESNET_IMAGE_SIZE) -> torch.Tensor:
    """Resize, center-crop, and normalize an RGB image for ResNet-18."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3)")

    height, width = image.shape[:2]
    scale = 256 / min(height, width)
    resized_width = max(image_size, round(width * scale))
    resized_height = max(image_size, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    top = (resized_height - image_size) // 2
    left = (resized_width - image_size) // 2
    cropped = resized[top : top + image_size, left : left + image_size]
    tensor = torch.from_numpy(cropped.copy()).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor(RESNET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(RESNET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def normalize_tensor(image: torch.Tensor) -> torch.Tensor:
    """Normalize a ``(3, height, width)`` RGB tensor in the ``[0, 1]`` range."""
    mean = torch.tensor(RESNET_MEAN, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(RESNET_STD, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return (image - mean) / std


def tensor_to_image(image: torch.Tensor) -> np.ndarray:
    """Convert a single RGB ``[0, 1]`` tensor to a ``uint8`` NumPy image."""
    clipped = image.detach().cpu().clamp(0, 1)
    return (clipped.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def encode_image(image: np.ndarray, extension: str = ".png") -> bytes:
    """Encode an RGB image as an image payload."""
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(extension, bgr_image)
    if not success:
        raise ValueError("Unable to encode generated image")
    return encoded.tobytes()
