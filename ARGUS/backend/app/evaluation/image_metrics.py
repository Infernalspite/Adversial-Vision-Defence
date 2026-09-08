"""Image distortion metrics."""

from typing import Any

import numpy as np


def image_metrics(original: np.ndarray, defended: np.ndarray) -> dict[str, float]:
    """Calculate MAE, MSE, and PSNR on normalized pixel ranges."""
    difference = defended.astype(np.float32) / 255 - original.astype(np.float32) / 255
    mae = float(np.abs(difference).mean()); mse = float(np.mean(difference ** 2))
    return {"mae": mae, "mse": mse, "psnr_db": float("inf") if mse == 0 else float(10 * np.log10(1 / mse))}
