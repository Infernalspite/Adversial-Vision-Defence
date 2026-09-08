"""Baseline inference API contracts."""

from pydantic import BaseModel, Field


class TopPredictionResponse(BaseModel):
    """One ranked baseline model prediction."""

    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)


class InferenceResponse(BaseModel):
    """Clean-image ResNet-18 inference response."""

    request_id: str
    model_name: str
    predicted_class: str
    predicted_class_id: int
    confidence: float = Field(ge=0, le=1)
    top_predictions: list[TopPredictionResponse] = Field(min_length=1, max_length=5)
    inference_time_ms: float = Field(ge=0)


class InferenceMetadata(BaseModel):
    """Optional metadata for an uploaded image request."""

    source: str | None = None
    filename: str | None = None
