import numpy as np
import torch

from app.attacks import get_attack
from app.audit.audit_logger import AuditLogger
from app.models.base_model import TopPrediction
from app.pipeline.pipeline import ArgusPipeline, PipelineContext


class FakeModel:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 1000))
        self.model.eval()

    def pixel_to_model_input(self, image: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(image.unsqueeze(0), size=(16, 16), mode="bilinear", align_corners=False).squeeze(0)

    def predict(self, image: np.ndarray):
        source = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
        values = torch.softmax(self.model(self.pixel_to_model_input(source).unsqueeze(0)), dim=1)[0]
        indices = torch.topk(values, 5).indices
        top = tuple(TopPrediction(int(index), str(int(index)), float(values[index].detach())) for index in indices)
        return type("Prediction", (), {"class_id": top[0].class_id, "class_name": top[0].class_name, "confidence": top[0].confidence, "top_predictions": top, "inference_time_ms": 0.1})()


def test_clean_and_all_phase2_attacks_reach_final_decision(tmp_path):
    model = FakeModel(); image = np.zeros((32, 32, 3), dtype=np.uint8); image[:, :, 1] = 220
    cases = [("clean", image)]
    cases.extend((name, get_attack(attack).generate(image, model, **params).adversarial_image) for name, attack, params in (("fgsm", "fgsm", {"epsilon": 0.01}), ("pgd", "pgd", {"epsilon": 0.03, "step_size": 0.01, "iterations": 1, "random_start": False}), ("patch", "adversarial_patch", {"patch_size": 0.2, "iterations": 1})))
    for name, sample in cases:
        result = ArgusPipeline(audit_logger=AuditLogger(tmp_path / f"{name}.json")).run(PipelineContext(name, sample, {"model": model, "mode": "evaluation"}))
        assert result.decision.final_state.value in {"TRUSTED", "DEFENDED", "ABSTAIN"}
        assert result.defense.detection is not None
        assert result.defense.trust is not None
        assert result.verification.verification.verification_score >= 0
        assert result.audit
        if result.decision.final_state.value == "DEFENDED":
            assert result.defense.orchestration.defense.defense_applied
