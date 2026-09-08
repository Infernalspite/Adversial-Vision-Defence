"""Adversarial detector registry and public interfaces."""

from collections.abc import Callable

from app.detectors.base_detector import BaseDetector
from app.detectors.confidence_instability import ConfidenceInstabilityDetector
from app.detectors.feature_squeezing import FeatureSqueezingDetector
from app.detectors.frequency_analysis import FrequencyAnalysisDetector
from app.detectors.saliency_analysis import SaliencyAnalysisDetector

_DETECTOR_FACTORIES: dict[str, Callable[[], BaseDetector]] = {
	"feature_squeezing": FeatureSqueezingDetector,
	"frequency_analysis": FrequencyAnalysisDetector,
	"confidence_instability": ConfidenceInstabilityDetector,
	"saliency_analysis": SaliencyAnalysisDetector,
}


def get_detector(name: str) -> BaseDetector:
	"""Create a registered detector by name."""
	try:
		return _DETECTOR_FACTORIES[name]()
	except KeyError as error:
		raise ValueError(f"Unknown detector: {name}") from error


def supported_detectors() -> list[str]:
	"""Return detector names in the default triage order."""
	return list(_DETECTOR_FACTORIES)
