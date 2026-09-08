"""Defense registry and strategy interfaces."""

from collections.abc import Callable

from app.defense.base_defense import BaseDefense, NoDefense
from app.defense.certified_inference import CertifiedInferenceDefense
from app.defense.masking import MaskingDefense
from app.defense.purification import PurificationDefense
from app.defense.transforms import TransformDefense

_DEFENSE_FACTORIES: dict[str, Callable[[], BaseDefense]] = {
	"transform": TransformDefense,
	"mask": MaskingDefense,
	"purification": PurificationDefense,
	"certified": CertifiedInferenceDefense,
	"none": NoDefense,
}


def get_defense(name: str) -> BaseDefense:
	"""Create a registered defense by name."""
	try:
		return _DEFENSE_FACTORIES[name]()
	except KeyError as error:
		raise ValueError(f"Unknown defense: {name}") from error


def supported_defenses() -> list[str]:
	"""Return active and roadmap defense names."""
	return ["transform", "mask", "purification", "certified"]
