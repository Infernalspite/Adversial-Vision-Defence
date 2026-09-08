"""Adversarial attack registry and public interfaces."""

from collections.abc import Callable

from app.attacks.adversarial_patch import AdversarialPatchAttack
from app.attacks.base_attack import BaseAttack
from app.attacks.fgsm import FGSMAttack
from app.attacks.pgd import PGDAttack

_ATTACK_FACTORIES: dict[str, Callable[[], BaseAttack]] = {
	"fgsm": FGSMAttack,
	"pgd": PGDAttack,
	"adversarial_patch": AdversarialPatchAttack,
}


def get_attack(name: str) -> BaseAttack:
	"""Create a supported attack by name."""
	try:
		return _ATTACK_FACTORIES[name]()
	except KeyError as error:
		raise ValueError(f"Unknown attack: {name}") from error


def supported_attacks() -> list[str]:
	"""Return the registered attack names."""
	return list(_ATTACK_FACTORIES)
