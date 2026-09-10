"""Adversarial attack registry and public interfaces."""

from collections.abc import Callable

from app.attacks.adversarial_patch import AdversarialPatchAttack
from app.attacks.base_attack import BaseAttack
from app.attacks.fgsm import FGSMAttack
from app.attacks.pgd import PGDAttack
from app.attacks.zoo import ZOO_ATTACK_NAMES, ZooAttack

_ATTACK_FACTORIES: dict[str, Callable[[], BaseAttack]] = {
	"fgsm": FGSMAttack,
	"pgd": PGDAttack,
	"adversarial_patch": AdversarialPatchAttack,
}


def get_attack(name: str) -> BaseAttack:
	"""Create a supported attack by name (native or zoo)."""
	factory = _ATTACK_FACTORIES.get(name)
	if factory is not None:
		return factory()
	if name in ZOO_ATTACK_NAMES:
		return ZooAttack(name)
	raise ValueError(f"Unknown attack: {name}")


def supported_attacks() -> list[str]:
	"""Return the registered attack names, including zoo attacks."""
	return list(_ATTACK_FACTORIES) + list(ZOO_ATTACK_NAMES)
