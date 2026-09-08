"""Safe fallback contract for abstention."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FallbackResponse:
    """Non-actuating fallback returned when evidence is insufficient."""

    action: str
    reason: str


def abstain_fallback() -> FallbackResponse:
    """Return the safe MVP fallback without performing an external action."""
    return FallbackResponse("NO_ACTION", "Insufficient confidence to safely proceed.")
