"""Domain exceptions shared by API adapters and services."""


class ArgusError(Exception):
    """Base exception for expected ARGUS-AEGIS failures."""


class NotImplementedDomainError(ArgusError):
    """Raised when a planned algorithm has not been implemented yet."""
