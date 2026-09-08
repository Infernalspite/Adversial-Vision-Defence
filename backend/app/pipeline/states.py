"""Externally meaningful and internal pipeline states."""

from enum import Enum


class FinalState(str, Enum):
    """Decision states exposed to clients."""

    TRUSTED = "TRUSTED"
    DEFENDED = "DEFENDED"
    ABSTAIN = "ABSTAIN"


class PipelineStage(str, Enum):
    """Internal stages used for tracing and audit records."""

    INGEST = "ingest"
    SANITIZE = "sanitize"
    INFERENCE = "inference"
    TRIAGE = "triage"
    TRUST_MAPPING = "trust_mapping"
    DEFENSE = "defense"
    REINFERENCE = "reinference"
    VERIFICATION = "verification"
    FUSION = "fusion"
    AUDIT = "audit"
