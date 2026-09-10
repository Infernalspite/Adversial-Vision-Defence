"""Markdown robustness-report rendering.

The renderer enforces the reporting discipline: clean and robust accuracy are
always shown together, detection numbers are broken down by attack family and
strength, FPR is explicit, unseen attack families get their own section, and
the threat-model limits are stated rather than omitted.
"""

from __future__ import annotations

import json
from typing import Any

ZOO_ATTACK_NAMES = ("cw_l2", "deepfool", "mim", "square", "hopskipjump", "autoattack")


def _pct(value: Any) -> str:
    try:
        return f"{100 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def render_robustness_report(summary: dict[str, Any]) -> str:
    """Render the final robustness report from the evaluation summary dict."""
    meta = summary.get("meta", {})
    lines: list[str] = []
    lines.append("# ARGUS-AEGIS Robustness Report")
    lines.append("")
    lines.append(f"- Generated: {meta.get('generated', 'unknown')}")
    lines.append(f"- Defended model: {meta.get('model', 'robust specialist (ResNet-18)')}")
    lines.append(f"- Detector calibration: `{meta.get('calibration', 'data/models/detector_calibration.json')}`")
    lines.append(f"- Evaluation split: **test only** (untouched during training and calibration)")
    lines.append(f"- Test samples: clean {meta.get('clean_count', 0)}, attacked {meta.get('attacked_count', 0)}")
    lines.append("")

    clean = summary.get("clean", {})
    lines.append("## Headline numbers")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Clean accuracy (defended model, unperturbed) | {_pct(clean.get('clean_accuracy'))} |")
    lines.append(f"| Clean FPR of the attack detector | {_pct(clean.get('fpr'))} |")
    lines.append(f"| Clean abstention rate | {_pct(clean.get('abstain_rate'))} |")
    lines.append(f"| Robust accuracy before defense (calibrated attacks) | {_pct(summary.get('robust_accuracy_before_defense'))} |")
    lines.append(f"| Robust accuracy after defense | {_pct(summary.get('robust_accuracy_after_defense'))} |")
    lines.append(f"| Detection F1 (calibrated attacks) | {_fmt(summary.get('detection', {}).get('f1'))} |")
    lines.append(f"| Detection FPR (calibrated attacks) | {_pct(summary.get('detection', {}).get('fpr'))} |")
    lines.append(f"| Unsafe acceptance rate (attacked, TRUSTED) | {_pct(summary.get('decisions', {}).get('unsafe_acceptance_rate'))} |")
    lines.append(f"| Abstention precision (ABSTAIN that are adversarial) | {_pct(summary.get('abstention_precision'))} |")
    lines.append(f"| Pipeline latency mean / P95 | {_fmt(summary.get('latency', {}).get('mean_ms'), 0)} ms / {_fmt(summary.get('latency', {}).get('p95_ms'), 0)} ms |")
    lines.append("")

    by_family = summary.get("by_attack_family", {})
    if by_family:
        lines.append("## Per attack family and strength")
        lines.append("")
        lines.append("| Family | Epsilon/Size | Samples | Attack success | Robust acc (pre) | Detected (TPR) | Recovery | Unsafe accept |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for family in sorted(by_family):
            for entry in sorted(by_family[family], key=lambda item: str(item.get("strength"))):
                lines.append(
                    f"| {family} | {entry.get('strength', '-')} | {entry.get('samples', 0)} "
                    f"| {_pct(entry.get('attack_success_rate'))} | {_pct(entry.get('robust_accuracy'))} "
                    f"| {_pct(entry.get('detection', {}).get('recall'))} | {_pct(entry.get('defense', {}).get('defense_recovery_rate'))} "
                    f"| {_pct(entry.get('decisions', {}).get('unsafe_acceptance_rate'))} |"
                )
        lines.append("")

    localization = summary.get("localization", {})
    if localization.get("samples"):
        lines.append("## Patch localization (ground-truth masks)")
        lines.append("")
        lines.append(f"- Mean IoU: {_fmt(localization.get('average_iou'))} over {localization['samples']} masked samples")
        lines.append(f"- Localization precision: {_fmt(localization.get('precision'))}, recall: {_fmt(localization.get('recall'))}")
        lines.append("")

    zoo = summary.get("unseen_attack_families", {})
    if zoo:
        lines.append("## Unseen attack families (never used for calibration)")
        lines.append("")
        lines.append("| Family | Samples | Attack success | Detected (TPR) | Recovery |")
        lines.append("|---|---|---|---|---|")
        for family in sorted(zoo):
            entry = zoo[family]
            lines.append(
                f"| {family} | {entry.get('samples', 0)} | {_pct(entry.get('attack_success_rate'))} "
                f"| {_pct(entry.get('detection', {}).get('recall'))} | {_pct(entry.get('defense', {}).get('defense_recovery_rate'))} |"
            )
        lines.append("")
        lines.append("These families were held out from detector calibration on purpose; the rates above are a")
        lines.append("generalization measurement, not an in-distribution result.")
        lines.append("")

    lines.append("## Threat-model limits (read before quoting these numbers)")
    lines.append("")
    lines.append("1. The defended claim covers the attacks evaluated here: FGSM, PGD, and adversarial patches")
    lines.append("   at the tested strengths, plus the listed unseen families. It does not establish general")
    lines.append("   adversarial robustness beyond them.")
    lines.append("2. Adaptive white-box attacks designed against this detector/defense combination (attacker")
    lines.append("   knows the pipeline) are expected to perform better than any attack evaluated here; they")
    lines.append("   are out of scope for this report.")
    lines.append("3. Clean and robust accuracy are reported together throughout: a system that abstains on")
    lines.append("   everything has perfect robust accuracy and zero utility.")
    lines.append("4. The robust specialist covers 10 defended classes; arbitrary uploads are recognized by the")
    lines.append("   frozen ImageNet-1000 tier with the same detection/purification stack.")
    lines.append("5. For a standardized external reference point, compare robust accuracy against published")
    lines.append("   entries at [RobustBench](https://robustbench.github.io/) only where the threat model and")
    lines.append("   class set overlap; Imagenette-derived numbers are not directly comparable to CIFAR/ImageNet benchmarks.")
    lines.append("")
    return "\n".join(lines)


def compact_json(value: Any) -> str:
    return json.dumps(value, default=str)
