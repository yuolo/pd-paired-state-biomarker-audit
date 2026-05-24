"""Abstract penalties for candidate compensatory targets."""

from __future__ import annotations


HIGH_SENSITIVITY_NODES = {"STN", "GPi", "GPe", "Thalamus"}


def risk_penalty_proxy(node: str, target_type: str) -> float:
    """Return an abstract risk penalty for optimization demos.

    This is not a clinical safety estimate. It is a conservative proxy that
    discourages fragile or highly central targets in the toy optimization.
    """

    penalty = 0.25
    if node in HIGH_SENSITIVITY_NODES:
        penalty += 0.35
    if target_type == "edge_coupling_modulation":
        penalty += 0.10
    return penalty


def complexity_penalty_proxy(target_type: str) -> float:
    """Return an abstract implementation complexity penalty."""

    if target_type == "edge_coupling_modulation":
        return 0.45
    return 0.25
