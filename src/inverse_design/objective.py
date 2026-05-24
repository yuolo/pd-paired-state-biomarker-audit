"""Objective functions for in silico compensatory target generation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.inverse_design.constraints import (
    complexity_penalty_proxy,
    risk_penalty_proxy,
)
from src.pathology_model.state_deviation import state_deviation_score


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights for the inverse-design objective."""

    lambda_energy: float = 0.15
    lambda_risk: float = 0.25
    lambda_complexity: float = 0.10
    lambda_stability: float = 0.20


def compensatory_objective(
    deviation_from_target_state: float,
    intervention_energy: float,
    risk_penalty: float,
    circuit_complexity: float,
    stability_score: float,
    weights: ObjectiveWeights,
) -> float:
    """Compute the scalar objective to minimize."""

    return float(
        deviation_from_target_state
        + weights.lambda_energy * intervention_energy
        + weights.lambda_risk * risk_penalty
        + weights.lambda_complexity * circuit_complexity
        - weights.lambda_stability * stability_score
    )


def parse_candidate_feature(feature_name: str) -> dict[str, str]:
    """Parse a feature name into an abstract target descriptor."""

    if "_stn_" in feature_name and feature_name.endswith("_coupling"):
        node, rest = feature_name.split("_stn_", maxsplit=1)
        band = rest.removesuffix("_coupling")
        return {
            "target_type": "edge_coupling_modulation",
            "node": node,
            "edge": f"{node}-STN",
            "band": band,
            "feature": feature_name,
        }

    parts = feature_name.split("_")
    if len(parts) >= 3 and parts[-1] == "power":
        return {
            "target_type": "node_band_modulation",
            "node": parts[0],
            "edge": "",
            "band": "_".join(parts[1:-1]),
            "feature": feature_name,
        }

    return {
        "target_type": "feature_modulation",
        "node": feature_name.split("_")[0],
        "edge": "",
        "band": "unknown",
        "feature": feature_name,
    }


def candidate_effect_vector(
    current_state: np.ndarray,
    target_state: np.ndarray,
    feature_index: int,
    effect_fraction: float,
) -> np.ndarray:
    """Create a one-feature abstract modulation vector toward the target."""

    effect = np.zeros_like(current_state, dtype=float)
    effect[feature_index] = effect_fraction * (target_state[feature_index] - current_state[feature_index])
    return effect


def stability_proxy(effect: np.ndarray) -> float:
    """Estimate whether a candidate makes a small, stable state update."""

    magnitude = float(np.linalg.norm(effect))
    return float(1.0 / (1.0 + magnitude))


def uncertainty_proxy(effect: np.ndarray, risk_penalty: float, floor: float = 0.05) -> float:
    """Return an intentionally conservative uncertainty proxy."""

    return float(floor + 0.05 * np.linalg.norm(effect) + 0.1 * risk_penalty)


def rank_candidate_targets(
    current_state: np.ndarray,
    target_state: np.ndarray,
    feature_columns: list[str],
    reference_scale: np.ndarray | None = None,
    weights: ObjectiveWeights | None = None,
    effect_fraction: float = 0.6,
    uncertainty_floor: float = 0.05,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Rank candidate nodes, edges, and bands by objective value."""

    current_state = np.asarray(current_state, dtype=float)
    target_state = np.asarray(target_state, dtype=float)
    reference_scale = np.ones_like(current_state) if reference_scale is None else reference_scale
    weights = weights or ObjectiveWeights()

    baseline_deviation = state_deviation_score(
        current_state,
        target_state,
        reference_scale,
        metric="zscore_euclidean",
    )
    rows = []

    for idx, feature in enumerate(feature_columns):
        descriptor = parse_candidate_feature(feature)
        effect = candidate_effect_vector(current_state, target_state, idx, effect_fraction)
        post_state = current_state + effect
        post_deviation = state_deviation_score(
            post_state,
            target_state,
            reference_scale,
            metric="zscore_euclidean",
        )
        energy = float(np.sum(effect**2))
        risk = risk_penalty_proxy(descriptor["node"], descriptor["target_type"])
        complexity = complexity_penalty_proxy(descriptor["target_type"])
        stability = stability_proxy(effect)
        uncertainty = uncertainty_proxy(effect, risk, floor=uncertainty_floor)
        objective = compensatory_objective(
            post_deviation,
            energy,
            risk,
            complexity,
            stability,
            weights,
        )
        rows.append(
            {
                **descriptor,
                "feature_index": idx,
                "objective": objective,
                "baseline_deviation": baseline_deviation,
                "post_deviation": post_deviation,
                "expected_delta_deviation": baseline_deviation - post_deviation,
                "intervention_energy": energy,
                "risk_penalty": risk,
                "complexity_penalty": complexity,
                "stability_estimate": stability,
                "uncertainty": uncertainty,
                "effect_size": float(effect[idx]),
            }
        )

    ranked = pd.DataFrame(rows).sort_values(
        ["objective", "expected_delta_deviation"],
        ascending=[True, False],
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    if top_k is not None:
        ranked = ranked.head(top_k)
    return ranked.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny inverse-design demo.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    feature_columns = ["M1_beta_power", "STN_beta_power", "M1_stn_beta_coupling"]
    current = np.array([1.2, 1.4, 1.1])
    target = np.zeros(3)
    ranked = rank_candidate_targets(current, target, feature_columns, top_k=args.top_k)
    print(ranked)


if __name__ == "__main__":
    main()
