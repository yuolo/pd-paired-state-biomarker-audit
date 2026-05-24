"""Deviation models for pathological electrophysiological states.

HCP-derived graphs must not be treated as direct electrophysiological ground
truth for ds004998. In v1, deviations are computed against synthetic
electrophysiology references, dataset-internal baselines, or explicitly
labeled methodological proxies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.normative_model.graph_reference import laplacian_spectrum


FORBIDDEN_EPHYS_REFERENCE_HINTS = ("hcp", "human_connectome_project", "s1200")


@dataclass(frozen=True)
class ReferenceStats:
    """Mean and scale for feature-wise deviation scoring."""

    mean: np.ndarray
    std: np.ndarray
    feature_columns: list[str]
    reference_mode: str


def validate_ephys_reference_mode(reference_mode: str) -> None:
    """Reject use of HCP-like labels as electrophysiological ground truth."""

    mode = reference_mode.lower()
    if any(hint in mode for hint in FORBIDDEN_EPHYS_REFERENCE_HINTS):
        if "connectomic_prior" not in mode and "graph_prior" not in mode:
            raise ValueError(
                "HCP may only be used as a connectomic prior in this prototype. "
                "Use synthetic_electrophysiology_reference, dataset_internal_baseline, "
                "or another explicitly labeled proxy for electrophysiological deviation."
            )


def build_reference_stats(
    reference_features: pd.DataFrame,
    feature_columns: list[str],
    reference_mode: str = "synthetic_electrophysiology_reference",
    min_std: float = 1e-6,
) -> ReferenceStats:
    """Build feature-wise mean/std reference statistics."""

    validate_ephys_reference_mode(reference_mode)
    values = reference_features[feature_columns].to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0)
    ddof = 1 if values.shape[0] > 1 else 0
    std = np.nanstd(values, axis=0, ddof=ddof)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=min_std, posinf=min_std, neginf=min_std)
    std = np.where(std < min_std, min_std, std)
    return ReferenceStats(mean=mean, std=std, feature_columns=feature_columns, reference_mode=reference_mode)


def build_internal_reference_stats(
    all_features: pd.DataFrame,
    reference_features: pd.DataFrame,
    feature_columns: list[str],
    reference_mode: str,
    min_std: float = 0.1,
) -> ReferenceStats:
    """Build internal stats with a dataset-scale fallback for tiny references."""

    reference_stats = build_reference_stats(
        reference_features,
        feature_columns,
        reference_mode=reference_mode,
        min_std=min_std,
    )
    if len(reference_features) >= 2:
        return reference_stats
    scale_stats = build_reference_stats(
        all_features,
        feature_columns,
        reference_mode="dataset_internal_scale_proxy",
        min_std=min_std,
    )
    return ReferenceStats(
        mean=reference_stats.mean,
        std=scale_stats.std,
        feature_columns=feature_columns,
        reference_mode=reference_mode,
    )


def zscore_state(state: np.ndarray, stats: ReferenceStats) -> np.ndarray:
    """Convert a feature state into reference-relative z scores."""

    return (np.asarray(state, dtype=float) - stats.mean) / stats.std


def state_deviation_score(
    state: np.ndarray,
    target_state: np.ndarray,
    scale: np.ndarray | None = None,
    metric: str = "zscore_euclidean",
) -> float:
    """Compute a scalar proxy deviation score between two feature states."""

    state = np.asarray(state, dtype=float)
    target_state = np.asarray(target_state, dtype=float)
    scale = np.ones_like(state) if scale is None else np.asarray(scale, dtype=float)
    scale = np.where(scale == 0, 1.0, scale)
    finite = np.isfinite(state) & np.isfinite(target_state) & np.isfinite(scale)
    if not np.any(finite):
        return float("nan")
    delta = (state[finite] - target_state[finite]) / scale[finite]

    if metric in {"zscore_euclidean", "euclidean"}:
        return float(np.sqrt(np.mean(delta**2)))
    if metric == "manhattan":
        return float(np.mean(np.abs(delta)))
    if metric == "cosine":
        state_finite = state[finite]
        target_finite = target_state[finite]
        denom = np.linalg.norm(state_finite) * np.linalg.norm(target_finite)
        if denom == 0:
            return 0.0 if np.linalg.norm(state_finite - target_finite) == 0 else 1.0
        return float(1.0 - np.dot(state_finite, target_finite) / denom)
    raise ValueError(f"Unsupported deviation metric: {metric}")


def compute_deviation_table(
    features: pd.DataFrame,
    stats: ReferenceStats,
    group_columns: list[str] | None = None,
    metric: str = "zscore_euclidean",
) -> pd.DataFrame:
    """Score each subject/condition row against reference mean."""

    group_columns = group_columns or ["subject", "condition", "medication"]
    rows = []
    for _, row in features.iterrows():
        state = row[stats.feature_columns].to_numpy(dtype=float)
        score = state_deviation_score(state, stats.mean, stats.std, metric=metric)
        record = {column: row[column] for column in group_columns if column in row}
        record.update(
            {
                "deviation_score": score,
                "metric": metric,
                "reference_mode": stats.reference_mode,
                "feature_dim": len(stats.feature_columns),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _normalized_text(series: pd.Series) -> pd.Series:
    """Return lower-case string labels for robust internal matching."""

    return series.fillna("unknown").astype(str).str.lower()


def select_internal_reference_rows(
    features: pd.DataFrame,
    row: pd.Series | None = None,
    strategy: str = "medon_condition_proxy",
) -> tuple[pd.DataFrame, str]:
    """Select dataset-internal electrophysiology reference rows.

    MedOn is treated only as an internal compensated proxy, not a healthy state.
    """

    validate_ephys_reference_mode(strategy)
    if features.empty:
        return features.copy(), "empty_reference"

    medication = _normalized_text(features["medication"]) if "medication" in features else pd.Series("unknown", index=features.index)
    condition = _normalized_text(features["condition"]) if "condition" in features else pd.Series("unknown", index=features.index)
    subject = _normalized_text(features["subject"]) if "subject" in features else pd.Series("unknown", index=features.index)

    row_condition = str(row.get("condition", "unknown")).lower() if row is not None else "unknown"
    row_subject = str(row.get("subject", "unknown")).lower() if row is not None else "unknown"

    medon = medication.eq("on")
    same_condition = condition.eq(row_condition)
    same_subject = subject.eq(row_subject)

    strategy = strategy.lower()
    candidates: list[tuple[pd.Series, str]]
    if strategy == "subject_medon_condition_proxy" and row is not None:
        candidates = [
            (same_subject & same_condition & medon, "subject_medon_same_condition_proxy_not_healthy"),
            (same_subject & medon, "subject_medon_proxy_not_healthy"),
            (same_condition & medon, "medon_same_condition_proxy_not_healthy"),
            (medon, "medon_global_proxy_not_healthy"),
            (same_subject, "subject_internal_proxy"),
        ]
    elif strategy == "condition_baseline_proxy" and row is not None:
        candidates = [
            (same_condition, "same_condition_internal_proxy"),
            (medon, "medon_global_proxy_not_healthy"),
        ]
    else:
        candidates = []
        if row is not None:
            candidates.append((same_condition & medon, "medon_same_condition_proxy_not_healthy"))
        candidates.extend(
            [
                (medon, "medon_global_proxy_not_healthy"),
                (pd.Series(True, index=features.index), "dataset_global_proxy"),
            ]
        )

    for mask, label in candidates:
        subset = features[mask]
        if not subset.empty:
            return subset, label
    return features.copy(), "dataset_global_proxy"


def compute_internal_deviation_table(
    features: pd.DataFrame,
    feature_columns: list[str],
    strategy: str = "medon_condition_proxy",
    metric: str = "zscore_euclidean",
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute deviation using ds004998-internal references only."""

    validate_ephys_reference_mode(strategy)
    group_columns = group_columns or ["subject", "session", "condition", "medication", "task", "run"]
    rows = []
    for _, row in features.iterrows():
        reference, reference_label = select_internal_reference_rows(features, row, strategy)
        stats = build_internal_reference_stats(features, reference, feature_columns, reference_label)
        state = row[feature_columns].to_numpy(dtype=float)
        score = state_deviation_score(state, stats.mean, stats.std, metric=metric)
        record = {column: row[column] for column in group_columns if column in row}
        record.update(
            {
                "deviation_score": score,
                "metric": metric,
                "reference_strategy": strategy,
                "reference_mode": reference_label,
                "reference_n": int(len(reference)),
                "feature_dim": int(len(feature_columns)),
                "methodological_note": "dataset-internal electrophysiology proxy; HCP not used as ephys reference",
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def graph_spectral_deviation(
    graph_a: np.ndarray,
    graph_b: np.ndarray,
    k: int = 5,
) -> float:
    """Compare two graph priors using low-order Laplacian spectra."""

    spec_a = laplacian_spectrum(graph_a, k=k)
    spec_b = laplacian_spectrum(graph_b, k=k)
    length = min(len(spec_a), len(spec_b))
    if length == 0:
        return 0.0
    return float(np.sqrt(np.mean((spec_a[:length] - spec_b[:length]) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny deviation scoring demo.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    feature_columns = ["M1_beta_power", "STN_beta_power", "M1_stn_beta_coupling"]
    reference = pd.DataFrame(rng.normal(size=(20, 3)), columns=feature_columns)
    features = pd.DataFrame(
        [["demo_subject", "rest", "off", 1.5, 1.2, 1.0]],
        columns=["subject", "condition", "medication"] + feature_columns,
    )
    stats = build_reference_stats(reference, feature_columns)
    print(compute_deviation_table(features, stats))


if __name__ == "__main__":
    main()
