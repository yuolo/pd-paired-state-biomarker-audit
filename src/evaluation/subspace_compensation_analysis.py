"""Restricted-subspace compensation analyses for ds004998 outputs.

This module evaluates whether MedOff-to-MedOn direction modeling improves when
restricted to reliability-supported and biologically motivated feature
subspaces. MedOn is used only as a dataset-internal compensated proxy; HCP is
not used as an electrophysiological reference.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, MultiTaskElasticNet, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings as py_warnings

from src.evaluation.clean_subset_analysis import (
    ANALYSIS_SUBSETS,
    attach_quality,
    complete_medoff_medon_pairs,
    pair_count,
    read_quality_table,
    read_state_vectors,
    rank_subset_candidates,
    subset_state_vectors,
    target_group_flags,
)
from src.evaluation.side_aware_analysis import SIDE_AWARE_STRATA, parse_task_side
from src.inverse_design.objective import ObjectiveWeights, parse_candidate_feature
from src.pathology_model.build_real_state_vectors import (
    BASE_REAL_STATE_FEATURE_COLUMNS,
    REAL_STATE_FEATURE_COLUMNS,
)
from src.pathology_model.state_deviation import (
    compute_internal_deviation_table,
    state_deviation_score,
)
from src.utils.io_utils import write_table


ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
SUBSPACE_QUALITY_SUBSETS = list(ANALYSIS_SUBSETS)
PREDICTOR_MODELS = [
    "no_action",
    "random_direction",
    "group_mean_direction",
    "task_family_group_mean_direction",
    "side_group_mean_direction",
    "reliability_weighted_direction",
    "ridge_regression_predictor",
    "elastic_net_predictor",
    "pls_regression_predictor",
    "bayesian_ridge_predictor",
    "kernel_ridge_predictor",
    "gaussian_process_predictor",
]


@dataclass(frozen=True)
class SubspaceRunConfig:
    """Runtime configuration for subspace analysis."""

    weights: ObjectiveWeights
    effect_fraction: float = 0.6
    uncertainty_floor: float = 0.05
    seed: int = 42
    top_k: int = 5
    lambda_magnitude: float = 0.01
    lambda_complexity: float = 0.005
    lambda_instability: float = 0.05
    n_bootstrap: int = 0


def _read_optional_csv(path: str | Path, warnings: list[str], dtype: dict[str, str] | None = None) -> pd.DataFrame:
    """Read an optional CSV table, recording a warning if it is unavailable."""

    path = Path(path)
    if not path.exists():
        warnings.append(f"Missing optional input: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype)
    except Exception as exc:  # noqa: BLE001 - report and continue with available inputs.
        warnings.append(f"Could not read {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def _numeric_features(data: pd.DataFrame) -> list[str]:
    """Return numeric feature-like columns in state-vector tables."""

    metadata = {
        "subject",
        "session",
        "condition",
        "medication",
        "task",
        "run",
        "task_original",
        "task_family",
        "side",
        "n_windows",
        "source",
        "methodological_label",
        "quality_flag",
        "quality_source",
        "preprocessing_stage_omitted_percent",
    }
    return [
        column
        for column in data.columns
        if column not in metadata and pd.api.types.is_numeric_dtype(data[column])
    ]


def _feature_meta(feature: str) -> dict[str, str]:
    """Parse a feature into node, band, and coarse feature type."""

    desc = parse_candidate_feature(feature)
    lower = feature.lower()
    if "coupling" in lower:
        feature_type = "cortico_stn_coupling_proxy"
        node = "motor-stn" if "motor_stn" in lower else desc.get("node", "")
    elif "index" in lower:
        feature_type = "derived_network_index"
        node = "cortico-stn"
    elif lower.endswith("_power"):
        feature_type = "band_power"
        node = desc.get("node", "")
    else:
        feature_type = desc.get("target_type", "feature")
        node = desc.get("node", "")

    band = desc.get("band", "unknown")
    if band == "unknown":
        for candidate in ["low_beta", "high_beta", "broad_beta", "gamma", "alpha", "beta"]:
            if candidate in lower:
                band = candidate
                break
    return {
        "node": node,
        "band": band,
        "target_type": desc.get("target_type", "feature_modulation"),
        "feature_type": feature_type,
    }


def _direction_label(delta: float, epsilon: float = 1e-8) -> str:
    """Map a signed delta to increase, decrease, or unchanged."""

    if not np.isfinite(delta) or abs(delta) <= epsilon:
        return "unchanged"
    return "increase" if delta > 0 else "decrease"


def _safe_scale(values: np.ndarray, min_scale: float = 0.1) -> np.ndarray:
    """Compute a robust feature scale for deviation metrics."""

    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    ddof = 1 if values.shape[0] > 1 else 0
    scale = np.nanstd(values, axis=0, ddof=ddof)
    scale = np.nan_to_num(scale, nan=min_scale, posinf=min_scale, neginf=min_scale)
    return np.where(scale < min_scale, min_scale, scale)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with a stable zero-vector convention."""

    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return float("nan")
    a = a[finite]
    b = b[finite]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _append_warning(warnings: list[str], message: str) -> None:
    """Append warning while avoiding duplicates."""

    if message not in warnings:
        warnings.append(message)


def enhance_state_vectors(
    state_vectors: pd.DataFrame,
    window_features_path: str | Path,
    warnings: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add lightweight derived coupling features available from current outputs."""

    data = state_vectors.copy()
    inventory_rows: list[dict[str, object]] = []
    original_features = [feature for feature in REAL_STATE_FEATURE_COLUMNS if feature in data]

    for feature in original_features:
        meta = _feature_meta(feature)
        inventory_rows.append(
            {
                "feature": feature,
                "available": True,
                "source": "ds004998_state_vectors",
                **meta,
                "derivation": "original_state_feature",
                "note": "Available in current state-vector table.",
            }
        )

    beta_coupling = [
        "motor_stn_low_beta_coupling",
        "motor_stn_high_beta_coupling",
        "motor_stn_broad_beta_coupling",
    ]
    if "cortico_stn_coupling_index" in data:
        pass
    elif all(feature in data for feature in beta_coupling):
        data["cortico_stn_coupling_index"] = data[beta_coupling].mean(axis=1)
        inventory_rows.append(
            {
                "feature": "cortico_stn_coupling_index",
                "available": True,
                "source": "derived_from_state_vectors",
                "node": "cortico-stn",
                "band": "beta",
                "target_type": "network_index_modulation",
                "feature_type": "derived_network_index",
                "derivation": "mean of available motor-STN low/high/broad beta coupling proxies",
                "note": "Derived without raw-signal recomputation.",
            }
        )
    else:
        _append_warning(warnings, "Could not derive cortico_stn_coupling_index because beta coupling columns were incomplete.")

    window_features = _read_optional_csv(window_features_path, warnings, dtype={"run": str})
    available_window_coupling = (
        [column for column in window_features.columns if "coupling" in column.lower()]
        if not window_features.empty
        else []
    )
    requested_extra = [
        "motor_stn_alpha_coupling",
        "motor_stn_gamma_coupling",
        "motor_stn_left_broad_beta_coupling",
        "motor_stn_right_broad_beta_coupling",
        "beta_coupling_asymmetry",
        "gamma_coupling_index",
    ]
    for feature in requested_extra:
        if feature in data:
            continue
        inventory_rows.append(
            {
                "feature": feature,
                "available": False,
                "source": "not_available_in_current_outputs",
                **_feature_meta(feature),
                "derivation": "requires richer window features or signal-level recomputation",
                "note": (
                    "Current window table exposes coupling columns "
                    f"{';'.join(available_window_coupling) if available_window_coupling else 'none'}."
                ),
            }
        )
    if "motor_stn_gamma_coupling" not in data:
        _append_warning(
            warnings,
            "Gamma and side-specific cortico-STN coupling cannot be recovered from current tables without richer extraction.",
        )
    return data, pd.DataFrame(inventory_rows)


def _feature_list(data: pd.DataFrame, features: list[str]) -> list[str]:
    """Keep features that are present and numeric."""

    return [
        feature
        for feature in features
        if feature in data and pd.api.types.is_numeric_dtype(data[feature])
    ]


def _top_reliability_features(table: pd.DataFrame, data: pd.DataFrame, score_columns: list[str], top_k: int) -> list[str]:
    """Select top reliability-ranked features from a table."""

    if table.empty or "feature" not in table:
        return []
    ranked = table.copy()
    for column in score_columns:
        if column in ranked:
            ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    sort_columns = [column for column in score_columns if column in ranked]
    if sort_columns:
        ranked = ranked.sort_values(sort_columns, ascending=[False] + [True] * (len(sort_columns) - 1))
    return _feature_list(data, list(dict.fromkeys(ranked["feature"].dropna().astype(str))))[:top_k]


def _clean_stable_features(clean_candidates: pd.DataFrame, data: pd.DataFrame) -> list[str]:
    """Features appearing in top 10 across at least two clean-subset contexts."""

    if clean_candidates.empty or "feature" not in clean_candidates or "analysis_subset" not in clean_candidates:
        return []
    table = clean_candidates.copy()
    table["rank"] = pd.to_numeric(table.get("rank", np.nan), errors="coerce")
    top = table[table["rank"] <= 10]
    counts = top.groupby("feature")["analysis_subset"].nunique().sort_values(ascending=False)
    return _feature_list(data, list(counts[counts >= 2].index))


def build_subspace_definitions(
    enhanced: pd.DataFrame,
    reliability: pd.DataFrame,
    reliability_sideaware: pd.DataFrame,
    clean_candidates: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """Create biologically motivated and reliability-supported subspaces."""

    specs: dict[str, tuple[list[str], str]] = {
        "full_28": (
            [feature for feature in BASE_REAL_STATE_FEATURE_COLUMNS if feature in enhanced],
            "All original 28 state features; retained as the noisy full-vector comparator.",
        ),
        "motor_beta": (
            ["motor_low_beta_power", "motor_high_beta_power", "motor_broad_beta_power"],
            "Sensor-level motor beta proxy features.",
        ),
        "motor_gamma": (["motor_gamma_power"], "Sensor-level motor gamma proxy feature."),
        "motor_beta_gamma": (
            ["motor_low_beta_power", "motor_high_beta_power", "motor_broad_beta_power", "motor_gamma_power"],
            "Motor beta features plus motor gamma proxy.",
        ),
        "stn_beta": (
            [
                "stn_low_beta_power",
                "stn_high_beta_power",
                "stn_broad_beta_power",
                "stn_left_low_beta_power",
                "stn_left_high_beta_power",
                "stn_left_broad_beta_power",
                "stn_right_low_beta_power",
                "stn_right_high_beta_power",
                "stn_right_broad_beta_power",
            ],
            "STN beta power proxies across pooled, left, and right LFP contacts.",
        ),
        "stn_gamma": (
            ["stn_gamma_power", "stn_left_gamma_power", "stn_right_gamma_power"],
            "STN gamma power proxies; interpreted cautiously because prior runs showed subject sensitivity.",
        ),
        "stn_beta_gamma": (
            [
                "stn_low_beta_power",
                "stn_high_beta_power",
                "stn_broad_beta_power",
                "stn_left_low_beta_power",
                "stn_left_high_beta_power",
                "stn_left_broad_beta_power",
                "stn_right_low_beta_power",
                "stn_right_high_beta_power",
                "stn_right_broad_beta_power",
                "stn_gamma_power",
                "stn_left_gamma_power",
                "stn_right_gamma_power",
            ],
            "Combined STN beta and gamma subspace.",
        ),
        "cortical_alpha_beta": (
            [
                "meg_alpha_power",
                "motor_alpha_power",
                "meg_low_beta_power",
                "meg_high_beta_power",
                "meg_broad_beta_power",
                "motor_low_beta_power",
                "motor_high_beta_power",
                "motor_broad_beta_power",
            ],
            "Cortical sensor-level alpha and beta proxies.",
        ),
        "cortico_stn_coupling": (
            [column for column in enhanced.columns if "coupling" in column.lower()] + ["cortico_stn_coupling_index"],
            "Available cortico-STN coupling proxies, including multi-band and side-aware coherence when extracted.",
        ),
        "reliability_top_k": (
            _top_reliability_features(
                reliability,
                enhanced,
                ["reliability_score", "top10_frequency", "mean_rank"],
                top_k,
            ),
            f"Top {top_k} features from the current target reliability table.",
        ),
        "sideaware_reliability_top_k": (
            _top_reliability_features(
                reliability_sideaware,
                enhanced,
                ["sideaware_reliability_score", "reliability_score", "mean_rank"],
                top_k,
            ),
            f"Top {top_k} features from side-aware reliability scoring.",
        ),
        "clean_stable_features": (
            _clean_stable_features(clean_candidates, enhanced),
            "Features that survive clean-subset filters and appear in top 10 across at least two quality subsets.",
        ),
        "literature_informed_beta_network": (
            [
                "motor_low_beta_power",
                "motor_high_beta_power",
                "motor_broad_beta_power",
                "stn_low_beta_power",
                "stn_high_beta_power",
                "stn_broad_beta_power",
                "stn_left_low_beta_power",
                "stn_left_high_beta_power",
                "stn_left_broad_beta_power",
                "stn_right_low_beta_power",
                "stn_right_high_beta_power",
                "stn_right_broad_beta_power",
                "motor_stn_low_beta_coupling",
                "motor_stn_high_beta_coupling",
                "motor_stn_broad_beta_coupling",
                "motor_stn_left_low_beta_coupling",
                "motor_stn_left_high_beta_coupling",
                "motor_stn_left_broad_beta_coupling",
                "motor_stn_right_low_beta_coupling",
                "motor_stn_right_high_beta_coupling",
                "motor_stn_right_broad_beta_coupling",
                "motor_stn_task_side_low_beta_coupling",
                "motor_stn_task_side_high_beta_coupling",
                "motor_stn_task_side_broad_beta_coupling",
                "motor_stn_opposite_task_side_low_beta_coupling",
                "motor_stn_opposite_task_side_high_beta_coupling",
                "motor_stn_opposite_task_side_broad_beta_coupling",
                "beta_coupling_index",
                "beta_coupling_asymmetry",
                "cortico_stn_coupling_index",
            ],
            "Motor beta, STN beta, and available motor-STN beta coupling proxies.",
        ),
        "literature_informed_fast_network": (
            [
                "stn_gamma_power",
                "stn_left_gamma_power",
                "stn_right_gamma_power",
                "motor_gamma_power",
                "motor_stn_gamma_coupling",
                "motor_stn_left_gamma_coupling",
                "motor_stn_right_gamma_coupling",
                "motor_stn_task_side_gamma_coupling",
                "motor_stn_opposite_task_side_gamma_coupling",
                "gamma_coupling_index",
                "gamma_coupling_asymmetry",
            ],
            "Fast-band motor/STN features plus gamma coupling if available.",
        ),
    }

    rows: list[dict[str, object]] = []
    for subspace_name, (features, rationale) in specs.items():
        present = _feature_list(enhanced, list(dict.fromkeys(features)))
        if subspace_name == "cortico_stn_coupling" and present:
            has_gamma = any("gamma" in feature.lower() for feature in present)
            has_alpha = any("alpha" in feature.lower() for feature in present)
            rationale += " Label: multi_band_coupling." if has_gamma or has_alpha else " Label: beta_coupling_only."
        for feature in present:
            meta = _feature_meta(feature)
            rows.append(
                {
                    "subspace_name": subspace_name,
                    "feature": feature,
                    **meta,
                    "rationale": rationale,
                    "n_features_in_subspace": len(present),
                }
            )
    return pd.DataFrame(rows)


def subspace_map(definitions: pd.DataFrame) -> dict[str, list[str]]:
    """Convert subspace definitions into a mapping."""

    if definitions.empty:
        return {}
    return {
        str(name): list(dict.fromkeys(group["feature"].astype(str)))
        for name, group in definitions.groupby("subspace_name", sort=False)
    }


def build_pairs(state_vectors: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Build complete subject/task MedOff-MedOn pairs for a feature subspace."""

    rows: list[dict[str, object]] = []
    if state_vectors.empty or not feature_columns:
        return pd.DataFrame()
    for (subject, task_original), subset in state_vectors.groupby(["subject", "task_original"], dropna=False):
        medication = subset["medication"].fillna("").astype(str).str.lower()
        off = subset[medication.eq("off")]
        on = subset[medication.eq("on")]
        if off.empty or on.empty:
            continue
        off_vec = off[feature_columns].mean(numeric_only=True)
        on_vec = on[feature_columns].mean(numeric_only=True)
        first = subset.iloc[0]
        row: dict[str, object] = {
            "subject": str(subject),
            "session": str(first.get("session", "unknown")),
            "task_original": str(task_original),
            "task_family": str(first.get("task_family", "unknown")),
            "side": str(first.get("side", "unknown")),
            "run_off": ";".join(off.get("run", pd.Series("unknown", index=off.index)).dropna().astype(str).unique()),
            "run_on": ";".join(on.get("run", pd.Series("unknown", index=on.index)).dropna().astype(str).unique()),
            "quality_flag": str(first.get("quality_flag", "unknown")),
        }
        for feature in feature_columns:
            row[f"x_{feature}"] = float(off_vec[feature])
            row[f"y_{feature}"] = float(on_vec[feature] - off_vec[feature])
            row[f"medon_{feature}"] = float(on_vec[feature])
        rows.append(row)
    return pd.DataFrame(rows)


def _x_matrix(pairs: pd.DataFrame, features: list[str]) -> np.ndarray:
    return pairs[[f"x_{feature}" for feature in features]].to_numpy(dtype=float)


def _y_matrix(pairs: pd.DataFrame, features: list[str]) -> np.ndarray:
    return pairs[[f"y_{feature}" for feature in features]].to_numpy(dtype=float)


def _medon_matrix(pairs: pd.DataFrame, features: list[str]) -> np.ndarray:
    return pairs[[f"medon_{feature}" for feature in features]].to_numpy(dtype=float)


def _direction_consistency(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Feature-wise sign consistency."""

    if values.size == 0:
        return np.zeros(0)
    signs = np.where(np.abs(values) <= epsilon, 0, np.sign(values))
    scores = []
    for idx in range(signs.shape[1]):
        _, counts = np.unique(signs[:, idx], return_counts=True)
        scores.append(float(counts.max() / max(1, signs.shape[0])) if len(counts) else 0.0)
    return np.asarray(scores, dtype=float)


def compute_subspace_direction_consistency(
    subset_name: str,
    subspace_name: str,
    subset: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute direction consistency and subspace-level quality summary."""

    pairs = build_pairs(subset, features)
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame(
            [
                {
                    "quality_subset": subset_name,
                    "subspace_name": subspace_name,
                    "n_features": len(features),
                    "n_subjects": 0,
                    "n_state_vectors": int(len(subset)),
                    "n_paired_examples": 0,
                    "mean_baseline_deviation": np.nan,
                    "mean_direction_norm": np.nan,
                    "mean_direction_consistency": np.nan,
                    "interpretation_note": "No complete MedOff-MedOn pairs available.",
                }
            ]
        )
    x = _x_matrix(pairs, features)
    y = _y_matrix(pairs, features)
    medon = _medon_matrix(pairs, features)
    scale = _safe_scale(np.vstack([x, medon]))
    consistency = _direction_consistency(y)
    rows = []
    for idx, feature in enumerate(features):
        deltas = y[:, idx]
        signs = [_direction_label(value) for value in deltas]
        counts = pd.Series(signs).value_counts().to_dict()
        dominant = max(["increase", "decrease", "unchanged"], key=lambda label: counts.get(label, 0))
        meta = _feature_meta(feature)
        rows.append(
            {
                "quality_subset": subset_name,
                "subspace_name": subspace_name,
                "feature": feature,
                **meta,
                "n_subjects": int(pairs["subject"].nunique()),
                "n_paired_examples": int(len(pairs)),
                "group_mean_signed_delta": float(np.nanmean(deltas)),
                "group_median_signed_delta": float(np.nanmedian(deltas)),
                "group_std_signed_delta": float(np.nanstd(deltas)),
                "group_direction": _direction_label(float(np.nanmean(deltas))),
                "increase_count": int(counts.get("increase", 0)),
                "decrease_count": int(counts.get("decrease", 0)),
                "unchanged_count": int(counts.get("unchanged", 0)),
                "consistency_direction": dominant,
                "consistency_fraction": float(consistency[idx]),
            }
        )
    baseline = [
        state_deviation_score(x[idx], medon[idx], scale)
        for idx in range(len(pairs))
    ]
    summary = pd.DataFrame(
        [
            {
                "quality_subset": subset_name,
                "subspace_name": subspace_name,
                "n_features": len(features),
                "n_subjects": int(pairs["subject"].nunique()),
                "n_state_vectors": int(len(subset)),
                "n_paired_examples": int(len(pairs)),
                "mean_baseline_deviation": float(np.nanmean(baseline)),
                "mean_direction_norm": float(np.nanmean(np.linalg.norm(y, axis=1))),
                "mean_direction_consistency": float(np.nanmean(consistency)),
                "interpretation_note": "Complete pairs only; MedOn is a dataset-internal compensated proxy.",
            }
        ]
    )
    return pd.DataFrame(rows), summary


def compute_subspace_deviation(
    subset_name: str,
    subspace_name: str,
    subset: pd.DataFrame,
    features: list[str],
    reference_strategy: str,
) -> pd.DataFrame:
    """Compute dataset-internal deviation table for a subspace."""

    if subset.empty or not features:
        return pd.DataFrame()
    group_columns = [
        "subject",
        "session",
        "condition",
        "medication",
        "task",
        "run",
        "task_original",
        "task_family",
        "side",
        "quality_flag",
    ]
    deviation = compute_internal_deviation_table(
        subset,
        features,
        strategy=reference_strategy,
        group_columns=[column for column in group_columns if column in subset],
    )
    if deviation.empty:
        return deviation
    deviation.insert(0, "subspace_name", subspace_name)
    deviation.insert(0, "quality_subset", subset_name)
    deviation["n_features"] = len(features)
    return deviation


def rank_subspace_candidates(
    subset_name: str,
    subspace_name: str,
    subset: pd.DataFrame,
    features: list[str],
    config: SubspaceRunConfig,
) -> pd.DataFrame:
    """Rank candidate targets inside one subspace."""

    ranked = rank_subset_candidates(
        subset,
        features,
        subset_name,
        config.weights,
        config.effect_fraction,
        config.uncertainty_floor,
    )
    if ranked.empty:
        return ranked
    ranked = target_group_flags(ranked)
    if "analysis_subset" in ranked and "quality_subset" not in ranked:
        ranked = ranked.rename(columns={"analysis_subset": "quality_subset"})
    ranked.insert(0, "subspace_name", subspace_name)
    ranked["n_features_in_subspace"] = len(features)
    return ranked


def summarize_side_strata(
    subset_name: str,
    subspace_name: str,
    subset: pd.DataFrame,
    features: list[str],
    config: SubspaceRunConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compute side/task-family candidate summaries for a subspace."""

    rows: list[dict[str, object]] = []
    tables: dict[str, pd.DataFrame] = {}
    for stratum_name, (_, column, value) in SIDE_AWARE_STRATA.items():
        stratum = subset[subset[column].astype(str).eq(value)].copy() if column in subset else subset.iloc[0:0].copy()
        targets = rank_subspace_candidates(subset_name, subspace_name, stratum, features, config)
        if not targets.empty:
            targets["stratum"] = stratum_name
            tables[stratum_name] = targets
        else:
            tables[stratum_name] = pd.DataFrame()
        top10 = targets[pd.to_numeric(targets.get("rank", np.nan), errors="coerce") <= 10] if not targets.empty else pd.DataFrame()
        task_original, task_family, side = parse_task_side(value)
        if stratum_name == "left_side":
            task_original, task_family, side = "pooled", "pooled", "L"
        elif stratum_name == "right_side":
            task_original, task_family, side = "pooled", "pooled", "R"
        elif stratum_name == "hold_family":
            task_original, task_family, side = "pooled", "Hold", "pooled"
        elif stratum_name == "move_family":
            task_original, task_family, side = "pooled", "Move", "pooled"
        rows.append(
            {
                "quality_subset": subset_name,
                "subspace_name": subspace_name,
                "stratum": stratum_name,
                "task_original": task_original,
                "task_family": task_family,
                "side": side,
                "n_subjects": int(stratum["subject"].nunique()) if "subject" in stratum else 0,
                "n_state_vectors": int(len(stratum)),
                "n_features": len(features),
                "top1_target": str(targets.sort_values("rank").iloc[0]["feature"]) if not targets.empty else "",
                "top10_targets": ";".join(top10["feature"].astype(str)) if not top10.empty else "",
                "top10_motor_beta_count": int(top10.get("is_motor_beta_target", pd.Series(dtype=bool)).sum()) if not top10.empty else 0,
                "top10_STN_count": int(top10.get("is_stn_target", pd.Series(dtype=bool)).sum()) if not top10.empty else 0,
                "top10_beta_count": int(top10.get("is_beta_target", pd.Series(dtype=bool)).sum()) if not top10.empty else 0,
                "top10_gamma_count": int(top10.get("is_gamma_target", pd.Series(dtype=bool)).sum()) if not top10.empty else 0,
                "top10_alpha_count": int(top10.get("is_alpha_target", pd.Series(dtype=bool)).sum()) if not top10.empty else 0,
                "top10_coupling_count": int(top10["feature"].astype(str).str.contains("coupling|cortico_stn").sum()) if not top10.empty else 0,
                "best_expected_delta_deviation": float(targets["expected_delta_deviation"].max()) if not targets.empty else np.nan,
                "mean_uncertainty": float(top10["uncertainty"].mean()) if not top10.empty else np.nan,
                "interpretation_note": "side/task-family stratum; dataset-internal proxy comparison",
            }
        )
    return pd.DataFrame(rows), tables


def compare_rank_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
    output_kind: str,
) -> pd.DataFrame:
    """Compare two candidate rank tables feature-by-feature."""

    features = sorted(set(left.get("feature", pd.Series(dtype=str)).astype(str)) | set(right.get("feature", pd.Series(dtype=str)).astype(str)))
    rows = []
    left_ranks = left.set_index("feature")["rank"].to_dict() if not left.empty and "feature" in left else {}
    right_ranks = right.set_index("feature")["rank"].to_dict() if not right.empty and "feature" in right else {}
    for feature in features:
        rank_left = float(left_ranks.get(feature, np.nan))
        rank_right = float(right_ranks.get(feature, np.nan))
        appears_left = bool(np.isfinite(rank_left) and rank_left <= 10)
        appears_right = bool(np.isfinite(rank_right) and rank_right <= 10)
        if appears_left and appears_right:
            note = "shared_high_priority"
        elif appears_left:
            note = f"{left_name}_only_top10"
        elif appears_right:
            note = f"{right_name}_only_top10"
        elif np.isfinite(rank_left) and np.isfinite(rank_right):
            note = f"{left_name}_dominant" if rank_left + 5 < rank_right else f"{right_name}_dominant" if rank_right + 5 < rank_left else "unstable_or_low_priority"
        else:
            note = "unstable_or_low_priority"
        meta = _feature_meta(feature)
        rows.append(
            {
                "feature": feature,
                **meta,
                f"rank_{left_name}": rank_left,
                f"rank_{right_name}": rank_right,
                "rank_difference": rank_left - rank_right if np.isfinite(rank_left) and np.isfinite(rank_right) else np.nan,
                f"appears_{left_name}_top10": appears_left,
                f"appears_{right_name}_top10": appears_right,
                "comparison_type": output_kind,
                "specificity_note": note,
            }
        )
    return pd.DataFrame(rows)


def _reliability_scores(reliability: pd.DataFrame, reliability_sideaware: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Build feature weights from reliability tables."""

    scores = pd.Series(1.0, index=features, dtype=float)
    tables = []
    if not reliability.empty and "feature" in reliability:
        table = reliability.copy()
        if "reliability_score" in table:
            table["_score"] = pd.to_numeric(table["reliability_score"], errors="coerce")
            tables.append(table[["feature", "_score"]])
    if not reliability_sideaware.empty and "feature" in reliability_sideaware:
        table = reliability_sideaware.copy()
        score_column = "sideaware_reliability_score" if "sideaware_reliability_score" in table else "reliability_score"
        if score_column in table:
            table["_score"] = pd.to_numeric(table[score_column], errors="coerce")
            tables.append(table[["feature", "_score"]])
    if tables:
        combined = pd.concat(tables, ignore_index=True).dropna()
        means = combined.groupby("feature")["_score"].mean()
        for feature in features:
            if feature in means and np.isfinite(means[feature]):
                scores.loc[feature] = float(means[feature])
    maximum = float(scores.max()) if scores.max() > 0 else 1.0
    return (scores / maximum).clip(0.05, 1.0).to_numpy(dtype=float)


def _best_features_from_candidates(candidates: pd.DataFrame, features: list[str], top_k: int) -> list[str]:
    """Select best features from candidate ranking."""

    if candidates.empty or "feature" not in candidates:
        return features[:top_k]
    table = candidates[candidates["feature"].isin(features)].copy()
    if table.empty:
        return features[:top_k]
    table["rank"] = pd.to_numeric(table["rank"], errors="coerce")
    return list(dict.fromkeys(table.sort_values("rank")["feature"].astype(str)))[:top_k]


def _compensation_direction(
    strategy: str,
    pair: pd.Series,
    pairs: pd.DataFrame,
    features: list[str],
    reliability_weights: np.ndarray,
    candidate_features: list[str],
    rng: np.random.Generator,
    top_k: int,
) -> tuple[np.ndarray, str]:
    """Return one strategy direction for subspace compensation simulation."""

    y = _y_matrix(pairs, features)
    group_mean = np.nanmean(y, axis=0) if len(y) else np.zeros(len(features))
    if strategy == "no_action":
        return np.zeros(len(features)), "zero_vector"
    if strategy == "random_subspace_direction":
        random = rng.normal(size=len(features))
        norm = float(np.linalg.norm(random))
        target_norm = float(np.nanmean(np.linalg.norm(y, axis=1))) if len(y) else 1.0
        return (random / norm * target_norm if norm else np.zeros(len(features))), "seeded_random_subspace_direction"
    if strategy == "group_mean_subspace_direction":
        return group_mean, "all_pairs_group_mean_direction"
    if strategy == "task_family_mean_subspace_direction":
        subset = pairs[pairs["task_family"].astype(str).eq(str(pair.get("task_family", "")))]
        if subset.empty:
            return group_mean, "fallback_group_mean_no_task_family_match"
        return np.nanmean(_y_matrix(subset, features), axis=0), "task_family_mean_direction"
    if strategy == "side_mean_subspace_direction":
        subset = pairs[pairs["side"].astype(str).eq(str(pair.get("side", "")))]
        if subset["subject"].nunique() < 2:
            return group_mean, "fallback_group_mean_underpowered_side"
        return np.nanmean(_y_matrix(subset, features), axis=0), "side_mean_direction"
    if strategy == "reliability_weighted_subspace_direction":
        return group_mean * reliability_weights, "group_mean_weighted_by_reliability"
    if strategy == "best_single_feature_in_subspace":
        direction = np.zeros(len(features))
        if candidate_features:
            idx = features.index(candidate_features[0])
            direction[idx] = group_mean[idx]
        return direction, "best_single_feature_masked_group_mean"
    if strategy == "top_k_features_in_subspace":
        direction = np.zeros(len(features))
        for feature in candidate_features[:top_k]:
            if feature in features:
                idx = features.index(feature)
                direction[idx] = group_mean[idx]
        return direction, f"top_{top_k}_features_masked_group_mean"
    raise ValueError(f"Unknown compensation strategy: {strategy}")


def run_subspace_compensation(
    subset_name: str,
    subspace_name: str,
    subset: pd.DataFrame,
    features: list[str],
    candidates: pd.DataFrame,
    reliability: pd.DataFrame,
    reliability_sideaware: pd.DataFrame,
    config: SubspaceRunConfig,
) -> pd.DataFrame:
    """Run feature-level compensation simulation inside a subspace."""

    pairs = build_pairs(subset, features)
    if pairs.empty:
        return pd.DataFrame()
    scale = _safe_scale(np.vstack([_x_matrix(pairs, features), _medon_matrix(pairs, features)]))
    consistency = _direction_consistency(_y_matrix(pairs, features))
    rel_weights = _reliability_scores(reliability, reliability_sideaware, features)
    candidate_features = _best_features_from_candidates(candidates, features, config.top_k)
    strategies = [
        "no_action",
        "random_subspace_direction",
        "group_mean_subspace_direction",
        "task_family_mean_subspace_direction",
        "side_mean_subspace_direction",
        "reliability_weighted_subspace_direction",
        "best_single_feature_in_subspace",
        "top_k_features_in_subspace",
    ]
    rng = np.random.default_rng(config.seed + sum(ord(ch) for ch in subspace_name + subset_name))
    rows: list[dict[str, object]] = []
    for _, pair in pairs.iterrows():
        x = pair[[f"x_{feature}" for feature in features]].to_numpy(dtype=float)
        subject_ref = pair[[f"medon_{feature}" for feature in features]].to_numpy(dtype=float)
        refs = [
            {
                "reference_scope": "subject_medon_proxy",
                "reference_n": 1,
                "state": subject_ref,
            }
        ]
        family_subset = pairs[pairs["task_family"].astype(str).eq(str(pair.get("task_family", "")))]
        if not family_subset.empty:
            refs.append(
                {
                    "reference_scope": "group_task_family_medon_proxy",
                    "reference_n": int(len(family_subset)),
                    "state": np.nanmean(_medon_matrix(family_subset, features), axis=0),
                }
            )
        for strategy in strategies:
            direction, note = _compensation_direction(
                strategy,
                pair,
                pairs,
                features,
                rel_weights,
                candidate_features,
                rng,
                config.top_k,
            )
            for ref in refs:
                baseline = state_deviation_score(x, np.asarray(ref["state"]), scale)
                for alpha in ALPHA_VALUES:
                    applied = float(alpha) * direction
                    shifted = x + applied
                    post = state_deviation_score(shifted, np.asarray(ref["state"]), scale)
                    reduction = baseline - post
                    n_features = int(np.sum(np.abs(applied) > 1e-12))
                    magnitude = float(np.sqrt(np.mean((applied / scale) ** 2))) if n_features else 0.0
                    active = np.abs(applied) > 1e-12
                    direction_score = float(np.nanmean(consistency[active])) if active.any() else 1.0
                    instability = max(0.0, 1.0 - direction_score) if n_features else 0.0
                    net = (
                        reduction
                        - config.lambda_magnitude * magnitude
                        - config.lambda_complexity * n_features
                        - config.lambda_instability * instability
                    )
                    rows.append(
                        {
                            "quality_subset": subset_name,
                            "subspace_name": subspace_name,
                            "subject": pair["subject"],
                            "session": pair["session"],
                            "task_original": pair["task_original"],
                            "task_family": pair["task_family"],
                            "side": pair["side"],
                            "reference_scope": ref["reference_scope"],
                            "reference_n": int(ref["reference_n"]),
                            "strategy": strategy,
                            "alpha": float(alpha),
                            "baseline_deviation": baseline,
                            "post_shift_deviation": post,
                            "absolute_deviation_reduction": reduction,
                            "percent_deviation_reduction": 100.0 * reduction / baseline if baseline > 0 else np.nan,
                            "net_compensation_score": net,
                            "n_features": len(features),
                            "n_shifted_features": n_features,
                            "intervention_magnitude": magnitude,
                            "direction_consistency_score": direction_score,
                            "quality_flag": pair.get("quality_flag", "unknown"),
                            "methodological_note": f"{note}; in silico subspace feature shift toward dataset-internal MedOn proxy",
                        }
                    )
    return pd.DataFrame(rows)


def _best_alpha_summary(data: pd.DataFrame, groups: list[str], model_col: str) -> pd.DataFrame:
    """Choose best alpha by mean reduction for each group."""

    if data.empty:
        return pd.DataFrame()
    primary = data[data["reference_scope"].astype(str).eq("subject_medon_proxy")].copy()
    if primary.empty:
        primary = data.copy()
    grouped = primary.groupby([*groups, model_col, "alpha"], dropna=False).agg(
        n_rows=("subject", "size"),
        n_subjects=("subject", "nunique"),
        mean_baseline_deviation=("baseline_deviation", "mean"),
        mean_post_shift_deviation=("post_shift_deviation", "mean"),
        mean_deviation_reduction=("absolute_deviation_reduction", "mean"),
        median_deviation_reduction=("absolute_deviation_reduction", "median"),
        improvement_rate=("absolute_deviation_reduction", lambda x: float((x > 0).mean())),
        mean_net_compensation_score=("net_compensation_score", "mean"),
        mean_intervention_magnitude=("intervention_magnitude", "mean"),
    ).reset_index()
    pieces = []
    for _, table in grouped.groupby([*groups, model_col], dropna=False):
        table = table.copy()
        label = str(table[model_col].iloc[0])
        if label != "no_action":
            active = table[table["alpha"] > 0]
            if not active.empty:
                table = active
        else:
            zero = table[np.isclose(table["alpha"].astype(float), 0.0)]
            if not zero.empty:
                table = zero
        pieces.append(table.sort_values(["mean_deviation_reduction", "mean_net_compensation_score"], ascending=[False, False]).iloc[[0]])
    return pd.concat(pieces, ignore_index=True).rename(columns={"alpha": "best_alpha"}) if pieces else pd.DataFrame()


def summarize_compensation(simulation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize compensation simulation overall, by task family, and by side."""

    summary = _best_alpha_summary(simulation, ["quality_subset", "subspace_name"], "strategy")
    by_task = _best_alpha_summary(simulation, ["quality_subset", "subspace_name", "task_family"], "strategy")
    by_side = _best_alpha_summary(simulation, ["quality_subset", "subspace_name", "side"], "strategy")
    return summary, by_task, by_side


def _fit_predict_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    group_mean: np.ndarray,
    warnings: list[str],
    subspace_name: str,
) -> tuple[np.ndarray, str]:
    """Fit a small-data model and predict a direction."""

    if model_name == "ridge_regression_predictor":
        model = Ridge(alpha=1.0)
    elif model_name == "elastic_net_predictor":
        if len(x_train) < 6:
            return group_mean, "fallback_group_mean_insufficient_train_pairs_for_elastic_net"
        model = MultiTaskElasticNet(alpha=0.01, l1_ratio=0.25, max_iter=20000, random_state=7)
    elif model_name == "pls_regression_predictor":
        n_components = max(1, min(2, x_train.shape[1], x_train.shape[0] - 1))
        model = PLSRegression(n_components=n_components)
    elif model_name == "bayesian_ridge_predictor":
        model = MultiOutputRegressor(BayesianRidge())
    elif model_name == "kernel_ridge_predictor":
        model = KernelRidge(alpha=1.0, kernel="rbf")
    elif model_name == "gaussian_process_predictor":
        return group_mean, "fallback_group_mean_gaussian_process_not_feasible_for_broad_subspace_sweep"
    else:
        raise ValueError(model_name)
    try:
        pipeline = make_pipeline(StandardScaler(), model)
        with py_warnings.catch_warnings():
            py_warnings.simplefilter("ignore", category=ConvergenceWarning)
            pipeline.fit(x_train, y_train)
        pred = np.atleast_1d(np.asarray(pipeline.predict(x_test.reshape(1, -1))[0], dtype=float))
        if pred.shape[0] != group_mean.shape[0]:
            pred = np.resize(pred, group_mean.shape[0])
        return pred, f"{model_name}_train_subjects_only"
    except Exception as exc:  # noqa: BLE001 - small-data models may fail; keep pipeline running.
        _append_warning(warnings, f"{model_name} fallback in {subspace_name}: {type(exc).__name__}: {exc}")
        return group_mean, f"fallback_group_mean_{model_name}_error"


def _predict_direction(
    model_name: str,
    train: pd.DataFrame,
    test_row: pd.Series,
    features: list[str],
    rel_weights: np.ndarray,
    rng: np.random.Generator,
    warnings: list[str],
    subspace_name: str,
) -> tuple[np.ndarray, str]:
    """Predict one held-out direction."""

    x_train = _x_matrix(train, features)
    y_train = _y_matrix(train, features)
    x_test = test_row[[f"x_{feature}" for feature in features]].to_numpy(dtype=float)
    group_mean = np.nanmean(y_train, axis=0) if len(y_train) else np.zeros(len(features))
    consistency = _direction_consistency(y_train)
    if model_name == "no_action":
        return np.zeros(len(features)), "zero_vector"
    if model_name == "random_direction":
        random = rng.normal(size=len(features))
        norm = float(np.linalg.norm(random))
        target_norm = float(np.nanmean(np.linalg.norm(y_train, axis=1))) if len(y_train) else 1.0
        return (random / norm * target_norm if norm else np.zeros(len(features))), "seeded_random_direction"
    if model_name == "group_mean_direction":
        return group_mean, "train_subject_group_mean_direction"
    if model_name == "task_family_group_mean_direction":
        family = str(test_row.get("task_family", "unknown"))
        subset = train[train["task_family"].astype(str).eq(family)]
        if subset.empty:
            return group_mean, "fallback_group_mean_no_task_family_match"
        return np.nanmean(_y_matrix(subset, features), axis=0), "train_task_family_mean_direction"
    if model_name == "side_group_mean_direction":
        side = str(test_row.get("side", "unknown"))
        subset = train[train["side"].astype(str).eq(side)]
        if subset["subject"].nunique() < 2:
            return group_mean, "fallback_group_mean_underpowered_side"
        return np.nanmean(_y_matrix(subset, features), axis=0), "train_side_mean_direction"
    if model_name == "reliability_weighted_direction":
        return group_mean * rel_weights * consistency, "train_group_mean_weighted_by_reliability_and_consistency"
    return _fit_predict_model(model_name, x_train, y_train, x_test, group_mean, warnings, subspace_name)


def _fit_predict_batch_model(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    warnings: list[str],
    subspace_name: str,
) -> tuple[np.ndarray, str]:
    """Fit one learned model once per held-out subject and predict all test rows."""

    x_train = _x_matrix(train, features)
    y_train = _y_matrix(train, features)
    x_test = _x_matrix(test, features)
    group_mean = np.nanmean(y_train, axis=0) if len(y_train) else np.zeros(len(features))
    fallback = np.tile(group_mean, (len(test), 1))
    if model_name == "ridge_regression_predictor":
        model = Ridge(alpha=1.0)
    elif model_name == "elastic_net_predictor":
        return fallback, "fallback_group_mean_elastic_net_not_feasible_for_broad_subspace_sweep"
    elif model_name == "pls_regression_predictor":
        return fallback, "fallback_group_mean_pls_not_feasible_for_broad_subspace_sweep"
    elif model_name == "bayesian_ridge_predictor":
        return fallback, "fallback_group_mean_bayesian_ridge_not_feasible_for_broad_subspace_sweep"
    elif model_name == "kernel_ridge_predictor":
        return fallback, "fallback_group_mean_kernel_ridge_not_feasible_for_broad_subspace_sweep"
    elif model_name == "gaussian_process_predictor":
        return fallback, "fallback_group_mean_gaussian_process_not_feasible_for_broad_subspace_sweep"
    else:
        raise ValueError(model_name)
    try:
        pipeline = make_pipeline(StandardScaler(), model)
        with py_warnings.catch_warnings():
            py_warnings.simplefilter("ignore", category=ConvergenceWarning)
            pipeline.fit(x_train, y_train)
        pred = np.asarray(pipeline.predict(x_test), dtype=float)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if pred.shape[1] != len(features):
            pred = np.resize(pred, (len(test), len(features)))
        return pred, f"{model_name}_train_subjects_only"
    except Exception as exc:  # noqa: BLE001 - keep broad sweep running.
        _append_warning(warnings, f"{model_name} batch fallback in {subspace_name}: {type(exc).__name__}: {exc}")
        return fallback, f"fallback_group_mean_{model_name}_error"


def run_subspace_predictor(
    subset_name: str,
    subspace_name: str,
    subset: pd.DataFrame,
    features: list[str],
    reliability: pd.DataFrame,
    reliability_sideaware: pd.DataFrame,
    config: SubspaceRunConfig,
) -> pd.DataFrame:
    """Run leave-one-subject-out direction prediction inside one subspace."""

    pairs = build_pairs(subset, features)
    if pairs.empty or pairs["subject"].nunique() < 2:
        return pd.DataFrame()
    rel_weights = _reliability_scores(reliability, reliability_sideaware, features)
    rng = np.random.default_rng(config.seed + sum(ord(ch) for ch in "predictor" + subspace_name + subset_name))
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    for held_subject in sorted(pairs["subject"].astype(str).unique()):
        train = pairs[~pairs["subject"].astype(str).eq(held_subject)].copy()
        test = pairs[pairs["subject"].astype(str).eq(held_subject)].copy()
        if train.empty:
            continue
        scale = _safe_scale(np.vstack([_x_matrix(train, features), _medon_matrix(train, features)]))
        train_consistency = _direction_consistency(_y_matrix(train, features))
        learned_models = [
            "ridge_regression_predictor",
            "elastic_net_predictor",
            "pls_regression_predictor",
            "bayesian_ridge_predictor",
            "kernel_ridge_predictor",
            "gaussian_process_predictor",
        ]
        batch_predictions = {
            model: _fit_predict_batch_model(model, train, test, features, warnings, subspace_name)
            for model in learned_models
        }
        for test_position, (_, test_row) in enumerate(test.iterrows()):
            x = test_row[[f"x_{feature}" for feature in features]].to_numpy(dtype=float)
            true = test_row[[f"y_{feature}" for feature in features]].to_numpy(dtype=float)
            subject_ref = test_row[[f"medon_{feature}" for feature in features]].to_numpy(dtype=float)
            refs = [
                {
                    "reference_scope": "subject_medon_proxy",
                    "reference_n": 1,
                    "state": subject_ref,
                }
            ]
            family_subset = train[train["task_family"].astype(str).eq(str(test_row.get("task_family", "")))]
            if family_subset.empty:
                family_subset = train
            refs.append(
                {
                    "reference_scope": "group_task_family_medon_proxy",
                    "reference_n": int(len(family_subset)),
                    "state": np.nanmean(_medon_matrix(family_subset, features), axis=0),
                }
            )
            predictions = {}
            for model in PREDICTOR_MODELS:
                if model in batch_predictions:
                    pred_matrix, note = batch_predictions[model]
                    predictions[model] = (np.asarray(pred_matrix[test_position], dtype=float), note)
                else:
                    predictions[model] = _predict_direction(
                        model,
                        train,
                        test_row,
                        features,
                        rel_weights,
                        rng,
                        warnings,
                        subspace_name,
                    )
            for ref in refs:
                baseline = state_deviation_score(x, np.asarray(ref["state"]), scale)
                for model_name, (direction, note) in predictions.items():
                    error = direction - true
                    cosine = _cosine(direction, true)
                    for alpha in ALPHA_VALUES:
                        applied = float(alpha) * direction
                        shifted = x + applied
                        post = state_deviation_score(shifted, np.asarray(ref["state"]), scale)
                        reduction = baseline - post
                        active = np.abs(applied) > 1e-12
                        n_nonzero = int(active.sum())
                        magnitude = float(np.sqrt(np.mean((applied / scale) ** 2))) if n_nonzero else 0.0
                        stability = float(np.nanmean(train_consistency[active])) if active.any() else 1.0
                        instability = max(0.0, 1.0 - stability) if n_nonzero else 0.0
                        net = (
                            reduction
                            - config.lambda_magnitude * magnitude
                            - config.lambda_complexity * n_nonzero
                            - config.lambda_instability * instability
                        )
                        rows.append(
                            {
                                "quality_subset": subset_name,
                                "subspace_name": subspace_name,
                                "subject": test_row["subject"],
                                "session": test_row["session"],
                                "task_original": test_row["task_original"],
                                "task_family": test_row["task_family"],
                                "side": test_row["side"],
                                "reference_scope": ref["reference_scope"],
                                "reference_n": int(ref["reference_n"]),
                                "model_name": model_name,
                                "alpha": float(alpha),
                                "baseline_deviation": baseline,
                                "post_shift_deviation": post,
                                "absolute_deviation_reduction": reduction,
                                "percent_deviation_reduction": 100.0 * reduction / baseline if baseline > 0 else np.nan,
                                "improvement_over_no_action": reduction,
                                "cosine_similarity_to_true_direction": cosine,
                                "direction_mse": float(np.nanmean(error**2)),
                                "direction_mae": float(np.nanmean(np.abs(error))),
                                "n_nonzero_shifted_features": n_nonzero,
                                "intervention_magnitude": magnitude,
                                "direction_stability_penalty": instability,
                                "net_compensation_score": net,
                                "quality_flag": test_row.get("quality_flag", "unknown"),
                                "methodological_note": f"{note}; held-out subject; dataset-internal MedOn proxy",
                            }
                        )
    if warnings:
        warning_rows = pd.DataFrame(rows)
        if not warning_rows.empty:
            warning_rows["model_warning_count"] = len(warnings)
        return warning_rows
    return pd.DataFrame(rows)


def summarize_predictor(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize subspace predictor results."""

    summary = _best_alpha_summary(results, ["quality_subset", "subspace_name"], "model_name")
    by_subject = _best_alpha_summary(results, ["quality_subset", "subspace_name", "subject"], "model_name")
    by_task = _best_alpha_summary(results, ["quality_subset", "subspace_name", "task_family"], "model_name")
    by_side = _best_alpha_summary(results, ["quality_subset", "subspace_name", "side"], "model_name")
    if not summary.empty:
        comparison = summary.set_index(["quality_subset", "subspace_name", "model_name"])

        def beats(row: pd.Series, baseline: str) -> bool:
            key = (row["quality_subset"], row["subspace_name"], baseline)
            if key not in comparison.index:
                return False
            return bool(float(row["mean_deviation_reduction"]) > float(comparison.loc[key, "mean_deviation_reduction"]))

        summary["beats_no_action"] = [beats(row, "no_action") for _, row in summary.iterrows()]
        summary["beats_group_mean"] = [beats(row, "group_mean_direction") for _, row in summary.iterrows()]
        summary["beats_random"] = [beats(row, "random_direction") for _, row in summary.iterrows()]
    return summary, by_subject, by_task, by_side


def bootstrap_predictor_ci(results: pd.DataFrame, summary: pd.DataFrame, seed: int, n_bootstrap: int) -> pd.DataFrame:
    """Bootstrap exploratory CIs over held-out subjects when possible."""

    if results.empty or summary.empty:
        return pd.DataFrame()
    if n_bootstrap <= 0:
        return pd.DataFrame(
            [
                {
                    "quality_subset": "",
                    "subspace_name": "",
                    "model_name": "",
                    "best_alpha": np.nan,
                    "metric": "mean_deviation_reduction",
                    "mean": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "bootstrap_unit": "not_run",
                    "n_units": 0,
                    "n_bootstrap": 0,
                    "methodological_note": "Bootstrap skipped for the broad all-subspace sweep; rerun with --n-bootstrap for exploratory intervals.",
                }
            ]
        )
    primary = results[results["reference_scope"].astype(str).eq("subject_medon_proxy")].copy()
    if primary.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    metrics = {
        "mean_deviation_reduction": "absolute_deviation_reduction",
        "mean_net_compensation_score": "net_compensation_score",
        "mean_cosine_similarity": "cosine_similarity_to_true_direction",
    }
    for _, row in summary.iterrows():
        subset_name = str(row["quality_subset"])
        subspace_name = str(row["subspace_name"])
        model_name = str(row["model_name"])
        alpha = float(row["best_alpha"])
        data = primary[
            primary["quality_subset"].astype(str).eq(subset_name)
            & primary["subspace_name"].astype(str).eq(subspace_name)
            & primary["model_name"].astype(str).eq(model_name)
            & np.isclose(primary["alpha"].astype(float), alpha)
        ].copy()
        if data.empty:
            continue
        units = sorted(data["subject"].dropna().astype(str).unique())
        unit_column = "subject"
        note = "bootstrap_over_held_out_subjects_exploratory"
        if len(units) < 4:
            data["_state_id"] = data["subject"].astype(str) + "::" + data["task_original"].astype(str)
            units = sorted(data["_state_id"].unique())
            unit_column = "_state_id"
            note = "bootstrap_over_held_out_states_due_to_small_subject_count_exploratory"
        for metric_name, source in metrics.items():
            observed = float(data[source].mean())
            boot = []
            for _ in range(n_bootstrap):
                sampled_units = rng.choice(units, size=len(units), replace=True)
                sampled = pd.concat([data[data[unit_column].eq(unit)] for unit in sampled_units], ignore_index=True)
                if not sampled.empty:
                    boot.append(float(sampled[source].mean()))
            lower, upper = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
            rows.append(
                {
                    "quality_subset": subset_name,
                    "subspace_name": subspace_name,
                    "model_name": model_name,
                    "best_alpha": alpha,
                    "metric": metric_name,
                    "mean": observed,
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "bootstrap_unit": unit_column,
                    "n_units": len(units),
                    "n_bootstrap": n_bootstrap,
                    "methodological_note": note,
                }
            )
    return pd.DataFrame(rows)


def clinical_anchor_tables(
    state_vectors: pd.DataFrame,
    direction_consistency: pd.DataFrame,
    compensation_summary: pd.DataFrame,
    raw_root: str | Path,
    warnings: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optionally link subspace summaries to local UPDRS medication-response metadata."""

    raw_root = Path(raw_root)
    off_path = raw_root / "participants_updrs_off.tsv"
    on_path = raw_root / "participants_updrs_on.tsv"
    if not off_path.exists() or not on_path.exists():
        _append_warning(warnings, "UPDRS off/on metadata files were not found locally; clinical anchor check skipped.")
        summary = pd.DataFrame(
            [
                {
                    "metadata_available": False,
                    "participants_updrs_off": str(off_path),
                    "participants_updrs_on": str(on_path),
                    "n_matched_subjects": 0,
                    "methodological_note": "UPDRS metadata unavailable or not matched.",
                }
            ]
        )
        return summary, pd.DataFrame(columns=["subspace_name", "metric", "spearman_rho", "p_value", "n_subjects"])
    try:
        off = pd.read_csv(off_path, sep="\t")
        on = pd.read_csv(on_path, sep="\t")
    except Exception as exc:  # noqa: BLE001
        _append_warning(warnings, f"Could not read UPDRS metadata: {type(exc).__name__}: {exc}")
        return pd.DataFrame([{"metadata_available": False, "n_matched_subjects": 0}]), pd.DataFrame()
    id_col = "participant_id" if "participant_id" in off and "participant_id" in on else off.columns[0]
    for table in [off, on]:
        for column in table.columns:
            if column != id_col:
                table[column] = pd.to_numeric(table[column], errors="coerce")
    score_cols = [column for column in off.columns if column != id_col and pd.api.types.is_numeric_dtype(off[column])]
    common_scores = [column for column in score_cols if column in on and pd.api.types.is_numeric_dtype(on[column])]
    if not common_scores:
        _append_warning(warnings, "UPDRS files were found but no common numeric score columns were detected.")
        return pd.DataFrame([{"metadata_available": True, "n_matched_subjects": 0, "score_column": ""}]), pd.DataFrame()
    score = "SUM" if "SUM" in common_scores else common_scores[0]
    response = off[[id_col, score]].merge(on[[id_col, score]], on=id_col, suffixes=("_off", "_on"))
    response["subject"] = response[id_col].astype(str)
    response["medication_response"] = response[f"{score}_off"] - response[f"{score}_on"]
    available_subjects = set(state_vectors["subject"].astype(str).unique()) if "subject" in state_vectors else set()
    response = response[response["subject"].isin(available_subjects)].copy()
    if response.empty:
        _append_warning(warnings, "UPDRS files were found but no subjects matched current state vectors.")
        return pd.DataFrame([{"metadata_available": True, "n_matched_subjects": 0, "score_column": score}]), pd.DataFrame()

    try:
        from scipy.stats import spearmanr
    except Exception:  # noqa: BLE001
        spearmanr = None

    anchor_specs = {
        "motor_beta": ["motor_low_beta_power", "motor_high_beta_power", "motor_broad_beta_power"],
        "stn_beta": [
            "stn_low_beta_power",
            "stn_high_beta_power",
            "stn_broad_beta_power",
            "stn_left_low_beta_power",
            "stn_left_high_beta_power",
            "stn_left_broad_beta_power",
            "stn_right_low_beta_power",
            "stn_right_high_beta_power",
            "stn_right_broad_beta_power",
        ],
        "cortico_stn_coupling": [
            "motor_stn_low_beta_coupling",
            "motor_stn_high_beta_coupling",
            "motor_stn_broad_beta_coupling",
            "motor_stn_alpha_coupling",
            "motor_stn_gamma_coupling",
            "motor_stn_left_broad_beta_coupling",
            "motor_stn_right_broad_beta_coupling",
            "motor_stn_task_side_broad_beta_coupling",
            "motor_stn_opposite_task_side_broad_beta_coupling",
            "beta_coupling_index",
            "gamma_coupling_index",
            "beta_coupling_asymmetry",
            "cortico_stn_coupling_index",
        ],
        "stn_gamma": ["stn_gamma_power", "stn_left_gamma_power", "stn_right_gamma_power"],
    }
    rows = []
    for subspace_name, features in anchor_specs.items():
        available = _feature_list(state_vectors, features)
        if not available:
            continue
        pairs = build_pairs(complete_medoff_medon_pairs(state_vectors), available)
        if pairs.empty:
            continue
        subject_rows = []
        for subject, table in pairs.groupby("subject", dropna=False):
            y = _y_matrix(table, available)
            x = _x_matrix(table, available)
            medon = _medon_matrix(table, available)
            scale = _safe_scale(np.vstack([x, medon]))
            baseline = [state_deviation_score(x[idx], medon[idx], scale) for idx in range(len(table))]
            subject_rows.append(
                {
                    "subject": str(subject),
                    "direction_norm": float(np.nanmean(np.linalg.norm(y, axis=1))),
                    "baseline_deviation": float(np.nanmean(baseline)),
                }
            )
        metrics = pd.DataFrame(subject_rows).merge(response[["subject", "medication_response"]], on="subject")
        for metric in ["direction_norm", "baseline_deviation"]:
            if len(metrics) < 3 or metrics[metric].nunique(dropna=True) < 2 or spearmanr is None:
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(metrics["medication_response"], metrics[metric], nan_policy="omit")
            rows.append(
                {
                    "subspace_name": subspace_name,
                    "metric": metric,
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "n_subjects": int(len(metrics)),
                    "methodological_note": "Exploratory subject-level UPDRS association; no causal interpretation.",
                }
            )
    summary = pd.DataFrame(
        [
            {
                "metadata_available": True,
                "score_column": score,
                "n_matched_subjects": int(len(response)),
                "mean_medication_response": float(response["medication_response"].mean()),
                "methodological_note": "Exploratory medication-response anchor; no causal interpretation.",
            }
        ]
    )
    return summary, pd.DataFrame(rows)


def _fmt(value: object, digits: int = 4) -> str:
    """Format a scalar for Markdown."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _best_row(table: pd.DataFrame, metric: str, active_only_col: str | None = None) -> pd.Series | None:
    """Return row with maximum metric."""

    if table.empty or metric not in table:
        return None
    data = table.dropna(subset=[metric]).copy()
    if active_only_col and active_only_col in data:
        data = data[~data[active_only_col].astype(str).eq("no_action")]
    if data.empty:
        return None
    return data.sort_values(metric, ascending=False).iloc[0]


def _top_candidate_line(candidates: pd.DataFrame, subspace_name: str, subset_name: str = "all_recordings") -> str:
    """Report top candidate line for a subspace."""

    table = candidates[
        candidates["quality_subset"].astype(str).eq(subset_name)
        & candidates["subspace_name"].astype(str).eq(subspace_name)
    ].copy()
    if table.empty:
        return f"- {subspace_name}: no candidate ranking available."
    top = table.sort_values("rank").iloc[0]
    top10 = table[pd.to_numeric(table["rank"], errors="coerce") <= 10]
    coupling_count = int(top10["feature"].astype(str).str.contains("coupling|cortico_stn").sum())
    return (
        f"- {subspace_name}: top1={top['feature']}, "
        f"motor_beta_top10={int(top10.get('is_motor_beta_target', pd.Series(dtype=bool)).sum())}, "
        f"stn_beta_top10={int(top10.get('is_stn_beta_target', pd.Series(dtype=bool)).sum())}, "
        f"stn_gamma_top10={int(top10.get('is_stn_gamma_target', pd.Series(dtype=bool)).sum())}, "
        f"coupling_top10={coupling_count}"
    )


def write_subspace_report(
    path: str | Path,
    enhanced: pd.DataFrame,
    inventory: pd.DataFrame,
    definitions: pd.DataFrame,
    quality_comparison: pd.DataFrame,
    candidates: pd.DataFrame,
    compensation_summary: pd.DataFrame,
    predictor_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    clinical_summary: pd.DataFrame,
    clinical_corr: pd.DataFrame,
    warnings: list[str],
) -> None:
    """Write the JNE-oriented subspace analysis report."""

    n_subjects = int(enhanced["subject"].nunique()) if "subject" in enhanced and not enhanced.empty else 0
    n_states = int(len(enhanced))
    n_pairs = pair_count(complete_medoff_medon_pairs(enhanced)) if not enhanced.empty else 0
    n_features_original = len([feature for feature in REAL_STATE_FEATURE_COLUMNS if feature in enhanced])
    n_features_enhanced = len(_numeric_features(enhanced))
    best_comp = _best_row(compensation_summary, "mean_deviation_reduction", "strategy")
    best_comp_net = _best_row(compensation_summary, "mean_net_compensation_score", "strategy")
    best_pred = _best_row(predictor_summary, "mean_deviation_reduction", "model_name")
    best_pred_net = _best_row(predictor_summary, "mean_net_compensation_score", "model_name")
    learned = predictor_summary[predictor_summary["model_name"].astype(str).str.contains("ridge|elastic|pls|bayesian|kernel|gaussian", regex=True)] if not predictor_summary.empty else pd.DataFrame()
    group = predictor_summary[predictor_summary["model_name"].eq("group_mean_direction")] if not predictor_summary.empty else pd.DataFrame()
    no_action = predictor_summary[predictor_summary["model_name"].eq("no_action")] if not predictor_summary.empty else pd.DataFrame()

    def beats(table: pd.DataFrame, baseline: pd.DataFrame) -> str:
        if table.empty or baseline.empty:
            return "not_available"
        return "yes" if float(table["mean_deviation_reduction"].max()) > float(baseline["mean_deviation_reduction"].max()) else "no"

    coupling_available = inventory[inventory["feature"].astype(str).str.contains("coupling|cortico_stn") & inventory["available"].astype(bool)]
    gamma_names = {"motor_gamma", "stn_gamma", "stn_beta_gamma", "literature_informed_fast_network"}
    gamma_sub = predictor_summary[predictor_summary["subspace_name"].astype(str).isin(gamma_names)] if not predictor_summary.empty else pd.DataFrame()
    clean_rows = quality_comparison[quality_comparison["quality_subset"].eq("good_only")] if not quality_comparison.empty else pd.DataFrame()

    lines = [
        "# Subspace Compensation Analysis Report",
        "",
        "## Purpose",
        "",
        "This JNE-oriented layer tests whether MedOff-to-MedOn direction modeling improves when restricted to stable feature subspaces and available cortico-STN network proxies. All comparisons are in silico and use MedOn only as a dataset-internal compensated proxy.",
        "",
        "## Dataset And Feature Inventory",
        "",
        f"- subjects: {n_subjects}",
        f"- state_vectors: {n_states}",
        f"- paired_medoff_medon_examples: {n_pairs}",
        f"- original_state_features: {n_features_original}",
        f"- enhanced_numeric_features: {n_features_enhanced}",
        f"- subspaces_defined: {definitions['subspace_name'].nunique() if not definitions.empty else 0}",
        f"- available_coupling_features: {';'.join(coupling_available['feature'].astype(str)) if not coupling_available.empty else 'none'}",
        "",
    ]
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in dict.fromkeys(warnings):
            lines.append(f"- {warning}")
        lines.append("")

    lines.extend(["## Subspace Candidate Overview", ""])
    for subspace_name in [
        "full_28",
        "motor_beta",
        "stn_beta",
        "stn_gamma",
        "cortico_stn_coupling",
        "literature_informed_beta_network",
        "literature_informed_fast_network",
    ]:
        lines.append(_top_candidate_line(candidates, subspace_name))

    lines.extend(["", "## Main Questions", ""])
    lines.append("1. Does restricting the target space improve direction modeling?")
    if best_pred is not None:
        lines.append(
            f"   Best predictor row: subspace={best_pred['subspace_name']}, subset={best_pred['quality_subset']}, "
            f"model={best_pred['model_name']}, mean_reduction={_fmt(best_pred['mean_deviation_reduction'])}."
        )
    else:
        lines.append("   Predictor summaries were unavailable.")
    lines.append("2. Which subspace is most stable?")
    if not quality_comparison.empty:
        stable = quality_comparison.sort_values("mean_direction_consistency", ascending=False).iloc[0]
        lines.append(
            f"   Highest mean direction consistency: {stable['subspace_name']} / {stable['quality_subset']} "
            f"({_fmt(stable['mean_direction_consistency'])})."
        )
    lines.append("3. Which subspace gives the best active compensation reduction?")
    if best_comp is not None:
        lines.append(
            f"   {best_comp['subspace_name']} / {best_comp['quality_subset']} using {best_comp['strategy']} "
            f"(mean_reduction={_fmt(best_comp['mean_deviation_reduction'])})."
        )
    lines.append("4. Does any subspace beat no_action under conservative scoring?")
    if best_comp_net is not None:
        no_action_net = compensation_summary[compensation_summary["strategy"].eq("no_action")]["mean_net_compensation_score"].max() if not compensation_summary.empty else np.nan
        lines.append(
            f"   Best active net row={best_comp_net['strategy']} in {best_comp_net['subspace_name']} "
            f"(net={_fmt(best_comp_net['mean_net_compensation_score'])}); no_action_max_net={_fmt(no_action_net)}."
        )
    lines.append(f"5. Do learned models beat group_mean inside any subspace? {beats(learned, group)}.")
    lines.append(f"6. Do learned models beat no_action inside any subspace? {beats(learned, no_action)}.")
    lines.append("7. Is motor beta more robust than STN beta/gamma?")
    lines.append("   Motor beta and STN beta are evaluated separately in the candidate and predictor tables. STN gamma is retained as exploratory because it remains sensitive to subject, side, task, and quality filters.")
    lines.append("8. Are STN-cortex coupling features useful?")
    if not coupling_available.empty:
        coupling_best = _best_row(
            predictor_summary[predictor_summary["subspace_name"].eq("cortico_stn_coupling")],
            "mean_deviation_reduction",
        )
        if coupling_best is not None:
            lines.append(
                f"   Available coupling subspace best predictor={coupling_best['model_name']} "
                f"mean_reduction={_fmt(coupling_best['mean_deviation_reduction'])}."
            )
        else:
            lines.append("   Available coupling proxies were ranked but predictor summaries were unavailable.")
    else:
        lines.append("   Expanded coupling features were not available in current tables.")
    lines.append("9. Are gamma effects still unstable after subspace restriction?")
    if not gamma_sub.empty:
        gamma_best = _best_row(gamma_sub, "mean_deviation_reduction")
        lines.append(
            f"   Best gamma/fast subspace predictor row={gamma_best['subspace_name']} / {gamma_best['model_name']} "
            f"with mean_reduction={_fmt(gamma_best['mean_deviation_reduction'])}."
        )
    else:
        lines.append("   Gamma subspace summaries were unavailable.")
    lines.append("10. Do quality filters change the conclusion?")
    if not clean_rows.empty:
        top_clean = clean_rows.sort_values("mean_baseline_deviation", ascending=True).iloc[0]
        lines.append(
            f"   good_only retains subspace rows; lowest baseline deviation is {top_clean['subspace_name']} "
            f"({_fmt(top_clean['mean_baseline_deviation'])}). Interpret with reduced pair count."
        )
    lines.append("11. Are there exploratory links with UPDRS medication response?")
    if clinical_summary.empty or not bool(clinical_summary.iloc[0].get("metadata_available", False)):
        lines.append("   Local UPDRS metadata were unavailable or not matched, so this check was skipped.")
    else:
        lines.append(
            f"   UPDRS metadata matched {int(clinical_summary.iloc[0].get('n_matched_subjects', 0))} subjects; correlations are exploratory."
        )

    lines.extend(
        [
            "",
            "## Quality-Subset Notes",
            "",
        ]
    )
    if quality_comparison.empty:
        lines.append("No quality-subset summaries were available.")
    else:
        for subset_name, table in quality_comparison.groupby("quality_subset", dropna=False):
            n_pairs_subset = int(table["n_paired_examples"].max()) if "n_paired_examples" in table else 0
            lines.append(f"- {subset_name}: paired_examples={n_pairs_subset}, subspaces={table['subspace_name'].nunique()}.")

    lines.extend(
        [
            "",
            "## Conservative Interpretation",
            "",
            "- Positive deviation reduction means a simulated feature-level shift moved a MedOff vector closer to a dataset-internal MedOn proxy under the current metric.",
            "- Restricted subspaces are candidate analysis spaces, not device settings.",
            "- Gamma and coupling results require validation on more subjects and richer perturbation-response data.",
            "- HCP is not used as an electrophysiological reference in this analysis.",
            "",
            "## JNE-Oriented Takeaway",
            "",
            "The subspace layer reframes the weak full-vector prediction result as a feature-space design problem: evaluate whether reliable motor beta, STN beta, gamma, and cortico-STN proxy subspaces produce more stable in silico direction estimates than the full 28-feature vector.",
        ]
    )
    if not bootstrap.empty and not bootstrap.get("bootstrap_unit", pd.Series(dtype=str)).astype(str).eq("not_run").all():
        lines.extend(["", "## Bootstrap Uncertainty", ""])
        for _, row in bootstrap[bootstrap["metric"].eq("mean_deviation_reduction")].head(10).iterrows():
            lines.append(
                f"- {row['subspace_name']} / {row['model_name']}: mean={_fmt(row['mean'])}, "
                f"95CI=[{_fmt(row['ci_lower'])}, {_fmt(row['ci_upper'])}], subset={row['quality_subset']}"
            )
    elif not bootstrap.empty:
        lines.extend(["", "## Bootstrap Uncertainty", ""])
        lines.append("- Bootstrap intervals were skipped for the broad all-subspace sweep; rerun with --n-bootstrap for exploratory intervals.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_integrated_summary_12(
    path: str | Path,
    enhanced: pd.DataFrame,
    quality_comparison: pd.DataFrame,
    candidates: pd.DataFrame,
    compensation_summary: pd.DataFrame,
    predictor_summary: pd.DataFrame,
) -> None:
    """Write current 12-subject integrated summary with subspace results."""

    n_subjects = int(enhanced["subject"].nunique()) if not enhanced.empty else 0
    n_states = int(len(enhanced))
    n_pairs = pair_count(complete_medoff_medon_pairs(enhanced)) if not enhanced.empty else 0
    tasks = sorted(enhanced["task_original"].dropna().astype(str).unique()) if "task_original" in enhanced else []
    best_comp = _best_row(compensation_summary, "mean_deviation_reduction", "strategy")
    best_pred = _best_row(predictor_summary, "mean_deviation_reduction", "model_name")
    best_learned = _best_row(
        predictor_summary[predictor_summary["model_name"].astype(str).str.contains("ridge|elastic|pls|bayesian|kernel|gaussian", regex=True)] if not predictor_summary.empty else pd.DataFrame(),
        "mean_deviation_reduction",
    )
    lines = [
        "# Real 12-Subject Integrated Summary",
        "",
        "## Dataset Subset",
        "",
        f"- subjects: {n_subjects}",
        f"- state_vectors: {n_states}",
        f"- paired_medoff_medon_examples: {n_pairs}",
        f"- tasks: {';'.join(tasks) if tasks else 'not_available'}",
        "- MedOn is used only as a dataset-internal compensated proxy.",
        "",
        "## Clean-Subset And Subspace Results",
        "",
    ]
    if not quality_comparison.empty:
        for subset_name, table in quality_comparison.groupby("quality_subset", dropna=False):
            lines.append(
                f"- {subset_name}: paired_examples={int(table['n_paired_examples'].max())}, "
                f"mean_direction_consistency={_fmt(table['mean_direction_consistency'].mean())}"
            )
    lines.extend(["", "## Candidate Findings", ""])
    for subspace in ["motor_beta", "stn_beta", "stn_gamma", "cortico_stn_coupling", "literature_informed_beta_network"]:
        lines.append(_top_candidate_line(candidates, subspace))
    lines.extend(["", "## In Silico Compensation", ""])
    if best_comp is not None:
        lines.append(
            f"- best_active_subspace_strategy_by_reduction: {best_comp['subspace_name']} / {best_comp['strategy']} "
            f"subset={best_comp['quality_subset']} mean_reduction={_fmt(best_comp['mean_deviation_reduction'])}"
        )
    lines.extend(["", "## Patient-Specific Prediction After Subspace Restriction", ""])
    if best_pred is not None:
        lines.append(
            f"- best_predictor_row: {best_pred['subspace_name']} / {best_pred['model_name']} "
            f"subset={best_pred['quality_subset']} mean_reduction={_fmt(best_pred['mean_deviation_reduction'])}"
        )
    if best_learned is not None:
        lines.append(
            f"- best_learned_predictor_row: {best_learned['subspace_name']} / {best_learned['model_name']} "
            f"subset={best_learned['quality_subset']} mean_reduction={_fmt(best_learned['mean_deviation_reduction'])}"
        )
    lines.extend(
        [
            "- These are exploratory feature-level direction estimates, not device settings.",
            "",
            "## JNE-Oriented Interpretation",
            "",
            "The current 12-subject layer tests whether biologically motivated subspaces improve over full-vector MedOff-to-MedOn prediction. Motor beta, STN beta, STN gamma, alpha, and cortico-STN coupling proxies are interpreted separately, with gamma and side-dependent findings treated as requiring validation.",
            "",
            "## Guardrails",
            "",
            "- HCP is not used as an electrophysiological reference.",
            "- MedOn is not a healthy state.",
            "- Candidate directions are in silico research outputs.",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_legacy_integrated_summary(path: str | Path) -> None:
    """Append a pointer from the older integrated summary to the 12-subject summary."""

    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else "# Real 8-Subject Integrated Summary\n"
    marker = "## Current 12-Subject Subspace Update"
    if marker in text:
        text = text.split(marker, maxsplit=1)[0].rstrip() + "\n"
    section = [
        "",
        marker,
        "",
        "- A newer 12-subject subspace compensation analysis has been generated.",
        "- See reports/real_12_subject_integrated_summary.md for the current subject count, clean-subset result, and restricted-subspace findings.",
    ]
    path.write_text(text.rstrip() + "\n" + "\n".join(section) + "\n", encoding="utf-8")


def run_subspace_compensation_analysis(
    state_vectors_path: str | Path,
    quality_table_path: str | Path,
    output_dir: str | Path = "outputs/tables",
    reports_dir: str | Path = "reports",
    window_features_path: str | Path = "outputs/tables/ds004998_window_features.csv",
    raw_root: str | Path = "data/raw/ds004998",
    reference_strategy: str = "medon_condition_proxy",
    config: SubspaceRunConfig | None = None,
) -> dict[str, Path]:
    """Run the full restricted-subspace analysis layer."""

    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    config = config or SubspaceRunConfig(weights=ObjectiveWeights())
    warnings: list[str] = []

    state_vectors = read_state_vectors(state_vectors_path)
    quality = read_quality_table(quality_table_path)
    reliability = _read_optional_csv(output_dir / "real_target_reliability.csv", warnings)
    reliability_sideaware = _read_optional_csv(output_dir / "real_target_reliability_sideaware.csv", warnings)
    clean_candidates = _read_optional_csv(output_dir / "clean_subset_candidate_targets.csv", warnings)
    _read_optional_csv(output_dir / "real_compensation_directions_group.csv", warnings)
    _read_optional_csv(output_dir / "real_compensation_directions_sideaware_group.csv", warnings)
    _read_optional_csv(output_dir / "real_patient_specific_pairs.csv", warnings)
    _read_optional_csv(output_dir / "clean_subset_counts.csv", warnings)
    _read_optional_csv(output_dir / "clean_subset_predictor_summary.csv", warnings)
    _read_optional_csv(output_dir / "clean_subset_compensation_summary.csv", warnings)

    enhanced, inventory = enhance_state_vectors(state_vectors, window_features_path, warnings)
    annotated = attach_quality(enhanced, quality)
    definitions = build_subspace_definitions(
        annotated,
        reliability,
        reliability_sideaware,
        clean_candidates,
        top_k=config.top_k,
    )
    subspaces = subspace_map(definitions)
    subsets = {
        subset_name: subset_state_vectors(annotated, flags)
        for subset_name, flags in ANALYSIS_SUBSETS.items()
    }

    direction_rows = []
    deviation_rows = []
    quality_rows = []
    candidate_rows = []
    side_rows = []
    left_right_rows = []
    hold_move_rows = []
    compensation_rows = []
    predictor_rows = []

    for subset_name in SUBSPACE_QUALITY_SUBSETS:
        subset = subsets[subset_name]
        for subspace_name, features in subspaces.items():
            if not features:
                continue
            print(f"Running subspace analysis: {subset_name} / {subspace_name}", flush=True)
            direction, quality_summary = compute_subspace_direction_consistency(subset_name, subspace_name, subset, features)
            direction_rows.append(direction)
            quality_rows.append(quality_summary)
            deviation_rows.append(compute_subspace_deviation(subset_name, subspace_name, subset, features, reference_strategy))
            candidates = rank_subspace_candidates(subset_name, subspace_name, subset, features, config)
            candidate_rows.append(candidates)
            side_summary, strata_tables = summarize_side_strata(subset_name, subspace_name, subset, features, config)
            side_rows.append(side_summary)
            left_right = compare_rank_tables(
                strata_tables.get("left_side", pd.DataFrame()),
                strata_tables.get("right_side", pd.DataFrame()),
                "left",
                "right",
                "left_right",
            )
            if not left_right.empty:
                left_right.insert(0, "subspace_name", subspace_name)
                left_right.insert(0, "quality_subset", subset_name)
                left_right_rows.append(left_right)
            hold_move = compare_rank_tables(
                strata_tables.get("hold_family", pd.DataFrame()),
                strata_tables.get("move_family", pd.DataFrame()),
                "hold_family",
                "move_family",
                "hold_move_family",
            )
            if not hold_move.empty:
                hold_move.insert(0, "subspace_name", subspace_name)
                hold_move.insert(0, "quality_subset", subset_name)
                hold_move_rows.append(hold_move)
            compensation_rows.append(
                run_subspace_compensation(
                    subset_name,
                    subspace_name,
                    subset,
                    features,
                    candidates,
                    reliability,
                    reliability_sideaware,
                    config,
                )
            )
            predictor_rows.append(
                run_subspace_predictor(
                    subset_name,
                    subspace_name,
                    subset,
                    features,
                    reliability,
                    reliability_sideaware,
                    config,
                )
            )

    direction_consistency = pd.concat([row for row in direction_rows if not row.empty], ignore_index=True) if direction_rows else pd.DataFrame()
    deviation = pd.concat([row for row in deviation_rows if not row.empty], ignore_index=True) if deviation_rows else pd.DataFrame()
    quality_comparison = pd.concat([row for row in quality_rows if not row.empty], ignore_index=True) if quality_rows else pd.DataFrame()
    candidates = pd.concat([row for row in candidate_rows if not row.empty], ignore_index=True) if candidate_rows else pd.DataFrame()
    side_summary = pd.concat([row for row in side_rows if not row.empty], ignore_index=True) if side_rows else pd.DataFrame()
    left_right = pd.concat(left_right_rows, ignore_index=True) if left_right_rows else pd.DataFrame()
    hold_move = pd.concat(hold_move_rows, ignore_index=True) if hold_move_rows else pd.DataFrame()
    compensation = pd.concat([row for row in compensation_rows if not row.empty], ignore_index=True) if compensation_rows else pd.DataFrame()
    comp_summary, comp_by_task, comp_by_side = summarize_compensation(compensation)
    predictor = pd.concat([row for row in predictor_rows if not row.empty], ignore_index=True) if predictor_rows else pd.DataFrame()
    pred_summary, pred_by_subject, pred_by_task, pred_by_side = summarize_predictor(predictor)
    bootstrap = bootstrap_predictor_ci(predictor, pred_summary, config.seed, config.n_bootstrap)
    clinical_summary, clinical_corr = clinical_anchor_tables(
        annotated,
        direction_consistency,
        comp_summary,
        raw_root,
        warnings,
    )

    paths = {
        "enhanced_state_vectors": output_dir / "subspace_enhanced_state_vectors.csv",
        "feature_inventory": output_dir / "subspace_feature_inventory.csv",
        "subspace_definitions": output_dir / "subspace_definitions.csv",
        "direction_consistency": output_dir / "subspace_direction_consistency.csv",
        "deviation_scores": output_dir / "subspace_deviation_scores.csv",
        "quality_comparison": output_dir / "subspace_quality_comparison.csv",
        "candidate_targets": output_dir / "subspace_candidate_targets.csv",
        "side_aware_summary": output_dir / "subspace_side_aware_summary.csv",
        "hold_move_comparison": output_dir / "subspace_hold_move_comparison.csv",
        "left_right_comparison": output_dir / "subspace_left_right_comparison.csv",
        "compensation_simulation": output_dir / "subspace_compensation_simulation.csv",
        "compensation_summary": output_dir / "subspace_compensation_summary.csv",
        "compensation_by_task_family": output_dir / "subspace_compensation_by_task_family.csv",
        "compensation_by_side": output_dir / "subspace_compensation_by_side.csv",
        "predictor_results": output_dir / "subspace_predictor_results.csv",
        "predictor_summary": output_dir / "subspace_predictor_summary.csv",
        "predictor_by_subject": output_dir / "subspace_predictor_by_subject.csv",
        "predictor_by_task_family": output_dir / "subspace_predictor_by_task_family.csv",
        "predictor_by_side": output_dir / "subspace_predictor_by_side.csv",
        "predictor_bootstrap_ci": output_dir / "subspace_predictor_bootstrap_ci.csv",
        "clinical_anchor_summary": output_dir / "subspace_clinical_anchor_summary.csv",
        "updrs_correlation": output_dir / "subspace_updrs_correlation.csv",
        "report": reports_dir / "subspace_compensation_analysis_report.md",
        "integrated_12": reports_dir / "real_12_subject_integrated_summary.md",
        "integrated_8": reports_dir / "real_8_subject_integrated_summary.md",
    }

    write_table(enhanced, paths["enhanced_state_vectors"], formats=("csv", "parquet"))
    write_table(inventory, paths["feature_inventory"], formats=("csv", "parquet"))
    write_table(definitions, paths["subspace_definitions"], formats=("csv", "parquet"))
    direction_consistency.to_csv(paths["direction_consistency"], index=False)
    deviation.to_csv(paths["deviation_scores"], index=False)
    quality_comparison.to_csv(paths["quality_comparison"], index=False)
    candidates.to_csv(paths["candidate_targets"], index=False)
    side_summary.to_csv(paths["side_aware_summary"], index=False)
    hold_move.to_csv(paths["hold_move_comparison"], index=False)
    left_right.to_csv(paths["left_right_comparison"], index=False)
    write_table(compensation, paths["compensation_simulation"], formats=("csv", "parquet"))
    comp_summary.to_csv(paths["compensation_summary"], index=False)
    comp_by_task.to_csv(paths["compensation_by_task_family"], index=False)
    comp_by_side.to_csv(paths["compensation_by_side"], index=False)
    write_table(predictor, paths["predictor_results"], formats=("csv", "parquet"))
    pred_summary.to_csv(paths["predictor_summary"], index=False)
    pred_by_subject.to_csv(paths["predictor_by_subject"], index=False)
    pred_by_task.to_csv(paths["predictor_by_task_family"], index=False)
    pred_by_side.to_csv(paths["predictor_by_side"], index=False)
    bootstrap.to_csv(paths["predictor_bootstrap_ci"], index=False)
    clinical_summary.to_csv(paths["clinical_anchor_summary"], index=False)
    clinical_corr.to_csv(paths["updrs_correlation"], index=False)

    write_subspace_report(
        paths["report"],
        annotated,
        inventory,
        definitions,
        quality_comparison,
        candidates,
        comp_summary,
        pred_summary,
        bootstrap,
        clinical_summary,
        clinical_corr,
        warnings,
    )
    write_integrated_summary_12(paths["integrated_12"], annotated, quality_comparison, candidates, comp_summary, pred_summary)
    update_legacy_integrated_summary(paths["integrated_8"])
    return paths
