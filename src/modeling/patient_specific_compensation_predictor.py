"""Patient-specific MedOff-to-MedOn compensation-direction prediction.

This module builds paired MedOff/MedOn examples from ds004998 state vectors and
evaluates whether subject-held-out models can predict feature-level directions
toward a dataset-internal MedOn compensated proxy. It is an exploratory
in silico analysis and does not model real interventions, device settings, or
clinical outcomes.
"""

from __future__ import annotations

import json
import os
import warnings as py_warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.side_aware_analysis import add_task_side_columns
from src.pathology_model.build_real_state_vectors import available_real_feature_columns
from src.pathology_model.state_deviation import state_deviation_score

_default_cache = Path.cwd() / ".cache" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(_default_cache))
os.environ.setdefault("MPLBACKEND", "Agg")
_default_cache.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402


ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
MODEL_NAMES = [
    "no_action",
    "random_direction",
    "group_mean_direction",
    "task_family_group_mean_direction",
    "side_group_mean_direction",
    "beta_only_direction",
    "stn_beta_only_direction",
    "gamma_only_direction",
    "reliability_weighted_direction",
    "ridge_regression_predictor",
    "elastic_net_predictor",
]
PAIR_METADATA_COLUMNS = [
    "subject",
    "session",
    "task_original",
    "task_family",
    "side",
    "run_off",
    "run_on",
    "n_medoff_rows",
    "n_medon_rows",
    "pairing_rule",
]
RESULT_COLUMNS = [
    "subject",
    "session",
    "task_original",
    "task_family",
    "side",
    "run_off",
    "run_on",
    "reference_scope",
    "reference_n",
    "model_name",
    "alpha",
    "baseline_deviation",
    "post_shift_deviation",
    "absolute_deviation_reduction",
    "percent_deviation_reduction",
    "improvement_over_no_action",
    "cosine_similarity_to_true_direction",
    "direction_mse",
    "direction_mae",
    "n_nonzero_shifted_features",
    "intervention_magnitude",
    "direction_stability_penalty",
    "net_compensation_score",
    "quality_flag",
    "methodological_note",
]
QUALITY_ORDER = {"good": 0, "acceptable": 1, "caution": 2, "low_quality": 3, "unknown": 4}


@dataclass(frozen=True)
class PredictorInputs:
    """Input tables used by the patient-specific predictor."""

    state_vectors: pd.DataFrame
    deviation_scores: pd.DataFrame
    reliability: pd.DataFrame
    reliability_sideaware: pd.DataFrame
    directions_by_subject: pd.DataFrame
    directions_group: pd.DataFrame
    directions_sideaware_by_subject: pd.DataFrame
    directions_sideaware_group: pd.DataFrame
    quality: pd.DataFrame
    side_summary: pd.DataFrame
    left_right: pd.DataFrame
    hold_move: pd.DataFrame
    warnings: list[str]


def read_optional_table(path: str | Path, warnings: list[str], dtype: dict[str, type | str] | None = None) -> pd.DataFrame:
    """Read an optional CSV table and record a warning if unavailable."""

    path = Path(path)
    if not path.exists():
        warnings.append(f"Missing optional input: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype)
    except Exception as exc:  # noqa: BLE001 - continue with available inputs.
        warnings.append(f"Could not read {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def read_state_vectors(path: str | Path, warnings: list[str]) -> pd.DataFrame:
    """Load state vectors and ensure task side metadata are present."""

    path = Path(path)
    if not path.exists():
        warnings.append(f"Missing required state vectors: {path}")
        return pd.DataFrame()
    data = pd.read_csv(path, dtype={"run": str})
    for column in ["subject", "session", "condition", "medication", "task", "run"]:
        if column in data:
            data[column] = data[column].fillna("unknown").astype(str)
    if "task_original" not in data or "task_family" not in data or "side" not in data:
        data = add_task_side_columns(data, "task")
    else:
        data["task_original"] = data["task_original"].fillna(data.get("task", "unknown")).astype(str)
        data["task_family"] = data["task_family"].fillna("unknown").astype(str)
        data["side"] = data["side"].fillna("unknown").astype(str)
    return data


def load_predictor_inputs(
    state_vectors_path: str | Path,
    table_dir: str | Path = "outputs/tables",
) -> PredictorInputs:
    """Load all requested predictor inputs."""

    warnings: list[str] = []
    table_dir = Path(table_dir)
    dtype = {"run": str}
    return PredictorInputs(
        state_vectors=read_state_vectors(state_vectors_path, warnings),
        deviation_scores=read_optional_table(table_dir / "real_deviation_scores.csv", warnings, dtype=dtype),
        reliability=read_optional_table(table_dir / "real_target_reliability.csv", warnings),
        reliability_sideaware=read_optional_table(table_dir / "real_target_reliability_sideaware.csv", warnings),
        directions_by_subject=read_optional_table(
            table_dir / "real_compensation_directions_by_subject.csv",
            warnings,
            dtype=dtype,
        ),
        directions_group=read_optional_table(table_dir / "real_compensation_directions_group.csv", warnings),
        directions_sideaware_by_subject=read_optional_table(
            table_dir / "real_compensation_directions_sideaware_by_subject.csv",
            warnings,
            dtype=dtype,
        ),
        directions_sideaware_group=read_optional_table(
            table_dir / "real_compensation_directions_sideaware_group.csv",
            warnings,
        ),
        quality=read_optional_table(table_dir / "real_recording_quality.csv", warnings, dtype=dtype),
        side_summary=read_optional_table(table_dir / "real_side_aware_summary.csv", warnings),
        left_right=read_optional_table(table_dir / "real_left_right_target_comparison.csv", warnings),
        hold_move=read_optional_table(table_dir / "real_hold_move_family_comparison.csv", warnings),
        warnings=warnings,
    )


def _join_unique(values: pd.Series) -> str:
    """Join unique non-empty values with a semicolon."""

    unique = [str(value) for value in values.dropna().astype(str).unique() if str(value)]
    return ";".join(unique) if unique else "unknown"


def _feature_json(values: pd.Series, feature_columns: list[str]) -> str:
    """Serialize a feature vector into a stable JSON mapping."""

    return json.dumps({feature: float(values[feature]) for feature in feature_columns}, sort_keys=True)


def build_patient_specific_pairs(
    state_vectors: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build paired MedOff/MedOn examples by subject and original task.

    If multiple rows exist for a subject/task/medication, the current state
    vectors are averaged and the contributing runs are preserved as metadata.
    """

    pair_rows: list[dict[str, object]] = []
    log_rows: list[dict[str, object]] = []
    if state_vectors.empty or not feature_columns:
        return pd.DataFrame(), pd.DataFrame(
            [
                {
                    "subject": "",
                    "task_original": "",
                    "pairing_status": "skipped",
                    "reason": "empty_state_vectors_or_no_features",
                }
            ]
        )

    required_columns = {"subject", "medication", "task_original"}
    missing = required_columns - set(state_vectors.columns)
    if missing:
        return pd.DataFrame(), pd.DataFrame(
            [
                {
                    "subject": "",
                    "task_original": "",
                    "pairing_status": "skipped",
                    "reason": f"missing_required_columns:{';'.join(sorted(missing))}",
                }
            ]
        )

    for (subject, task_original), subset in state_vectors.groupby(["subject", "task_original"], dropna=False):
        medication = subset["medication"].fillna("").astype(str).str.lower()
        off = subset[medication.eq("off")]
        on = subset[medication.eq("on")]
        session = _join_unique(subset.get("session", pd.Series("unknown", index=subset.index)))
        if off.empty or on.empty:
            log_rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "task_original": task_original,
                    "task_family": _join_unique(subset.get("task_family", pd.Series("unknown", index=subset.index))),
                    "side": _join_unique(subset.get("side", pd.Series("unknown", index=subset.index))),
                    "pairing_status": "skipped",
                    "reason": "missing_medoff" if off.empty else "missing_medon",
                    "n_medoff_rows": int(len(off)),
                    "n_medon_rows": int(len(on)),
                }
            )
            continue
        off_values = off[feature_columns].mean(numeric_only=True)
        on_values = on[feature_columns].mean(numeric_only=True)
        direction = on_values - off_values
        first = subset.iloc[0]
        row: dict[str, object] = {
            "subject": subject,
            "session": session,
            "task_original": task_original,
            "task_family": str(first.get("task_family", "unknown")),
            "side": str(first.get("side", "unknown")),
            "run_off": _join_unique(off.get("run", pd.Series("unknown", index=off.index))),
            "run_on": _join_unique(on.get("run", pd.Series("unknown", index=on.index))),
            "n_medoff_rows": int(len(off)),
            "n_medon_rows": int(len(on)),
            "pairing_rule": "mean_state_vector_by_subject_task_medication",
            "x_medoff_features_json": _feature_json(off_values, feature_columns),
            "y_true_direction_json": _feature_json(direction, feature_columns),
            "medon_reference_features_json": _feature_json(on_values, feature_columns),
            "true_direction_norm": float(np.linalg.norm(direction.to_numpy(dtype=float))),
        }
        for feature in feature_columns:
            row[f"x_{feature}"] = float(off_values[feature])
            row[f"y_{feature}"] = float(direction[feature])
            row[f"medon_{feature}"] = float(on_values[feature])
        pair_rows.append(row)
        log_rows.append(
            {
                "subject": subject,
                "session": session,
                "task_original": task_original,
                "task_family": str(first.get("task_family", "unknown")),
                "side": str(first.get("side", "unknown")),
                "pairing_status": "paired",
                "reason": "paired_by_subject_task_original",
                "n_medoff_rows": int(len(off)),
                "n_medon_rows": int(len(on)),
            }
        )

    pairs = pd.DataFrame(pair_rows)
    log = pd.DataFrame(log_rows)
    if not pairs.empty:
        ordered = [
            *PAIR_METADATA_COLUMNS,
            "x_medoff_features_json",
            "y_true_direction_json",
            "medon_reference_features_json",
            "true_direction_norm",
        ]
        feature_ordered = [f"x_{feature}" for feature in feature_columns]
        feature_ordered += [f"y_{feature}" for feature in feature_columns]
        feature_ordered += [f"medon_{feature}" for feature in feature_columns]
        pairs = pairs[[column for column in ordered + feature_ordered if column in pairs.columns]]
    return pairs, log


def _matrix(pairs: pd.DataFrame, feature_columns: list[str], prefix: str) -> np.ndarray:
    """Return a feature matrix from wide prefixed columns."""

    columns = [f"{prefix}{feature}" for feature in feature_columns]
    return pairs[columns].to_numpy(dtype=float)


def x_matrix(pairs: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Return MedOff input features."""

    return _matrix(pairs, feature_columns, "x_")


def y_matrix(pairs: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Return true MedOn-minus-MedOff directions."""

    return _matrix(pairs, feature_columns, "y_")


def medon_matrix(pairs: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Return MedOn reference features."""

    return _matrix(pairs, feature_columns, "medon_")


def feature_scale(train_pairs: pd.DataFrame, feature_columns: list[str], min_scale: float = 0.1) -> np.ndarray:
    """Compute train-fold feature scale from MedOff and MedOn vectors only."""

    if train_pairs.empty:
        return np.ones(len(feature_columns), dtype=float)
    values = np.vstack([x_matrix(train_pairs, feature_columns), medon_matrix(train_pairs, feature_columns)])
    ddof = 1 if values.shape[0] > 1 else 0
    scale = np.nanstd(values, axis=0, ddof=ddof)
    scale = np.nan_to_num(scale, nan=min_scale, posinf=min_scale, neginf=min_scale)
    return np.where(scale < min_scale, min_scale, scale)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with a stable zero-vector convention."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return float("nan")
    a = a[finite]
    b = b[finite]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def direction_consistency(y_train: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Estimate train-fold sign consistency for each feature direction."""

    if y_train.size == 0:
        return np.zeros(0, dtype=float)
    signs = np.where(np.abs(y_train) <= epsilon, 0, np.sign(y_train))
    consistency = []
    for column in range(signs.shape[1]):
        values, counts = np.unique(signs[:, column], return_counts=True)
        if len(values) == 0:
            consistency.append(0.0)
        else:
            consistency.append(float(counts.max() / max(1, signs.shape[0])))
    return np.asarray(consistency, dtype=float)


def feature_mask(feature_columns: list[str], mask_type: str) -> np.ndarray:
    """Feature mask for beta, STN beta, and gamma baselines."""

    text = np.asarray([feature.lower() for feature in feature_columns])
    beta = np.asarray(["low_beta" in feature or "high_beta" in feature or "broad_beta" in feature for feature in text])
    stn = np.asarray(["stn" in feature for feature in text])
    gamma = np.asarray(["gamma" in feature for feature in text])
    if mask_type == "beta":
        return beta
    if mask_type == "stn_beta":
        return beta & stn
    if mask_type == "gamma":
        return gamma
    return np.ones(len(feature_columns), dtype=bool)


def reliability_weights(
    feature_columns: list[str],
    reliability: pd.DataFrame,
    reliability_sideaware: pd.DataFrame,
) -> np.ndarray:
    """Build feature-prior weights from available reliability tables."""

    scores = pd.Series(1.0, index=feature_columns, dtype=float)
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
    if not tables:
        return scores.to_numpy(dtype=float)
    combined = pd.concat(tables, ignore_index=True)
    combined = combined.dropna(subset=["feature", "_score"])
    if combined.empty:
        return scores.to_numpy(dtype=float)
    mean_scores = combined.groupby("feature")["_score"].mean()
    for feature in feature_columns:
        if feature in mean_scores:
            scores.loc[feature] = float(mean_scores.loc[feature])
    maximum = float(scores.max()) if np.isfinite(scores.max()) and scores.max() > 0 else 1.0
    return (scores / maximum).clip(0.05, 1.0).to_numpy(dtype=float)


def _safe_pipeline_predict(model, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Fit a sklearn pipeline and return a single prediction."""

    pipeline = make_pipeline(StandardScaler(), model)
    with py_warnings.catch_warnings():
        py_warnings.simplefilter("ignore", category=ConvergenceWarning)
        pipeline.fit(x_train, y_train)
    return np.asarray(pipeline.predict(x_test.reshape(1, -1))[0], dtype=float)


def predict_direction(
    model_name: str,
    train_pairs: pd.DataFrame,
    test_row: pd.Series,
    feature_columns: list[str],
    external_weights: np.ndarray,
    rng: np.random.Generator,
    warnings: list[str],
) -> tuple[np.ndarray, str]:
    """Predict a MedOff-to-MedOn feature direction for one held-out state."""

    y_train = y_matrix(train_pairs, feature_columns)
    x_train = x_matrix(train_pairs, feature_columns)
    x_test = test_row[[f"x_{feature}" for feature in feature_columns]].to_numpy(dtype=float)
    group_mean = np.nanmean(y_train, axis=0) if len(y_train) else np.zeros(len(feature_columns), dtype=float)
    consistency = direction_consistency(y_train)

    if model_name == "no_action":
        return np.zeros(len(feature_columns), dtype=float), "zero_vector_baseline"
    if model_name == "random_direction":
        random = rng.normal(size=len(feature_columns))
        random_norm = float(np.linalg.norm(random))
        train_norm = float(np.mean(np.linalg.norm(y_train, axis=1))) if len(y_train) else 1.0
        if random_norm == 0.0:
            return np.zeros(len(feature_columns), dtype=float), "seeded_random_zero_fallback"
        return random * (train_norm / random_norm), "seeded_random_direction_scaled_to_train_direction_norm"
    if model_name == "group_mean_direction":
        return group_mean, "train_subject_group_mean_direction"
    if model_name == "task_family_group_mean_direction":
        family = str(test_row.get("task_family", "unknown"))
        subset = train_pairs[train_pairs["task_family"].astype(str).eq(family)]
        if subset.empty:
            warnings.append(f"task_family_group_mean fallback to group mean for {family}.")
            return group_mean, "fallback_group_mean_no_matching_train_task_family"
        return np.nanmean(y_matrix(subset, feature_columns), axis=0), "train_task_family_group_mean_direction"
    if model_name == "side_group_mean_direction":
        side = str(test_row.get("side", "unknown"))
        subset = train_pairs[train_pairs["side"].astype(str).eq(side)]
        if subset["subject"].nunique() < 2:
            warnings.append(f"side_group_mean fallback to group mean for side={side}; fewer than two train subjects.")
            return group_mean, "fallback_group_mean_underpowered_side"
        return np.nanmean(y_matrix(subset, feature_columns), axis=0), "train_side_group_mean_direction"
    if model_name == "beta_only_direction":
        direction = np.zeros(len(feature_columns), dtype=float)
        mask = feature_mask(feature_columns, "beta")
        direction[mask] = group_mean[mask]
        return direction, "train_group_mean_beta_features_only"
    if model_name == "stn_beta_only_direction":
        direction = np.zeros(len(feature_columns), dtype=float)
        mask = feature_mask(feature_columns, "stn_beta")
        direction[mask] = group_mean[mask]
        return direction, "train_group_mean_stn_beta_features_only"
    if model_name == "gamma_only_direction":
        direction = np.zeros(len(feature_columns), dtype=float)
        mask = feature_mask(feature_columns, "gamma")
        direction[mask] = group_mean[mask]
        return direction, "train_group_mean_gamma_features_only"
    if model_name == "reliability_weighted_direction":
        if len(consistency) != len(group_mean):
            consistency = np.ones(len(group_mean), dtype=float)
        direction = group_mean * external_weights * consistency
        return direction, "train_group_mean_weighted_by_existing_reliability_prior_and_train_consistency"
    if model_name == "ridge_regression_predictor":
        if len(train_pairs) < 3:
            warnings.append("ridge_regression_predictor fallback to group mean; fewer than three training pairs.")
            return group_mean, "fallback_group_mean_insufficient_train_pairs"
        try:
            return _safe_pipeline_predict(Ridge(alpha=1.0), x_train, y_train, x_test), "ridge_regression_train_subjects_only"
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"ridge_regression_predictor fallback to group mean: {type(exc).__name__}: {exc}")
            return group_mean, "fallback_group_mean_ridge_error"
    if model_name == "elastic_net_predictor":
        if len(train_pairs) < 6:
            warnings.append("elastic_net_predictor fallback to group mean; fewer than six training pairs.")
            return group_mean, "fallback_group_mean_insufficient_train_pairs"
        try:
            model = MultiTaskElasticNet(alpha=0.01, l1_ratio=0.25, max_iter=20000, random_state=7)
            return _safe_pipeline_predict(model, x_train, y_train, x_test), "elastic_net_train_subjects_only"
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"elastic_net_predictor fallback to group mean: {type(exc).__name__}: {exc}")
            return group_mean, "fallback_group_mean_elastic_net_error"
    raise ValueError(f"Unknown model_name: {model_name}")


def reference_states(
    test_row: pd.Series,
    train_pairs: pd.DataFrame,
    feature_columns: list[str],
) -> list[dict[str, object]]:
    """Return subject-specific and train-fold group proxy references."""

    subject_reference = test_row[[f"medon_{feature}" for feature in feature_columns]].to_numpy(dtype=float)
    refs = [
        {
            "reference_scope": "subject_medon_proxy",
            "reference_n": 1,
            "state": subject_reference,
            "note": "held_out_subject_medon_used_for_evaluation_only",
        }
    ]
    family = str(test_row.get("task_family", "unknown"))
    family_subset = train_pairs[train_pairs["task_family"].astype(str).eq(family)]
    if family_subset.empty:
        family_subset = train_pairs
        family_note = "fallback_all_train_medon_proxy"
    else:
        family_note = "train_task_family_medon_proxy"
    if not family_subset.empty:
        refs.append(
            {
                "reference_scope": "group_task_family_medon_proxy",
                "reference_n": int(len(family_subset)),
                "state": np.nanmean(medon_matrix(family_subset, feature_columns), axis=0),
                "note": family_note,
            }
        )
    side = str(test_row.get("side", "unknown"))
    side_task = train_pairs[
        train_pairs["side"].astype(str).eq(side) & train_pairs["task_family"].astype(str).eq(family)
    ]
    if not side_task.empty:
        refs.append(
            {
                "reference_scope": "side_task_medon_proxy",
                "reference_n": int(len(side_task)),
                "state": np.nanmean(medon_matrix(side_task, feature_columns), axis=0),
                "note": "train_side_task_family_medon_proxy",
            }
        )
    return refs


def worst_quality_flag(flags: pd.Series) -> str:
    """Return the most cautious quality flag from matching rows."""

    values = [str(flag) for flag in flags.dropna().astype(str) if str(flag)]
    if not values:
        return "unknown"
    return sorted(values, key=lambda flag: QUALITY_ORDER.get(flag, 4), reverse=True)[0]


def quality_lookup(quality: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    """Build a MedOff quality lookup by subject, task, and run."""

    if quality.empty:
        return {}
    data = quality.copy()
    for column in ["subject", "task_original", "task", "run", "medication", "quality_flag"]:
        if column not in data:
            data[column] = "unknown"
        data[column] = data[column].fillna("unknown").astype(str)
    if "task_original" not in quality.columns or data["task_original"].eq("unknown").all():
        data = add_task_side_columns(data, "task")
    off = data[data["medication"].str.lower().eq("off")]
    lookup: dict[tuple[str, str, str], str] = {}
    for (subject, task, run), subset in off.groupby(["subject", "task_original", "run"], dropna=False):
        lookup[(str(subject), str(task), str(run))] = worst_quality_flag(subset["quality_flag"])
    for (subject, task), subset in off.groupby(["subject", "task_original"], dropna=False):
        lookup[(str(subject), str(task), "*")] = worst_quality_flag(subset["quality_flag"])
    return lookup


def _quality_for_pair(row: pd.Series, lookup: dict[tuple[str, str, str], str]) -> str:
    """Return quality flag for a paired MedOff row."""

    subject = str(row.get("subject", "unknown"))
    task = str(row.get("task_original", "unknown"))
    run = str(row.get("run_off", "unknown"))
    if (subject, task, run) in lookup:
        return lookup[(subject, task, run)]
    return lookup.get((subject, task, "*"), "unknown")


def run_loso_predictions(
    pairs: pd.DataFrame,
    feature_columns: list[str],
    reliability: pd.DataFrame,
    reliability_sideaware: pd.DataFrame,
    quality: pd.DataFrame,
    seed: int = 42,
    alpha_values: list[float] | None = None,
    lambda_magnitude: float = 0.01,
    lambda_complexity: float = 0.005,
    lambda_instability: float = 0.05,
) -> tuple[pd.DataFrame, list[str]]:
    """Run leave-one-subject-out prediction and alpha-shift evaluation."""

    alpha_values = alpha_values or ALPHA_VALUES
    warnings: list[str] = []
    if pairs.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS), ["No paired MedOff-MedOn examples were available."]
    subjects = sorted(pairs["subject"].dropna().astype(str).unique())
    external_weights = reliability_weights(feature_columns, reliability, reliability_sideaware)
    quality_flags = quality_lookup(quality)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for held_subject in subjects:
        train = pairs[~pairs["subject"].astype(str).eq(held_subject)].copy()
        test = pairs[pairs["subject"].astype(str).eq(held_subject)].copy()
        if train.empty or test.empty:
            warnings.append(f"Skipping held-out subject {held_subject}; train or test pairs are empty.")
            continue
        scale = feature_scale(train, feature_columns)
        train_consistency = direction_consistency(y_matrix(train, feature_columns))
        for _, test_row in test.iterrows():
            x_state = test_row[[f"x_{feature}" for feature in feature_columns]].to_numpy(dtype=float)
            true_direction = test_row[[f"y_{feature}" for feature in feature_columns]].to_numpy(dtype=float)
            refs = reference_states(test_row, train, feature_columns)
            quality_flag = _quality_for_pair(test_row, quality_flags)
            predictions: dict[str, tuple[np.ndarray, str]] = {}
            for model_name in MODEL_NAMES:
                predictions[model_name] = predict_direction(
                    model_name,
                    train,
                    test_row,
                    feature_columns,
                    external_weights,
                    rng,
                    warnings,
                )
            for ref in refs:
                reference_state = np.asarray(ref["state"], dtype=float)
                baseline = state_deviation_score(x_state, reference_state, scale)
                for model_name, (predicted_direction, model_note) in predictions.items():
                    cosine = cosine_similarity(predicted_direction, true_direction)
                    direction_error = predicted_direction - true_direction
                    direction_mse = float(np.nanmean(direction_error**2))
                    direction_mae = float(np.nanmean(np.abs(direction_error)))
                    for alpha in alpha_values:
                        applied = float(alpha) * predicted_direction
                        shifted = x_state + applied
                        post = state_deviation_score(shifted, reference_state, scale)
                        reduction = baseline - post
                        percent = 100.0 * reduction / baseline if np.isfinite(baseline) and baseline > 0 else np.nan
                        n_nonzero = int(np.sum(np.abs(applied) > 1e-12))
                        safe_scale = np.where(scale <= 0, 1.0, scale)
                        magnitude = float(np.sqrt(np.mean((applied / safe_scale) ** 2))) if n_nonzero else 0.0
                        if n_nonzero and len(train_consistency) == len(applied):
                            active_consistency = train_consistency[np.abs(applied) > 1e-12]
                            instability = float(max(0.0, 1.0 - np.nanmean(active_consistency)))
                        else:
                            instability = 0.0
                        net = (
                            reduction
                            - lambda_magnitude * magnitude
                            - lambda_complexity * n_nonzero
                            - lambda_instability * instability
                        )
                        rows.append(
                            {
                                "subject": test_row["subject"],
                                "session": test_row["session"],
                                "task_original": test_row["task_original"],
                                "task_family": test_row["task_family"],
                                "side": test_row["side"],
                                "run_off": test_row["run_off"],
                                "run_on": test_row["run_on"],
                                "reference_scope": ref["reference_scope"],
                                "reference_n": int(ref["reference_n"]),
                                "model_name": model_name,
                                "alpha": float(alpha),
                                "baseline_deviation": baseline,
                                "post_shift_deviation": post,
                                "absolute_deviation_reduction": reduction,
                                "percent_deviation_reduction": percent,
                                "improvement_over_no_action": reduction,
                                "cosine_similarity_to_true_direction": cosine,
                                "direction_mse": direction_mse,
                                "direction_mae": direction_mae,
                                "n_nonzero_shifted_features": n_nonzero,
                                "intervention_magnitude": magnitude,
                                "direction_stability_penalty": instability,
                                "net_compensation_score": net,
                                "quality_flag": quality_flag,
                                "methodological_note": (
                                    f"{model_note}; {ref['note']}; in silico feature-level shift toward "
                                    "dataset-internal MedOn proxy"
                                ),
                            }
                        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS), warnings


def _best_alpha_rows(results: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Choose best alpha by mean deviation reduction for each group."""

    if results.empty:
        return pd.DataFrame()
    primary = results[results["reference_scope"].eq("subject_medon_proxy")].copy()
    if primary.empty:
        primary = results.copy()
    grouped = primary.groupby([*group_columns, "alpha"], dropna=False).agg(
        n_rows=("subject", "size"),
        n_held_out_subjects=("subject", "nunique"),
        n_held_out_states=("task_original", "size"),
        mean_cosine_similarity=("cosine_similarity_to_true_direction", "mean"),
        mean_direction_mse=("direction_mse", "mean"),
        mean_direction_mae=("direction_mae", "mean"),
        mean_deviation_reduction=("absolute_deviation_reduction", "mean"),
        median_deviation_reduction=("absolute_deviation_reduction", "median"),
        improvement_rate=("absolute_deviation_reduction", lambda x: float((x > 0).mean())),
        mean_net_compensation_score=("net_compensation_score", "mean"),
        mean_intervention_magnitude=("intervention_magnitude", "mean"),
    ).reset_index()
    if grouped.empty:
        return grouped
    best_parts = []
    for _, table in grouped.groupby(group_columns, dropna=False):
        table = table.copy()
        model_name = str(table["model_name"].iloc[0]) if "model_name" in table else ""
        if model_name and model_name != "no_action":
            active = table[table["alpha"] > 0].copy()
            if not active.empty:
                table = active
        elif model_name == "no_action":
            zero = table[np.isclose(table["alpha"].astype(float), 0.0)].copy()
            if not zero.empty:
                table = zero
        table["_alpha_sort"] = table["alpha"].where(table["alpha"] > 0, -1.0)
        best_parts.append(
            table.sort_values(
                ["mean_deviation_reduction", "mean_net_compensation_score", "_alpha_sort"],
                ascending=[False, False, False],
            ).iloc[[0]]
        )
    best = pd.concat(best_parts, ignore_index=True).drop(columns=["_alpha_sort"]) if best_parts else pd.DataFrame()
    best = best.rename(columns={"alpha": "best_alpha"})
    return best.reset_index(drop=True)


def summarize_models(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize model performance at each model's best reduction alpha."""

    if results.empty:
        return pd.DataFrame()
    best = _best_alpha_rows(results, ["model_name"])
    if best.empty:
        return best
    primary = results[results["reference_scope"].eq("subject_medon_proxy")].copy()
    net_alpha = primary.groupby(["model_name", "alpha"], as_index=False)["net_compensation_score"].mean()
    net_parts = []
    for model_name, table in net_alpha.groupby("model_name", dropna=False):
        table = table.copy()
        if str(model_name) != "no_action":
            active = table[table["alpha"] > 0].copy()
            if not active.empty:
                table = active
        else:
            zero = table[np.isclose(table["alpha"].astype(float), 0.0)].copy()
            if not zero.empty:
                table = zero
        net_parts.append(table.sort_values("net_compensation_score", ascending=False).iloc[[0]])
    net_best = pd.concat(net_parts, ignore_index=True) if net_parts else pd.DataFrame()
    net_best = net_best.rename(
        columns={"alpha": "best_net_alpha", "net_compensation_score": "best_net_compensation_score"}
    )
    summary = best.merge(net_best, on="model_name", how="left")
    comparison = summary.set_index("model_name")

    def _beats(model: str, baseline: str, metric: str = "mean_deviation_reduction") -> bool:
        if model not in comparison.index or baseline not in comparison.index:
            return False
        return bool(float(comparison.loc[model, metric]) > float(comparison.loc[baseline, metric]))

    summary["beats_group_mean_direction"] = [
        _beats(model, "group_mean_direction") for model in summary["model_name"].astype(str)
    ]
    summary["beats_random_direction"] = [
        _beats(model, "random_direction") for model in summary["model_name"].astype(str)
    ]
    summary["beats_no_action"] = [_beats(model, "no_action") for model in summary["model_name"].astype(str)]
    summary["methodological_note"] = "leave-one-subject-out subject_medon_proxy summary; MedOn is not a healthy state"
    return summary.sort_values(
        ["mean_deviation_reduction", "mean_net_compensation_score"],
        ascending=[False, False],
    ).reset_index(drop=True)


def summarize_by_group(results: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Summarize best-alpha model performance by subject, task family, or side."""

    if results.empty or group_column not in results:
        return pd.DataFrame()
    summary = _best_alpha_rows(results, [group_column, "model_name"])
    if summary.empty:
        return summary
    return summary.sort_values(
        [group_column, "mean_deviation_reduction", "mean_net_compensation_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def bootstrap_ci(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    seed: int = 42,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    """Bootstrap simple 95% confidence intervals over held-out subjects."""

    if results.empty or summary.empty:
        return pd.DataFrame()
    primary = results[results["reference_scope"].eq("subject_medon_proxy")].copy()
    rng = np.random.default_rng(seed)
    rows = []
    units = sorted(primary["subject"].dropna().astype(str).unique())
    if len(units) >= 4:
        unit_column = "subject"
        note = "bootstrap_over_held_out_subjects_exploratory"
    else:
        unit_column = "_state_id"
        primary["_state_id"] = primary["subject"].astype(str) + "::" + primary["task_original"].astype(str)
        units = sorted(primary["_state_id"].unique())
        note = "bootstrap_over_held_out_states_due_to_small_subject_count_exploratory"
    metrics = {
        "mean_deviation_reduction": "absolute_deviation_reduction",
        "mean_net_compensation_score": "net_compensation_score",
        "mean_cosine_similarity": "cosine_similarity_to_true_direction",
    }
    for _, row in summary.iterrows():
        model = str(row["model_name"])
        alpha = float(row["best_alpha"])
        data = primary[primary["model_name"].eq(model) & np.isclose(primary["alpha"].astype(float), alpha)]
        if data.empty:
            continue
        for metric_name, source_column in metrics.items():
            observed = float(data[source_column].mean())
            boot_values = []
            for _ in range(n_bootstrap):
                sampled_units = rng.choice(units, size=len(units), replace=True)
                sampled = pd.concat([data[data[unit_column].eq(unit)] for unit in sampled_units], ignore_index=True)
                if sampled.empty:
                    continue
                boot_values.append(float(sampled[source_column].mean()))
            if boot_values:
                lower, upper = np.percentile(boot_values, [2.5, 97.5])
            else:
                lower, upper = np.nan, np.nan
            rows.append(
                {
                    "model_name": model,
                    "best_alpha": alpha,
                    "metric": metric_name,
                    "mean": observed,
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "bootstrap_unit": unit_column,
                    "n_units": int(len(units)),
                    "n_bootstrap": int(n_bootstrap),
                    "methodological_note": note,
                }
            )
    return pd.DataFrame(rows)


def _bar_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    output_path: str | Path,
    color: str = "#437c90",
) -> None:
    """Write a simple bar plot if data are available."""

    if data.empty or x not in data or y not in data:
        return
    plot_data = data.copy()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(plot_data[x].astype(str), plot_data[y].astype(float), color=color)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_grouped_summary(data: pd.DataFrame, group_column: str, output_path: str | Path, title: str) -> None:
    """Plot best model per group."""

    if data.empty or group_column not in data:
        return
    top = data.sort_values(
        [group_column, "mean_deviation_reduction", "mean_net_compensation_score"],
        ascending=[True, False, False],
    ).groupby(group_column, as_index=False).head(1)
    labels = top[group_column].astype(str) + "\n" + top["model_name"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(labels, top["mean_deviation_reduction"].astype(float), color="#8a6f3d")
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("Mean proxy deviation reduction")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _fmt_float(value: object, digits: int = 4) -> str:
    """Format floats for markdown."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _best_row(summary: pd.DataFrame, metric: str) -> pd.Series | None:
    """Return row with highest metric."""

    if summary.empty or metric not in summary:
        return None
    data = summary.dropna(subset=[metric])
    if data.empty:
        return None
    return data.sort_values(metric, ascending=False).iloc[0]


def _best_active_row(summary: pd.DataFrame, metric: str) -> pd.Series | None:
    """Return the best non-no-action row for a metric."""

    if summary.empty or "model_name" not in summary:
        return None
    return _best_row(summary[~summary["model_name"].astype(str).eq("no_action")], metric)


def write_report(
    path: str | Path,
    inputs: PredictorInputs,
    pairs: pd.DataFrame,
    pairing_log: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    by_task_family: pd.DataFrame,
    by_side: pd.DataFrame,
    bootstrap: pd.DataFrame,
    warnings: list[str],
) -> None:
    """Write patient-specific predictor report."""

    n_subjects = int(pairs["subject"].nunique()) if not pairs.empty else 0
    n_pairs = int(len(pairs))
    tasks = sorted(pairs["task_original"].dropna().astype(str).unique()) if not pairs.empty else []
    sides = sorted(pairs["side"].dropna().astype(str).unique()) if not pairs.empty else []
    feature_count = len(available_real_feature_columns(inputs.state_vectors)) if not inputs.state_vectors.empty else 0
    best_reduction = _best_row(summary, "mean_deviation_reduction")
    best_active_reduction = _best_active_row(summary, "mean_deviation_reduction")
    best_net = _best_row(summary, "best_net_compensation_score")
    best_active_net = _best_active_row(summary, "best_net_compensation_score")
    best_cosine = _best_row(summary, "mean_cosine_similarity")
    learned = summary[summary["model_name"].isin(["ridge_regression_predictor", "elastic_net_predictor"])]
    group = summary[summary["model_name"].eq("group_mean_direction")]
    random = summary[summary["model_name"].eq("random_direction")]
    no_action = summary[summary["model_name"].eq("no_action")]

    def _compare_against(table: pd.DataFrame, baseline: pd.DataFrame, metric: str) -> str:
        if table.empty or baseline.empty:
            return "not_available"
        return "yes" if float(table[metric].max()) > float(baseline[metric].max()) else "no"

    lines = [
        "# Real Patient-Specific Compensation-Direction Predictor Report",
        "",
        "## Purpose",
        "",
        "This module evaluates patient-specific compensation-direction prediction from MedOff state vectors toward paired dataset-internal MedOn proxies. It is an exploratory in silico feature-level prediction module.",
        "",
        "## Dataset Subset",
        "",
        f"- subjects: {n_subjects}",
        f"- paired_medoff_medon_examples: {n_pairs}",
        f"- tasks: {';'.join(tasks) if tasks else 'not_available'}",
        f"- sides: {';'.join(sides) if sides else 'not_available'}",
        f"- state_features_used: {feature_count}",
        f"- pairing_log_rows: {len(pairing_log)}",
        "",
        "## Methods",
        "",
        "- Paired examples are built by subject and task_original.",
        "- true_direction = MedOn_features - MedOff_features.",
        "- Leave-one-subject-out validation is used for every model and baseline.",
        "- Group means, task-family means, side means, and learned models use training subjects only.",
        "- Subject-specific MedOn rows are used only as held-out evaluation references.",
        "- Alpha shift simulation tests alpha values 0.0, 0.25, 0.5, 0.75, and 1.0.",
        "- Net score formula: deviation_reduction - lambda_magnitude * intervention_magnitude - lambda_complexity * n_nonzero_shifted_features - lambda_instability * direction_stability_penalty.",
        "",
    ]
    if warnings or inputs.warnings:
        lines.extend(["## Warnings", ""])
        for warning in dict.fromkeys([*inputs.warnings, *warnings]):
            lines.append(f"- {warning}")
        lines.append("")

    lines.extend(["## Main Results", ""])
    if best_reduction is not None:
        lines.append(
            f"- best_model_by_deviation_reduction: {best_reduction['model_name']} "
            f"(alpha={best_reduction['best_alpha']}, mean_reduction={_fmt_float(best_reduction['mean_deviation_reduction'])})"
        )
    if best_active_reduction is not None:
        lines.append(
            f"- best_active_model_by_deviation_reduction: {best_active_reduction['model_name']} "
            f"(alpha={best_active_reduction['best_alpha']}, "
            f"mean_reduction={_fmt_float(best_active_reduction['mean_deviation_reduction'])})"
        )
    if best_net is not None:
        lines.append(
            f"- best_model_by_net_score: {best_net['model_name']} "
            f"(best_net_alpha={best_net['best_net_alpha']}, net_score={_fmt_float(best_net['best_net_compensation_score'])})"
        )
    if best_active_net is not None:
        lines.append(
            f"- best_active_model_by_net_score: {best_active_net['model_name']} "
            f"(best_net_alpha={best_active_net['best_net_alpha']}, "
            f"net_score={_fmt_float(best_active_net['best_net_compensation_score'])})"
        )
    if best_cosine is not None:
        lines.append(
            f"- best_model_by_cosine_similarity: {best_cosine['model_name']} "
            f"(mean_cosine={_fmt_float(best_cosine['mean_cosine_similarity'])})"
        )
    lines.append(
        f"- learned_models_beat_group_mean_by_reduction: {_compare_against(learned, group, 'mean_deviation_reduction')}"
    )
    lines.append(
        f"- learned_models_beat_random_by_reduction: {_compare_against(learned, random, 'mean_deviation_reduction')}"
    )
    lines.append(
        f"- learned_models_beat_no_action_by_reduction: {_compare_against(learned, no_action, 'mean_deviation_reduction')}"
    )
    for model in ["beta_only_direction", "stn_beta_only_direction", "gamma_only_direction", "group_mean_direction"]:
        rows = summary[summary["model_name"].eq(model)]
        if not rows.empty:
            row = rows.iloc[0]
            lines.append(
                f"- {model}: mean_reduction={_fmt_float(row['mean_deviation_reduction'])}, "
                f"mean_net={_fmt_float(row['mean_net_compensation_score'])}, "
                f"mean_cosine={_fmt_float(row['mean_cosine_similarity'])}"
            )
    lines.extend(["", "## Side-Aware Findings", ""])
    if by_side.empty:
        lines.append("Side-specific summaries were unavailable.")
    else:
        for side, table in by_side.groupby("side", dropna=False):
            top = table.sort_values("mean_deviation_reduction", ascending=False).iloc[0]
            n_side_subjects = pairs[pairs["side"].astype(str).eq(str(side))]["subject"].nunique()
            note = "descriptive_underpowered" if n_side_subjects < 2 else "exploratory_side_summary"
            lines.append(
                f"- side={side}: top_model={top['model_name']}, mean_reduction={_fmt_float(top['mean_deviation_reduction'])}, "
                f"subjects={n_side_subjects}, note={note}"
            )
    lines.extend(["", "## Task-Family Findings", ""])
    if by_task_family.empty:
        lines.append("Task-family summaries were unavailable.")
    else:
        for family, table in by_task_family.groupby("task_family", dropna=False):
            top = table.sort_values("mean_deviation_reduction", ascending=False).iloc[0]
            lines.append(
                f"- task_family={family}: top_model={top['model_name']}, "
                f"mean_reduction={_fmt_float(top['mean_deviation_reduction'])}, "
                f"mean_cosine={_fmt_float(top['mean_cosine_similarity'])}"
            )

    lines.extend(
        [
            "",
            "## Conservative Interpretation",
            "",
            "- Results are patient-specific compensation-direction predictions evaluated against dataset-internal MedOn proxies.",
            "- A positive deviation reduction means a simulated feature-level shift moved the held-out MedOff vector closer to the selected proxy under the current metric.",
            "- Candidate directions are computational outputs and require validation on more subjects and perturbation-response datasets.",
            "",
            "## Limitations",
            "",
            "- The processed subset remains small.",
            "- Right-side estimates are underpowered when only one subject contributes right-side tasks.",
            "- MedOn is not a healthy state; it is only a dataset-internal compensated proxy.",
            "- Feature-level shifts are not real interventions.",
            "- No causal perturbation-response data are available in this module.",
            "- Learned models may be unstable with the current sample size.",
            "- Larger subject samples are needed before strong claims.",
            "",
            "## Relevance To Neural Engineering",
            "",
            "This module adds a learnable patient-specific layer for modeling compensatory state directions from paired cortico-subthalamic recordings while remaining explicitly in silico and non-clinical.",
            "",
            "## Methodological Guardrails",
            "",
            "- HCP is not used as an electrophysiological reference.",
            "- HCP may only be considered a structural or connectomic prior elsewhere in the project.",
            "- MedOn is a dataset-internal compensated proxy, not a healthy state.",
            "- Candidate directions are not device settings or real-world guidance.",
        ]
    )
    if not bootstrap.empty:
        lines.extend(["", "## Bootstrap Uncertainty", ""])
        for _, row in bootstrap[bootstrap["metric"].eq("mean_deviation_reduction")].head(8).iterrows():
            lines.append(
                f"- {row['model_name']}: mean={_fmt_float(row['mean'])}, "
                f"95CI=[{_fmt_float(row['ci_lower'])}, {_fmt_float(row['ci_upper'])}], "
                f"unit={row['bootstrap_unit']}"
            )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_integrated_summary(path: str | Path, summary: pd.DataFrame) -> None:
    """Append or replace the patient-specific prediction section."""

    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else "# Real 8-Subject Integrated Summary\n"
    marker = "## Patient-Specific Prediction Layer"
    if marker in text:
        text = text.split(marker, maxsplit=1)[0].rstrip() + "\n"
    best_reduction = _best_row(summary, "mean_deviation_reduction")
    best_active_reduction = _best_active_row(summary, "mean_deviation_reduction")
    best_net = _best_row(summary, "best_net_compensation_score")
    best_active_net = _best_active_row(summary, "best_net_compensation_score")
    learned = summary[summary["model_name"].isin(["ridge_regression_predictor", "elastic_net_predictor"])]
    group = summary[summary["model_name"].eq("group_mean_direction")]
    random = summary[summary["model_name"].eq("random_direction")]
    no_action = summary[summary["model_name"].eq("no_action")]

    def _beats(table: pd.DataFrame, baseline: pd.DataFrame) -> str:
        if table.empty or baseline.empty:
            return "not_available"
        return "yes" if float(table["mean_deviation_reduction"].max()) > float(baseline["mean_deviation_reduction"].max()) else "no"

    section = [
        "",
        marker,
        "",
        "- This layer evaluates leave-one-subject-out patient-specific compensation-direction prediction from MedOff vectors.",
    ]
    if best_reduction is not None:
        section.append(
            f"- best_model_by_deviation_reduction: {best_reduction['model_name']} "
            f"alpha={best_reduction['best_alpha']} mean_reduction={_fmt_float(best_reduction['mean_deviation_reduction'])}"
        )
    if best_active_reduction is not None:
        section.append(
            f"- best_active_model_by_deviation_reduction: {best_active_reduction['model_name']} "
            f"alpha={best_active_reduction['best_alpha']} "
            f"mean_reduction={_fmt_float(best_active_reduction['mean_deviation_reduction'])}"
        )
    if best_net is not None:
        section.append(
            f"- best_model_by_net_score: {best_net['model_name']} "
            f"alpha={best_net['best_net_alpha']} net_score={_fmt_float(best_net['best_net_compensation_score'])}"
        )
    if best_active_net is not None:
        section.append(
            f"- best_active_model_by_net_score: {best_active_net['model_name']} "
            f"alpha={best_active_net['best_net_alpha']} net_score={_fmt_float(best_active_net['best_net_compensation_score'])}"
        )
    section.extend(
        [
            f"- learned_model_beats_group_mean_by_reduction: {_beats(learned, group)}",
            f"- learned_model_beats_random_by_reduction: {_beats(learned, random)}",
            f"- learned_model_beats_no_action_by_reduction: {_beats(learned, no_action)}",
            "- This is an exploratory in silico feature-level prediction result using dataset-internal MedOn proxies.",
            "- Next: compare subject-specific and group-level directions with uncertainty intervals before expanding claims.",
        ]
    )
    path.write_text(text.rstrip() + "\n" + "\n".join(section) + "\n", encoding="utf-8")


def run_patient_specific_predictor(
    state_vectors_path: str | Path,
    output_dir: str | Path = "outputs/tables",
    figures_dir: str | Path = "outputs/figures",
    reports_dir: str | Path = "reports",
    seed: int = 42,
    alpha_values: list[float] | None = None,
    lambda_magnitude: float = 0.01,
    lambda_complexity: float = 0.005,
    lambda_instability: float = 0.05,
    n_bootstrap: int = 500,
) -> dict[str, Path]:
    """Run the full patient-specific predictor workflow."""

    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_predictor_inputs(state_vectors_path, output_dir)
    feature_columns = available_real_feature_columns(inputs.state_vectors) if not inputs.state_vectors.empty else []
    pairs, pairing_log = build_patient_specific_pairs(inputs.state_vectors, feature_columns)
    results, model_warnings = run_loso_predictions(
        pairs,
        feature_columns,
        inputs.reliability,
        inputs.reliability_sideaware,
        inputs.quality,
        seed=seed,
        alpha_values=alpha_values or ALPHA_VALUES,
        lambda_magnitude=lambda_magnitude,
        lambda_complexity=lambda_complexity,
        lambda_instability=lambda_instability,
    )
    warnings = model_warnings
    warnings.append("small_mlp_predictor skipped: current paired sample size is too small for a stable neural-network comparison.")
    summary = summarize_models(results)
    by_subject = summarize_by_group(results, "subject")
    by_task_family = summarize_by_group(results, "task_family")
    by_side = summarize_by_group(results, "side")
    bootstrap = bootstrap_ci(results, summary, seed=seed, n_bootstrap=n_bootstrap)

    paths = {
        "pairs": output_dir / "real_patient_specific_pairs.csv",
        "pairing_log": output_dir / "real_patient_specific_pairing_log.csv",
        "results": output_dir / "real_patient_specific_predictor_results.csv",
        "summary": output_dir / "real_patient_specific_predictor_summary.csv",
        "by_subject": output_dir / "real_patient_specific_predictor_by_subject.csv",
        "by_task_family": output_dir / "real_patient_specific_predictor_by_task_family.csv",
        "by_side": output_dir / "real_patient_specific_predictor_by_side.csv",
        "bootstrap_ci": output_dir / "real_patient_specific_predictor_bootstrap_ci.csv",
        "model_comparison_figure": figures_dir / "patient_specific_predictor_model_comparison.png",
        "deviation_reduction_figure": figures_dir / "patient_specific_predictor_deviation_reduction.png",
        "cosine_similarity_figure": figures_dir / "patient_specific_predictor_cosine_similarity.png",
        "task_family_figure": figures_dir / "patient_specific_predictor_by_task_family.png",
        "side_figure": figures_dir / "patient_specific_predictor_by_side.png",
        "report": reports_dir / "real_patient_specific_predictor_report.md",
        "integrated_summary": reports_dir / "real_8_subject_integrated_summary.md",
    }
    pairs.to_csv(paths["pairs"], index=False)
    pairing_log.to_csv(paths["pairing_log"], index=False)
    results.to_csv(paths["results"], index=False)
    summary.to_csv(paths["summary"], index=False)
    by_subject.to_csv(paths["by_subject"], index=False)
    by_task_family.to_csv(paths["by_task_family"], index=False)
    by_side.to_csv(paths["by_side"], index=False)
    bootstrap.to_csv(paths["bootstrap_ci"], index=False)

    _bar_plot(
        summary,
        "model_name",
        "mean_net_compensation_score",
        "Patient-specific predictor model comparison",
        "Mean net compensation score",
        paths["model_comparison_figure"],
        color="#6f8f4e",
    )
    _bar_plot(
        summary,
        "model_name",
        "mean_deviation_reduction",
        "Deviation reduction by predictor",
        "Mean proxy deviation reduction",
        paths["deviation_reduction_figure"],
        color="#437c90",
    )
    _bar_plot(
        summary,
        "model_name",
        "mean_cosine_similarity",
        "Direction cosine similarity",
        "Mean cosine similarity to true direction",
        paths["cosine_similarity_figure"],
        color="#7c5f8f",
    )
    plot_grouped_summary(
        by_task_family,
        "task_family",
        paths["task_family_figure"],
        "Best predictor by task family",
    )
    plot_grouped_summary(by_side, "side", paths["side_figure"], "Best predictor by side")
    write_report(
        paths["report"],
        inputs,
        pairs,
        pairing_log,
        results,
        summary,
        by_task_family,
        by_side,
        bootstrap,
        warnings,
    )
    update_integrated_summary(paths["integrated_summary"], summary)
    return paths
