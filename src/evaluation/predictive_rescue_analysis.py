"""Expanded-cohort paired-identification rescue analysis for ds004998.

This module tests a conservative predictive framing for paired MedOff/MedOn
state vectors using matched-pair retrieval rather than continuous full-vector
direction regression. MedOn is used only as a dataset-internal compensated
proxy. No HCP electrophysiology reference or clinical intervention claim is
used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import math
import os

import numpy as np
import pandas as pd

from src.evaluation.side_aware_analysis import add_task_side_columns
from src.evaluation.subspace_compensation_analysis import subspace_map
from src.pathology_model.build_real_state_vectors import available_real_feature_columns

try:  # pragma: no cover - optional in lightweight environments.
    from scipy.stats import binom, spearmanr
except Exception:  # noqa: BLE001
    binom = None
    spearmanr = None

try:  # pragma: no cover - sklearn is part of the intended environment.
    from sklearn.linear_model import Ridge
except Exception:  # noqa: BLE001
    Ridge = None


QUALITY_SEVERITY = {"good": 0, "acceptable": 1, "caution": 2, "low_quality": 3, "unknown": 4}
EVAL_SUBSPACES = [
    "full_28",
    "cortical_alpha_beta",
    "motor_beta",
    "motor_beta_gamma",
    "stn_beta",
    "stn_gamma",
    "cortico_stn_coupling",
    "literature_informed_beta_network",
    "clean_stable_features",
    "sideaware_reliability_top_k",
]
SCORING_METHODS = [
    "euclidean_train_normalized",
    "cosine_distance",
    "reliability_weighted_distance",
    "mahalanobis_distance",
    "linear_ridge_scoring",
]
NULL_STRATEGIES = [
    "task_side_quality_matched_medon_shuffle",
    "task_side_matched_medon_shuffle",
    "task_family_side_matched_medon_shuffle",
    "subject_label_shuffle_within_matched_strata",
    "sign_flip_sensitivity",
]


@dataclass(frozen=True)
class RescueConfig:
    """Configuration for paired-identification rescue analysis."""

    n_permutations: int = 100
    random_seed: int = 42
    min_distractors: int = 2
    ridge_alpha: float = 1.0


def _read_csv(path: str | Path, warnings: list[str], **kwargs: object) -> pd.DataFrame:
    """Read a CSV/TSV with warning-on-failure behavior."""

    path = Path(path)
    if not path.exists():
        warnings.append(f"Missing input: {path}")
        return pd.DataFrame()
    try:
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t", **kwargs)
        return pd.read_csv(path, **kwargs)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Could not read {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def _write_csv(data: pd.DataFrame, path: str | Path) -> Path:
    """Write CSV with parent-directory creation."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return path


def _write_text(lines: Iterable[str], path: str | Path) -> Path:
    """Write a text report."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_state_vectors(path: str | Path, warnings: list[str]) -> pd.DataFrame:
    """Read state vectors and preserve side-aware labels."""

    data = _read_csv(path, warnings, dtype={"run": str})
    if data.empty:
        return data
    for column in ["subject", "session", "condition", "medication", "task", "run"]:
        if column in data:
            data[column] = data[column].fillna("unknown").astype(str)
    if not {"task_original", "task_family", "side"}.issubset(data.columns):
        data = add_task_side_columns(data, "task")
    else:
        for column in ["task_original", "task_family", "side"]:
            data[column] = data[column].fillna("unknown").astype(str)
    return data


def read_quality_table(path: str | Path, warnings: list[str]) -> pd.DataFrame:
    """Read quality table with side labels."""

    quality = _read_csv(path, warnings, dtype={"run": str})
    if quality.empty:
        return quality
    for column in ["subject", "session", "task", "medication", "run", "quality_flag"]:
        if column in quality:
            quality[column] = quality[column].fillna("unknown").astype(str)
    if not {"task_original", "task_family", "side"}.issubset(quality.columns):
        quality = add_task_side_columns(quality, "task" if "task" in quality else None)
    return quality


def verify_downloaded_files(data_root: str | Path, list_path: str | Path, warnings: list[str]) -> pd.DataFrame:
    """Verify locally downloaded files listed in download_next_best_new_subjects.txt."""

    data_root = Path(data_root)
    list_path = Path(list_path)
    if not list_path.exists():
        warnings.append(f"Missing downloaded-file list: {list_path}")
        return pd.DataFrame(
            columns=[
                "listed_path",
                "expected_path",
                "exists",
                "is_symlink",
                "resolved_exists",
                "subject",
                "task_original",
                "medication",
                "run",
                "split",
            ]
        )
    rows = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = data_root / line
        name = path.name
        parts = {piece.split("-", 1)[0]: piece.split("-", 1)[1] for piece in name.split("_") if "-" in piece}
        rows.append(
            {
                "listed_path": line,
                "expected_path": str(path),
                "exists": bool(path.exists()),
                "is_symlink": bool(path.is_symlink()),
                "resolved_exists": bool(path.resolve().exists()) if path.exists() else False,
                "subject": parts.get("sub", path.parts[-4] if len(path.parts) >= 4 else ""),
                "task_original": parts.get("task", ""),
                "medication": "off" if "MedOff" in name else ("on" if "MedOn" in name else "unknown"),
                "run": parts.get("run", "").removesuffix(".fif"),
                "split": parts.get("split", "").removesuffix(".fif"),
            }
        )
    return pd.DataFrame(rows)


def summarize_recording_manifest(recordings: pd.DataFrame, state_vectors: pd.DataFrame) -> pd.DataFrame:
    """Summarize expanded BIDS inspection and complete pair counts."""

    rows = []
    if recordings.empty:
        return pd.DataFrame(columns=["metric", "value"])
    usable = recordings[recordings.get("usable", False).astype(bool)] if "usable" in recordings else recordings
    for metric, value in [
        ("recording_rows", len(recordings)),
        ("usable_recording_rows", len(usable)),
        ("subjects_in_recording_manifest", recordings["subject"].nunique() if "subject" in recordings else 0),
        ("subjects_in_state_vectors", state_vectors["subject"].nunique() if not state_vectors.empty else 0),
        ("state_vectors", len(state_vectors)),
        ("complete_medoff_medon_pairs", count_complete_pairs(state_vectors)),
    ]:
        rows.append({"metric": metric, "value": value})
    for column in ["task", "medication"]:
        if column in usable:
            for key, value in usable[column].fillna("unknown").astype(str).value_counts().sort_index().items():
                rows.append({"metric": f"usable_{column}_{key}", "value": int(value)})
    if not state_vectors.empty and "side" in state_vectors:
        for key, value in state_vectors["side"].fillna("unknown").astype(str).value_counts().sort_index().items():
            rows.append({"metric": f"state_vector_side_{key}", "value": int(value)})
    split_count = int(recordings["file_path"].fillna("").astype(str).str.contains("split-", regex=False).sum()) if "file_path" in recordings else 0
    rows.append({"metric": "split_file_rows_detected", "value": split_count})
    return pd.DataFrame(rows)


def _worst_quality(values: Iterable[object]) -> str:
    """Return worst quality flag by severity."""

    flags = [str(value) for value in values if str(value) and str(value).lower() != "nan"]
    if not flags:
        return "unknown"
    return sorted(flags, key=lambda flag: QUALITY_SEVERITY.get(flag, 4), reverse=True)[0]


def attach_pair_quality(pairs: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Attach pair-level worst quality flags."""

    if pairs.empty:
        return pairs
    data = pairs.copy()
    if quality.empty:
        data["quality_flag"] = "unknown"
        data["quality_match_source"] = "missing_quality_table"
        return data
    q = quality.copy()
    for column in ["subject", "task_original", "medication", "quality_flag"]:
        if column not in q:
            q[column] = "unknown"
        q[column] = q[column].fillna("unknown").astype(str)
    lookup = (
        q.groupby(["subject", "task_original"], dropna=False)["quality_flag"]
        .agg(_worst_quality)
        .to_dict()
    )
    data["quality_flag"] = [
        lookup.get((str(row.subject), str(row.task_original)), "unknown")
        for row in data.itertuples()
    ]
    data["quality_match_source"] = "subject_task_worst_recording_quality"
    return data


def numeric_state_features(state_vectors: pd.DataFrame) -> list[str]:
    """Return numeric state feature columns."""

    features = available_real_feature_columns(state_vectors)
    if features:
        return features
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
    }
    return [
        column
        for column in state_vectors.columns
        if column not in metadata and pd.api.types.is_numeric_dtype(state_vectors[column])
    ]


def build_paired_examples(state_vectors: pd.DataFrame, quality: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build complete subject/task MedOff-MedOn paired examples."""

    features = numeric_state_features(state_vectors)
    rows: list[dict[str, object]] = []
    log_rows: list[dict[str, object]] = []
    if state_vectors.empty:
        return pd.DataFrame(), pd.DataFrame()
    for (subject, task_original), subset in state_vectors.groupby(["subject", "task_original"], dropna=False):
        meds = subset["medication"].fillna("").astype(str).str.lower()
        off = subset[meds.eq("off")]
        on = subset[meds.eq("on")]
        status = "paired" if not off.empty and not on.empty else "missing_medoff_or_medon"
        log_rows.append(
            {
                "subject": subject,
                "task_original": task_original,
                "n_medoff_rows": int(len(off)),
                "n_medon_rows": int(len(on)),
                "status": status,
            }
        )
        if off.empty or on.empty:
            continue
        off_values = off[features].mean(numeric_only=True)
        on_values = on[features].mean(numeric_only=True)
        first = subset.iloc[0]
        row: dict[str, object] = {
            "pair_id": f"{subject}|{task_original}",
            "subject": str(subject),
            "session": str(first.get("session", "unknown")),
            "task_original": str(task_original),
            "task_family": str(first.get("task_family", "unknown")),
            "side": str(first.get("side", "unknown")),
            "run_off": ";".join(off.get("run", pd.Series("unknown", index=off.index)).dropna().astype(str).unique()),
            "run_on": ";".join(on.get("run", pd.Series("unknown", index=on.index)).dropna().astype(str).unique()),
        }
        for feature in features:
            row[f"x_{feature}"] = float(off_values[feature])
            row[f"medon_{feature}"] = float(on_values[feature])
            row[f"y_{feature}"] = float(on_values[feature] - off_values[feature])
        rows.append(row)
    pairs = attach_pair_quality(pd.DataFrame(rows), quality)
    return pairs, pd.DataFrame(log_rows)


def count_complete_pairs(state_vectors: pd.DataFrame) -> int:
    """Count complete subject/task MedOff-MedOn pairs."""

    if state_vectors.empty or "subject" not in state_vectors:
        return 0
    count = 0
    for _, subset in state_vectors.groupby(["subject", "task_original"], dropna=False):
        meds = subset["medication"].fillna("").astype(str).str.lower()
        count += int(meds.eq("off").any() and meds.eq("on").any())
    return count


def read_subspace_definitions(path: str | Path, state_vectors: pd.DataFrame, warnings: list[str]) -> dict[str, list[str]]:
    """Read subspace definitions and keep only available state features."""

    definitions = _read_csv(path, warnings)
    available = set(numeric_state_features(state_vectors))
    if definitions.empty:
        return {"full_28": sorted(available)}
    subspaces = subspace_map(definitions)
    filtered: dict[str, list[str]] = {}
    for name, features in subspaces.items():
        kept = [feature for feature in features if feature in available]
        if kept:
            filtered[name] = kept
    if "full_28" not in filtered:
        filtered["full_28"] = sorted(available)
    return filtered


def pair_feature_matrix(pairs: pd.DataFrame, prefix: str, features: list[str]) -> np.ndarray:
    """Return a pair matrix for a feature prefix."""

    return pairs[[f"{prefix}{feature}" for feature in features]].to_numpy(dtype=float)


def fit_train_scaler(train: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fit train-only scaler from MedOff and MedOn states."""

    values = np.vstack([pair_feature_matrix(train, "x_", features), pair_feature_matrix(train, "medon_", features)])
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0, ddof=1)
    std = np.where(~np.isfinite(std) | (std < 1e-8), 1.0, std)
    return mean, std


def transform(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply train-only standardization."""

    return (np.asarray(values, dtype=float) - mean) / std


def train_reliability_weights(train: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Compute train-only direction-consistency weights."""

    y = pair_feature_matrix(train, "y_", features)
    if y.size == 0:
        return np.ones(len(features), dtype=float)
    signs = np.sign(y)
    weights = []
    for idx in range(signs.shape[1]):
        column = signs[:, idx]
        nonzero = column[column != 0]
        if len(nonzero) == 0:
            weights.append(0.5)
            continue
        positive = float((nonzero > 0).mean())
        weights.append(max(positive, 1.0 - positive))
    weights = np.asarray(weights, dtype=float)
    return np.where(np.isfinite(weights), weights, 0.5)


def mahalanobis_inverse(train: pd.DataFrame, features: list[str], mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, str]:
    """Compute regularized covariance inverse for train-only Mahalanobis scoring."""

    values = np.vstack([pair_feature_matrix(train, "x_", features), pair_feature_matrix(train, "medon_", features)])
    z = transform(values, mean, std)
    if z.shape[0] <= len(features) + 2:
        cov = np.cov(z, rowvar=False) if z.shape[0] > 1 else np.eye(len(features))
        status = "regularized_small_sample"
    else:
        cov = np.cov(z, rowvar=False)
        status = "covariance_from_training_subjects"
    cov = np.atleast_2d(cov)
    cov = cov + 0.1 * np.eye(cov.shape[0])
    return np.linalg.pinv(cov), status


def fit_ridge_medon_predictor(
    train: pd.DataFrame,
    features: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    alpha: float,
) -> object | None:
    """Fit train-only ridge model mapping MedOff state to MedOn state."""

    if Ridge is None or len(train) < 3:
        return None
    x_train = transform(pair_feature_matrix(train, "x_", features), mean, std)
    y_train = transform(pair_feature_matrix(train, "medon_", features), mean, std)
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    return model


def candidate_pool_for_query(pairs: pd.DataFrame, query: pd.Series, min_distractors: int) -> tuple[pd.DataFrame, str]:
    """Choose matched MedOn distractors for one MedOff query."""

    other = pairs[~pairs["subject"].astype(str).eq(str(query["subject"]))].copy()
    task_side = other[
        other["task_original"].astype(str).eq(str(query["task_original"]))
        & other["side"].astype(str).eq(str(query["side"]))
    ]
    match_level = "task_original_side"
    pool = task_side
    if len(pool) < min_distractors:
        pool = other[
            other["task_family"].astype(str).eq(str(query["task_family"]))
            & other["side"].astype(str).eq(str(query["side"]))
        ]
        match_level = "task_family_side"
    if "quality_flag" in pool and len(pool) >= min_distractors:
        same_quality = pool[pool["quality_flag"].astype(str).eq(str(query.get("quality_flag", "unknown")))]
        if len(same_quality) >= min_distractors:
            pool = same_quality
            match_level += "_quality"
    return pool.copy(), match_level


def build_base_candidate_sets(pairs: pd.DataFrame, min_distractors: int) -> pd.DataFrame:
    """Build base candidate metadata before subspace-specific scoring."""

    rows: list[dict[str, object]] = []
    for _, query in pairs.iterrows():
        pool, match_level = candidate_pool_for_query(pairs, query, min_distractors)
        candidate_rows = [query]
        candidate_rows.extend([row for _, row in pool.iterrows()])
        for position, candidate in enumerate(candidate_rows):
            rows.append(
                {
                    "query_pair_id": query["pair_id"],
                    "query_subject": query["subject"],
                    "query_task_original": query["task_original"],
                    "query_task_family": query["task_family"],
                    "query_side": query["side"],
                    "query_quality_flag": query.get("quality_flag", "unknown"),
                    "candidate_pair_id": candidate["pair_id"],
                    "candidate_subject": candidate["subject"],
                    "candidate_task_original": candidate["task_original"],
                    "candidate_task_family": candidate["task_family"],
                    "candidate_side": candidate["side"],
                    "candidate_quality_flag": candidate.get("quality_flag", "unknown"),
                    "is_true_pair": bool(position == 0),
                    "match_level": match_level,
                    "number_of_candidates": int(len(candidate_rows)),
                }
            )
    return pd.DataFrame(rows)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance with stable zero-vector convention."""

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 1.0
    return float(1.0 - (np.dot(a, b) / denom))


def _rank_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Add candidate ranks to scores."""

    ranked = scores.sort_values(["query_pair_id", "subspace", "scoring_method", "score"], ascending=True).copy()
    ranked["rank"] = ranked.groupby(["query_pair_id", "subspace", "scoring_method"], dropna=False).cumcount() + 1
    return ranked


def score_candidate_sets(
    pairs: pd.DataFrame,
    base_candidates: pd.DataFrame,
    subspaces: dict[str, list[str]],
    config: RescueConfig,
) -> pd.DataFrame:
    """Score matched MedOn candidate sets under LOSO train-only preprocessing."""

    pair_lookup = pairs.set_index("pair_id", drop=False)
    rows: list[dict[str, object]] = []
    selected_subspaces = {name: subspaces[name] for name in EVAL_SUBSPACES if name in subspaces and subspaces[name]}
    rng = np.random.default_rng(config.random_seed)
    for subspace, features in selected_subspaces.items():
        for heldout_subject in sorted(pairs["subject"].astype(str).unique()):
            train = pairs[~pairs["subject"].astype(str).eq(heldout_subject)].copy()
            test_ids = pairs.loc[pairs["subject"].astype(str).eq(heldout_subject), "pair_id"].astype(str).tolist()
            if train.empty or not test_ids:
                continue
            mean, std = fit_train_scaler(train, features)
            weights = train_reliability_weights(train, features)
            cov_inv, mahal_status = mahalanobis_inverse(train, features, mean, std)
            ridge = fit_ridge_medon_predictor(train, features, mean, std, config.ridge_alpha)
            for query_id in test_ids:
                query = pair_lookup.loc[query_id]
                qx = transform(query[[f"x_{feature}" for feature in features]].to_numpy(dtype=float), mean, std)
                if ridge is None:
                    ridge_target = None
                else:
                    ridge_target = np.asarray(ridge.predict(qx.reshape(1, -1))[0], dtype=float)
                candidates = base_candidates[base_candidates["query_pair_id"].astype(str).eq(str(query_id))]
                for _, candidate_meta in candidates.iterrows():
                    candidate = pair_lookup.loc[str(candidate_meta["candidate_pair_id"])]
                    cm = transform(candidate[[f"medon_{feature}" for feature in features]].to_numpy(dtype=float), mean, std)
                    diff = cm - qx
                    raw_scores = {
                        "euclidean_train_normalized": float(np.linalg.norm(diff)),
                        "cosine_distance": _cosine_distance(qx, cm),
                        "reliability_weighted_distance": float(np.sqrt(np.nanmean(weights * diff**2))),
                        "mahalanobis_distance": float(np.sqrt(max(0.0, diff.T @ cov_inv @ diff))),
                    }
                    if ridge_target is None:
                        raw_scores["linear_ridge_scoring"] = raw_scores["euclidean_train_normalized"]
                        ridge_status = "fallback_euclidean_insufficient_training_or_sklearn_missing"
                    else:
                        raw_scores["linear_ridge_scoring"] = float(np.linalg.norm(cm - ridge_target))
                        ridge_status = "train_only_ridge_medon_prediction"
                    for method, score in raw_scores.items():
                        rows.append(
                            {
                                **candidate_meta.to_dict(),
                                "heldout_subject": heldout_subject,
                                "subspace": subspace,
                                "n_features": len(features),
                                "scoring_method": method,
                                "score": score,
                                "mahalanobis_status": mahal_status,
                                "ridge_status": ridge_status,
                                "methodological_note": "LOSO matched-pair retrieval; MedOn is dataset-internal proxy",
                            }
                        )
            # Keep deterministic RNG state use explicit for future extensions.
            _ = rng.random()
    return _rank_scores(pd.DataFrame(rows))


def summarize_retrieval(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build query-level and grouped retrieval summaries."""

    if scored.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    true_rows = scored[scored["is_true_pair"].astype(bool)].copy()
    true_rows["top1_success"] = true_rows["rank"].eq(1)
    true_rows["top2_success"] = true_rows["rank"].le(2)
    true_rows["top3_success"] = true_rows["rank"].le(3)
    true_rows["reciprocal_rank"] = 1.0 / true_rows["rank"].astype(float)
    true_rows["chance_top1"] = 1.0 / true_rows["number_of_candidates"].astype(float)
    true_rows["improvement_over_chance"] = true_rows["top1_success"].astype(float) - true_rows["chance_top1"]
    true_rows["percentile_rank"] = 1.0 - (
        (true_rows["rank"].astype(float) - 1.0) / (true_rows["number_of_candidates"].astype(float) - 1.0).clip(lower=1.0)
    )
    best_distractor = (
        scored[~scored["is_true_pair"].astype(bool)]
        .sort_values("score")
        .groupby(["query_pair_id", "subspace", "scoring_method"], as_index=False)
        .first()[["query_pair_id", "subspace", "scoring_method", "score"]]
        .rename(columns={"score": "best_distractor_score"})
    )
    true_rows = true_rows.merge(best_distractor, on=["query_pair_id", "subspace", "scoring_method"], how="left")
    true_rows["margin"] = true_rows["best_distractor_score"] - true_rows["score"]
    group_cols = ["subspace", "scoring_method"]
    summary = (
        true_rows.groupby(group_cols, dropna=False)
        .agg(
            top1_accuracy=("top1_success", "mean"),
            top2_accuracy=("top2_success", "mean"),
            top3_accuracy=("top3_success", "mean"),
            mean_reciprocal_rank=("reciprocal_rank", "mean"),
            median_true_rank=("rank", "median"),
            mean_percentile_rank=("percentile_rank", "mean"),
            mean_number_of_candidates=("number_of_candidates", "mean"),
            mean_chance_top1=("chance_top1", "mean"),
            mean_improvement_over_chance=("improvement_over_chance", "mean"),
            mean_margin=("margin", "mean"),
            n_queries=("query_pair_id", "nunique"),
            n_subjects=("query_subject", "nunique"),
        )
        .reset_index()
    )
    by_subject = (
        true_rows.groupby(["query_subject", "subspace", "scoring_method"], dropna=False)
        .agg(
            top1_accuracy=("top1_success", "mean"),
            mean_reciprocal_rank=("reciprocal_rank", "mean"),
            mean_percentile_rank=("percentile_rank", "mean"),
            mean_margin=("margin", "mean"),
            n_queries=("query_pair_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"query_subject": "subject"})
    )
    by_task_side = (
        true_rows.groupby(["query_task_original", "query_task_family", "query_side", "subspace", "scoring_method"], dropna=False)
        .agg(
            top1_accuracy=("top1_success", "mean"),
            mean_reciprocal_rank=("reciprocal_rank", "mean"),
            mean_percentile_rank=("percentile_rank", "mean"),
            mean_margin=("margin", "mean"),
            n_queries=("query_pair_id", "nunique"),
            n_subjects=("query_subject", "nunique"),
        )
        .reset_index()
    )
    return true_rows, summary, by_subject.merge(by_task_side, how="outer")


def _pseudo_true_mask(group: pd.DataFrame, rng: np.random.Generator, strategy: str) -> pd.Series:
    """Choose one pseudo-true candidate for a query under a matched null."""

    candidates = group.copy()
    original_true = candidates["is_true_pair"].astype(bool)
    pool = candidates[~original_true].copy()
    if pool.empty:
        pool = candidates.copy()
    if strategy == "task_side_quality_matched_medon_shuffle":
        query_quality = str(candidates["query_quality_flag"].iloc[0])
        same_quality = pool[pool["candidate_quality_flag"].astype(str).eq(query_quality)]
        if not same_quality.empty:
            pool = same_quality
    elif strategy == "task_side_matched_medon_shuffle":
        same_task = pool[
            pool["candidate_task_original"].astype(str).eq(str(candidates["query_task_original"].iloc[0]))
            & pool["candidate_side"].astype(str).eq(str(candidates["query_side"].iloc[0]))
        ]
        if not same_task.empty:
            pool = same_task
    elif strategy == "task_family_side_matched_medon_shuffle":
        same_family = pool[
            pool["candidate_task_family"].astype(str).eq(str(candidates["query_task_family"].iloc[0]))
            & pool["candidate_side"].astype(str).eq(str(candidates["query_side"].iloc[0]))
        ]
        if not same_family.empty:
            pool = same_family
    elif strategy == "subject_label_shuffle_within_matched_strata":
        pass
    elif strategy == "sign_flip_sensitivity":
        pass
    chosen_index = rng.choice(pool.index.to_numpy())
    return candidates.index.to_series().eq(chosen_index)


def run_matched_nulls(
    scored: pd.DataFrame,
    real_summary: pd.DataFrame,
    config: RescueConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run matched-label permutation nulls from scored candidate sets."""

    if scored.empty:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(config.random_seed)
    null_rows: list[dict[str, object]] = []
    keys = ["query_pair_id", "subspace", "scoring_method"]
    precomputed_groups = []
    for group_key, group in scored.groupby(keys, dropna=False, sort=False):
        group = group.reset_index(drop=True)
        original_true = group["is_true_pair"].astype(bool).to_numpy()
        non_true_positions = np.flatnonzero(~original_true)
        if len(non_true_positions) == 0:
            non_true_positions = np.arange(len(group))
        query_quality = str(group["query_quality_flag"].iloc[0])
        query_task = str(group["query_task_original"].iloc[0])
        query_family = str(group["query_task_family"].iloc[0])
        query_side = str(group["query_side"].iloc[0])
        pools: dict[str, np.ndarray] = {}
        same_quality = [
            idx
            for idx in non_true_positions
            if str(group.loc[idx, "candidate_quality_flag"]) == query_quality
        ]
        pools["task_side_quality_matched_medon_shuffle"] = (
            np.asarray(same_quality, dtype=int) if same_quality else non_true_positions
        )
        same_task = [
            idx
            for idx in non_true_positions
            if str(group.loc[idx, "candidate_task_original"]) == query_task
            and str(group.loc[idx, "candidate_side"]) == query_side
        ]
        pools["task_side_matched_medon_shuffle"] = (
            np.asarray(same_task, dtype=int) if same_task else non_true_positions
        )
        same_family = [
            idx
            for idx in non_true_positions
            if str(group.loc[idx, "candidate_task_family"]) == query_family
            and str(group.loc[idx, "candidate_side"]) == query_side
        ]
        pools["task_family_side_matched_medon_shuffle"] = (
            np.asarray(same_family, dtype=int) if same_family else non_true_positions
        )
        pools["subject_label_shuffle_within_matched_strata"] = non_true_positions
        pools["sign_flip_sensitivity"] = np.arange(len(group))
        scores = group["score"].to_numpy(dtype=float)
        ranks = group["rank"].to_numpy(dtype=float)
        precomputed_groups.append(
            {
                "query_pair_id": group_key[0],
                "subspace": group_key[1],
                "scoring_method": group_key[2],
                "number_of_candidates": float(group["number_of_candidates"].iloc[0]),
                "scores": scores,
                "ranks": ranks,
                "pools": pools,
            }
        )
    for permutation_index in range(config.n_permutations):
        for strategy in NULL_STRATEGIES:
            buckets: dict[tuple[object, object], list[dict[str, float]]] = {}
            for group in precomputed_groups:
                pool = group["pools"].get(strategy, np.arange(len(group["scores"])))
                chosen_position = int(rng.choice(pool))
                rank = float(group["ranks"][chosen_position])
                scores = group["scores"]
                nonchosen = np.delete(scores, chosen_position)
                best_nonchosen = float(np.nanmin(nonchosen)) if len(nonchosen) else np.nan
                n_candidates = float(group["number_of_candidates"])
                item = {
                    "top1_success": float(rank == 1.0),
                    "reciprocal_rank": 1.0 / rank if rank > 0 else np.nan,
                    "percentile_rank": 1.0 - ((rank - 1.0) / max(1.0, n_candidates - 1.0)),
                    "margin": best_nonchosen - float(scores[chosen_position]) if np.isfinite(best_nonchosen) else np.nan,
                }
                buckets.setdefault((group["subspace"], group["scoring_method"]), []).append(item)
            for (subspace, method), values in buckets.items():
                value_df = pd.DataFrame(values)
                null_rows.append(
                    {
                        "subspace": subspace,
                        "scoring_method": method,
                        "null_strategy": strategy,
                        "permutation_index": permutation_index,
                        "top1_accuracy": float(value_df["top1_success"].mean()),
                        "mean_reciprocal_rank": float(value_df["reciprocal_rank"].mean()),
                        "mean_percentile_rank": float(value_df["percentile_rank"].mean()),
                        "mean_margin": float(value_df["margin"].mean()),
                        "n_queries": int(len(value_df)),
                    }
                )
    null_distribution = pd.DataFrame(null_rows)
    summary_rows = []
    metrics = ["top1_accuracy", "mean_reciprocal_rank", "mean_percentile_rank", "mean_margin"]
    for (subspace, method, strategy), subset in null_distribution.groupby(["subspace", "scoring_method", "null_strategy"], dropna=False):
        real = real_summary[
            real_summary["subspace"].astype(str).eq(str(subspace))
            & real_summary["scoring_method"].astype(str).eq(str(method))
        ]
        row: dict[str, object] = {
            "subspace": subspace,
            "scoring_method": method,
            "null_strategy": strategy,
            "n_permutations": int(subset["permutation_index"].nunique()),
        }
        for metric in metrics:
            real_value = float(real[metric].iloc[0]) if not real.empty and metric in real else np.nan
            null_values = subset[metric].to_numpy(dtype=float)
            mean_null = float(np.nanmean(null_values)) if len(null_values) else np.nan
            std_null = float(np.nanstd(null_values, ddof=1)) if len(null_values) > 1 else np.nan
            empirical_p = (1 + int(np.nansum(null_values >= real_value))) / (1 + len(null_values)) if np.isfinite(real_value) else np.nan
            row[f"real_{metric}"] = real_value
            row[f"null_mean_{metric}"] = mean_null
            row[f"effect_vs_null_{metric}"] = real_value - mean_null if np.isfinite(real_value) else np.nan
            row[f"z_vs_null_{metric}"] = (real_value - mean_null) / std_null if std_null and std_null > 0 and np.isfinite(real_value) else np.nan
            row[f"empirical_p_{metric}"] = empirical_p
        summary_rows.append(row)
    null_summary = pd.DataFrame(summary_rows)
    if not null_summary.empty:
        p = null_summary["empirical_p_top1_accuracy"].to_numpy(dtype=float)
        order = np.argsort(np.nan_to_num(p, nan=1.0))
        adjusted = np.full(len(p), np.nan)
        m = max(1, len(p))
        running = 1.0
        for rank_pos, idx in enumerate(order[::-1], start=1):
            original_rank = m - rank_pos + 1
            value = min(running, p[idx] * m / max(1, original_rank)) if np.isfinite(p[idx]) else np.nan
            adjusted[idx] = value
            if np.isfinite(value):
                running = value
        null_summary["bh_fdr_p_top1_accuracy"] = adjusted
    return null_distribution, null_summary


def clinical_anchor_audit(
    raw_root: str | Path,
    pairs: pd.DataFrame,
    subspaces: dict[str, list[str]],
    warnings: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inventory and match UPDRS medication-response variables if available."""

    raw_root = Path(raw_root)
    candidates = list(raw_root.glob("*participants*")) + list(raw_root.glob("**/*UPDRS*")) + list(raw_root.glob("**/*updrs*"))
    inventory = pd.DataFrame(
        [
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "file_type": path.suffix.lower(),
            }
            for path in sorted(set(candidates))
        ]
    )
    off_path = raw_root / "participants_updrs_off.tsv"
    on_path = raw_root / "participants_updrs_on.tsv"
    off = _read_csv(off_path, warnings)
    on = _read_csv(on_path, warnings)
    if off.empty or on.empty or "participant_id" not in off or "participant_id" not in on or "SUM" not in off or "SUM" not in on:
        warnings.append("UPDRS off/on SUM scores were not available or not matchable.")
        empty = pd.DataFrame()
        return inventory, empty, empty, empty
    off = off[["participant_id", "SUM"]].rename(columns={"SUM": "updrs_off"})
    on = on[["participant_id", "SUM"]].rename(columns={"SUM": "updrs_on"})
    matched = off.merge(on, on="participant_id", how="outer")
    matched["updrs_off"] = pd.to_numeric(matched["updrs_off"], errors="coerce")
    matched["updrs_on"] = pd.to_numeric(matched["updrs_on"], errors="coerce")
    matched["medication_response"] = matched["updrs_off"] - matched["updrs_on"]
    matched["percent_response"] = matched["medication_response"] / matched["updrs_off"].replace(0, np.nan)
    matched["in_state_vectors"] = matched["participant_id"].astype(str).isin(set(pairs["subject"].astype(str))) if not pairs.empty else False
    response_targets = []
    if not pairs.empty:
        for subject, subject_pairs in pairs.groupby("subject", dropna=False):
            row = matched[matched["participant_id"].astype(str).eq(str(subject))]
            if row.empty:
                continue
            base = {
                "subject": subject,
                "updrs_off": float(row["updrs_off"].iloc[0]),
                "updrs_on": float(row["updrs_on"].iloc[0]),
                "medication_response": float(row["medication_response"].iloc[0]),
                "percent_response": float(row["percent_response"].iloc[0]),
                "n_pairs": int(len(subject_pairs)),
            }
            for subspace in [
                "motor_beta",
                "cortical_alpha_beta",
                "stn_beta",
                "cortico_stn_coupling",
                "full_28",
            ]:
                features = [feature for feature in subspaces.get(subspace, []) if f"y_{feature}" in subject_pairs]
                if not features:
                    base[f"{subspace}_delta_norm"] = np.nan
                    continue
                y = subject_pairs[[f"y_{feature}" for feature in features]].to_numpy(dtype=float)
                base[f"{subspace}_delta_norm"] = float(np.nanmean(np.linalg.norm(y, axis=1)))
            response_targets.append(base)
    response = pd.DataFrame(response_targets)
    corr_rows = []
    if spearmanr is not None and not response.empty:
        for column in [col for col in response.columns if col.endswith("_delta_norm")]:
            valid = response[["medication_response", column]].dropna()
            if len(valid) >= 4:
                rho, p_value = spearmanr(valid["medication_response"], valid[column])
            else:
                rho, p_value = np.nan, np.nan
            corr_rows.append(
                {
                    "target": column,
                    "n_subjects": int(len(valid)),
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "interpretation": "exploratory_clinical_anchor_not_outcome_claim",
                }
            )
    return inventory, matched, response, pd.DataFrame(corr_rows)


def variance_decomposition(values: pd.DataFrame, value_col: str, grouping_cols: list[str], label: str) -> pd.DataFrame:
    """Compute manual variance components by one-way group means."""

    rows = []
    if values.empty or value_col not in values:
        return pd.DataFrame()
    total_var = float(values[value_col].var(ddof=1)) if len(values) > 1 else 0.0
    for group_col in grouping_cols:
        if group_col not in values:
            continue
        means = values.groupby(group_col, dropna=False)[value_col].mean()
        counts = values[group_col].value_counts(dropna=False)
        grand = float(values[value_col].mean())
        weighted = float(sum(counts.get(idx, 0) * (mean - grand) ** 2 for idx, mean in means.items()) / max(1, len(values) - 1))
        rows.append(
            {
                "analysis": label,
                "value_column": value_col,
                "component": group_col,
                "variance_component": weighted,
                "total_variance": total_var,
                "fraction_of_total_variance": weighted / total_var if total_var > 0 else np.nan,
                "method": "manual_one_way_variance_decomposition_small_n",
            }
        )
    return pd.DataFrame(rows)


def build_variance_tables(
    pairs: pd.DataFrame,
    retrieval_results: pd.DataFrame,
    subspaces: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build manual variance decomposition summaries."""

    delta_rows = []
    for _, row in pairs.iterrows():
        base = {
            "subject": row["subject"],
            "task_family": row["task_family"],
            "side": row["side"],
            "quality_flag": row.get("quality_flag", "unknown"),
        }
        for subspace in EVAL_SUBSPACES:
            features = [feature for feature in subspaces.get(subspace, []) if f"y_{feature}" in pairs]
            if not features:
                continue
            y = row[[f"y_{feature}" for feature in features]].to_numpy(dtype=float)
            delta_rows.append({**base, "subspace": subspace, "delta_norm": float(np.linalg.norm(y))})
    deltas = pd.DataFrame(delta_rows)
    components = []
    for subspace, subset in deltas.groupby("subspace", dropna=False):
        components.append(variance_decomposition(subset, "delta_norm", ["subject", "task_family", "side", "quality_flag"], f"feature_delta_{subspace}"))
    if not retrieval_results.empty:
        retrieval = retrieval_results.copy()
        retrieval["top1_numeric"] = retrieval["top1_success"].astype(float)
        for (subspace, method), subset in retrieval.groupby(["subspace", "scoring_method"], dropna=False):
            components.append(
                variance_decomposition(
                    subset.rename(
                        columns={
                            "query_subject": "subject",
                            "query_task_family": "task_family",
                            "query_side": "side",
                            "query_quality_flag": "quality_flag",
                        }
                    ),
                    "percentile_rank",
                    ["subject", "task_family", "side", "quality_flag"],
                    f"retrieval_percentile_{subspace}_{method}",
                )
            )
    variance = pd.concat([item for item in components if item is not None and not item.empty], ignore_index=True) if components else pd.DataFrame()
    mixed = variance.copy()
    if not mixed.empty:
        mixed["mixed_effects_status"] = "manual_variance_decomposition_used_due_small_n"
        mixed["model_formula_proxy"] = "outcome ~ task_family + side + quality + subject_component"
    return variance, mixed


def _binomial_power(n: int, p0: float, p1: float, alpha: float = 0.05) -> tuple[float, int]:
    """Approximate exact binomial power if scipy is available."""

    if n <= 0:
        return np.nan, 0
    if binom is not None:
        critical = 0
        for k in range(n + 1):
            if 1.0 - binom.cdf(k - 1, n, p0) <= alpha:
                critical = k
                break
        return float(1.0 - binom.cdf(critical - 1, n, p1)), int(critical)
    z = 1.6448536269514722
    threshold = n * p0 + z * math.sqrt(max(1e-12, n * p0 * (1.0 - p0)))
    mean1 = n * p1
    sd1 = math.sqrt(max(1e-12, n * p1 * (1.0 - p1)))
    power = 1.0 - 0.5 * (1.0 + math.erf((threshold - mean1) / (sd1 * math.sqrt(2.0))))
    return float(power), int(math.ceil(threshold))


def power_analysis(summary: pd.DataFrame, retrieval_results: pd.DataFrame) -> pd.DataFrame:
    """Power analysis for paired identification top1 accuracy."""

    if summary.empty:
        return pd.DataFrame()
    n = int(summary["n_queries"].max())
    chance = float(summary["mean_chance_top1"].mean())
    rows = []
    for target in [0.10, 0.15, 0.20, 0.25, 0.33, 0.50]:
        power, critical = _binomial_power(n, chance, target)
        rows.append(
            {
                "analysis": "top1_accuracy_binomial_vs_average_chance",
                "n_queries": n,
                "chance_top1": chance,
                "target_top1_accuracy": target,
                "critical_successes_alpha_0_05": critical,
                "estimated_power": power,
            }
        )
    detectable = np.nan
    for candidate in np.linspace(max(chance, 0.01), 0.95, 500):
        power, _ = _binomial_power(n, chance, float(candidate))
        if power >= 0.80:
            detectable = float(candidate)
            break
    rows.append(
        {
            "analysis": "minimum_detectable_top1_for_80_percent_power",
            "n_queries": n,
            "chance_top1": chance,
            "target_top1_accuracy": detectable,
            "critical_successes_alpha_0_05": np.nan,
            "estimated_power": 0.80 if np.isfinite(detectable) else np.nan,
        }
    )
    if not retrieval_results.empty and "margin" in retrieval_results:
        margin_sd = float(retrieval_results["margin"].std(ddof=1))
        margin_mean = float(retrieval_results["margin"].mean())
        rows.append(
            {
                "analysis": "observed_margin_signal_to_noise_descriptive",
                "n_queries": int(retrieval_results["query_pair_id"].nunique()),
                "chance_top1": chance,
                "target_top1_accuracy": np.nan,
                "critical_successes_alpha_0_05": np.nan,
                "estimated_power": np.nan,
                "observed_mean_margin": margin_mean,
                "observed_margin_sd": margin_sd,
                "margin_signal_to_noise": margin_mean / margin_sd if margin_sd > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def decide_predictive_status(null_summary: pd.DataFrame, retrieval_summary: pd.DataFrame) -> tuple[str, str]:
    """Apply conservative decision logic for predictive-framing rescue."""

    if null_summary.empty or retrieval_summary.empty:
        return "methodological_only", "Null or retrieval summaries were unavailable."
    primary = null_summary[null_summary["null_strategy"].eq("task_side_quality_matched_medon_shuffle")]
    if primary.empty:
        return "methodological_only", "Primary strict quality-matched null was unavailable."
    corrected = primary["bh_fdr_p_top1_accuracy"] if "bh_fdr_p_top1_accuracy" in primary else primary["empirical_p_top1_accuracy"]
    if (corrected < 0.05).any():
        return "predictive_claim_supported", "At least one retrieval configuration beat the strict matched null after correction."
    chance_beating = retrieval_summary["mean_improvement_over_chance"].fillna(0).max() > 0
    if chance_beating:
        return "weak_predictive_signal_only", "Some retrieval configurations beat average chance but not the strict matched null."
    return "methodological_only", "Retrieval did not exceed chance or strict matched null sufficiently."


def write_inspection_report(
    verification: pd.DataFrame,
    manifest_summary: pd.DataFrame,
    path: str | Path,
    warnings: list[str],
) -> Path:
    """Write expanded inspection report."""

    missing = int((~verification["exists"].astype(bool)).sum()) if not verification.empty and "exists" in verification else 0
    lines = [
        "# Expanded 15-Subject Inspection Report",
        "",
        "This report verifies locally downloaded ds004998 files and summarizes the expanded BIDS manifest. It does not download data.",
        "",
        f"- listed_new_files: {len(verification)}",
        f"- missing_listed_files: {missing}",
        "",
        "## Manifest Summary",
        "",
    ]
    for _, row in manifest_summary.iterrows():
        lines.append(f"- {row['metric']}: {row['value']}")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- HCP is not used as an electrophysiological reference.",
            "- MedOn is a dataset-internal compensated proxy, not a healthy state.",
            "- Candidate outputs remain exploratory in silico research outputs.",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])
    return _write_text(lines, path)


def write_clinical_report(
    inventory: pd.DataFrame,
    updrs_match: pd.DataFrame,
    updrs_response: pd.DataFrame,
    correlations: pd.DataFrame,
    path: str | Path,
    warnings: list[str],
) -> Path:
    """Write clinical anchor audit report."""

    matched = int(updrs_match["in_state_vectors"].sum()) if not updrs_match.empty and "in_state_vectors" in updrs_match else 0
    lines = [
        "# Expanded Clinical Anchor Audit Report",
        "",
        "This audit checks whether local ds004998 UPDRS medication-response variables are available and matchable to processed subjects.",
        "",
        f"- clinical_files_found: {len(inventory)}",
        f"- updrs_subjects_matched_to_state_vectors: {matched}",
        f"- response_target_rows: {len(updrs_response)}",
        "",
        "## Exploratory UPDRS Correlations",
        "",
    ]
    if correlations.empty:
        lines.append("No sufficiently populated UPDRS correlation table was available.")
    else:
        for _, row in correlations.iterrows():
            lines.append(
                f"- {row['target']}: n={row['n_subjects']}, rho={row['spearman_rho']:.3f}, p={row['p_value']:.3f}"
            )
    lines.extend(
        [
            "",
            "These correlations are exploratory anchors only and are not treatment or efficacy claims.",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])
    return _write_text(lines, path)


def write_predictive_report(
    pairs: pd.DataFrame,
    retrieval_summary: pd.DataFrame,
    null_summary: pd.DataFrame,
    decision: tuple[str, str],
    path: str | Path,
    warnings: list[str],
) -> Path:
    """Write paired-identification rescue report."""

    best = retrieval_summary.sort_values(["top1_accuracy", "mean_reciprocal_rank"], ascending=False).head(1)
    best_text = "No retrieval result was available."
    if not best.empty:
        row = best.iloc[0]
        best_text = (
            f"{row['subspace']} / {row['scoring_method']} "
            f"(top1={row['top1_accuracy']:.3f}, MRR={row['mean_reciprocal_rank']:.3f}, "
            f"chance={row['mean_chance_top1']:.3f})"
        )
    primary = null_summary[null_summary["null_strategy"].eq("task_side_quality_matched_medon_shuffle")] if not null_summary.empty else pd.DataFrame()
    primary_best = primary.sort_values("empirical_p_top1_accuracy").head(1) if not primary.empty else pd.DataFrame()
    primary_text = "Primary strict null was unavailable."
    if not primary_best.empty:
        row = primary_best.iloc[0]
        primary_text = (
            f"{row['subspace']} / {row['scoring_method']} "
            f"p={row['empirical_p_top1_accuracy']:.3f}, "
            f"FDR={row.get('bh_fdr_p_top1_accuracy', np.nan):.3f}, "
            f"effect_vs_null={row['effect_vs_null_top1_accuracy']:.3f}"
        )
    lines = [
        "# Predictive Rescue Analysis Report",
        "",
        "This report replaces continuous MedOn-minus-MedOff vector regression with subject-held-out matched-pair retrieval.",
        "",
        "## Dataset Subset",
        "",
        f"- subjects: {pairs['subject'].nunique() if not pairs.empty else 0}",
        f"- paired_examples: {len(pairs)}",
        f"- tasks: {pairs['task_original'].nunique() if not pairs.empty else 0}",
        f"- sides: {pairs['side'].nunique() if not pairs.empty else 0}",
        "",
        "## Main Retrieval Result",
        "",
        f"- best_observed_retrieval: {best_text}",
        f"- best_primary_null_comparison: {primary_text}",
        "",
        "## Decision",
        "",
        f"- status: {decision[0]}",
        f"- rationale: {decision[1]}",
        "",
        "## Interpretation Guardrails",
        "",
        "- MedOn is used only as a dataset-internal compensated proxy.",
        "- This is paired-state identification, not clinical control or device-setting optimization.",
        "- Results are exploratory and require validation in larger paired neurophysiology cohorts.",
    ]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])
    return _write_text(lines, path)


def write_power_report(power: pd.DataFrame, path: str | Path) -> Path:
    """Write power analysis report."""

    lines = [
        "# Predictive Rescue Power Analysis Report",
        "",
        "Power estimates use the observed number of matched-pair retrieval queries and candidate-set sizes.",
        "",
    ]
    if power.empty:
        lines.append("Power analysis could not be computed.")
    else:
        for _, row in power.iterrows():
            if row["analysis"] == "top1_accuracy_binomial_vs_average_chance":
                lines.append(
                    f"- target_top1={row['target_top1_accuracy']:.2f}: estimated_power={row['estimated_power']:.3f}, "
                    f"critical_successes={row['critical_successes_alpha_0_05']}"
                )
            elif row["analysis"] == "minimum_detectable_top1_for_80_percent_power":
                lines.append(f"- minimum_detectable_top1_for_80_percent_power: {row['target_top1_accuracy']:.3f}")
    lines.extend(
        [
            "",
            "These estimates are exploratory and do not imply clinical utility.",
        ]
    )
    return _write_text(lines, path)


def write_decision_report(
    decision: tuple[str, str],
    previous_pairs: int,
    pairs: pd.DataFrame,
    updrs_match: pd.DataFrame,
    retrieval_summary: pd.DataFrame,
    null_summary: pd.DataFrame,
    power: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Write a clear rescue/abandon decision report."""

    right_pairs = int(pairs[pairs["side"].astype(str).eq("R")]["pair_id"].nunique()) if not pairs.empty else 0
    updrs_available = not updrs_match.empty and bool(updrs_match.get("in_state_vectors", pd.Series(dtype=bool)).any())
    strict = null_summary[null_summary["null_strategy"].eq("task_side_quality_matched_medon_shuffle")] if not null_summary.empty else pd.DataFrame()
    beats_strict = bool((strict.get("bh_fdr_p_top1_accuracy", pd.Series(dtype=float)) < 0.05).any()) if not strict.empty else False
    beats_chance = bool(retrieval_summary["mean_improvement_over_chance"].fillna(0).max() > 0) if not retrieval_summary.empty else False
    best_subspace = ""
    if not retrieval_summary.empty:
        best = retrieval_summary.sort_values(["top1_accuracy", "mean_reciprocal_rank"], ascending=False).iloc[0]
        best_subspace = f"{best['subspace']} / {best['scoring_method']}"
    underpowered = True
    if not power.empty:
        detectable = power[power["analysis"].eq("minimum_detectable_top1_for_80_percent_power")]
        underpowered = bool(not detectable.empty and float(detectable["target_top1_accuracy"].iloc[0]) > 0.33)
    lines = [
        "# Predictive Rescue Decision Report",
        "",
        "## Questions",
        "",
        f"1. Did expansion increase paired examples? Previous={previous_pairs}; expanded={len(pairs)}.",
        f"2. Did right-side examples improve? right_side_pairs={right_pairs}.",
        f"3. Is UPDRS usable? {'yes' if updrs_available else 'no'}.",
        f"4. Does paired identification beat chance? {'yes' if beats_chance else 'no'}.",
        f"5. Does paired identification beat the strict task/side/quality null? {'yes' if beats_strict else 'no'}.",
        f"6. Best retrieval subspace/method: {best_subspace or 'unavailable'}.",
        "7. LOSO was used for train-normalization and train-only scoring.",
        "8. Subject/task/side/quality effects are summarized in the variance-component tables.",
        f"9. Is the expanded dataset powered enough for a predictive claim? {'no' if underpowered else 'possibly'}.",
        f"10. Can the old predictive framing be rescued? {decision[0]}.",
        "",
        "## Final Status",
        "",
        f"- status: {decision[0]}",
        f"- rationale: {decision[1]}",
        "",
        "The result should not be framed as treatment, clinical control, DBS optimization, or patient recommendation.",
    ]
    return _write_text(lines, path)


def write_manuscript_framing(decision: tuple[str, str], path: str | Path) -> Path:
    """Write manuscript framing paths after expansion."""

    lines = [
        "# Manuscript Framing After Expansion",
        "",
        "## Path A: Predictive Rescue Succeeds",
        "",
        "If paired identification beats strict task/side/quality-matched nulls after correction, the main claim can be that paired MedOff/MedOn states are identifiable above matched-null chance under strict constraints. This should remain a modest in silico paired-state modeling claim.",
        "",
        "## Path B: Predictive Rescue Fails",
        "",
        "If retrieval does not beat strict matched nulls, the strongest manuscript framing is methodological: the project provides an audited validation framework for small paired clinical neurophysiology datasets, showing where continuous vector prediction and matched-pair identification fail strong controls.",
        "",
        "## Current Decision",
        "",
        f"- status: {decision[0]}",
        f"- rationale: {decision[1]}",
        "",
        "No clinical treatment, DBS prescription, therapeutic efficacy, or patient recommendation claim should be made.",
    ]
    return _write_text(lines, path)


def write_integrated_summary(
    manifest_summary: pd.DataFrame,
    retrieval_summary: pd.DataFrame,
    null_summary: pd.DataFrame,
    decision: tuple[str, str],
    path: str | Path,
) -> Path:
    """Write 15-subject integrated summary."""

    metrics = {str(row["metric"]): row["value"] for _, row in manifest_summary.iterrows()} if not manifest_summary.empty else {}
    best = retrieval_summary.sort_values(["top1_accuracy", "mean_reciprocal_rank"], ascending=False).head(1)
    best_line = "No retrieval summary available."
    if not best.empty:
        row = best.iloc[0]
        best_line = f"{row['subspace']} / {row['scoring_method']} top1={row['top1_accuracy']:.3f}, MRR={row['mean_reciprocal_rank']:.3f}"
    strict = null_summary[null_summary["null_strategy"].eq("task_side_quality_matched_medon_shuffle")] if not null_summary.empty else pd.DataFrame()
    strict_line = "Strict matched null unavailable."
    if not strict.empty:
        row = strict.sort_values("empirical_p_top1_accuracy").iloc[0]
        strict_line = f"{row['subspace']} / {row['scoring_method']} p={row['empirical_p_top1_accuracy']:.3f}, FDR={row.get('bh_fdr_p_top1_accuracy', np.nan):.3f}"
    lines = [
        "# Real 15-Subject Integrated Summary",
        "",
        "This summary compares the prior 12-subject state with the expanded local ds004998 cohort.",
        "",
        "## Expanded Cohort",
        "",
        f"- subjects_in_state_vectors: {metrics.get('subjects_in_state_vectors', 'unknown')}",
        f"- state_vectors: {metrics.get('state_vectors', 'unknown')}",
        f"- complete_medoff_medon_pairs: {metrics.get('complete_medoff_medon_pairs', 'unknown')}",
        f"- split_file_rows_detected: {metrics.get('split_file_rows_detected', 'unknown')}",
        "",
        "## Predictive Rescue",
        "",
        f"- best_retrieval: {best_line}",
        f"- best_strict_null_comparison: {strict_line}",
        f"- decision_status: {decision[0]}",
        f"- decision_rationale: {decision[1]}",
        "",
        "## Interpretation",
        "",
        "The expanded analysis keeps MedOn as a dataset-internal compensated proxy and evaluates paired-state identification under subject-held-out validation. It does not support clinical control, therapeutic efficacy, DBS settings, or patient recommendations.",
    ]
    return _write_text(lines, path)


def run_predictive_rescue_analysis(
    state_vectors_path: str | Path,
    quality_table_path: str | Path,
    data_root: str | Path = "data/raw/ds004998",
    downloaded_list: str | Path = "data/raw/ds004998/download_next_best_new_subjects.txt",
    subspace_definitions_path: str | Path = "outputs/tables/subspace_definitions.csv",
    recording_manifest_path: str | Path = "outputs/tables/ds004998_recording_manifest.csv",
    output_dir: str | Path = "outputs/tables",
    reports_dir: str | Path = "reports",
    config: RescueConfig | None = None,
) -> dict[str, Path]:
    """Run expanded-cohort predictive-rescue analysis."""

    config = config or RescueConfig()
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    warnings: list[str] = []
    state_vectors = read_state_vectors(state_vectors_path, warnings)
    quality = read_quality_table(quality_table_path, warnings)
    subspaces = read_subspace_definitions(subspace_definitions_path, state_vectors, warnings)
    verification = verify_downloaded_files(data_root, downloaded_list, warnings)
    recordings = _read_csv(recording_manifest_path, warnings, dtype={"run": str})
    manifest_summary = summarize_recording_manifest(recordings, state_vectors)
    pairs, pairing_log = build_paired_examples(state_vectors, quality)
    base_candidates = build_base_candidate_sets(pairs, config.min_distractors) if not pairs.empty else pd.DataFrame()
    scored = score_candidate_sets(pairs, base_candidates, subspaces, config) if not pairs.empty else pd.DataFrame()
    retrieval_results, retrieval_summary, by_subject_and_task = summarize_retrieval(scored)
    by_subject = (
        retrieval_results.groupby(["query_subject", "subspace", "scoring_method"], dropna=False)
        .agg(
            top1_accuracy=("top1_success", "mean"),
            mean_reciprocal_rank=("reciprocal_rank", "mean"),
            mean_percentile_rank=("percentile_rank", "mean"),
            mean_margin=("margin", "mean"),
            n_queries=("query_pair_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"query_subject": "subject"})
        if not retrieval_results.empty
        else pd.DataFrame()
    )
    by_task_side = (
        retrieval_results.groupby(["query_task_original", "query_task_family", "query_side", "subspace", "scoring_method"], dropna=False)
        .agg(
            top1_accuracy=("top1_success", "mean"),
            mean_reciprocal_rank=("reciprocal_rank", "mean"),
            mean_percentile_rank=("percentile_rank", "mean"),
            mean_margin=("margin", "mean"),
            n_queries=("query_pair_id", "nunique"),
            n_subjects=("query_subject", "nunique"),
        )
        .reset_index()
        if not retrieval_results.empty
        else pd.DataFrame()
    )
    null_distribution, null_summary = run_matched_nulls(scored, retrieval_summary, config)
    inventory, updrs_match, updrs_response, updrs_corr = clinical_anchor_audit(data_root, pairs, subspaces, warnings)
    variance, mixed = build_variance_tables(pairs, retrieval_results, subspaces)
    power = power_analysis(retrieval_summary, retrieval_results)
    decision = decide_predictive_status(null_summary, retrieval_summary)

    paths = {
        "expanded_download_verification": _write_csv(verification, output_dir / "expanded_download_verification.csv"),
        "expanded_recording_manifest_summary": _write_csv(manifest_summary, output_dir / "expanded_recording_manifest_summary.csv"),
        "predictive_rescue_pairs": _write_csv(pairs, output_dir / "predictive_rescue_pairs_expanded.csv"),
        "predictive_rescue_pairing_log": _write_csv(pairing_log, output_dir / "predictive_rescue_pairing_log_expanded.csv"),
        "predictive_rescue_candidate_sets": _write_csv(scored, output_dir / "predictive_rescue_candidate_sets_expanded.csv"),
        "predictive_rescue_retrieval_results": _write_csv(retrieval_results, output_dir / "predictive_rescue_retrieval_results_expanded.csv"),
        "predictive_rescue_retrieval_summary": _write_csv(retrieval_summary, output_dir / "predictive_rescue_retrieval_summary_expanded.csv"),
        "predictive_rescue_by_subject": _write_csv(by_subject, output_dir / "predictive_rescue_by_subject_expanded.csv"),
        "predictive_rescue_by_task_side": _write_csv(by_task_side, output_dir / "predictive_rescue_by_task_side_expanded.csv"),
        "predictive_rescue_null_summary": _write_csv(null_summary, output_dir / "predictive_rescue_null_summary_expanded.csv"),
        "predictive_rescue_null_distributions": _write_csv(null_distribution, output_dir / "predictive_rescue_null_distributions_expanded.csv"),
        "expanded_clinical_anchor_inventory": _write_csv(inventory, output_dir / "expanded_clinical_anchor_inventory.csv"),
        "expanded_updrs_subject_match": _write_csv(updrs_match, output_dir / "expanded_updrs_subject_match.csv"),
        "expanded_updrs_response_targets": _write_csv(updrs_response, output_dir / "expanded_updrs_response_targets.csv"),
        "expanded_updrs_correlation": _write_csv(updrs_corr, output_dir / "expanded_updrs_correlation.csv"),
        "predictive_rescue_variance_components": _write_csv(variance, output_dir / "predictive_rescue_variance_components_expanded.csv"),
        "predictive_rescue_mixed_effects_summary": _write_csv(mixed, output_dir / "predictive_rescue_mixed_effects_summary_expanded.csv"),
        "predictive_rescue_power_analysis": _write_csv(power, output_dir / "predictive_rescue_power_analysis_expanded.csv"),
    }
    paths.update(
        {
            "inspection_report": write_inspection_report(
                verification,
                manifest_summary,
                reports_dir / "expanded_15_subject_inspection_report.md",
                warnings,
            ),
            "clinical_report": write_clinical_report(
                inventory,
                updrs_match,
                updrs_response,
                updrs_corr,
                reports_dir / "expanded_clinical_anchor_audit_report.md",
                warnings,
            ),
            "predictive_report": write_predictive_report(
                pairs,
                retrieval_summary,
                null_summary,
                decision,
                reports_dir / "predictive_rescue_analysis_report.md",
                warnings,
            ),
            "power_report": write_power_report(power, reports_dir / "predictive_rescue_power_analysis_report.md"),
            "decision_report": write_decision_report(
                decision,
                previous_pairs=24,
                pairs=pairs,
                updrs_match=updrs_match,
                retrieval_summary=retrieval_summary,
                null_summary=null_summary,
                power=power,
                path=reports_dir / "predictive_rescue_decision_report.md",
            ),
            "manuscript_framing": write_manuscript_framing(
                decision,
                reports_dir / "manuscript_framing_after_expansion.md",
            ),
            "integrated_summary": write_integrated_summary(
                manifest_summary,
                retrieval_summary,
                null_summary,
                decision,
                reports_dir / "real_15_subject_integrated_summary.md",
            ),
        }
    )
    return paths
