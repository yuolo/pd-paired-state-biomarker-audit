"""Run single-feature STN biomarker retrieval baselines.

This downstream audit compares classical STN biomarker baselines against the
existing paired-state retrieval task. It reads cached ds004998/OXF pair tables
only; no raw signal extraction, scorer tuning, or project outputs from the main
methods are modified.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path("outputs/single_feature_biomarker_retrieval_baselines")
MANUSCRIPT_TABLES_DIR = Path("manuscript/tables")
DS_PAIRS_PATH = Path("outputs/v2A_frozen_18subjects_34pairs/frozen_v2a_pairs.csv")
OXF_PAIRS_PATH = Path("outputs/oxf_external_stn_retrieval_validation/oxf_onoff_pairs.csv")

RANDOM_SEED = 42
N_PERM = 5000
N_BOOT = 5000
MIN_DISTRACTORS = 2
PRIMARY_CONDITION = "original_frozen_matched_pool"
ALL_MEDON_CONDITION = "all_medon_candidates"
EVALUATED_CONDITIONS = [PRIMARY_CONDITION, ALL_MEDON_CONDITION]
REFERENCE_METRICS = {
    ("ds004998", PRIMARY_CONDITION): ("full_v2A", 0.8529411764705882, 0.8958333333333333),
    ("ds004998", ALL_MEDON_CONDITION): ("full_v2D", 0.8235294117647058, 0.8804271708683473),
    ("OXF", ALL_MEDON_CONDITION): ("MAGE_G_source_span_soft_v2D", 0.6, 0.720079365079365),
}


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset-specific pair table and feature-name mapping."""

    name: str
    pairs_path: Path
    beta_feature: str
    slope_feature: str
    primary_pool_note: str


@dataclass(frozen=True)
class BaselineSpec:
    """Single-feature or compact two-feature baseline definition."""

    name: str
    label: str
    role: str
    feature_keys: tuple[str, ...]
    distance: str


DATASETS = [
    DatasetSpec(
        name="ds004998",
        pairs_path=DS_PAIRS_PATH,
        beta_feature="stn_broad_beta_power",
        slope_feature="stn_aperiodic_slope",
        primary_pool_note="original frozen v2A matched task/side pool",
    ),
    DatasetSpec(
        name="OXF",
        pairs_path=OXF_PAIRS_PATH,
        beta_feature="stn_broad_beta_log_power",
        slope_feature="aperiodic_slope",
        primary_pool_note="matched hemisphere analogue: true ON plus other-subject same-side ON candidates",
    ),
]

BASELINES = [
    BaselineSpec(
        name="single_stn_beta_band_power",
        label="STN beta band power",
        role="classic single-feature dopaminergic-state biomarker",
        feature_keys=("beta_feature",),
        distance="absolute_difference",
    ),
    BaselineSpec(
        name="single_stn_aperiodic_slope",
        label="STN aperiodic slope",
        role="single-feature aperiodic 1/f biomarker",
        feature_keys=("slope_feature",),
        distance="absolute_difference",
    ),
    BaselineSpec(
        name="two_feature_beta_plus_slope",
        label="STN beta power + aperiodic slope",
        role="minimal two-feature classical-plus-aperiodic baseline",
        feature_keys=("beta_feature", "slope_feature"),
        distance="cosine",
    ),
]


def to_builtin(value: object) -> object:
    """Convert numpy/pandas values into JSON-safe Python values."""

    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    return value


def read_csv_required(path: Path, **kwargs: object) -> pd.DataFrame:
    """Read a required CSV file."""

    if not path.exists():
        raise FileNotFoundError(f"Required cached file is missing: {path}")
    return pd.read_csv(path, **kwargs)


def percentile_ci(values: pd.Series | np.ndarray, low: float = 2.5, high: float = 97.5) -> tuple[float, float]:
    """Return percentile CI after finite filtering."""

    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    return float(np.percentile(arr, low)), float(np.percentile(arr, high))


def query_metrics(diagnostics: pd.DataFrame) -> dict[str, object]:
    """Compute top-k and rank metrics from query diagnostics."""

    if diagnostics.empty:
        return {
            "n_pairs": 0,
            "top1": np.nan,
            "mrr": np.nan,
            "top3": np.nan,
            "top5": np.nan,
            "percentile_rank": np.nan,
            "mean_true_rank": np.nan,
            "median_true_rank": np.nan,
            "failures": 0,
            "retrieval_margin": np.nan,
        }
    ranks = pd.to_numeric(diagnostics["true_medon_rank"], errors="coerce")
    success = diagnostics["top_ranked_is_true_pair"].astype(bool)
    return {
        "n_pairs": int(len(diagnostics)),
        "top1": float(success.mean()),
        "mrr": float(pd.to_numeric(diagnostics["reciprocal_rank"], errors="coerce").mean()),
        "top3": float((ranks <= 3).mean()),
        "top5": float((ranks <= 5).mean()),
        "percentile_rank": float(pd.to_numeric(diagnostics["percentile_rank"], errors="coerce").mean()),
        "mean_true_rank": float(ranks.mean()),
        "median_true_rank": float(ranks.median()),
        "failures": int((~success).sum()),
        "retrieval_margin": float(pd.to_numeric(diagnostics["retrieval_margin"], errors="coerce").mean()),
    }


def write_json(data: dict[str, object], path: Path) -> Path:
    """Write JSON output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(data), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(data: pd.DataFrame, path: Path) -> Path:
    """Write CSV output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return path


def write_text(lines: Iterable[str], path: Path) -> Path:
    """Write a text/Markdown output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance with stable zero-vector fallback."""

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12 or not np.isfinite(denom):
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def fit_scaler(train: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fit train-subject-only scaler over OFF and ON states."""

    values = np.vstack(
        [
            train[[f"x_{feature}" for feature in features]].to_numpy(dtype=float),
            train[[f"medon_{feature}" for feature in features]].to_numpy(dtype=float),
        ]
    )
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0, ddof=1)
    std = np.where(~np.isfinite(std) | (std < 1e-8), 1.0, std)
    return mean, std


def transform(row: pd.Series, prefix: str, features: list[str], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Extract and standardize a vector from one pair row."""

    values = row[[f"{prefix}{feature}" for feature in features]].to_numpy(dtype=float)
    return (values - mean) / std


def finite_feature_subset(pairs: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Drop rows with missing values for the requested baseline features."""

    columns: list[str] = []
    for feature in features:
        columns.extend([f"x_{feature}", f"medon_{feature}"])
    missing = [column for column in columns if column not in pairs.columns]
    if missing:
        raise KeyError(f"Missing required pair columns: {missing}")
    out = pairs.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    mask = np.isfinite(out[columns].to_numpy(dtype=float)).all(axis=1)
    return out.loc[mask].reset_index(drop=True), columns


def candidate_row(query: pd.Series, candidate: pd.Series, is_true: bool, match_level: str) -> dict[str, object]:
    """Build one candidate-metadata row in the shared retrieval schema."""

    return {
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
        "is_true_pair": bool(is_true),
        "match_level": match_level,
    }


def deduplicate_candidates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep first occurrence for each candidate id."""

    seen: set[str] = set()
    out = []
    for row in rows:
        candidate_id = str(row["candidate_pair_id"])
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        out.append(row)
    return out


def finalize_candidate_pool(rows: list[dict[str, object]], condition: str) -> pd.DataFrame:
    """Add candidate counts and condition labels."""

    data = pd.DataFrame(rows)
    if data.empty:
        return data
    data["candidate_pool_condition"] = condition
    data["number_of_candidates"] = (
        data.groupby("query_pair_id", dropna=False)["candidate_pair_id"].transform("count").astype(int)
    )
    return data


def ds_primary_distractor_pool(pairs: pd.DataFrame, query: pd.Series) -> tuple[pd.DataFrame, str]:
    """Choose frozen-v2A-style task/side matched distractors for ds004998."""

    other = pairs[~pairs["subject"].astype(str).eq(str(query["subject"]))].copy()
    pool = other[
        other["task_original"].astype(str).eq(str(query["task_original"]))
        & other["side"].astype(str).eq(str(query["side"]))
    ]
    match_level = "task_original_side"
    if len(pool) < MIN_DISTRACTORS:
        pool = other[
            other["task_family"].astype(str).eq(str(query["task_family"]))
            & other["side"].astype(str).eq(str(query["side"]))
        ]
        match_level = "task_family_side"
    if "quality_flag" in pool and len(pool) >= MIN_DISTRACTORS:
        same_quality = pool[pool["quality_flag"].astype(str).eq(str(query.get("quality_flag", "unknown")))]
        if len(same_quality) >= MIN_DISTRACTORS:
            pool = same_quality
            match_level += "_quality"
    return pool.copy(), match_level


def build_ds_candidate_pool(pairs: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Build ds004998 candidate pools matching the frozen v2A/v2D studies."""

    rows: list[dict[str, object]] = []
    if condition == PRIMARY_CONDITION:
        for _, query in pairs.iterrows():
            pool, match_level = ds_primary_distractor_pool(pairs, query)
            local = [candidate_row(query, query, True, match_level)]
            for _, candidate in pool.iterrows():
                local.append(candidate_row(query, candidate, False, match_level))
            rows.extend(deduplicate_candidates(local))
        return finalize_candidate_pool(rows, condition)
    if condition == ALL_MEDON_CONDITION:
        pair_rows = [row for _, row in pairs.iterrows()]
        for _, query in pairs.iterrows():
            local = [candidate_row(query, query, True, "all_medon_candidates")]
            for candidate in pair_rows:
                local.append(
                    candidate_row(
                        query,
                        candidate,
                        str(candidate["pair_id"]) == str(query["pair_id"]),
                        "all_medon_candidates",
                    )
                )
            rows.extend(deduplicate_candidates(local))
        return finalize_candidate_pool(rows, condition)
    raise ValueError(f"Unsupported ds004998 condition: {condition}")


def build_oxf_candidate_pool(pairs: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Build OXF matched-side and all-MedOn candidate pools."""

    rows: list[dict[str, object]] = []
    for _, query in pairs.iterrows():
        query_id = str(query["pair_id"])
        subject = str(query["subject"])
        side = str(query["side"])
        local = [candidate_row(query, query, True, "true_pair")]
        if condition == PRIMARY_CONDITION:
            pool = pairs[~pairs["subject"].astype(str).eq(subject) & pairs["side"].astype(str).eq(side)]
            for _, candidate in pool.iterrows():
                local.append(candidate_row(query, candidate, False, "other_subject_same_hemisphere"))
        elif condition == ALL_MEDON_CONDITION:
            for _, candidate in pairs.iterrows():
                local.append(candidate_row(query, candidate, str(candidate["pair_id"]) == query_id, "all_on_candidates"))
        else:
            raise ValueError(f"Unsupported OXF condition: {condition}")
        rows.extend(deduplicate_candidates(local))
    return finalize_candidate_pool(rows, condition)


def build_pool(dataset: str, pairs: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Build the dataset-specific candidate pool."""

    if dataset == "OXF":
        return build_oxf_candidate_pool(pairs, condition)
    return build_ds_candidate_pool(pairs, condition)


def resolve_features(dataset: DatasetSpec, baseline: BaselineSpec) -> list[str]:
    """Resolve baseline feature keys for a dataset."""

    out = []
    for key in baseline.feature_keys:
        out.append(getattr(dataset, key))
    return out


def score_baseline(
    dataset: DatasetSpec,
    pairs: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    condition: str,
    baseline: BaselineSpec,
    features: list[str],
) -> pd.DataFrame:
    """Score one baseline over one candidate pool using LOSO train-only scaling."""

    pair_lookup = pairs.set_index("pair_id", drop=False)
    rows: list[dict[str, object]] = []
    for heldout_subject in sorted(pairs["subject"].astype(str).unique()):
        train = pairs[~pairs["subject"].astype(str).eq(heldout_subject)].copy()
        test_ids = pairs.loc[pairs["subject"].astype(str).eq(heldout_subject), "pair_id"].astype(str).tolist()
        if train.empty or not test_ids:
            continue
        mean, std = fit_scaler(train, features)
        for query_id in test_ids:
            query = pair_lookup.loc[str(query_id)]
            qx = transform(query, "x_", features, mean, std)
            candidates = candidate_pool[candidate_pool["query_pair_id"].astype(str).eq(str(query_id))]
            for _, meta in candidates.iterrows():
                candidate = pair_lookup.loc[str(meta["candidate_pair_id"])]
                cm = transform(candidate, "medon_", features, mean, std)
                if baseline.distance == "absolute_difference":
                    score = float(abs(qx[0] - cm[0]))
                elif baseline.distance == "cosine":
                    score = cosine_distance(qx, cm)
                else:
                    raise ValueError(f"Unsupported distance: {baseline.distance}")
                rows.append(
                    {
                        **meta.to_dict(),
                        "dataset": dataset.name,
                        "heldout_subject": heldout_subject,
                        "variant_name": baseline.name,
                        "baseline_label": baseline.label,
                        "baseline_role": baseline.role,
                        "feature_space": "+".join(features),
                        "score_kind": "single_feature_baseline" if len(features) == 1 else "two_feature_baseline",
                        "distance": baseline.distance,
                        "n_features": int(len(features)),
                        "score": score,
                        "candidate_pool_condition": condition,
                    }
                )
    scored = pd.DataFrame(rows)
    if scored.empty:
        return scored
    scored = scored.sort_values(["dataset", "candidate_pool_condition", "variant_name", "query_pair_id", "score"]).copy()
    scored["rank"] = (
        scored.groupby(["dataset", "candidate_pool_condition", "variant_name", "query_pair_id"], dropna=False)
        .cumcount()
        .astype(int)
        + 1
    )
    return scored


def classify_failure(row: pd.Series) -> str:
    """Classify top-1 failures for compact diagnostics."""

    if bool(row.get("top_ranked_is_true_pair", False)):
        return "success"
    same_subject = str(row.get("top_ranked_candidate_subject")) == str(row.get("query_subject"))
    same_task = str(row.get("top_ranked_candidate_task")) == str(row.get("query_task"))
    same_side = str(row.get("top_ranked_candidate_side")) == str(row.get("query_side"))
    if same_subject and not same_task:
        return "same_subject_wrong_task"
    if same_subject and not same_side:
        return "same_subject_wrong_side"
    if (not same_subject) and same_task and same_side:
        return "other_subject_same_task_side"
    if not same_task:
        return "other_subject_wrong_task"
    return "unclear"


def query_diagnostics(scores: pd.DataFrame) -> pd.DataFrame:
    """Create one row per query, dataset, condition, and baseline."""

    rows: list[dict[str, object]] = []
    group_cols = ["dataset", "candidate_pool_condition", "variant_name", "query_pair_id"]
    for (dataset, condition, variant, query_id), group in scores.groupby(group_cols, dropna=False):
        true = group[group["is_true_pair"].astype(bool)]
        if true.empty:
            continue
        true_row = true.iloc[0]
        top = group.sort_values("rank").iloc[0]
        distractors = group[~group["is_true_pair"].astype(bool)]
        nearest = distractors.sort_values("score").iloc[0] if not distractors.empty else pd.Series(dtype=object)
        candidate_set_size = int(true_row["number_of_candidates"])
        rank = int(true_row["rank"])
        row = {
            "dataset": dataset,
            "candidate_pool_condition": condition,
            "variant_name": variant,
            "baseline_label": true_row.get("baseline_label", ""),
            "feature_space": true_row.get("feature_space", ""),
            "distance": true_row.get("distance", ""),
            "n_features": int(true_row.get("n_features", 0)),
            "query_pair_id": query_id,
            "query_subject": true_row["query_subject"],
            "query_task": true_row["query_task_original"],
            "query_side": true_row["query_side"],
            "query_quality": true_row.get("query_quality_flag", "unknown"),
            "true_medon_rank": rank,
            "top_ranked_candidate_pair_id": top["candidate_pair_id"],
            "top_ranked_candidate_subject": top["candidate_subject"],
            "top_ranked_candidate_task": top["candidate_task_original"],
            "top_ranked_candidate_side": top["candidate_side"],
            "top_ranked_is_true_pair": bool(rank == 1),
            "distance_to_true_medon": float(true_row["score"]),
            "distance_to_nearest_distractor": float(nearest.get("score", np.nan)),
            "retrieval_margin": float(nearest.get("score", np.nan) - true_row["score"]),
            "candidate_set_size": candidate_set_size,
            "chance_top1": 1.0 / float(candidate_set_size),
            "reciprocal_rank": 1.0 / float(rank),
            "percentile_rank": 1.0 - ((float(rank) - 1.0) / max(1.0, float(candidate_set_size) - 1.0)),
        }
        row["failure_type"] = classify_failure(pd.Series(row))
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed retrieval metrics."""

    rows = []
    group_cols = ["dataset", "candidate_pool_condition", "variant_name"]
    for (dataset, condition, variant), subset in diagnostics.groupby(group_cols, dropna=False):
        metrics = query_metrics(subset)
        reference_label, reference_top1, reference_mrr = REFERENCE_METRICS.get(
            (str(dataset), str(condition)), ("", np.nan, np.nan)
        )
        rows.append(
            {
                "dataset": dataset,
                "candidate_pool_condition": condition,
                "variant_name": variant,
                "baseline_label": subset["baseline_label"].iloc[0],
                "feature_space": subset["feature_space"].iloc[0],
                "distance": subset["distance"].iloc[0],
                **metrics,
                "reference_method": reference_label,
                "reference_top1": reference_top1,
                "reference_mrr": reference_mrr,
                "delta_top1_vs_reference": float(metrics["top1"]) - float(reference_top1)
                if np.isfinite(reference_top1)
                else np.nan,
                "delta_mrr_vs_reference": float(metrics["mrr"]) - float(reference_mrr)
                if np.isfinite(reference_mrr)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "candidate_pool_condition", "variant_name"]).reset_index(drop=True)


def subject_bootstrap_ci(
    diagnostics: pd.DataFrame,
    n_boot: int = N_BOOT,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subject-clustered bootstrap CI for top-1 and MRR."""

    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    group_cols = ["dataset", "candidate_pool_condition", "variant_name"]
    for (dataset, condition, variant), subset in diagnostics.groupby(group_cols, dropna=False):
        subjects = sorted(subset["query_subject"].astype(str).unique())
        if not subjects:
            continue
        per_subject = {}
        for subject in subjects:
            rows = subset[subset["query_subject"].astype(str).eq(subject)]
            per_subject[subject] = {
                "n": int(len(rows)),
                "top1": float(rows["top_ranked_is_true_pair"].astype(float).sum()),
                "mrr": float(pd.to_numeric(rows["reciprocal_rank"], errors="coerce").sum()),
            }
        local_rows = []
        for index in range(n_boot):
            draw = rng.choice(subjects, size=len(subjects), replace=True)
            n = 0
            top1 = 0.0
            mrr = 0.0
            for subject in draw:
                item = per_subject[str(subject)]
                n += int(item["n"])
                top1 += float(item["top1"])
                mrr += float(item["mrr"])
            row = {
                "bootstrap_index": int(index),
                "dataset": dataset,
                "candidate_pool_condition": condition,
                "variant_name": variant,
                "top1": float(top1 / n) if n else np.nan,
                "mrr": float(mrr / n) if n else np.nan,
                "n_query_rows": int(n),
                "seed": int(seed),
                "bootstrap_unit": "subject",
            }
            local_rows.append(row)
            sample_rows.append(row)
        local = pd.DataFrame(local_rows)
        observed = query_metrics(subset)
        for metric in ["top1", "mrr"]:
            low, high = percentile_ci(local[metric])
            summary_rows.append(
                {
                    "dataset": dataset,
                    "candidate_pool_condition": condition,
                    "variant_name": variant,
                    "metric": metric,
                    "observed": float(observed[metric]),
                    "ci_lower_95": low,
                    "ci_upper_95": high,
                    "n_boot": int(n_boot),
                    "seed": int(seed),
                    "bootstrap_unit": "subject",
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def random_label_null(
    diagnostics: pd.DataFrame,
    scores: pd.DataFrame,
    n_perm: int = N_PERM,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random-label null preserving each query candidate-set size."""

    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    group_cols = ["dataset", "candidate_pool_condition", "variant_name"]
    for (dataset, condition, variant), diag in diagnostics.groupby(group_cols, dropna=False):
        subset_scores = scores[
            scores["dataset"].astype(str).eq(str(dataset))
            & scores["candidate_pool_condition"].astype(str).eq(str(condition))
            & scores["variant_name"].astype(str).eq(str(variant))
        ]
        sizes = subset_scores.groupby("query_pair_id", dropna=False)["candidate_pair_id"].size().to_numpy(dtype=int)
        if len(sizes) == 0:
            continue
        observed = query_metrics(diag)
        local_rows = []
        for index in range(n_perm):
            ranks = np.asarray([rng.integers(1, size + 1) for size in sizes], dtype=float)
            row = {
                "permutation_index": int(index),
                "dataset": dataset,
                "candidate_pool_condition": condition,
                "variant_name": variant,
                "top1": float(np.mean(ranks == 1)),
                "mrr": float(np.mean(1.0 / ranks)),
                "percentile_rank": float(np.mean(1.0 - ((ranks - 1.0) / np.maximum(1.0, sizes - 1.0)))),
                "seed": int(seed),
                "null_type": "random_label",
            }
            local_rows.append(row)
            sample_rows.append(row)
        local = pd.DataFrame(local_rows)
        for metric in ["top1", "mrr", "percentile_rank"]:
            values = local[metric].to_numpy(dtype=float)
            obs = float(observed[metric])
            summary_rows.append(
                {
                    "dataset": dataset,
                    "candidate_pool_condition": condition,
                    "variant_name": variant,
                    "metric": metric,
                    "observed": obs,
                    "null_mean": float(np.nanmean(values)),
                    "null_sd": float(np.nanstd(values, ddof=1)),
                    "empirical_p_greater_equal": float((np.sum(values >= obs) + 1.0) / (len(values) + 1.0)),
                    "n_perm": int(len(values)),
                    "seed": int(seed),
                    "null_type": "random_label",
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def eligible_matched_false_ranks(group: pd.DataFrame) -> np.ndarray:
    """Return ranks of matched false candidates for task/side-aware null."""

    false = group[~group["is_true_pair"].astype(bool)].copy()
    if false.empty:
        return np.asarray([], dtype=int)
    query_task = str(group["query_task_original"].iloc[0])
    query_family = str(group["query_task_family"].iloc[0])
    query_side = str(group["query_side"].iloc[0])
    exact = false[
        false["candidate_task_original"].astype(str).eq(query_task)
        & false["candidate_side"].astype(str).eq(query_side)
    ]
    if not exact.empty:
        return exact["rank"].to_numpy(dtype=int)
    family = false[
        false["candidate_task_family"].astype(str).eq(query_family)
        & false["candidate_side"].astype(str).eq(query_side)
    ]
    if not family.empty:
        return family["rank"].to_numpy(dtype=int)
    side = false[false["candidate_side"].astype(str).eq(query_side)]
    if not side.empty:
        return side["rank"].to_numpy(dtype=int)
    return false["rank"].to_numpy(dtype=int)


def matched_task_side_null(
    diagnostics: pd.DataFrame,
    scores: pd.DataFrame,
    n_perm: int = N_PERM,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matched false-positive null using task/side when available."""

    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    group_cols = ["dataset", "candidate_pool_condition", "variant_name"]
    for (dataset, condition, variant), subset in scores.groupby(group_cols, dropna=False):
        per_query = []
        for query_id, group in subset.groupby("query_pair_id", dropna=False):
            true = group[group["is_true_pair"].astype(bool)]
            if true.empty:
                continue
            false_ranks = eligible_matched_false_ranks(group)
            if len(false_ranks) == 0:
                continue
            per_query.append(
                {
                    "query_pair_id": query_id,
                    "candidate_set_size": int(group["candidate_pair_id"].nunique()),
                    "true_rank": int(true.iloc[0]["rank"]),
                    "false_ranks": false_ranks,
                }
            )
        if not per_query:
            continue
        true_ranks = np.asarray([item["true_rank"] for item in per_query], dtype=float)
        sizes = np.asarray([item["candidate_set_size"] for item in per_query], dtype=float)
        observed = {
            "top1": float(np.mean(true_ranks == 1)),
            "mrr": float(np.mean(1.0 / true_ranks)),
            "percentile_rank": float(np.mean(1.0 - ((true_ranks - 1.0) / np.maximum(1.0, sizes - 1.0)))),
        }
        local_rows = []
        for index in range(n_perm):
            ranks = np.asarray([rng.choice(item["false_ranks"]) for item in per_query], dtype=float)
            row = {
                "permutation_index": int(index),
                "dataset": dataset,
                "candidate_pool_condition": condition,
                "variant_name": variant,
                "top1": float(np.mean(ranks == 1)),
                "mrr": float(np.mean(1.0 / ranks)),
                "percentile_rank": float(np.mean(1.0 - ((ranks - 1.0) / np.maximum(1.0, sizes - 1.0)))),
                "seed": int(seed),
                "null_type": "matched_task_side_false_positive",
                "n_queries_with_matched_false_positive": int(len(per_query)),
            }
            local_rows.append(row)
            sample_rows.append(row)
        local = pd.DataFrame(local_rows)
        for metric in ["top1", "mrr", "percentile_rank"]:
            values = local[metric].to_numpy(dtype=float)
            obs = float(observed[metric])
            summary_rows.append(
                {
                    "dataset": dataset,
                    "candidate_pool_condition": condition,
                    "variant_name": variant,
                    "metric": metric,
                    "observed": obs,
                    "null_mean": float(np.nanmean(values)),
                    "null_sd": float(np.nanstd(values, ddof=1)),
                    "empirical_p_greater_equal": float((np.sum(values >= obs) + 1.0) / (len(values) + 1.0)),
                    "n_perm": int(len(values)),
                    "n_queries": int(len(per_query)),
                    "seed": int(seed),
                    "null_type": "matched_task_side_false_positive",
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def feature_mapping() -> pd.DataFrame:
    """Return the exact dataset-specific columns used by each baseline."""

    rows = []
    for dataset in DATASETS:
        for baseline in BASELINES:
            features = resolve_features(dataset, baseline)
            rows.append(
                {
                    "dataset": dataset.name,
                    "variant_name": baseline.name,
                    "baseline_label": baseline.label,
                    "baseline_role": baseline.role,
                    "features": ";".join(features),
                    "distance": baseline.distance,
                    "primary_pool_note": dataset.primary_pool_note,
                    "beta_definition_note": "broad beta band log-power proxy from cached STN feature table",
                    "slope_definition_note": "aperiodic 1/f slope from cached compact aperiodic feature table",
                }
            )
    return pd.DataFrame(rows)


def write_report(summary: dict[str, object], path: Path) -> Path:
    """Write a concise Markdown report."""

    metrics = pd.DataFrame(summary["key_metrics"])
    lines = [
        "# Single-Feature STN Biomarker Retrieval Baselines",
        "",
        "Scope: classical single-feature and two-feature STN baselines evaluated with the same paired-state retrieval target.",
        "",
        "## Configuration",
        f"- n_boot: {summary['config']['n_boot']}",
        f"- n_perm: {summary['config']['n_perm']}",
        f"- seed: {summary['config']['seed']}",
        "- scalar distance: absolute difference after LOSO train-only scaling",
        "- two-feature distance: cosine distance after LOSO train-only scaling",
        "",
        "## Key Metrics",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"- {row['dataset']} / {row['candidate_pool_condition']} / {row['variant_name']}: "
            f"top1={float(row['top1']):.3f}, MRR={float(row['mrr']):.3f}, "
            f"reference={row['reference_method'] or 'not_primary'}"
        )
    lines.extend(
        [
            "",
            "Boundary: these baselines are biomarker-comparison retrieval controls only; they are not clinical prediction, treatment, DBS optimization, or causal medication-effect estimates.",
        ]
    )
    return write_text(lines, path)


def run(n_perm: int = N_PERM, n_boot: int = N_BOOT, seed: int = RANDOM_SEED) -> dict[str, object]:
    """Run all single-feature baselines and controls."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    score_tables: list[pd.DataFrame] = []
    validation_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        raw_pairs = read_csv_required(dataset.pairs_path, dtype={"run_off": str, "run_on": str})
        for baseline in BASELINES:
            features = resolve_features(dataset, baseline)
            pairs, used_columns = finite_feature_subset(raw_pairs, features)
            if pairs.empty:
                raise AssertionError(f"No usable rows for {dataset.name} / {baseline.name}")
            validation_rows.append(
                {
                    "dataset": dataset.name,
                    "variant_name": baseline.name,
                    "n_pairs": int(len(pairs)),
                    "n_subjects": int(pairs["subject"].nunique()),
                    "features": ";".join(features),
                    "used_columns": ";".join(used_columns),
                    "dropped_pairs_missing_features": int(len(raw_pairs) - len(pairs)),
                }
            )
            for condition in EVALUATED_CONDITIONS:
                pool = build_pool(dataset.name, pairs, condition)
                scores = score_baseline(dataset, pairs, pool, condition, baseline, features)
                score_tables.append(scores)
    scores = pd.concat(score_tables, ignore_index=True)
    diagnostics = query_diagnostics(scores)
    observed = metrics_table(diagnostics)
    bootstrap_ci, bootstrap_samples = subject_bootstrap_ci(diagnostics, n_boot=n_boot, seed=seed)
    random_summary, random_samples = random_label_null(diagnostics, scores, n_perm=n_perm, seed=seed)
    matched_summary, matched_samples = matched_task_side_null(diagnostics, scores, n_perm=n_perm, seed=seed)
    mapping = feature_mapping()
    validation = pd.DataFrame(validation_rows).drop_duplicates().sort_values(["dataset", "variant_name"])

    key_conditions = [
        (("ds004998", PRIMARY_CONDITION)),
        (("ds004998", ALL_MEDON_CONDITION)),
        (("OXF", ALL_MEDON_CONDITION)),
    ]
    key_mask = pd.Series(False, index=observed.index)
    for dataset_name, condition in key_conditions:
        key_mask |= observed["dataset"].astype(str).eq(dataset_name) & observed["candidate_pool_condition"].astype(str).eq(condition)
    key_metrics = observed.loc[key_mask].copy().sort_values(["dataset", "candidate_pool_condition", "variant_name"])

    summary = {
        "config": {"n_boot": int(n_boot), "n_perm": int(n_perm), "seed": int(seed)},
        "validation": validation.to_dict("records"),
        "key_metrics": key_metrics.to_dict("records"),
        "claim_boundary": {
            "baseline_comparison_only": True,
            "raw_signal_reextraction": False,
            "clinical_prediction_or_treatment_claim": False,
            "parameter_tuning": False,
        },
    }

    paths = {
        "summary_json": write_json(summary, OUTPUT_DIR / "single_feature_baseline_summary.json"),
        "summary_md": write_report(summary, OUTPUT_DIR / "single_feature_baseline_summary.md"),
        "feature_mapping": write_csv(mapping, OUTPUT_DIR / "single_feature_baseline_feature_mapping.csv"),
        "validation": write_csv(validation, OUTPUT_DIR / "single_feature_baseline_validation.csv"),
        "candidate_scores": write_csv(scores, OUTPUT_DIR / "single_feature_baseline_candidate_scores.csv"),
        "query_diagnostics": write_csv(diagnostics, OUTPUT_DIR / "single_feature_baseline_query_diagnostics.csv"),
        "observed_metrics": write_csv(observed, OUTPUT_DIR / "single_feature_baseline_observed_metrics.csv"),
        "key_metrics": write_csv(key_metrics, OUTPUT_DIR / "single_feature_baseline_key_metrics.csv"),
        "bootstrap_ci": write_csv(bootstrap_ci, OUTPUT_DIR / "single_feature_baseline_subject_bootstrap_ci.csv"),
        "bootstrap_samples": write_csv(bootstrap_samples, OUTPUT_DIR / "single_feature_baseline_subject_bootstrap_samples.csv"),
        "random_label_null_summary": write_csv(random_summary, OUTPUT_DIR / "single_feature_baseline_random_label_null_summary.csv"),
        "random_label_null_samples": write_csv(random_samples, OUTPUT_DIR / "single_feature_baseline_random_label_null_samples.csv"),
        "matched_task_side_null_summary": write_csv(
            matched_summary, OUTPUT_DIR / "single_feature_baseline_matched_task_side_null_summary.csv"
        ),
        "matched_task_side_null_samples": write_csv(
            matched_samples, OUTPUT_DIR / "single_feature_baseline_matched_task_side_null_samples.csv"
        ),
        "manuscript_key_metrics": write_csv(
            key_metrics, MANUSCRIPT_TABLES_DIR / "journal_single_feature_baseline_key_metrics.csv"
        ),
        "manuscript_bootstrap_ci": write_csv(
            bootstrap_ci, MANUSCRIPT_TABLES_DIR / "journal_single_feature_baseline_subject_bootstrap_ci.csv"
        ),
        "manuscript_random_label_null": write_csv(
            random_summary, MANUSCRIPT_TABLES_DIR / "journal_single_feature_baseline_random_label_null_summary.csv"
        ),
        "manuscript_matched_task_side_null": write_csv(
            matched_summary, MANUSCRIPT_TABLES_DIR / "journal_single_feature_baseline_matched_task_side_null_summary.csv"
        ),
        "manuscript_feature_mapping": write_csv(
            mapping, MANUSCRIPT_TABLES_DIR / "journal_single_feature_baseline_feature_mapping.csv"
        ),
    }

    print("Single-feature STN biomarker baselines complete.")
    for _, row in key_metrics.iterrows():
        print(
            f"{row['dataset']} / {row['candidate_pool_condition']} / {row['variant_name']}: "
            f"top1={float(row['top1']):.3f}, MRR={float(row['mrr']):.3f}"
        )
    print(f"output folder: {OUTPUT_DIR}")
    return {"paths": {name: str(path) for name, path in paths.items()}, "summary": summary}


def main() -> None:
    """Entry point."""

    run()


if __name__ == "__main__":
    main()
