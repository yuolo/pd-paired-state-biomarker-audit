"""Experimental v2B hard-negative metric study for ds004998 paired retrieval.

This script tests literature-motivated algorithmic variants intended to reduce
subject-fingerprint confounding and improve hard-negative candidate-pool
performance. It does not modify the frozen v2A pipeline; v2B variants are
diagnostic experimental candidates evaluated against frozen v2A gates.
"""

from __future__ import annotations

import json
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

import matplotlib.pyplot as plt  # noqa: E402

from scripts.run_retrieval_distance_geometry_improvement import (  # noqa: E402
    build_feature_groups,
    cosine_distance,
    query_diagnostics_from_scores,
    to_builtin,
)
from scripts.run_v2a_scientific_audit_17subjects_33pairs import (  # noqa: E402
    APERIODIC_ALPHA,
    REQUIRED_CACHE_FILES,
    RANDOM_SEED,
    TOP_K,
    V2A_NAME,
    base_feature_names,
    load_subspace_inventory,
    query_metrics,
    read_csv_required,
    read_json_required,
    validate_cached_inputs,
    write_csv,
    write_json,
)
from scripts.run_paired_state_identifiability_study_17subjects_33pairs import (  # noqa: E402
    build_custom_candidate_pool,
    evaluate_frozen_on_candidate_pool,
    subject_fingerprint_cases,
)


OUTPUT_DIR = Path("outputs/v2B_hard_negative_metric_study_17subjects_33pairs")
V2B_TOP_K = TOP_K
V2B_ALPHA = APERIODIC_ALPHA
SUBJECT_COMPONENTS_TO_REMOVE = 2
HARD_NEGATIVE_EPS = 0.05
HARD_NEGATIVE_WEIGHT_MIN = 0.25
HARD_NEGATIVE_WEIGHT_MAX = 4.0
LOW_RANK_DIM = 10
LOW_RANK_SHRINKAGE = 0.25


@dataclass(frozen=True)
class V2BVariant:
    """A fixed experimental v2B variant."""

    name: str
    description: str
    remove_subject_components: int = 0
    hard_negative_weights: bool = False
    gated_aperiodic: bool = False
    low_rank_mahalanobis: bool = False
    transition_beta: float = 0.0
    use_aperiodic: bool = True


V2B_VARIANTS = [
    V2BVariant(
        name="v2B_subject_residualized_v2A",
        description="Remove top train-fold subject-centroid directions, then apply v2A-style top-5 reranking.",
        remove_subject_components=SUBJECT_COMPONENTS_TO_REMOVE,
    ),
    V2BVariant(
        name="v2B_hard_negative_diagonal_v2A",
        description="Train-fold diagonal feature weights from true-vs-hard-negative margins, then v2A reranking.",
        hard_negative_weights=True,
    ),
    V2BVariant(
        name="v2B_subject_residualized_hn_v2A",
        description="Subject-direction residualization plus train-fold hard-negative diagonal weights.",
        remove_subject_components=SUBJECT_COMPONENTS_TO_REMOVE,
        hard_negative_weights=True,
    ),
    V2BVariant(
        name="v2B_gated_aperiodic_top5",
        description="Frozen-style v2 candidate generator with aperiodic reranking gated to same task/side top-5 candidates.",
        gated_aperiodic=True,
    ),
    V2BVariant(
        name="v2B_subject_residualized_hn_gated_v2A",
        description="Subject residualization, hard-negative weights, and task/side-gated compact reranking.",
        remove_subject_components=SUBJECT_COMPONENTS_TO_REMOVE,
        hard_negative_weights=True,
        gated_aperiodic=True,
    ),
    V2BVariant(
        name="v2B_low_rank_mahalanobis_gated",
        description="Subject residualization with low-rank shrinkage Mahalanobis distance and gated compact reranking.",
        remove_subject_components=SUBJECT_COMPONENTS_TO_REMOVE,
        low_rank_mahalanobis=True,
        gated_aperiodic=True,
    ),
    V2BVariant(
        name="v2B_transition_consistency_gated",
        description="Subject residualization, hard-negative weights, train-fold transition consistency, and gated compact reranking.",
        remove_subject_components=SUBJECT_COMPONENTS_TO_REMOVE,
        hard_negative_weights=True,
        gated_aperiodic=True,
        transition_beta=0.25,
    ),
]


def write_text(lines: Iterable[str], path: Path) -> Path:
    """Write Markdown/text output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def stable_zscore(values: np.ndarray) -> np.ndarray:
    """Z-score with small-sample and constant-vector fallback."""

    arr = np.asarray(values, dtype=float)
    mean = float(np.nanmean(arr)) if len(arr) else 0.0
    std = float(np.nanstd(arr)) if len(arr) else 1.0
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return (arr - mean) / std


def pair_value(row: pd.Series, prefix: str, features: list[str]) -> np.ndarray:
    """Extract a pair vector by prefix."""

    return row[[f"{prefix}{feature}" for feature in features]].to_numpy(dtype=float)


def train_state_matrix(train: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return MedOff+MedOn train states and subject labels."""

    rows = []
    subjects = []
    for _, pair in train.iterrows():
        rows.append(pair_value(pair, "x_", features))
        subjects.append(str(pair["subject"]))
        rows.append(pair_value(pair, "medon_", features))
        subjects.append(str(pair["subject"]))
    return np.vstack(rows), subjects


def fit_subject_components(z_states: np.ndarray, subjects: list[str], n_components: int) -> np.ndarray:
    """Fit top between-subject centroid directions in train space."""

    if n_components <= 0 or z_states.size == 0:
        return np.empty((0, z_states.shape[1] if z_states.ndim == 2 else 0))
    unique_subjects = sorted(set(subjects))
    centroids = []
    for subject in unique_subjects:
        mask = np.asarray([label == subject for label in subjects], dtype=bool)
        if mask.any():
            centroids.append(np.nanmean(z_states[mask], axis=0))
    if len(centroids) < 2:
        return np.empty((0, z_states.shape[1]))
    centroid_matrix = np.vstack(centroids)
    centroid_matrix = centroid_matrix - np.nanmean(centroid_matrix, axis=0)
    _, _s, vt = np.linalg.svd(np.nan_to_num(centroid_matrix), full_matrices=False)
    n_keep = min(int(n_components), vt.shape[0])
    return vt[:n_keep]


def remove_components(values: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project out fitted orthonormal components."""

    arr = np.asarray(values, dtype=float)
    if components.size == 0:
        return arr
    return arr - (arr @ components.T) @ components


def hard_negative_rows(train: pd.DataFrame, query: pd.Series) -> pd.DataFrame:
    """Return same-subject and same-task/side hard negatives inside train."""

    subject = str(query["subject"])
    query_id = str(query["pair_id"])
    same_subject_wrong = train[
        train["subject"].astype(str).eq(subject)
        & ~train["pair_id"].astype(str).eq(query_id)
    ]
    other_same_task_side = train[
        ~train["subject"].astype(str).eq(subject)
        & train["task_original"].astype(str).eq(str(query["task_original"]))
        & train["side"].astype(str).eq(str(query["side"]))
    ]
    return pd.concat([same_subject_wrong, other_same_task_side], ignore_index=True).drop_duplicates("pair_id")


def feature_group_for_name(feature: str) -> str:
    """Compact feature-family label for outputs."""

    text = feature.lower()
    if "coupling" in text or "coherence" in text:
        return "coupling"
    if "aperiodic" in text or "residual" in text or "peak" in text:
        return "compact_aperiodic_residual"
    if text.startswith("stn_"):
        return "STN_features"
    if text.startswith("meg_") or text.startswith("motor_"):
        return "MEG_motor_power_features" if text.endswith("_power") else "MEG_motor_other"
    return "other"


def fit_hard_negative_weights(
    train: pd.DataFrame,
    features: list[str],
    transform_func: object,
) -> np.ndarray:
    """Fit fixed diagonal weights using train-only positive-vs-hard-negative ratios."""

    log_ratios = []
    for _, query in train.iterrows():
        qx = transform_func(query, "x_")
        positive = transform_func(query, "medon_")
        pos_abs = np.abs(positive - qx)
        negatives = hard_negative_rows(train, query)
        for _, negative in negatives.iterrows():
            neg_vec = transform_func(negative, "medon_")
            neg_abs = np.abs(neg_vec - qx)
            ratio = (neg_abs + HARD_NEGATIVE_EPS) / (pos_abs + HARD_NEGATIVE_EPS)
            log_ratios.append(np.log(np.clip(ratio, 1e-6, 1e6)))
    if not log_ratios:
        return np.ones(len(features), dtype=float)
    raw = np.exp(np.nanmedian(np.vstack(log_ratios), axis=0))
    weights = np.clip(raw, HARD_NEGATIVE_WEIGHT_MIN, HARD_NEGATIVE_WEIGHT_MAX)
    mean_weight = float(np.nanmean(weights))
    if np.isfinite(mean_weight) and mean_weight > 1e-8:
        weights = weights / mean_weight
    return np.where(np.isfinite(weights), weights, 1.0)


def fit_low_rank_metric(z_states: np.ndarray) -> dict[str, np.ndarray]:
    """Fit a low-rank shrinkage inverse covariance transform."""

    if z_states.shape[0] < 3 or z_states.shape[1] == 0:
        return {"components": np.empty((0, z_states.shape[1])), "inv_sqrt": np.empty(0)}
    centered = z_states - np.nanmean(z_states, axis=0)
    cov = np.cov(np.nan_to_num(centered), rowvar=False)
    cov = np.atleast_2d(cov)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    n_keep = min(LOW_RANK_DIM, eigvecs.shape[1], max(1, z_states.shape[0] - 2))
    eigvals = eigvals[:n_keep]
    eigvecs = eigvecs[:, :n_keep]
    mean_eig = float(np.nanmean(np.maximum(eigvals, 1e-6)))
    shrunk = (1.0 - LOW_RANK_SHRINKAGE) * np.maximum(eigvals, 1e-6) + LOW_RANK_SHRINKAGE * mean_eig
    return {"components": eigvecs.T, "inv_sqrt": 1.0 / np.sqrt(shrunk)}


def fit_metric_context(train: pd.DataFrame, features: list[str], variant: V2BVariant) -> dict[str, object]:
    """Fit all train-fold transforms for one variant and feature set."""

    states, subjects = train_state_matrix(train, features)
    mean = np.nanmean(states, axis=0)
    std = np.nanstd(states, axis=0, ddof=1)
    std = np.where(~np.isfinite(std) | (std < 1e-8), 1.0, std)
    z_states = (states - mean) / std
    subject_components = fit_subject_components(z_states, subjects, variant.remove_subject_components)

    def transform_raw(row: pd.Series, prefix: str) -> np.ndarray:
        z = (pair_value(row, prefix, features) - mean) / std
        return remove_components(z, subject_components)

    transformed_states = np.vstack([transform_raw(row, prefix) for _, row in train.iterrows() for prefix in ["x_", "medon_"]])
    weights = fit_hard_negative_weights(train, features, transform_raw) if variant.hard_negative_weights else np.ones(len(features))
    low_rank = fit_low_rank_metric(transformed_states) if variant.low_rank_mahalanobis else None
    deltas = np.vstack([transform_raw(row, "medon_") - transform_raw(row, "x_") for _, row in train.iterrows()])
    mean_delta = np.nanmean(deltas, axis=0)
    return {
        "features": features,
        "mean": mean,
        "std": std,
        "subject_components": subject_components,
        "weights": weights,
        "low_rank": low_rank,
        "mean_delta": mean_delta,
        "n_subject_components_removed": int(subject_components.shape[0]),
    }


def transform_with_context(row: pd.Series, prefix: str, context: dict[str, object]) -> np.ndarray:
    """Apply a fitted metric context to one row."""

    features = list(context["features"])
    mean = np.asarray(context["mean"], dtype=float)
    std = np.asarray(context["std"], dtype=float)
    components = np.asarray(context["subject_components"], dtype=float)
    z = (pair_value(row, prefix, features) - mean) / std
    return remove_components(z, components)


def grouped_weighted_cosine(
    qx: np.ndarray,
    cm: np.ndarray,
    features: list[str],
    clean_features: list[str],
    weights: np.ndarray,
) -> float:
    """Group-balanced weighted cosine distance."""

    groups = build_feature_groups(features, clean_features)
    if not groups:
        return cosine_distance(qx * weights, cm * weights)
    values = []
    for indices in groups.values():
        idx = np.asarray(indices, dtype=int)
        values.append(cosine_distance(qx[idx] * weights[idx], cm[idx] * weights[idx]))
    return float(np.nanmean(values)) if values else float("nan")


def low_rank_distance(qx: np.ndarray, cm: np.ndarray, low_rank: dict[str, np.ndarray] | None) -> float:
    """Low-rank shrinkage Mahalanobis distance."""

    if not low_rank:
        return float(np.linalg.norm(cm - qx))
    components = np.asarray(low_rank["components"], dtype=float)
    inv_sqrt = np.asarray(low_rank["inv_sqrt"], dtype=float)
    if components.size == 0:
        return float(np.linalg.norm(cm - qx))
    projected = components @ (cm - qx)
    return float(np.linalg.norm(projected * inv_sqrt))


def transition_similarity(qx: np.ndarray, cm: np.ndarray, mean_delta: np.ndarray) -> float:
    """Cosine similarity between candidate transition and train mean transition."""

    delta = cm - qx
    denom = float(np.linalg.norm(delta) * np.linalg.norm(mean_delta))
    if denom <= 1e-12 or not np.isfinite(denom):
        return 0.0
    return float(np.dot(delta, mean_delta) / denom)


def evaluate_metric_scores(
    pairs: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    features: list[str],
    clean_features: list[str],
    variant: V2BVariant,
    score_kind: str,
    score_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one train-fold metric over an arbitrary candidate pool."""

    pair_lookup = pairs.set_index("pair_id", drop=False)
    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for heldout_subject in sorted(pairs["subject"].astype(str).unique()):
        train = pairs[~pairs["subject"].astype(str).eq(heldout_subject)].copy()
        test_ids = pairs.loc[pairs["subject"].astype(str).eq(heldout_subject), "pair_id"].astype(str).tolist()
        if train.empty:
            continue
        context = fit_metric_context(train, features, variant)
        weights = np.asarray(context["weights"], dtype=float)
        for feature, weight in zip(features, weights, strict=False):
            weight_rows.append(
                {
                    "variant_name": variant.name,
                    "score_name": score_name,
                    "heldout_subject": heldout_subject,
                    "feature": feature,
                    "feature_group": feature_group_for_name(feature),
                    "weight": float(weight),
                    "n_subject_components_removed": int(context["n_subject_components_removed"]),
                }
            )
        for query_id in test_ids:
            query = pair_lookup.loc[str(query_id)]
            qx = transform_with_context(query, "x_", context)
            candidates = candidate_pool[candidate_pool["query_pair_id"].astype(str).eq(str(query_id))]
            query_rows: list[dict[str, object]] = []
            for _, candidate_meta in candidates.iterrows():
                candidate = pair_lookup.loc[str(candidate_meta["candidate_pair_id"])]
                cm = transform_with_context(candidate, "medon_", context)
                if score_kind == "standard":
                    score = cosine_distance(qx * weights, cm * weights)
                elif variant.low_rank_mahalanobis:
                    score = low_rank_distance(qx, cm, context.get("low_rank"))
                else:
                    score = grouped_weighted_cosine(qx, cm, features, clean_features, weights)
                query_rows.append(
                    {
                        **candidate_meta.to_dict(),
                        "heldout_subject": heldout_subject,
                        "variant_name": variant.name,
                        "score_name": score_name,
                        "score_kind": score_kind,
                        "n_features": int(len(features)),
                        "score_base": float(score),
                        "transition_similarity": transition_similarity(qx, cm, np.asarray(context["mean_delta"], dtype=float)),
                    }
                )
            if variant.transition_beta > 0 and query_rows:
                base = stable_zscore(np.asarray([row["score_base"] for row in query_rows], dtype=float))
                trans = stable_zscore(np.asarray([row["transition_similarity"] for row in query_rows], dtype=float))
                combined = base - float(variant.transition_beta) * trans
                for row, score in zip(query_rows, combined, strict=False):
                    row["score"] = float(score)
            else:
                for row in query_rows:
                    row["score"] = float(row["score_base"])
            rows.extend(query_rows)
    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores, pd.DataFrame(weight_rows)
    scores = scores.sort_values(["variant_name", "query_pair_id", "score"], ascending=True).copy()
    scores["rank"] = scores.groupby(["variant_name", "query_pair_id"], dropna=False).cumcount() + 1
    return scores, pd.DataFrame(weight_rows)


def combine_v2b_scores(v2_scores: pd.DataFrame, aper_scores: pd.DataFrame, variant: V2BVariant) -> pd.DataFrame:
    """Apply fixed top-k compact reranking, optionally gated by task/side."""

    if not variant.use_aperiodic or aper_scores.empty:
        out = v2_scores.copy()
        out["variant_name"] = variant.name
        return out
    key_cols = ["query_pair_id", "candidate_pair_id"]
    aper = aper_scores[key_cols + ["score"]].rename(columns={"score": "aperiodic_score"})
    merged = v2_scores.drop(columns=["rank"], errors="ignore").merge(aper, on=key_cols, how="left")
    merged["v2_score"] = pd.to_numeric(merged["score"], errors="coerce")
    merged["aperiodic_score"] = pd.to_numeric(merged["aperiodic_score"], errors="coerce")
    fill_value = float(merged["aperiodic_score"].max()) if merged["aperiodic_score"].notna().any() else 1.0
    merged["aperiodic_score"] = merged["aperiodic_score"].fillna(fill_value)
    rows = []
    for _, group in merged.groupby("query_pair_id", dropna=False):
        group = group.sort_values("v2_score", ascending=True).copy()
        group["v2_rank_for_stage"] = np.arange(1, len(group) + 1)
        group["true_in_v2_top_k"] = bool(
            group.loc[group["is_true_pair"].astype(bool), "v2_rank_for_stage"].iloc[0] <= V2B_TOP_K
        )
        inside = group["v2_rank_for_stage"] <= V2B_TOP_K
        if variant.gated_aperiodic:
            inside = (
                inside
                & group["candidate_task_original"].astype(str).eq(group["query_task_original"].astype(str))
                & group["candidate_side"].astype(str).eq(group["query_side"].astype(str))
            )
        group["score"] = 1000.0 + group["v2_rank_for_stage"].astype(float)
        if inside.any():
            stage = group.loc[inside].copy()
            combined_similarity = stable_zscore(-stage["v2_score"].to_numpy(dtype=float)) + V2B_ALPHA * stable_zscore(
                -stage["aperiodic_score"].to_numpy(dtype=float)
            )
            group.loc[stage.index, "score"] = -combined_similarity
        group["variant_name"] = variant.name
        group["gated_aperiodic"] = bool(variant.gated_aperiodic)
        rows.append(group)
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["variant_name", "query_pair_id", "score"], ascending=True).copy()
    out["rank"] = out.groupby(["variant_name", "query_pair_id"], dropna=False).cumcount() + 1
    return out


def evaluate_v2b_variant(
    pairs: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
    variant: V2BVariant,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate one v2B variant and return candidate scores, diagnostics, weights."""

    v2_scores, v2_weights = evaluate_metric_scores(
        pairs,
        candidate_pool,
        v2_features,
        clean_features,
        variant,
        "group_balanced",
        "v2_metric",
    )
    aper_scores, aper_weights = evaluate_metric_scores(
        pairs,
        candidate_pool,
        compact_features,
        [],
        variant,
        "standard",
        "compact_aperiodic_metric",
    )
    scores = combine_v2b_scores(v2_scores, aper_scores, variant)
    diagnostics = query_diagnostics_from_scores(scores)
    weights = pd.concat([v2_weights, aper_weights], ignore_index=True)
    return scores, diagnostics, weights


def metrics_row(condition: str, variant_name: str, diagnostics: pd.DataFrame, scores: pd.DataFrame) -> dict[str, object]:
    """Build one metrics row with subject-fingerprint diagnostics."""

    metrics = query_metrics(diagnostics)
    top = scores.sort_values("rank").groupby("query_pair_id", dropna=False).first().reset_index()
    top_same_subject = top["candidate_subject"].astype(str).eq(top["query_subject"].astype(str))
    top_same_subject_wrong = top_same_subject & ~top["is_true_pair"].astype(bool)
    top_same_task_side = (
        top["candidate_task_original"].astype(str).eq(top["query_task_original"].astype(str))
        & top["candidate_side"].astype(str).eq(top["query_side"].astype(str))
    )
    pool_sizes = scores.groupby("query_pair_id", dropna=False)["candidate_pair_id"].size()
    return {
        "candidate_pool_condition": condition,
        "variant_name": variant_name,
        **metrics,
        "candidate_set_size_mean": float(pool_sizes.mean()) if len(pool_sizes) else float("nan"),
        "candidate_set_size_min": int(pool_sizes.min()) if len(pool_sizes) else 0,
        "candidate_set_size_max": int(pool_sizes.max()) if len(pool_sizes) else 0,
        "top1_same_subject_rate": float(top_same_subject.mean()) if len(top) else float("nan"),
        "top1_same_subject_wrong_rate": float(top_same_subject_wrong.mean()) if len(top) else float("nan"),
        "top1_same_task_side_rate": float(top_same_task_side.mean()) if len(top) else float("nan"),
        "subject_fingerprint_gap": float(top_same_subject.mean()) - float(metrics["top1"]) if len(top) else float("nan"),
    }


def run_candidate_pool_ladder(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate frozen v2A and all v2B variants over severity-ladder pools."""

    conditions = [
        "original_frozen_matched_pool",
        "task_side_quality_ignored",
        "original_plus_same_subject_wrong_task_side",
        "all_medon_candidates",
        "true_plus_all_other_subject_medon",
    ]
    metric_rows = []
    diag_tables = []
    score_tables = []
    weight_tables = []
    for condition in conditions:
        candidate_pool = build_custom_candidate_pool(pairs, condition)
        _v2_scores, _v2_diag, frozen_scores, frozen_diag = evaluate_frozen_on_candidate_pool(
            pairs, candidate_pool, v2_features, compact_features, clean_features
        )
        frozen_scores = frozen_scores.copy()
        frozen_scores["candidate_pool_condition"] = condition
        frozen_diag = frozen_diag.copy()
        frozen_diag.insert(0, "candidate_pool_condition", condition)
        score_tables.append(frozen_scores)
        diag_tables.append(frozen_diag)
        metric_rows.append(metrics_row(condition, V2A_NAME, frozen_diag, frozen_scores))
        for variant in V2B_VARIANTS:
            scores, diagnostics, weights = evaluate_v2b_variant(
                pairs, candidate_pool, v2_features, compact_features, clean_features, variant
            )
            scores = scores.copy()
            scores["candidate_pool_condition"] = condition
            diagnostics = diagnostics.copy()
            diagnostics.insert(0, "candidate_pool_condition", condition)
            weights = weights.copy()
            weights["candidate_pool_condition"] = condition
            score_tables.append(scores)
            diag_tables.append(diagnostics)
            weight_tables.append(weights)
            metric_rows.append(metrics_row(condition, variant.name, diagnostics, scores))
    metrics = pd.DataFrame(metric_rows)
    diagnostics = pd.concat(diag_tables, ignore_index=True) if diag_tables else pd.DataFrame()
    scores = pd.concat(score_tables, ignore_index=True) if score_tables else pd.DataFrame()
    weights = pd.concat(weight_tables, ignore_index=True) if weight_tables else pd.DataFrame()
    return metrics, diagnostics, scores, weights


def build_gate_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare v2B variants against frozen v2A gates."""

    rows = []
    baseline = metrics[metrics["variant_name"].eq(V2A_NAME)].set_index("candidate_pool_condition", drop=False)
    primary_condition = "original_frozen_matched_pool"
    hard_condition = "original_plus_same_subject_wrong_task_side"
    all_condition = "all_medon_candidates"
    for variant in sorted(set(metrics["variant_name"]) - {V2A_NAME}):
        subset = metrics[metrics["variant_name"].eq(variant)].set_index("candidate_pool_condition", drop=False)
        if primary_condition not in subset.index:
            continue
        primary = subset.loc[primary_condition]
        hard = subset.loc[hard_condition] if hard_condition in subset.index else pd.Series(dtype=object)
        all_pool = subset.loc[all_condition] if all_condition in subset.index else pd.Series(dtype=object)
        base_primary = baseline.loc[primary_condition]
        base_hard = baseline.loc[hard_condition]
        base_all = baseline.loc[all_condition]
        gates = {
            "primary_top1_not_worse_than_v2A_minus_0_03": float(primary["top1"]) >= float(base_primary["top1"]) - 0.03,
            "primary_mrr_not_worse_than_v2A_minus_0_03": float(primary["mrr"]) >= float(base_primary["mrr"]) - 0.03,
            "hard_negative_top1_improves_v2A": float(hard.get("top1", np.nan)) > float(base_hard["top1"]),
            "all_medon_top1_improves_v2A": float(all_pool.get("top1", np.nan)) > float(base_all["top1"]),
            "all_medon_subject_gap_reduces": float(all_pool.get("subject_fingerprint_gap", np.nan))
            < float(base_all["subject_fingerprint_gap"]),
        }
        rows.append(
            {
                "variant_name": variant,
                **gates,
                "n_gates_passed": int(sum(gates.values())),
                "selected_as_v2B_candidate": bool(all(gates.values())),
                "primary_top1": float(primary["top1"]),
                "primary_mrr": float(primary["mrr"]),
                "hard_negative_top1": float(hard.get("top1", np.nan)),
                "all_medon_top1": float(all_pool.get("top1", np.nan)),
                "all_medon_mrr": float(all_pool.get("mrr", np.nan)),
                "all_medon_subject_fingerprint_gap": float(all_pool.get("subject_fingerprint_gap", np.nan)),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(
        ["selected_as_v2B_candidate", "n_gates_passed", "hard_negative_top1", "all_medon_top1", "primary_top1"],
        ascending=[False, False, False, False, False],
    )


def summarize_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """Summarize hard-negative weights by variant and feature group."""

    if weights.empty:
        return pd.DataFrame()
    return (
        weights.groupby(["candidate_pool_condition", "variant_name", "score_name", "feature_group"], dropna=False)
        .agg(
            n_features=("feature", "nunique"),
            mean_weight=("weight", "mean"),
            median_weight=("weight", "median"),
            max_weight=("weight", "max"),
            min_weight=("weight", "min"),
        )
        .reset_index()
    )


def plot_metric(metrics: pd.DataFrame, metric: str, path: Path) -> None:
    """Plot ladder metric for frozen v2A and v2B variants."""

    fig, ax = plt.subplots(figsize=(12, 5.6))
    if metrics.empty:
        ax.text(0.5, 0.5, "No metrics", ha="center", va="center")
        ax.set_axis_off()
    else:
        conditions = list(metrics["candidate_pool_condition"].drop_duplicates())
        variants = [V2A_NAME, *[variant.name for variant in V2B_VARIANTS]]
        x = np.arange(len(conditions))
        width = min(0.11, 0.8 / max(1, len(variants)))
        palette = plt.cm.tab10(np.linspace(0, 1, len(variants)))
        for idx, variant in enumerate(variants):
            values = []
            for condition in conditions:
                row = metrics[
                    metrics["candidate_pool_condition"].eq(condition)
                    & metrics["variant_name"].eq(variant)
                ]
                values.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (idx - (len(variants) - 1) / 2) * width, values, width=width, label=variant, color=palette[idx])
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.set_title(f"v2B Candidate-Pool Ladder: {metric}")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_gate_summary(gates: pd.DataFrame, path: Path) -> None:
    """Plot number of gates passed per v2B variant."""

    fig, ax = plt.subplots(figsize=(10, 4.8))
    if gates.empty:
        ax.text(0.5, 0.5, "No gate rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        data = gates.sort_values("n_gates_passed", ascending=False)
        x = np.arange(len(data))
        ax.bar(x, data["n_gates_passed"], color="#4C78A8")
        ax.set_ylim(0, 5)
        ax.set_ylabel("Gates passed")
        ax.set_xticks(x)
        ax.set_xticklabels(data["variant_name"], rotation=35, ha="right")
        ax.set_title("Experimental v2B Gate Summary")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(summary: dict[str, object], path: Path) -> Path:
    """Write concise Markdown report."""

    gates = pd.DataFrame(summary["gate_comparison"])
    selected = gates[gates["selected_as_v2B_candidate"].astype(bool)] if not gates.empty else pd.DataFrame()
    lines = [
        "# v2B Hard-Negative Metric Study",
        "",
        "Scope: experimental algorithmic variants only. Frozen v2A is unchanged.",
        "",
        "## Validation",
        f"- complete_pairs: {summary['validation']['complete_pairs']}",
        f"- subjects_with_complete_pairs: {summary['validation']['subjects_with_complete_pairs']}",
        f"- Rest excluded: {summary['validation']['rest_excluded']}",
        f"- top_k: {summary['frozen_settings']['top_k']}",
        f"- aperiodic_alpha: {summary['frozen_settings']['aperiodic_alpha']}",
        "",
        "## Candidate Selection",
    ]
    if selected.empty:
        lines.append("- No experimental v2B variant passed all pre-specified gates.")
    else:
        for _, row in selected.iterrows():
            lines.append(f"- {row['variant_name']} passed all gates.")
    lines.extend(["", "## Gate Ranking"])
    for _, row in gates.iterrows():
        lines.append(
            f"- {row['variant_name']}: gates={int(row['n_gates_passed'])}/5, "
            f"primary_top1={float(row['primary_top1']):.3f}, "
            f"hard_top1={float(row['hard_negative_top1']):.3f}, "
            f"allMedOn_top1={float(row['all_medon_top1']):.3f}, "
            f"gap={float(row['all_medon_subject_fingerprint_gap']):.3f}"
        )
    lines.extend(["", "## Warnings"])
    warnings = summary.get("warnings", [])
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- none")
    return write_text(lines, path)


def run() -> dict[str, object]:
    """Run the experimental v2B study."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = read_csv_required(REQUIRED_CACHE_FILES["pairs"], dtype={"run_off": str, "run_on": str})
    logical = read_csv_required(REQUIRED_CACHE_FILES["logical_manifest"])
    usable = read_csv_required(REQUIRED_CACHE_FILES["usable_pair_inventory"])
    excluded = read_csv_required(REQUIRED_CACHE_FILES["excluded_subjects"])
    cohort_summary = read_json_required(REQUIRED_CACHE_FILES["cohort_summary"])
    frozen_summary = read_json_required(REQUIRED_CACHE_FILES["frozen_summary"])
    extraction_failure_log = read_csv_required(REQUIRED_CACHE_FILES["extraction_failure_log"])
    subspaces = load_subspace_inventory(REQUIRED_CACHE_FILES["subspaces"])
    compact_inventory = read_json_required(REQUIRED_CACHE_FILES["compact_inventory"])
    validation = validate_cached_inputs(
        pairs, logical, usable, excluded, cohort_summary, frozen_summary, extraction_failure_log
    )

    all_features = base_feature_names(pairs)
    v2_features = [feature for feature in subspaces.get("v2_reference", []) if feature in all_features]
    compact_features = [str(feature) for feature in compact_inventory.get("selected_features", []) if str(feature) in all_features]
    clean_features = [feature for feature in subspaces.get("clean_stable_features", []) if feature in all_features]
    if not v2_features:
        raise AssertionError("Frozen v2_reference feature list is empty.")
    if not compact_features:
        raise AssertionError("Frozen compact aperiodic/residual feature list is empty.")

    metrics, diagnostics, scores, weights = run_candidate_pool_ladder(
        pairs, v2_features, compact_features, clean_features
    )
    fingerprint_summary, fingerprint_cases = subject_fingerprint_cases(
        scores[scores["candidate_pool_condition"].isin(["all_medon_candidates", "original_plus_same_subject_wrong_task_side"])]
    )
    gates = build_gate_table(metrics)
    weight_summary = summarize_weights(weights)

    selected = gates[gates["selected_as_v2B_candidate"].astype(bool)] if not gates.empty else pd.DataFrame()
    warnings = []
    if selected.empty:
        warnings.append("No experimental v2B variant passed all pre-specified gates.")
    best = gates.iloc[0].to_dict() if not gates.empty else {}
    if best and int(best.get("n_gates_passed", 0)) < 5:
        warnings.append(f"Best experimental variant passed {int(best['n_gates_passed'])}/5 gates.")

    summary = {
        "validation": validation,
        "frozen_settings": {
            "top_k": TOP_K,
            "aperiodic_alpha": APERIODIC_ALPHA,
            "frozen_v2A_changed": False,
        },
        "experimental_constants": {
            "subject_components_to_remove": SUBJECT_COMPONENTS_TO_REMOVE,
            "hard_negative_weight_min": HARD_NEGATIVE_WEIGHT_MIN,
            "hard_negative_weight_max": HARD_NEGATIVE_WEIGHT_MAX,
            "low_rank_dim": LOW_RANK_DIM,
            "low_rank_shrinkage": LOW_RANK_SHRINKAGE,
        },
        "variant_descriptions": [variant.__dict__ for variant in V2B_VARIANTS],
        "gate_comparison": gates.to_dict("records"),
        "selected_v2B_candidates": selected.to_dict("records") if not selected.empty else [],
        "warnings": warnings,
        "external_oxford_mrc_dataset_used": False,
        "raw_fif_reread": False,
    }

    paths = {
        "summary_json": write_json(summary, OUTPUT_DIR / "v2b_hard_negative_metric_summary.json"),
        "summary_md": write_report(summary, OUTPUT_DIR / "v2b_hard_negative_metric_summary.md"),
        "metrics": write_csv(metrics, OUTPUT_DIR / "v2b_candidate_pool_ladder_metrics.csv"),
        "diagnostics": write_csv(diagnostics, OUTPUT_DIR / "v2b_candidate_pool_ladder_query_diagnostics.csv"),
        "scores": write_csv(scores, OUTPUT_DIR / "v2b_candidate_pool_ladder_candidate_scores.csv"),
        "fingerprint_summary": write_csv(fingerprint_summary, OUTPUT_DIR / "v2b_subject_fingerprint_summary.csv"),
        "fingerprint_cases": write_csv(fingerprint_cases, OUTPUT_DIR / "v2b_subject_fingerprint_cases.csv"),
        "gates": write_csv(gates, OUTPUT_DIR / "v2b_gate_comparison.csv"),
        "feature_weights": write_csv(weights, OUTPUT_DIR / "v2b_feature_weights_by_fold.csv"),
        "feature_weight_summary": write_csv(weight_summary, OUTPUT_DIR / "v2b_feature_weight_summary.csv"),
    }
    plot_metric(metrics, "top1", OUTPUT_DIR / "v2b_candidate_pool_ladder_top1.png")
    plot_metric(metrics, "mrr", OUTPUT_DIR / "v2b_candidate_pool_ladder_mrr.png")
    plot_gate_summary(gates, OUTPUT_DIR / "v2b_gate_summary.png")

    frozen_original = metrics[
        metrics["candidate_pool_condition"].eq("original_frozen_matched_pool")
        & metrics["variant_name"].eq(V2A_NAME)
    ].iloc[0]
    frozen_all = metrics[
        metrics["candidate_pool_condition"].eq("all_medon_candidates")
        & metrics["variant_name"].eq(V2A_NAME)
    ].iloc[0]
    print(f"complete_pairs: {validation['complete_pairs']}")
    print(f"subjects_with_complete_pairs: {validation['subjects_with_complete_pairs']}")
    print(f"Rest excluded: {validation['rest_excluded']}")
    print(f"sub-BYJoWR excluded: {validation['sub_BYJoWR_excluded']}")
    print(f"split files collapsed logically: {validation['split_files_collapsed_logically']}")
    print(f"top_k: {TOP_K}")
    print(f"aperiodic_alpha: {APERIODIC_ALPHA}")
    print(f"frozen original v2A top1/MRR: {float(frozen_original['top1']):.3f} / {float(frozen_original['mrr']):.3f}")
    print(f"frozen all-MedOn v2A top1/MRR: {float(frozen_all['top1']):.3f} / {float(frozen_all['mrr']):.3f}")
    if not gates.empty:
        top = gates.iloc[0]
        print(
            "best experimental variant: "
            f"{top['variant_name']} ({int(top['n_gates_passed'])}/5 gates; "
            f"primary top1={float(top['primary_top1']):.3f}, "
            f"hard top1={float(top['hard_negative_top1']):.3f}, "
            f"all-MedOn top1={float(top['all_medon_top1']):.3f})"
        )
    print(f"output folder: {OUTPUT_DIR}")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("warnings: none")
    return {"paths": {key: str(path) for key, path in paths.items()}, "warnings": warnings}


def main() -> None:
    """Entry point."""

    run()


if __name__ == "__main__":
    main()
