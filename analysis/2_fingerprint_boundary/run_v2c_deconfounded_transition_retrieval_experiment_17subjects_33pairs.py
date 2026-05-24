"""Experimental v2C deconfounded transition-retrieval study for ds004998.

This script is an exploratory study layer around the frozen v2A and selected
experimental v2B-gated outputs. It does not modify frozen v2A/v2B logic, does
not read raw FIF files, does not use the Oxford/MRC external dataset, and does
not make clinical prediction, treatment, DBS, or causal medication-effect
claims.
"""

from __future__ import annotations

import json
import math
import os
import sys
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

from scripts.run_paired_state_identifiability_study_17subjects_33pairs import (  # noqa: E402
    build_custom_candidate_pool,
)
from scripts.run_retrieval_distance_geometry_improvement import (  # noqa: E402
    cosine_distance,
    query_diagnostics_from_scores,
    to_builtin,
)
from scripts.run_v2a_scientific_audit_17subjects_33pairs import (  # noqa: E402
    APERIODIC_ALPHA,
    N_BOOT,
    RANDOM_SEED,
    REQUIRED_CACHE_FILES,
    TOP_K,
    V2A_NAME,
    base_feature_names,
    load_subspace_inventory,
    percentile_ci,
    query_metrics,
    read_csv_required,
    read_json_required,
    validate_cached_inputs,
    write_csv,
    write_json,
)
from scripts.run_v2b_gated_scientific_audit_17subjects_33pairs import (  # noqa: E402
    ALL_MEDON_CONDITION,
    HARD_NEGATIVE_CONDITION,
    PRIMARY_CONDITION,
    V2B_NAME,
)
from scripts.run_v2b_hard_negative_metric_study_17subjects_33pairs import (  # noqa: E402
    V2BVariant,
    combine_v2b_scores,
    evaluate_metric_scores,
    fit_metric_context,
    hard_negative_rows,
    transform_with_context,
)


OUTPUT_DIR = Path("outputs/v2C_deconfounded_transition_retrieval_experiment_17subjects_33pairs")
V2B_AUDIT_DIR = Path("outputs/v2B_gated_scientific_audit_17subjects_33pairs")
DTSI_DIR = Path("outputs/dopaminergic_transition_specificity_index_17subjects_33pairs")

FORWARD_SCORES_PATH = V2B_AUDIT_DIR / "v2b_candidate_pool_ladder_candidate_scores.csv"
REVERSE_SCORES_PATH = DTSI_DIR / "reverse_medon_to_medoff_v2b_scores.csv"

TRUE_PLUS_OTHER_CONDITION = "true_plus_all_other_subject_medon"
V2C_FAMILY_NAME = "v2C_family_reweighted"
V2C_CYCLE_NAME = "v2C_cycle_consistent"
V2C_DECONFOUNDED_NAME = "v2C_deconfounded_cycle"

EVALUATED_CONDITIONS = [
    PRIMARY_CONDITION,
    HARD_NEGATIVE_CONDITION,
    ALL_MEDON_CONDITION,
    TRUE_PLUS_OTHER_CONDITION,
]
SUMMARY_VARIANTS = [V2A_NAME, V2B_NAME, V2C_FAMILY_NAME, V2C_CYCLE_NAME, V2C_DECONFOUNDED_NAME]

CYCLE_WEIGHT = 1.0
BROAD_POOL_MIN_CANDIDATES = 20
WRONG_TASK_SIDE_PENALTY = 2.0
SAME_SUBJECT_WRONG_PENALTY = 2.0
FAMILY_WEIGHT_MIN = 0.5
FAMILY_WEIGHT_MAX = 1.5


def write_text(lines: Iterable[str], path: Path) -> Path:
    """Write Markdown/text output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def as_bool(series: pd.Series) -> pd.Series:
    """Convert bool-like CSV values to bool."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def stable_zscore(values: Iterable[float]) -> np.ndarray:
    """Candidate-set z-score with constant-vector fallback."""

    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    mean = float(np.nanmean(arr)) if len(arr) else 0.0
    std = float(np.nanstd(arr)) if len(arr) else 1.0
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return (arr - mean) / std


def normalize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Normalize cached score dtypes and helper flags."""

    out = scores.copy()
    for column in [
        "is_true_pair",
        "true_in_v2_top_k",
        "gated_aperiodic",
        "residualized",
        "same_subject_candidate",
        "same_task_side_candidate",
    ]:
        if column in out:
            out[column] = as_bool(out[column])
    for column in [
        "score",
        "rank",
        "aperiodic_score",
        "v2_score",
        "v2_rank_for_stage",
        "score_base",
        "transition_similarity",
    ]:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def rank_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank scores by query and variant."""

    ranked = scores.sort_values(["candidate_pool_condition", "variant_name", "query_pair_id", "score"]).copy()
    ranked["rank"] = ranked.groupby(
        ["candidate_pool_condition", "variant_name", "query_pair_id"], dropna=False
    ).cumcount() + 1
    return ranked


def feature_family(feature: str) -> str:
    """Assign a compact feature-family label."""

    text = feature.lower()
    if "coupling" in text or "coherence" in text or "cortico_stn" in text:
        return "coupling"
    if text.startswith("stn_"):
        return "STN_features"
    if (text.startswith("meg_") or text.startswith("motor_")) and "power" in text:
        return "MEG_motor_power_features"
    if "beta" in text or "gamma" in text:
        return "raw_beta_gamma_power"
    return "other_v2_features"


def feature_family_indices(features: list[str]) -> dict[str, list[int]]:
    """Return feature-family index lists."""

    groups: dict[str, list[int]] = {}
    for idx, feature in enumerate(features):
        groups.setdefault(feature_family(feature), []).append(idx)
    return {name: indices for name, indices in groups.items() if indices}


def family_weighted_distance(
    qx: np.ndarray,
    candidate: np.ndarray,
    groups: dict[str, list[int]],
    weights: dict[str, float],
) -> float:
    """Weighted mean of per-family cosine distances."""

    numerator = 0.0
    denominator = 0.0
    for family, indices in groups.items():
        idx = np.asarray(indices, dtype=int)
        weight = float(weights.get(family, 1.0))
        numerator += weight * cosine_distance(qx[idx], candidate[idx])
        denominator += weight
    if denominator <= 1e-12:
        return cosine_distance(qx, candidate)
    return float(numerator / denominator)


def fit_family_reliability_weights(
    train: pd.DataFrame,
    groups: dict[str, list[int]],
    context: dict[str, object],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit train-fold family reliability weights from true-vs-hard-negative margins."""

    margin_rows: list[dict[str, object]] = []
    for _, query in train.iterrows():
        negatives = hard_negative_rows(train, query)
        if negatives.empty:
            continue
        qx = transform_with_context(query, "x_", context)
        true = transform_with_context(query, "medon_", context)
        neg_vectors = [transform_with_context(row, "medon_", context) for _, row in negatives.iterrows()]
        for family, indices in groups.items():
            idx = np.asarray(indices, dtype=int)
            true_distance = cosine_distance(qx[idx], true[idx])
            wrong_distances = [cosine_distance(qx[idx], candidate[idx]) for candidate in neg_vectors]
            best_wrong = float(np.nanmin(wrong_distances)) if wrong_distances else float("nan")
            margin = best_wrong - true_distance
            margin_rows.append(
                {
                    "family": family,
                    "query_pair_id": str(query["pair_id"]),
                    "hard_negative_margin": margin,
                    "true_beats_best_hard_negative": bool(np.isfinite(margin) and margin > 0),
                }
            )
    margins = pd.DataFrame(margin_rows)
    raw_weights: dict[str, float] = {}
    for family in groups:
        subset = margins[margins["family"].astype(str).eq(family)] if not margins.empty else pd.DataFrame()
        if subset.empty:
            success_rate = 0.5
        else:
            success_rate = float(subset["true_beats_best_hard_negative"].astype(float).mean())
        raw_weights[family] = float(np.clip(0.5 + success_rate, FAMILY_WEIGHT_MIN, FAMILY_WEIGHT_MAX))
    mean_weight = float(np.mean(list(raw_weights.values()))) if raw_weights else 1.0
    if not np.isfinite(mean_weight) or mean_weight <= 1e-8:
        mean_weight = 1.0
    weights = {family: float(weight / mean_weight) for family, weight in raw_weights.items()}
    return weights, margins


def evaluate_family_reweighted_v2_scores(
    pairs: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate the train-fold family-reweighted v2 candidate generator."""

    pair_lookup = pairs.set_index("pair_id", drop=False)
    groups = feature_family_indices(features)
    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    margin_tables: list[pd.DataFrame] = []
    null_variant = V2BVariant(name=V2C_FAMILY_NAME, description="family reweighted context")

    for heldout_subject in sorted(pairs["subject"].astype(str).unique()):
        train = pairs[~pairs["subject"].astype(str).eq(heldout_subject)].copy()
        test_ids = pairs.loc[pairs["subject"].astype(str).eq(heldout_subject), "pair_id"].astype(str).tolist()
        if train.empty:
            continue
        context = fit_metric_context(train, features, null_variant)
        weights, margins = fit_family_reliability_weights(train, groups, context)
        if not margins.empty:
            margins = margins.copy()
            margins["heldout_subject"] = heldout_subject
            margin_tables.append(margins)
        for family, indices in groups.items():
            weight_rows.append(
                {
                    "variant_name": V2C_FAMILY_NAME,
                    "score_name": "v2_family_reweighted_metric",
                    "heldout_subject": heldout_subject,
                    "feature_group": family,
                    "n_features": int(len(indices)),
                    "family_weight": float(weights.get(family, 1.0)),
                }
            )
        for query_id in test_ids:
            query = pair_lookup.loc[str(query_id)]
            qx = transform_with_context(query, "x_", context)
            candidates = candidate_pool[candidate_pool["query_pair_id"].astype(str).eq(str(query_id))]
            for _, candidate_meta in candidates.iterrows():
                candidate = pair_lookup.loc[str(candidate_meta["candidate_pair_id"])]
                cm = transform_with_context(candidate, "medon_", context)
                score = family_weighted_distance(qx, cm, groups, weights)
                rows.append(
                    {
                        **candidate_meta.to_dict(),
                        "heldout_subject": heldout_subject,
                        "variant_name": V2C_FAMILY_NAME,
                        "score_name": "v2_family_reweighted_metric",
                        "score_kind": "family_reweighted_group_balanced",
                        "n_features": int(len(features)),
                        "score": float(score),
                    }
                )
    scores = pd.DataFrame(rows)
    if not scores.empty:
        scores = scores.sort_values(["variant_name", "query_pair_id", "score"]).copy()
        scores["rank"] = scores.groupby(["variant_name", "query_pair_id"], dropna=False).cumcount() + 1
    margins = pd.concat(margin_tables, ignore_index=True) if margin_tables else pd.DataFrame()
    return scores, pd.DataFrame(weight_rows), margins


def evaluate_v2c_family_reweighted(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    conditions: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate family-reweighted v2 plus frozen top-5 compact reranking."""

    score_tables: list[pd.DataFrame] = []
    weight_tables: list[pd.DataFrame] = []
    margin_tables: list[pd.DataFrame] = []
    variant = V2BVariant(
        name=V2C_FAMILY_NAME,
        description="Train-fold feature-family reliability weighting plus frozen-style gated compact rerank.",
        gated_aperiodic=True,
    )
    for condition in conditions:
        candidate_pool = build_custom_candidate_pool(pairs, condition)
        v2_scores, weights, margins = evaluate_family_reweighted_v2_scores(pairs, candidate_pool, v2_features)
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
        scores["candidate_pool_condition"] = condition
        score_tables.append(scores)

        weights = weights.copy()
        weights["candidate_pool_condition"] = condition
        weight_tables.append(weights)
        if not aper_weights.empty:
            compact_weight_summary = (
                aper_weights.groupby(["heldout_subject", "score_name", "feature_group"], dropna=False)
                .agg(n_features=("feature", "size"), family_weight=("weight", "mean"))
                .reset_index()
            )
            compact_weight_summary["variant_name"] = V2C_FAMILY_NAME
            compact_weight_summary["candidate_pool_condition"] = condition
            weight_tables.append(compact_weight_summary)
        if not margins.empty:
            margins = margins.copy()
            margins["candidate_pool_condition"] = condition
            margin_tables.append(margins)
    all_scores = pd.concat(score_tables, ignore_index=True) if score_tables else pd.DataFrame()
    all_weights = pd.concat(weight_tables, ignore_index=True) if weight_tables else pd.DataFrame()
    all_margins = pd.concat(margin_tables, ignore_index=True) if margin_tables else pd.DataFrame()
    return all_scores, all_weights, all_margins


def reverse_lookup(reverse_scores: pd.DataFrame) -> pd.DataFrame:
    """Return reverse MedOn-to-MedOff scores aligned to forward query/candidate pairs."""

    focus = reverse_scores[
        reverse_scores["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & reverse_scores["variant_name"].astype(str).eq(V2B_NAME)
    ].copy()
    keep = [
        "query_pair_id",
        "candidate_pair_id",
        "score",
        "rank",
        "v2_score",
        "aperiodic_score",
    ]
    lookup = focus[keep].rename(
        columns={
            "query_pair_id": "candidate_pair_id",
            "candidate_pair_id": "query_pair_id",
            "score": "reverse_score",
            "rank": "reverse_rank",
            "v2_score": "reverse_v2_score",
            "aperiodic_score": "reverse_aperiodic_score",
        }
    )
    return lookup


def cycle_rerank_scores(
    forward_scores: pd.DataFrame,
    reverse_scores: pd.DataFrame,
    variant_name: str,
    deconfounded: bool = False,
) -> pd.DataFrame:
    """Build cycle-consistent v2C scores from cached v2B forward and reverse scores."""

    rev = reverse_lookup(reverse_scores)
    focus = forward_scores[
        forward_scores["candidate_pool_condition"].astype(str).isin(EVALUATED_CONDITIONS)
        & forward_scores["variant_name"].astype(str).eq(V2B_NAME)
    ].copy()
    merged = focus.merge(rev, on=["query_pair_id", "candidate_pair_id"], how="left")
    rows: list[pd.DataFrame] = []
    for (_, query_id), group in merged.groupby(["candidate_pool_condition", "query_pair_id"], dropna=False):
        group = group.copy()
        candidate_count = int(pd.to_numeric(group["number_of_candidates"], errors="coerce").max())
        use_cycle = (not deconfounded) or candidate_count >= BROAD_POOL_MIN_CANDIDATES
        if not use_cycle:
            group["forward_score"] = pd.to_numeric(group["score"], errors="coerce")
            group["score"] = group["forward_score"]
            group["cycle_score_component"] = 0.0
            group["deconfound_penalty"] = 0.0
        else:
            forward_z = stable_zscore(group["score"])
            reverse_fill = float(pd.to_numeric(group["reverse_score"], errors="coerce").max())
            if not np.isfinite(reverse_fill):
                reverse_fill = float(pd.to_numeric(group["score"], errors="coerce").max())
            reverse_z = stable_zscore(group["reverse_score"].fillna(reverse_fill))
            wrong_task_side = (
                (
                    group["candidate_task_original"].astype(str).ne(group["query_task_original"].astype(str))
                    | group["candidate_side"].astype(str).ne(group["query_side"].astype(str))
                )
                & ~group["is_true_pair"].astype(bool)
            )
            same_subject_wrong = (
                group["candidate_subject"].astype(str).eq(group["query_subject"].astype(str))
                & ~group["is_true_pair"].astype(bool)
            )
            penalty = np.zeros(len(group), dtype=float)
            if deconfounded:
                penalty += WRONG_TASK_SIDE_PENALTY * wrong_task_side.astype(float).to_numpy()
                penalty += SAME_SUBJECT_WRONG_PENALTY * same_subject_wrong.astype(float).to_numpy()
            group["forward_score"] = pd.to_numeric(group["score"], errors="coerce")
            group["cycle_score_component"] = CYCLE_WEIGHT * reverse_z
            group["deconfound_penalty"] = penalty
            group["score"] = forward_z + group["cycle_score_component"].to_numpy(dtype=float) + penalty
        group["variant_name"] = variant_name
        group["v2c_score_mode"] = "broad_pool_deconfounded_cycle" if deconfounded else "cycle_consistent"
        rows.append(group)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return rank_scores(out)


def metrics_row(condition: str, variant_name: str, scores: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    """Build one metrics row plus diagnostics."""

    diagnostics = query_diagnostics_from_scores(scores)
    metrics = query_metrics(diagnostics)
    top = scores.sort_values("rank").groupby("query_pair_id", dropna=False).first().reset_index()
    if top.empty:
        top_same_subject_rate = float("nan")
        top_same_subject_wrong_rate = float("nan")
        top_same_task_side_rate = float("nan")
        top_wrong_task_side_rate = float("nan")
    else:
        top_same_subject = top["candidate_subject"].astype(str).eq(top["query_subject"].astype(str))
        top_same_subject_wrong = top_same_subject & ~top["is_true_pair"].astype(bool)
        top_same_task_side = (
            top["candidate_task_original"].astype(str).eq(top["query_task_original"].astype(str))
            & top["candidate_side"].astype(str).eq(top["query_side"].astype(str))
        )
        top_wrong_task_side = ~top_same_task_side & ~top["is_true_pair"].astype(bool)
        top_same_subject_rate = float(top_same_subject.astype(float).mean())
        top_same_subject_wrong_rate = float(top_same_subject_wrong.astype(float).mean())
        top_same_task_side_rate = float(top_same_task_side.astype(float).mean())
        top_wrong_task_side_rate = float(top_wrong_task_side.astype(float).mean())
    row = {
        "candidate_pool_condition": condition,
        "variant_name": variant_name,
        **metrics,
        "top1_same_subject_rate": top_same_subject_rate,
        "top1_same_subject_wrong_rate": top_same_subject_wrong_rate,
        "top1_same_task_side_rate": top_same_task_side_rate,
        "top1_wrong_task_or_side_rate": top_wrong_task_side_rate,
        "subject_fingerprint_gap": top_same_subject_rate - float(metrics["top1"])
        if np.isfinite(top_same_subject_rate) and np.isfinite(float(metrics["top1"]))
        else float("nan"),
    }
    diagnostics = diagnostics.copy()
    diagnostics.insert(0, "candidate_pool_condition", condition)
    return row, diagnostics


def build_metrics_tables(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build metrics and query diagnostics for all variants and conditions."""

    metric_rows: list[dict[str, object]] = []
    diagnostics_tables: list[pd.DataFrame] = []
    focus = scores[
        scores["candidate_pool_condition"].astype(str).isin(EVALUATED_CONDITIONS)
        & scores["variant_name"].astype(str).isin(SUMMARY_VARIANTS)
    ].copy()
    for (condition, variant), subset in focus.groupby(["candidate_pool_condition", "variant_name"], dropna=False):
        row, diagnostics = metrics_row(str(condition), str(variant), subset)
        metric_rows.append(row)
        diagnostics_tables.append(diagnostics)
    metrics = pd.DataFrame(metric_rows)
    diagnostics = pd.concat(diagnostics_tables, ignore_index=True) if diagnostics_tables else pd.DataFrame()
    return metrics, diagnostics


def comparison_to_v2b(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare v2C variants against selected v2B-gated metrics."""

    rows: list[dict[str, object]] = []
    baseline = metrics[metrics["variant_name"].astype(str).eq(V2B_NAME)].set_index("candidate_pool_condition", drop=False)
    for _, row in metrics.iterrows():
        condition = str(row["candidate_pool_condition"])
        variant = str(row["variant_name"])
        if variant == V2B_NAME or condition not in baseline.index:
            continue
        base = baseline.loc[condition]
        rows.append(
            {
                "candidate_pool_condition": condition,
                "variant_name": variant,
                "delta_top1_vs_v2B": float(row["top1"]) - float(base["top1"]),
                "delta_mrr_vs_v2B": float(row["mrr"]) - float(base["mrr"]),
                "delta_failures_vs_v2B": int(row["failures"]) - int(base["failures"]),
                "delta_subject_fingerprint_gap_vs_v2B": float(row["subject_fingerprint_gap"])
                - float(base["subject_fingerprint_gap"]),
                "v2B_top1": float(base["top1"]),
                "v2C_top1": float(row["top1"]),
                "v2B_mrr": float(base["mrr"]),
                "v2C_mrr": float(row["mrr"]),
            }
        )
    return pd.DataFrame(rows)


def subject_bootstrap_ci(diagnostics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subject-aware bootstrap CIs for top1 and MRR."""

    rng = np.random.default_rng(RANDOM_SEED)
    sample_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for (condition, variant), subset in diagnostics.groupby(["candidate_pool_condition", "variant_name"], dropna=False):
        subjects = sorted(subset["query_subject"].astype(str).unique())
        if not subjects:
            continue
        subject_stats = {}
        for subject in subjects:
            subject_rows = subset[subset["query_subject"].astype(str).eq(subject)]
            subject_stats[subject] = {
                "n": int(len(subject_rows)),
                "top1_sum": float(subject_rows["top_ranked_is_true_pair"].astype(float).sum()),
                "mrr_sum": float(pd.to_numeric(subject_rows["reciprocal_rank"], errors="coerce").sum()),
            }
        observed = query_metrics(subset)
        for bootstrap_index in range(N_BOOT):
            draw = rng.choice(subjects, size=len(subjects), replace=True)
            n_rows = 0
            top1_sum = 0.0
            mrr_sum = 0.0
            for subject in draw:
                stats = subject_stats[str(subject)]
                n_rows += int(stats["n"])
                top1_sum += float(stats["top1_sum"])
                mrr_sum += float(stats["mrr_sum"])
            sample_rows.append(
                {
                    "bootstrap_index": int(bootstrap_index),
                    "candidate_pool_condition": condition,
                    "variant_name": variant,
                    "top1": float(top1_sum / n_rows) if n_rows else float("nan"),
                    "mrr": float(mrr_sum / n_rows) if n_rows else float("nan"),
                    "n_query_rows": int(n_rows),
                    "seed": RANDOM_SEED,
                    "bootstrap_unit": "subject",
                }
            )
        samples_so_far = pd.DataFrame([row for row in sample_rows if row["candidate_pool_condition"] == condition and row["variant_name"] == variant])
        for metric in ["top1", "mrr"]:
            ci_low, ci_high = percentile_ci(samples_so_far[metric])
            summary_rows.append(
                {
                    "candidate_pool_condition": condition,
                    "variant_name": variant,
                    "metric": metric,
                    "observed": observed[metric],
                    "bootstrap_mean": float(pd.to_numeric(samples_so_far[metric], errors="coerce").mean()),
                    "ci_lower_95": ci_low,
                    "ci_upper_95": ci_high,
                    "n_subjects": int(len(subjects)),
                    "n_queries": int(len(subset)),
                    "n_bootstrap": N_BOOT,
                    "seed": RANDOM_SEED,
                    "bootstrap_unit": "subject",
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def failure_cases(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Return failed v2B/v2C cases for broad-pool inspection."""

    focus = diagnostics[
        diagnostics["candidate_pool_condition"].astype(str).isin([ALL_MEDON_CONDITION, TRUE_PLUS_OTHER_CONDITION])
        & diagnostics["variant_name"].astype(str).isin([V2B_NAME, V2C_CYCLE_NAME, V2C_DECONFOUNDED_NAME])
        & ~diagnostics["top_ranked_is_true_pair"].astype(bool)
    ].copy()
    keep = [
        "candidate_pool_condition",
        "variant_name",
        "query_pair_id",
        "query_subject",
        "query_task",
        "query_side",
        "true_medon_rank",
        "top_ranked_candidate_pair_id",
        "top_ranked_candidate_subject",
        "top_ranked_candidate_task",
        "top_ranked_candidate_side",
        "failure_type",
        "retrieval_margin",
    ]
    return focus[[column for column in keep if column in focus.columns]].sort_values(
        ["candidate_pool_condition", "variant_name", "query_pair_id"]
    )


def cycle_pair_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Return compact cycle component table for v2C variants."""

    focus = scores[
        scores["variant_name"].astype(str).isin([V2C_CYCLE_NAME, V2C_DECONFOUNDED_NAME])
        & scores["candidate_pool_condition"].astype(str).isin([ALL_MEDON_CONDITION, TRUE_PLUS_OTHER_CONDITION])
    ].copy()
    keep = [
        "candidate_pool_condition",
        "variant_name",
        "query_pair_id",
        "candidate_pair_id",
        "is_true_pair",
        "score",
        "rank",
        "forward_score",
        "reverse_score",
        "reverse_rank",
        "cycle_score_component",
        "deconfound_penalty",
        "candidate_subject",
        "candidate_task_original",
        "candidate_side",
    ]
    return focus[[column for column in keep if column in focus.columns]].sort_values(
        ["candidate_pool_condition", "variant_name", "query_pair_id", "rank"]
    )


def plot_metric(metrics: pd.DataFrame, metric: str, path: Path) -> None:
    """Plot a metric by condition and variant."""

    fig, ax = plt.subplots(figsize=(12, 5.4))
    data = metrics[
        metrics["candidate_pool_condition"].astype(str).isin(EVALUATED_CONDITIONS)
        & metrics["variant_name"].astype(str).isin([V2B_NAME, V2C_FAMILY_NAME, V2C_CYCLE_NAME, V2C_DECONFOUNDED_NAME])
    ].copy()
    if data.empty:
        ax.text(0.5, 0.5, "No metrics", ha="center", va="center")
        ax.set_axis_off()
    else:
        conditions = EVALUATED_CONDITIONS
        variants = [V2B_NAME, V2C_FAMILY_NAME, V2C_CYCLE_NAME, V2C_DECONFOUNDED_NAME]
        colors = {
            V2B_NAME: "#4C78A8",
            V2C_FAMILY_NAME: "#F58518",
            V2C_CYCLE_NAME: "#54A24B",
            V2C_DECONFOUNDED_NAME: "#B279A2",
        }
        x = np.arange(len(conditions))
        width = 0.18
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(variants))
        for offset, variant in zip(offsets, variants, strict=False):
            values = []
            for condition in conditions:
                row = data[data["candidate_pool_condition"].eq(condition) & data["variant_name"].eq(variant)]
                values.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + offset, values, width=width, label=variant, color=colors[variant])
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=25, ha="right")
        if metric in {"top1", "mrr"}:
            ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.set_title(f"Experimental v2C {metric}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_delta(comparison: pd.DataFrame, path: Path) -> None:
    """Plot all-MedOn top1 deltas versus v2B."""

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    data = comparison[
        comparison["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & comparison["variant_name"].astype(str).str.startswith("v2C")
    ].copy()
    if data.empty:
        ax.text(0.5, 0.5, "No comparison rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.arange(len(data))
        ax.bar(x, data["delta_top1_vs_v2B"].to_numpy(dtype=float), color="#72B7B2")
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(data["variant_name"], rotation=25, ha="right")
        ax.set_ylabel("Delta top1 vs v2B")
        ax.set_title("All-MedOn v2C Delta")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_summary_md(summary: dict[str, object], path: Path) -> Path:
    """Write concise Markdown summary."""

    lines = [
        "# v2C Deconfounded Transition-Retrieval Experiment",
        "",
        "Scope: exploratory v2C study over cached ds004998 outputs. Frozen v2A/v2B are unchanged.",
        "",
        "## Validation",
        f"- complete_pairs: {summary['validation']['complete_pairs']}",
        f"- subjects_with_complete_pairs: {summary['validation']['subjects_with_complete_pairs']}",
        f"- Rest excluded: {summary['validation']['rest_excluded']}",
        f"- sub-BYJoWR excluded: {summary['validation']['sub_BYJoWR_excluded']}",
        f"- split files collapsed logically: {summary['validation']['split_files_collapsed_logically']}",
        f"- top_k: {summary['validation']['top_k']}",
        f"- aperiodic_alpha: {summary['validation']['aperiodic_alpha']}",
        "",
        "## Key Metrics",
    ]
    for row in summary["key_metrics"]:
        lines.append(
            f"- {row['candidate_pool_condition']} / {row['variant_name']}: "
            f"top1={float(row['top1']):.3f}, MRR={float(row['mrr']):.3f}, "
            f"failures={int(row['failures'])}"
        )
    lines.extend(["", "## Warnings"])
    if summary["warnings"]:
        lines.extend([f"- {warning}" for warning in summary["warnings"]])
    else:
        lines.append("- none")
    return write_text(lines, path)


def load_inputs() -> dict[str, object]:
    """Load cached inputs and validate the verified cohort."""

    missing = [str(path) for path in [FORWARD_SCORES_PATH, REVERSE_SCORES_PATH] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required v2C inputs: " + "; ".join(missing))

    pairs = read_csv_required(REQUIRED_CACHE_FILES["pairs"], dtype={"run_off": str, "run_on": str})
    logical = read_csv_required(REQUIRED_CACHE_FILES["logical_manifest"])
    usable = read_csv_required(REQUIRED_CACHE_FILES["usable_pair_inventory"])
    excluded = read_csv_required(REQUIRED_CACHE_FILES["excluded_subjects"])
    cohort_summary = read_json_required(REQUIRED_CACHE_FILES["cohort_summary"])
    frozen_summary = read_json_required(REQUIRED_CACHE_FILES["frozen_summary"])
    extraction_failure_log = read_csv_required(REQUIRED_CACHE_FILES["extraction_failure_log"])
    validation = validate_cached_inputs(
        pairs, logical, usable, excluded, cohort_summary, frozen_summary, extraction_failure_log
    )
    if TOP_K != 5:
        raise AssertionError(f"top_k changed: {TOP_K}")
    if not math.isclose(APERIODIC_ALPHA, 0.5):
        raise AssertionError(f"aperiodic_alpha changed: {APERIODIC_ALPHA}")

    subspaces = load_subspace_inventory(REQUIRED_CACHE_FILES["subspaces"])
    compact_inventory = read_json_required(REQUIRED_CACHE_FILES["compact_inventory"])
    all_features = base_feature_names(pairs)
    v2_features = [feature for feature in subspaces.get("v2_reference", []) if feature in all_features]
    compact_features = [str(feature) for feature in compact_inventory.get("selected_features", []) if str(feature) in all_features]
    if not v2_features or not compact_features:
        raise AssertionError("Required v2 or compact features are missing.")

    forward_scores = normalize_scores(read_csv_required(FORWARD_SCORES_PATH))
    reverse_scores = normalize_scores(read_csv_required(REVERSE_SCORES_PATH))
    return {
        "pairs": pairs,
        "validation": validation,
        "v2_features": v2_features,
        "compact_features": compact_features,
        "forward_scores": forward_scores,
        "reverse_scores": reverse_scores,
    }


def run() -> dict[str, object]:
    """Run the experimental v2C study."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    pairs = inputs["pairs"]
    validation = inputs["validation"]
    forward_scores = inputs["forward_scores"]
    reverse_scores = inputs["reverse_scores"]

    family_scores, family_weights, family_margin_training = evaluate_v2c_family_reweighted(
        pairs,
        inputs["v2_features"],
        inputs["compact_features"],
        EVALUATED_CONDITIONS,
    )
    cycle_scores = cycle_rerank_scores(forward_scores, reverse_scores, V2C_CYCLE_NAME, deconfounded=False)
    deconfounded_scores = cycle_rerank_scores(forward_scores, reverse_scores, V2C_DECONFOUNDED_NAME, deconfounded=True)

    baseline_scores = forward_scores[
        forward_scores["candidate_pool_condition"].astype(str).isin(EVALUATED_CONDITIONS)
        & forward_scores["variant_name"].astype(str).isin([V2A_NAME, V2B_NAME])
    ].copy()
    all_scores = pd.concat([baseline_scores, family_scores, cycle_scores, deconfounded_scores], ignore_index=True)
    all_scores = normalize_scores(all_scores)
    all_scores = rank_scores(all_scores)

    metrics, diagnostics = build_metrics_tables(all_scores)
    comparison = comparison_to_v2b(metrics)
    bootstrap_summary, bootstrap_samples = subject_bootstrap_ci(diagnostics)
    failures = failure_cases(diagnostics)
    cycle_pairs = cycle_pair_scores(all_scores)

    key_metrics = metrics[
        metrics["candidate_pool_condition"].astype(str).isin([PRIMARY_CONDITION, HARD_NEGATIVE_CONDITION, ALL_MEDON_CONDITION])
        & metrics["variant_name"].astype(str).isin([V2B_NAME, V2C_FAMILY_NAME, V2C_CYCLE_NAME, V2C_DECONFOUNDED_NAME])
    ].sort_values(["candidate_pool_condition", "variant_name"])
    warnings = [
        "v2C is experimental and exploratory; frozen v2A and selected v2B-gated outputs are unchanged.",
        "No Oxford/MRC external dataset and no raw FIF rereading were used.",
        "No hyperparameter tuning or clinical prediction claim is made; cycle weight and penalties are fixed audit settings.",
        "Clinical and medication-response interpretation remains out of scope without independent validation.",
    ]

    summary = {
        "validation": validation,
        "v2c_fixed_settings": {
            "cycle_weight": CYCLE_WEIGHT,
            "broad_pool_min_candidates": BROAD_POOL_MIN_CANDIDATES,
            "wrong_task_side_penalty": WRONG_TASK_SIDE_PENALTY,
            "same_subject_wrong_penalty": SAME_SUBJECT_WRONG_PENALTY,
            "family_weight_min": FAMILY_WEIGHT_MIN,
            "family_weight_max": FAMILY_WEIGHT_MAX,
        },
        "key_metrics": key_metrics.to_dict("records"),
        "all_medon_comparison_to_v2b": comparison[
            comparison["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        ].to_dict("records"),
        "warnings": warnings,
        "external_oxford_mrc_dataset_used": False,
        "raw_fif_reread": False,
        "frozen_v2A_changed": False,
        "v2B_logic_changed": False,
        "not_clinical_prediction_or_treatment": True,
    }

    paths = {
        "summary_json": write_json(summary, OUTPUT_DIR / "v2c_experiment_summary.json"),
        "metrics": write_csv(metrics, OUTPUT_DIR / "v2c_metrics.csv"),
        "candidate_scores": write_csv(all_scores, OUTPUT_DIR / "v2c_candidate_scores.csv"),
        "query_diagnostics": write_csv(diagnostics, OUTPUT_DIR / "v2c_query_diagnostics.csv"),
        "comparison": write_csv(comparison, OUTPUT_DIR / "v2c_comparison_to_v2b.csv"),
        "bootstrap_ci": write_csv(bootstrap_summary, OUTPUT_DIR / "v2c_bootstrap_ci.csv"),
        "bootstrap_samples": write_csv(bootstrap_samples, OUTPUT_DIR / "v2c_bootstrap_samples.csv"),
        "failure_cases": write_csv(failures, OUTPUT_DIR / "v2c_failure_cases.csv"),
        "family_weights": write_csv(family_weights, OUTPUT_DIR / "v2c_family_weights_by_fold.csv"),
        "family_training_margins": write_csv(family_margin_training, OUTPUT_DIR / "v2c_family_training_margins.csv"),
        "cycle_pair_scores": write_csv(cycle_pairs, OUTPUT_DIR / "v2c_cycle_pair_scores.csv"),
    }
    paths["summary_md"] = write_summary_md(to_builtin(summary), OUTPUT_DIR / "v2c_experiment_summary.md")

    plot_metric(metrics, "top1", OUTPUT_DIR / "v2c_top1_by_condition.png")
    plot_metric(metrics, "mrr", OUTPUT_DIR / "v2c_mrr_by_condition.png")
    plot_delta(comparison, OUTPUT_DIR / "v2c_all_medon_delta_top1_vs_v2b.png")

    v2b_all = metrics[
        metrics["candidate_pool_condition"].eq(ALL_MEDON_CONDITION) & metrics["variant_name"].eq(V2B_NAME)
    ].iloc[0]
    v2c_all = metrics[
        metrics["candidate_pool_condition"].eq(ALL_MEDON_CONDITION)
        & metrics["variant_name"].eq(V2C_DECONFOUNDED_NAME)
    ].iloc[0]
    cycle_all = metrics[
        metrics["candidate_pool_condition"].eq(ALL_MEDON_CONDITION) & metrics["variant_name"].eq(V2C_CYCLE_NAME)
    ].iloc[0]

    print(f"complete_pairs: {validation['complete_pairs']}")
    print(f"subjects_with_complete_pairs: {validation['subjects_with_complete_pairs']}")
    print(f"Rest excluded: {validation['rest_excluded']}")
    print(f"sub-BYJoWR excluded: {validation['sub_BYJoWR_excluded']}")
    print(f"split files collapsed logically: {validation['split_files_collapsed_logically']}")
    print(f"top_k: {TOP_K}")
    print(f"aperiodic_alpha: {APERIODIC_ALPHA}")
    print(f"v2B all-MedOn top1/MRR: {float(v2b_all['top1']):.3f}/{float(v2b_all['mrr']):.3f}")
    print(f"v2C cycle all-MedOn top1/MRR: {float(cycle_all['top1']):.3f}/{float(cycle_all['mrr']):.3f}")
    print(f"v2C deconfounded all-MedOn top1/MRR: {float(v2c_all['top1']):.3f}/{float(v2c_all['mrr']):.3f}")
    print(f"output folder: {OUTPUT_DIR}")
    print("warnings:")
    for warning in warnings:
        print(f"- {warning}")
    return {"paths": {key: str(path) for key, path in paths.items()}, "warnings": warnings}


def main() -> None:
    """Entry point."""

    run()


if __name__ == "__main__":
    main()
