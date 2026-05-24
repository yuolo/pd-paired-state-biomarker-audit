"""Scientific audit for the selected experimental v2B-gated variant.

This script audits the selected v2B_gated_aperiodic_top5 candidate against
frozen v2/v2A on the verified ds004998 17-subject/33-pair Hold/Move cohort.
Frozen v2A is not modified. The Oxford/MRC external dataset is not used.
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
    evaluate_frozen_on_candidate_pool,
    subject_fingerprint_cases,
)
from scripts.run_retrieval_distance_geometry_improvement import (  # noqa: E402
    query_diagnostics_from_scores,
    to_builtin,
)
from scripts.run_retrieval_signal_level_compact_v3 import MIN_DISTRACTORS  # noqa: E402
from scripts.run_v2a_scientific_audit_17subjects_33pairs import (  # noqa: E402
    APERIODIC_ALPHA,
    N_BOOT,
    N_PERM,
    RANDOM_SEED,
    REQUIRED_CACHE_FILES,
    TOP_K,
    V2A_NAME,
    V2_NAME,
    base_feature_names,
    build_score_lookup,
    empirical_p_larger,
    load_subspace_inventory,
    matched_task_side_null,
    normalize_cached_scores,
    percentile_ci,
    query_metrics,
    rank_scores,
    random_label_null,
    read_csv_required,
    read_json_required,
    rebuild_v2_scores_from_frozen_v2a,
    validate_cached_inputs,
    write_csv,
    write_json,
)
from scripts.run_v2b_hard_negative_metric_study_17subjects_33pairs import (  # noqa: E402
    V2BVariant,
    V2B_VARIANTS,
    evaluate_v2b_variant,
)
from src.evaluation.predictive_rescue_analysis import build_base_candidate_sets  # noqa: E402


OUTPUT_DIR = Path("outputs/v2B_gated_scientific_audit_17subjects_33pairs")
V2B_NAME = "v2B_gated_aperiodic_top5"
LADDER_CONDITIONS = [
    "original_frozen_matched_pool",
    "task_side_quality_ignored",
    "task_side_quality_strict",
    "task_family_side_quality_ignored",
    "original_plus_same_subject_wrong_task_side",
    "all_medon_candidates",
    "true_plus_all_other_subject_medon",
]
PRIMARY_CONDITION = "original_frozen_matched_pool"
HARD_NEGATIVE_CONDITION = "original_plus_same_subject_wrong_task_side"
ALL_MEDON_CONDITION = "all_medon_candidates"
COMPARISONS = [
    (V2A_NAME, V2_NAME),
    (V2B_NAME, V2A_NAME),
    (V2B_NAME, V2_NAME),
]


def write_text(lines: Iterable[str], path: Path) -> Path:
    """Write text output with parent directory creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def selected_variant() -> V2BVariant:
    """Return the pre-selected experimental v2B variant."""

    for variant in V2B_VARIANTS:
        if variant.name == V2B_NAME:
            return variant
    raise AssertionError(f"Selected v2B variant is not defined: {V2B_NAME}")


def add_condition_column(table: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Return a copy with candidate_pool_condition set to condition."""

    out = table.copy()
    if "candidate_pool_condition" in out.columns:
        out["candidate_pool_condition"] = condition
    else:
        out.insert(0, "candidate_pool_condition", condition)
    return out


def annotate_pool_context(scores: pd.DataFrame) -> pd.DataFrame:
    """Add same-subject/task-side top-candidate context to candidate scores."""

    out = scores.copy()
    out["same_subject_candidate"] = out["candidate_subject"].astype(str).eq(out["query_subject"].astype(str))
    out["same_task_side_candidate"] = (
        out["candidate_task_original"].astype(str).eq(out["query_task_original"].astype(str))
        & out["candidate_side"].astype(str).eq(out["query_side"].astype(str))
    )
    out["same_subject_wrong_candidate"] = out["same_subject_candidate"] & ~out["is_true_pair"].astype(bool)
    out["wrong_task_or_side_candidate"] = (
        ~out["is_true_pair"].astype(bool)
        & (
            ~out["candidate_task_original"].astype(str).eq(out["query_task_original"].astype(str))
            | ~out["candidate_side"].astype(str).eq(out["query_side"].astype(str))
        )
    )
    return out


def change_counts(base_diag: pd.DataFrame, new_diag: pd.DataFrame, base_name: str, new_name: str) -> dict[str, object]:
    """Count per-query success/failure transitions between two variants."""

    base = base_diag[["query_pair_id", "top_ranked_is_true_pair"]].rename(
        columns={"top_ranked_is_true_pair": "base_success"}
    )
    new = new_diag[["query_pair_id", "top_ranked_is_true_pair"]].rename(
        columns={"top_ranked_is_true_pair": "new_success"}
    )
    merged = base.merge(new, on="query_pair_id", how="inner")
    base_success = merged["base_success"].astype(bool)
    new_success = merged["new_success"].astype(bool)
    return {
        f"{base_name}_failure_to_{new_name}_success": int((~base_success & new_success).sum()),
        f"{base_name}_success_to_{new_name}_failure": int((base_success & ~new_success).sum()),
        f"{base_name}_{new_name}_unchanged_success": int((base_success & new_success).sum()),
        f"{base_name}_{new_name}_unchanged_failure": int((~base_success & ~new_success).sum()),
    }


def observed_metrics_table(
    diagnostics_by_variant: dict[str, pd.DataFrame],
    scores_by_variant: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build observed metrics for frozen v2/v2A and selected v2B."""

    rows = []
    all_failure_types: set[str] = set()
    failure_counts_by_variant: dict[str, dict[str, int]] = {}
    for variant, diag in diagnostics_by_variant.items():
        failures = diag[~diag["top_ranked_is_true_pair"].astype(bool)]
        counts = failures["failure_type"].fillna("unknown").astype(str).value_counts().sort_index().to_dict()
        counts = {str(key): int(value) for key, value in counts.items()}
        failure_counts_by_variant[variant] = counts
        all_failure_types.update(counts)

    metrics_by_variant = {variant: query_metrics(diag) for variant, diag in diagnostics_by_variant.items()}
    v2_metrics = metrics_by_variant[V2_NAME]
    v2a_metrics = metrics_by_variant[V2A_NAME]
    v2b_metrics = metrics_by_variant[V2B_NAME]
    v2_to_v2a = change_counts(diagnostics_by_variant[V2_NAME], diagnostics_by_variant[V2A_NAME], "v2", "v2A")
    v2a_to_v2b = change_counts(diagnostics_by_variant[V2A_NAME], diagnostics_by_variant[V2B_NAME], "v2A", "v2B")
    v2_to_v2b = change_counts(diagnostics_by_variant[V2_NAME], diagnostics_by_variant[V2B_NAME], "v2", "v2B")

    for variant in [V2_NAME, V2A_NAME, V2B_NAME]:
        row = {"variant_name": variant, **metrics_by_variant[variant]}
        row["failure_type_counts_json"] = json.dumps(failure_counts_by_variant[variant], sort_keys=True)
        for failure_type in sorted(all_failure_types):
            row[f"failure_type_{failure_type}"] = int(failure_counts_by_variant[variant].get(failure_type, 0))
        row["v2_failures"] = int(v2_metrics["failures"])
        row["v2A_failures"] = int(v2a_metrics["failures"])
        row["v2B_failures"] = int(v2b_metrics["failures"])
        row.update(v2_to_v2a)
        row.update(v2a_to_v2b)
        row.update(v2_to_v2b)
        row["delta_top1_vs_v2"] = float(metrics_by_variant[variant]["top1"]) - float(v2_metrics["top1"])
        row["delta_mrr_vs_v2"] = float(metrics_by_variant[variant]["mrr"]) - float(v2_metrics["mrr"])
        row["delta_top1_vs_v2A"] = float(metrics_by_variant[variant]["top1"]) - float(v2a_metrics["top1"])
        row["delta_mrr_vs_v2A"] = float(metrics_by_variant[variant]["mrr"]) - float(v2a_metrics["mrr"])
        scores = annotate_pool_context(scores_by_variant[variant])
        top = scores.sort_values("rank").groupby("query_pair_id", dropna=False).first().reset_index()
        row["top1_same_subject_rate"] = float(top["same_subject_candidate"].astype(float).mean()) if len(top) else math.nan
        row["top1_same_subject_wrong_rate"] = (
            float(top["same_subject_wrong_candidate"].astype(float).mean()) if len(top) else math.nan
        )
        row["top1_wrong_task_or_side_rate"] = (
            float(top["wrong_task_or_side_candidate"].astype(float).mean()) if len(top) else math.nan
        )
        row["subject_fingerprint_gap"] = float(row["top1_same_subject_rate"]) - float(row["top1"])
        rows.append(row)
    return pd.DataFrame(rows)


def subject_bootstrap_multi(
    diagnostics_by_variant: dict[str, pd.DataFrame],
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subject-aware bootstrap for v2/v2A/v2B metrics and deltas."""

    rng = np.random.default_rng(seed)
    subjects = sorted(
        {
            subject
            for diag in diagnostics_by_variant.values()
            for subject in diag["query_subject"].astype(str).unique().tolist()
        }
    )
    by_subject = {
        variant: {
            subject: diag[diag["query_subject"].astype(str).eq(subject)].copy()
            for subject in subjects
        }
        for variant, diag in diagnostics_by_variant.items()
    }
    sample_rows = []
    for bootstrap_index in range(n_boot):
        draw = rng.choice(subjects, size=len(subjects), replace=True)
        row: dict[str, object] = {
            "bootstrap_index": int(bootstrap_index),
            "n_subjects_sampled": int(len(draw)),
        }
        metrics_by_variant = {}
        for variant in [V2_NAME, V2A_NAME, V2B_NAME]:
            sample = pd.concat([by_subject[variant][str(subject)] for subject in draw], ignore_index=True)
            metrics = query_metrics(sample)
            metrics_by_variant[variant] = metrics
            row[f"{variant}_top1"] = metrics["top1"]
            row[f"{variant}_mrr"] = metrics["mrr"]
            row[f"{variant}_percentile_rank"] = metrics["percentile_rank"]
            row[f"{variant}_n_query_rows"] = int(len(sample))
        for new_name, base_name in COMPARISONS:
            for metric in ["top1", "mrr", "percentile_rank"]:
                row[f"delta_{metric}_{new_name}_minus_{base_name}"] = (
                    float(metrics_by_variant[new_name][metric]) - float(metrics_by_variant[base_name][metric])
                )
        sample_rows.append(row)
    samples = pd.DataFrame(sample_rows)

    observed_by_variant = {variant: query_metrics(diag) for variant, diag in diagnostics_by_variant.items()}
    summary_rows = []
    for variant in [V2_NAME, V2A_NAME, V2B_NAME]:
        for metric in ["top1", "mrr", "percentile_rank"]:
            column = f"{variant}_{metric}"
            ci_low, ci_high = percentile_ci(samples[column])
            summary_rows.append(
                {
                    "metric": column,
                    "observed": observed_by_variant[variant][metric],
                    "bootstrap_mean": float(samples[column].mean()),
                    "ci_lower_95": ci_low,
                    "ci_upper_95": ci_high,
                    "n_bootstrap": int(n_boot),
                    "seed": int(seed),
                    "bootstrap_unit": "subject",
                }
            )
    for new_name, base_name in COMPARISONS:
        for metric in ["top1", "mrr", "percentile_rank"]:
            column = f"delta_{metric}_{new_name}_minus_{base_name}"
            ci_low, ci_high = percentile_ci(samples[column])
            observed = float(observed_by_variant[new_name][metric]) - float(observed_by_variant[base_name][metric])
            summary_rows.append(
                {
                    "metric": column,
                    "observed": observed,
                    "bootstrap_mean": float(samples[column].mean()),
                    "ci_lower_95": ci_low,
                    "ci_upper_95": ci_high,
                    "n_bootstrap": int(n_boot),
                    "seed": int(seed),
                    "bootstrap_unit": "subject",
                }
            )
    return pd.DataFrame(summary_rows), samples


def evaluate_all_on_candidate_pool(
    pairs: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
    variant: V2BVariant,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Evaluate frozen v2/v2A and selected v2B on one candidate pool."""

    v2_scores, v2_diag, v2a_scores, v2a_diag = evaluate_frozen_on_candidate_pool(
        pairs, candidate_pool, v2_features, compact_features, clean_features
    )
    v2b_scores, v2b_diag, weights = evaluate_v2b_variant(
        pairs, candidate_pool, v2_features, compact_features, clean_features, variant
    )
    return (
        {V2_NAME: v2_scores, V2A_NAME: v2a_scores, V2B_NAME: v2b_scores},
        {V2_NAME: v2_diag, V2A_NAME: v2a_diag, V2B_NAME: v2b_diag},
        weights,
    )


def evaluate_subset_all(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
    variant: V2BVariant,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Evaluate all variants on the standard base candidate set for a subset."""

    if pairs.empty:
        return ({}, {}, pd.DataFrame())
    candidate_pool = build_base_candidate_sets(pairs, MIN_DISTRACTORS)
    return evaluate_all_on_candidate_pool(pairs, candidate_pool, v2_features, compact_features, clean_features, variant)


def candidate_pool_metrics_row(
    condition: str,
    variant_name: str,
    diagnostics: pd.DataFrame,
    scores: pd.DataFrame,
    candidate_pool: pd.DataFrame,
) -> dict[str, object]:
    """Build one ladder metrics row with pool and fingerprint context."""

    metrics = query_metrics(diagnostics)
    annotated = annotate_pool_context(scores)
    top = annotated.sort_values("rank").groupby("query_pair_id", dropna=False).first().reset_index()
    pool = candidate_pool.copy()
    pool["same_subject_wrong_distractor"] = (
        pool["candidate_subject"].astype(str).eq(pool["query_subject"].astype(str))
        & ~pool["is_true_pair"].astype(bool)
    )
    pool["same_task_side_distractor"] = (
        pool["candidate_task_original"].astype(str).eq(pool["query_task_original"].astype(str))
        & pool["candidate_side"].astype(str).eq(pool["query_side"].astype(str))
        & ~pool["is_true_pair"].astype(bool)
    )
    sizes = pool.groupby("query_pair_id", dropna=False)["candidate_pair_id"].size()
    same_subject_wrong = pool.groupby("query_pair_id", dropna=False)["same_subject_wrong_distractor"].sum()
    same_task_side = pool.groupby("query_pair_id", dropna=False)["same_task_side_distractor"].sum()
    return {
        "candidate_pool_condition": condition,
        "variant_name": variant_name,
        **metrics,
        "candidate_set_size_mean": float(sizes.mean()) if len(sizes) else math.nan,
        "candidate_set_size_min": int(sizes.min()) if len(sizes) else 0,
        "candidate_set_size_max": int(sizes.max()) if len(sizes) else 0,
        "queries_below_min_distractors": int(((sizes - 1) < MIN_DISTRACTORS).sum()) if len(sizes) else 0,
        "same_subject_wrong_distractors_mean": float(same_subject_wrong.mean()) if len(same_subject_wrong) else math.nan,
        "same_task_side_distractors_mean": float(same_task_side.mean()) if len(same_task_side) else math.nan,
        "top1_same_subject_rate": float(top["same_subject_candidate"].astype(float).mean()) if len(top) else math.nan,
        "top1_same_subject_wrong_rate": (
            float(top["same_subject_wrong_candidate"].astype(float).mean()) if len(top) else math.nan
        ),
        "top1_same_task_side_rate": float(top["same_task_side_candidate"].astype(float).mean()) if len(top) else math.nan,
        "top1_wrong_task_or_side_rate": (
            float(top["wrong_task_or_side_candidate"].astype(float).mean()) if len(top) else math.nan
        ),
        "subject_fingerprint_gap": (
            float(top["same_subject_candidate"].astype(float).mean()) - float(metrics["top1"]) if len(top) else math.nan
        ),
    }


def candidate_pool_ladder(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
    variant: V2BVariant,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run severity-ladder audit for v2/v2A/v2B-gated."""

    metric_rows = []
    diag_tables = []
    score_tables = []
    weight_tables = []
    for condition in LADDER_CONDITIONS:
        candidate_pool = build_custom_candidate_pool(pairs, condition)
        scores_by_variant, diag_by_variant, weights = evaluate_all_on_candidate_pool(
            pairs, candidate_pool, v2_features, compact_features, clean_features, variant
        )
        for variant_name in [V2_NAME, V2A_NAME, V2B_NAME]:
            scores = add_condition_column(scores_by_variant[variant_name], condition)
            diag = add_condition_column(diag_by_variant[variant_name], condition)
            score_tables.append(scores)
            diag_tables.append(diag)
            metric_rows.append(candidate_pool_metrics_row(condition, variant_name, diag, scores, candidate_pool))
        if not weights.empty:
            weight_tables.append(add_condition_column(weights, condition))
    return (
        pd.DataFrame(metric_rows),
        pd.concat(diag_tables, ignore_index=True) if diag_tables else pd.DataFrame(),
        pd.concat(score_tables, ignore_index=True) if score_tables else pd.DataFrame(),
        pd.concat(weight_tables, ignore_index=True) if weight_tables else pd.DataFrame(),
    )


def loso_sensitivity(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
    variant: V2BVariant,
    observed_v2b: dict[str, object],
) -> pd.DataFrame:
    """Remove one subject at a time and recompute v2/v2A/v2B-gated."""

    rows = []
    observed_top1 = float(observed_v2b["top1"])
    observed_mrr = float(observed_v2b["mrr"])
    collapse_rule = "v2B_top1_drop>=0.25 or v2B_mrr_drop>=0.20 or v2B_top1<=0.50"
    for subject in sorted(pairs["subject"].astype(str).unique()):
        subset = pairs[~pairs["subject"].astype(str).eq(subject)].copy()
        _scores_by_variant, diag_by_variant, _weights = evaluate_subset_all(
            subset, v2_features, compact_features, clean_features, variant
        )
        metrics_by_variant = {name: query_metrics(diag_by_variant[name]) for name in [V2_NAME, V2A_NAME, V2B_NAME]}
        v2b_top1 = float(metrics_by_variant[V2B_NAME]["top1"])
        v2b_mrr = float(metrics_by_variant[V2B_NAME]["mrr"])
        top1_drop = observed_top1 - v2b_top1
        mrr_drop = observed_mrr - v2b_mrr
        row = {
            "removed_subject": subject,
            "n_pairs_remaining": int(len(subset)),
            "n_subjects_remaining": int(subset["subject"].nunique()),
            "collapse_flag": bool(top1_drop >= 0.25 or mrr_drop >= 0.20 or v2b_top1 <= 0.50),
            "collapse_rule": collapse_rule,
            "v2B_top1_drop_from_observed": top1_drop,
            "v2B_mrr_drop_from_observed": mrr_drop,
        }
        for name in [V2_NAME, V2A_NAME, V2B_NAME]:
            row[f"{name}_top1"] = metrics_by_variant[name]["top1"]
            row[f"{name}_mrr"] = metrics_by_variant[name]["mrr"]
        row["delta_top1_v2A_minus_v2"] = float(metrics_by_variant[V2A_NAME]["top1"]) - float(
            metrics_by_variant[V2_NAME]["top1"]
        )
        row["delta_mrr_v2A_minus_v2"] = float(metrics_by_variant[V2A_NAME]["mrr"]) - float(
            metrics_by_variant[V2_NAME]["mrr"]
        )
        row["delta_top1_v2B_minus_v2A"] = float(metrics_by_variant[V2B_NAME]["top1"]) - float(
            metrics_by_variant[V2A_NAME]["top1"]
        )
        row["delta_mrr_v2B_minus_v2A"] = float(metrics_by_variant[V2B_NAME]["mrr"]) - float(
            metrics_by_variant[V2A_NAME]["mrr"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def task_side_sensitivity(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
    variant: V2BVariant,
) -> pd.DataFrame:
    """Evaluate Hold, Move, Left, and Right subsets for all variants."""

    subsets = {
        "Hold_only": pairs[pairs["task_family"].astype(str).eq("Hold")].copy(),
        "Move_only": pairs[pairs["task_family"].astype(str).eq("Move")].copy(),
        "Left_only": pairs[pairs["side"].astype(str).eq("L")].copy(),
        "Right_only": pairs[pairs["side"].astype(str).eq("R")].copy(),
    }
    rows = []
    for label, subset in subsets.items():
        if subset.empty:
            rows.append({"subset": label, "status": "skipped_empty", "n_pairs": 0, "n_subjects": 0})
            continue
        _scores_by_variant, diag_by_variant, _weights = evaluate_subset_all(
            subset, v2_features, compact_features, clean_features, variant
        )
        metrics_by_variant = {name: query_metrics(diag_by_variant[name]) for name in [V2_NAME, V2A_NAME, V2B_NAME]}
        row = {
            "subset": label,
            "status": "ok",
            "n_pairs": int(len(subset)),
            "n_subjects": int(subset["subject"].nunique()),
        }
        for name in [V2_NAME, V2A_NAME, V2B_NAME]:
            row[f"{name}_top1"] = metrics_by_variant[name]["top1"]
            row[f"{name}_mrr"] = metrics_by_variant[name]["mrr"]
            row[f"{name}_failures"] = metrics_by_variant[name]["failures"]
        row["delta_top1_v2A_minus_v2"] = float(metrics_by_variant[V2A_NAME]["top1"]) - float(
            metrics_by_variant[V2_NAME]["top1"]
        )
        row["delta_mrr_v2A_minus_v2"] = float(metrics_by_variant[V2A_NAME]["mrr"]) - float(
            metrics_by_variant[V2_NAME]["mrr"]
        )
        row["delta_top1_v2B_minus_v2A"] = float(metrics_by_variant[V2B_NAME]["top1"]) - float(
            metrics_by_variant[V2A_NAME]["top1"]
        )
        row["delta_mrr_v2B_minus_v2A"] = float(metrics_by_variant[V2B_NAME]["mrr"]) - float(
            metrics_by_variant[V2A_NAME]["mrr"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def mechanism_cases(ladder_scores: pd.DataFrame) -> pd.DataFrame:
    """Compare v2A and v2B-gated per query under harder candidate pools."""

    focus = ladder_scores[
        ladder_scores["candidate_pool_condition"].isin(
            [PRIMARY_CONDITION, HARD_NEGATIVE_CONDITION, ALL_MEDON_CONDITION]
        )
        & ladder_scores["variant_name"].isin([V2A_NAME, V2B_NAME])
    ].copy()
    rows = []
    for (condition, query_id), group in focus.groupby(["candidate_pool_condition", "query_pair_id"], dropna=False):
        by_variant = {
            variant_name: table.sort_values("rank").copy()
            for variant_name, table in group.groupby("variant_name", dropna=False)
        }
        if V2A_NAME not in by_variant or V2B_NAME not in by_variant:
            continue
        v2a = by_variant[V2A_NAME]
        v2b = by_variant[V2B_NAME]
        v2a_top = v2a.iloc[0]
        v2b_top = v2b.iloc[0]
        v2a_true = v2a[v2a["is_true_pair"].astype(bool)].iloc[0]
        v2b_true = v2b[v2b["is_true_pair"].astype(bool)].iloc[0]
        v2a_success = bool(v2a_top["is_true_pair"])
        v2b_success = bool(v2b_top["is_true_pair"])
        if (not v2a_success) and v2b_success:
            change_type = "v2A_failure_to_v2B_success"
        elif v2a_success and (not v2b_success):
            change_type = "v2A_success_to_v2B_failure"
        elif v2a_success and v2b_success:
            change_type = "unchanged_success"
        elif str(v2a_top["candidate_pair_id"]) != str(v2b_top["candidate_pair_id"]):
            change_type = "unchanged_failure_different_wrong_top1"
        else:
            change_type = "unchanged_failure_same_wrong_top1"
        v2a_wrong_same_subject = (
            str(v2a_top["candidate_subject"]) == str(v2a_top["query_subject"]) and not bool(v2a_top["is_true_pair"])
        )
        v2b_wrong_same_subject = (
            str(v2b_top["candidate_subject"]) == str(v2b_top["query_subject"]) and not bool(v2b_top["is_true_pair"])
        )
        v2a_wrong_task_or_side = (
            not bool(v2a_top["is_true_pair"])
            and (
                str(v2a_top["candidate_task_original"]) != str(v2a_top["query_task_original"])
                or str(v2a_top["candidate_side"]) != str(v2a_top["query_side"])
            )
        )
        v2b_wrong_task_or_side = (
            not bool(v2b_top["is_true_pair"])
            and (
                str(v2b_top["candidate_task_original"]) != str(v2b_top["query_task_original"])
                or str(v2b_top["candidate_side"]) != str(v2b_top["query_side"])
            )
        )
        rows.append(
            {
                "candidate_pool_condition": condition,
                "query_pair_id": query_id,
                "query_subject": v2a_top["query_subject"],
                "query_task": v2a_top["query_task_original"],
                "query_side": v2a_top["query_side"],
                "candidate_set_size": int(v2a_top["number_of_candidates"]),
                "change_type": change_type,
                "v2A_true_rank": int(v2a_true["rank"]),
                "v2B_true_rank": int(v2b_true["rank"]),
                "rank_improvement_v2A_minus_v2B": int(v2a_true["rank"]) - int(v2b_true["rank"]),
                "v2A_top1_candidate_pair_id": v2a_top["candidate_pair_id"],
                "v2A_top1_subject": v2a_top["candidate_subject"],
                "v2A_top1_task": v2a_top["candidate_task_original"],
                "v2A_top1_side": v2a_top["candidate_side"],
                "v2A_top1_same_subject_wrong": bool(v2a_wrong_same_subject),
                "v2A_top1_wrong_task_or_side": bool(v2a_wrong_task_or_side),
                "v2B_top1_candidate_pair_id": v2b_top["candidate_pair_id"],
                "v2B_top1_subject": v2b_top["candidate_subject"],
                "v2B_top1_task": v2b_top["candidate_task_original"],
                "v2B_top1_side": v2b_top["candidate_side"],
                "v2B_top1_same_subject_wrong": bool(v2b_wrong_same_subject),
                "v2B_top1_wrong_task_or_side": bool(v2b_wrong_task_or_side),
                "gate_prevented_same_subject_wrong_task_side_top1": bool(
                    v2a_wrong_same_subject and v2a_wrong_task_or_side and v2b_success
                ),
                "gate_removed_wrong_task_or_side_top1": bool(v2a_wrong_task_or_side and not v2b_wrong_task_or_side),
            }
        )
    return pd.DataFrame(rows)


def v2b_failure_taxonomy(
    v2_scores: pd.DataFrame,
    v2_diag: pd.DataFrame,
    v2a_diag: pd.DataFrame,
    v2b_scores: pd.DataFrame,
    v2b_diag: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-query primary-pool taxonomy for selected v2B."""

    v2_by_query = build_score_lookup(v2_scores)
    v2b_by_query = build_score_lookup(v2b_scores)
    v2d = v2_diag.set_index("query_pair_id", drop=False)
    v2ad = v2a_diag.set_index("query_pair_id", drop=False)
    v2bd = v2b_diag.set_index("query_pair_id", drop=False)
    rows = []
    for query_id in sorted(set(v2bd.index)):
        v2_row = v2d.loc[query_id]
        v2a_row = v2ad.loc[query_id]
        v2b_row = v2bd.loc[query_id]
        v2_group = v2_by_query.get(str(query_id), pd.DataFrame())
        v2b_group = v2b_by_query.get(str(query_id), pd.DataFrame())
        true_in_candidate_pool = bool((not v2b_group.empty) and v2b_group["is_true_pair"].astype(bool).any())
        v2b_true = v2b_group[v2b_group["is_true_pair"].astype(bool)] if true_in_candidate_pool else pd.DataFrame()
        v2b_stage_rank = (
            int(v2b_true.iloc[0].get("v2_rank_for_stage", v2b_true.iloc[0].get("rank", 10**9)))
            if not v2b_true.empty
            else 10**9
        )
        v2b_success = bool(v2b_row.get("top_ranked_is_true_pair", False))
        if not true_in_candidate_pool:
            audit_stage = "candidate-pool issue"
        elif v2b_success:
            audit_stage = "success"
        elif v2b_stage_rank > TOP_K:
            audit_stage = "v2B stage1 unrecoverable"
        else:
            audit_stage = "v2B gated rerank failure"
        v2b_top = v2b_group.sort_values("rank").iloc[0] if not v2b_group.empty else pd.Series(dtype=object)
        v2a_success = bool(v2a_row.get("top_ranked_is_true_pair", False))
        if (not v2a_success) and v2b_success:
            change_type = "v2A_failure_to_v2B_success"
        elif v2a_success and (not v2b_success):
            change_type = "v2A_success_to_v2B_failure"
        elif v2a_success and v2b_success:
            change_type = "unchanged_success"
        else:
            change_type = "unchanged_failure"
        wrong_subject = str(v2b_top.get("candidate_subject", "")) if not bool(v2b_top.get("is_true_pair", False)) else ""
        wrong_task = str(v2b_top.get("candidate_task_original", "")) if wrong_subject else ""
        wrong_side = str(v2b_top.get("candidate_side", "")) if wrong_subject else ""
        query_subject = str(v2b_row.get("query_subject", ""))
        query_task = str(v2b_row.get("query_task", ""))
        query_side = str(v2b_row.get("query_side", ""))
        true_v2 = v2_group[v2_group.get("is_true_pair", pd.Series(dtype=bool)).astype(bool)] if not v2_group.empty else pd.DataFrame()
        rows.append(
            {
                "query_pair_id": query_id,
                "query_subject": query_subject,
                "query_task": query_task,
                "query_side": query_side,
                "query_quality_flag": v2b_row.get("query_quality", ""),
                "true_medon_quality_flag": v2b_row.get("true_medon_quality", ""),
                "v2_true_rank": v2_row.get("true_medon_rank", math.nan),
                "v2A_true_rank": v2a_row.get("true_medon_rank", math.nan),
                "v2B_true_rank": v2b_row.get("true_medon_rank", math.nan),
                "v2B_v2_stage_rank": v2b_stage_rank,
                "v2B_true_in_stage_top5": bool(v2b_stage_rank <= TOP_K),
                "v2_true_in_frozen_top5": bool((not true_v2.empty) and int(true_v2.iloc[0]["rank"]) <= TOP_K),
                "v2_top1_candidate_pair_id": v2_row.get("top_ranked_candidate_pair_id", ""),
                "v2_top1_candidate_subject": v2_row.get("top_ranked_candidate_subject", ""),
                "v2_top1_candidate_task": v2_row.get("top_ranked_candidate_task", ""),
                "v2_top1_candidate_side": v2_row.get("top_ranked_candidate_side", ""),
                "v2A_top1_candidate_pair_id": v2a_row.get("top_ranked_candidate_pair_id", ""),
                "v2A_top1_candidate_subject": v2a_row.get("top_ranked_candidate_subject", ""),
                "v2A_top1_candidate_task": v2a_row.get("top_ranked_candidate_task", ""),
                "v2A_top1_candidate_side": v2a_row.get("top_ranked_candidate_side", ""),
                "v2B_top1_candidate_pair_id": v2b_row.get("top_ranked_candidate_pair_id", ""),
                "v2B_top1_candidate_subject": v2b_row.get("top_ranked_candidate_subject", ""),
                "v2B_top1_candidate_task": v2b_row.get("top_ranked_candidate_task", ""),
                "v2B_top1_candidate_side": v2b_row.get("top_ranked_candidate_side", ""),
                "v2_success": bool(v2_row.get("top_ranked_is_true_pair", False)),
                "v2A_success": v2a_success,
                "v2B_success": v2b_success,
                "v2B_audit_failure_stage": audit_stage,
                "v2A_to_v2B_change_type": change_type,
                "v2_failure_type": v2_row.get("failure_type", ""),
                "v2A_failure_type": v2a_row.get("failure_type", ""),
                "v2B_failure_type": v2b_row.get("failure_type", ""),
                "wrong_candidate_subject": wrong_subject,
                "wrong_candidate_task": wrong_task,
                "wrong_candidate_side": wrong_side,
                "wrong_same_subject": bool(wrong_subject and wrong_subject == query_subject),
                "wrong_same_task": bool(wrong_task and wrong_task == query_task),
                "wrong_same_side": bool(wrong_side and wrong_side == query_side),
                "wrong_task_or_side": bool(wrong_subject and (wrong_task != query_task or wrong_side != query_side)),
                "candidate_set_size": v2b_row.get("candidate_set_size", math.nan),
            }
        )
    return pd.DataFrame(rows)


def v2b_null_comparison_summary(
    observed: pd.DataFrame,
    null_samples: pd.DataFrame,
    null_name: str,
) -> pd.DataFrame:
    """Add delta p-values for v2B-v2A on top1 and MRR."""

    if null_samples.empty:
        return pd.DataFrame()
    observed_index = observed.set_index("variant_name", drop=False)
    observed_delta_top1 = float(observed_index.loc[V2B_NAME, "top1"]) - float(observed_index.loc[V2A_NAME, "top1"])
    observed_delta_mrr = float(observed_index.loc[V2B_NAME, "mrr"]) - float(observed_index.loc[V2A_NAME, "mrr"])
    wide = null_samples.pivot_table(index="permutation_index", columns="variant_name", values=["top1", "mrr"])
    delta_top1 = wide["top1"][V2B_NAME] - wide["top1"][V2A_NAME]
    delta_mrr = wide["mrr"][V2B_NAME] - wide["mrr"][V2A_NAME]
    return pd.DataFrame(
        [
            {
                "null_name": null_name,
                "metric": "delta_top1_v2B_minus_v2A",
                "observed": observed_delta_top1,
                "null_mean": float(delta_top1.mean()),
                "empirical_p_larger": empirical_p_larger(observed_delta_top1, delta_top1),
                "n_permutations": int(len(delta_top1)),
                "seed": RANDOM_SEED,
            },
            {
                "null_name": null_name,
                "metric": "delta_mrr_v2B_minus_v2A",
                "observed": observed_delta_mrr,
                "null_mean": float(delta_mrr.mean()),
                "empirical_p_larger": empirical_p_larger(observed_delta_mrr, delta_mrr),
                "n_permutations": int(len(delta_mrr)),
                "seed": RANDOM_SEED,
            },
        ]
    )


def plot_primary_metrics(observed: pd.DataFrame, path: Path) -> None:
    """Plot observed top1 and MRR for v2/v2A/v2B."""

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    order = [V2_NAME, V2A_NAME, V2B_NAME]
    labels = ["v2", "v2A", "v2B-gated"]
    x = np.arange(len(order))
    width = 0.34
    values_top1 = [float(observed.loc[observed["variant_name"].eq(name), "top1"].iloc[0]) for name in order]
    values_mrr = [float(observed.loc[observed["variant_name"].eq(name), "mrr"].iloc[0]) for name in order]
    ax.bar(x - width / 2, values_top1, width=width, label="top1", color="#4C78A8")
    ax.bar(x + width / 2, values_mrr, width=width, label="MRR", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Primary Matched Pool Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ladder_metric(metrics: pd.DataFrame, metric: str, path: Path) -> None:
    """Plot candidate-pool ladder metric."""

    fig, ax = plt.subplots(figsize=(12, 5.4))
    if metrics.empty:
        ax.text(0.5, 0.5, "No ladder metrics", ha="center", va="center")
        ax.set_axis_off()
    else:
        conditions = LADDER_CONDITIONS
        variants = [V2_NAME, V2A_NAME, V2B_NAME]
        labels = {"v2_reference": "v2", V2A_NAME: "v2A", V2B_NAME: "v2B-gated"}
        colors = {V2_NAME: "#4C78A8", V2A_NAME: "#F58518", V2B_NAME: "#54A24B"}
        x = np.arange(len(conditions))
        width = 0.24
        for idx, variant in enumerate(variants):
            values = []
            for condition in conditions:
                row = metrics[
                    metrics["candidate_pool_condition"].eq(condition)
                    & metrics["variant_name"].eq(variant)
                ]
                values.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (idx - 1) * width, values, width=width, label=labels[variant], color=colors[variant])
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.set_title(f"Candidate-Pool Severity Ladder: {metric}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_subject_fingerprint_gap(metrics: pd.DataFrame, path: Path) -> None:
    """Plot subject-fingerprint gap across hard pools."""

    fig, ax = plt.subplots(figsize=(9, 4.8))
    data = metrics[
        metrics["candidate_pool_condition"].isin([HARD_NEGATIVE_CONDITION, ALL_MEDON_CONDITION])
        & metrics["variant_name"].isin([V2A_NAME, V2B_NAME])
    ].copy()
    if data.empty:
        ax.text(0.5, 0.5, "No subject-fingerprint rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        conditions = [HARD_NEGATIVE_CONDITION, ALL_MEDON_CONDITION]
        variants = [V2A_NAME, V2B_NAME]
        labels = {V2A_NAME: "v2A", V2B_NAME: "v2B-gated"}
        colors = {V2A_NAME: "#F58518", V2B_NAME: "#54A24B"}
        x = np.arange(len(conditions))
        width = 0.32
        for idx, variant in enumerate(variants):
            values = []
            for condition in conditions:
                row = data[data["candidate_pool_condition"].eq(condition) & data["variant_name"].eq(variant)]
                values.append(float(row["subject_fingerprint_gap"].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (idx - 0.5) * width, values, width=width, label=labels[variant], color=colors[variant])
        ax.axhline(0.10, color="#333333", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=25, ha="right")
        ax.set_ylabel("Same-subject top1 minus true-pair top1")
        ax.set_title("Subject-Fingerprint Gap")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_mechanism_counts(cases: pd.DataFrame, path: Path) -> None:
    """Plot v2A-to-v2B change counts by candidate-pool condition."""

    fig, ax = plt.subplots(figsize=(10, 4.8))
    if cases.empty:
        ax.text(0.5, 0.5, "No mechanism cases", ha="center", va="center")
        ax.set_axis_off()
    else:
        pivot = cases.pivot_table(
            index="candidate_pool_condition", columns="change_type", values="query_pair_id", aggfunc="count", fill_value=0
        )
        order = [condition for condition in [PRIMARY_CONDITION, HARD_NEGATIVE_CONDITION, ALL_MEDON_CONDITION] if condition in pivot.index]
        pivot = pivot.loc[order]
        bottom = np.zeros(len(pivot), dtype=float)
        x = np.arange(len(pivot))
        colors = ["#54A24B", "#E45756", "#4C78A8", "#F58518", "#72B7B2"]
        for idx, column in enumerate(pivot.columns):
            values = pivot[column].to_numpy(dtype=float)
            ax.bar(x, values, bottom=bottom, label=str(column), color=colors[idx % len(colors)])
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=25, ha="right")
        ax.set_ylabel("Query count")
        ax.set_title("v2A to v2B-Gated Case Changes")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(summary: dict[str, object], path: Path) -> Path:
    """Write concise Markdown audit report."""

    observed = pd.DataFrame(summary["observed_metrics"])
    ladder = pd.DataFrame(summary["candidate_pool_ladder_metrics"])
    lines = [
        "# v2B-Gated Scientific Audit",
        "",
        "Scope: selected experimental v2B-gated variant audited against frozen v2/v2A on cached ds004998 Hold/Move pairs only.",
        "",
        "## Cohort Checks",
        f"- complete_pairs: {summary['validation']['complete_pairs']}",
        f"- subjects_with_complete_pairs: {summary['validation']['subjects_with_complete_pairs']}",
        f"- Rest excluded: {summary['validation']['rest_excluded']}",
        f"- sub-BYJoWR excluded: {summary['validation']['sub_BYJoWR_excluded']}",
        f"- split files collapsed logically: {summary['validation']['split_files_collapsed_logically']}",
        f"- top_k: {summary['validation']['top_k']}",
        f"- aperiodic_alpha: {summary['validation']['aperiodic_alpha']}",
        "",
        "## Primary Observed Metrics",
    ]
    for _, row in observed.iterrows():
        lines.append(
            f"- {row['variant_name']}: top1={float(row['top1']):.3f}, "
            f"MRR={float(row['mrr']):.3f}, failures={int(row['failures'])}, "
            f"subject_gap={float(row['subject_fingerprint_gap']):.3f}"
        )
    lines.extend(["", "## Severity Ladder"])
    ladder_focus = ladder[ladder["variant_name"].isin([V2A_NAME, V2B_NAME])]
    for _, row in ladder_focus.iterrows():
        lines.append(
            f"- {row['candidate_pool_condition']} / {row['variant_name']}: "
            f"top1={float(row['top1']):.3f}, MRR={float(row['mrr']):.3f}, "
            f"subject_gap={float(row['subject_fingerprint_gap']):.3f}"
        )
    lines.extend(["", "## Warnings"])
    warnings = summary.get("warnings", [])
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Boundary statement: v2B-gated remains experimental; this is paired-state identifiability audit evidence, not clinical prediction, treatment recommendation, DBS optimization, or causal medication-effect estimation.",
        ]
    )
    return write_text(lines, path)


def run() -> dict[str, object]:
    """Run the v2B-gated scientific audit."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    variant = selected_variant()

    pairs = read_csv_required(REQUIRED_CACHE_FILES["pairs"], dtype={"run_off": str, "run_on": str})
    cached_v2a_scores = normalize_cached_scores(read_csv_required(REQUIRED_CACHE_FILES["candidate_scores"]))
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
    if TOP_K != 5:
        raise AssertionError(f"top_k changed: {TOP_K}")
    if not math.isclose(APERIODIC_ALPHA, 0.5):
        raise AssertionError(f"aperiodic_alpha changed: {APERIODIC_ALPHA}")

    v2a_scores = rank_scores(cached_v2a_scores.drop(columns=["rank"], errors="ignore"))
    v2a_diag = query_diagnostics_from_scores(v2a_scores)
    v2_scores = rebuild_v2_scores_from_frozen_v2a(v2a_scores)
    v2_diag = query_diagnostics_from_scores(v2_scores)
    primary_pool = build_custom_candidate_pool(pairs, PRIMARY_CONDITION)
    v2b_scores, v2b_diag, primary_weights = evaluate_v2b_variant(
        pairs, primary_pool, v2_features, compact_features, clean_features, variant
    )
    v2b_scores = add_condition_column(v2b_scores, PRIMARY_CONDITION)
    v2b_diag = add_condition_column(v2b_diag, PRIMARY_CONDITION)
    v2_scores = add_condition_column(v2_scores, PRIMARY_CONDITION)
    v2_diag = add_condition_column(v2_diag, PRIMARY_CONDITION)
    v2a_scores = add_condition_column(v2a_scores, PRIMARY_CONDITION)
    v2a_diag = add_condition_column(v2a_diag, PRIMARY_CONDITION)

    diagnostics_by_variant = {V2_NAME: v2_diag, V2A_NAME: v2a_diag, V2B_NAME: v2b_diag}
    scores_by_variant = {V2_NAME: v2_scores, V2A_NAME: v2a_scores, V2B_NAME: v2b_scores}
    observed = observed_metrics_table(diagnostics_by_variant, scores_by_variant)

    random_summary, random_samples = random_label_null(diagnostics_by_variant, scores_by_variant, N_PERM, RANDOM_SEED)
    matched_summary, matched_samples = matched_task_side_null(
        diagnostics_by_variant, scores_by_variant, N_PERM, RANDOM_SEED
    )
    random_delta_summary = v2b_null_comparison_summary(observed, random_samples, "random_label_null")
    matched_delta_summary = v2b_null_comparison_summary(observed, matched_samples, "matched_task_side_null")
    bootstrap_summary, bootstrap_samples = subject_bootstrap_multi(diagnostics_by_variant, N_BOOT, RANDOM_SEED)
    loso = loso_sensitivity(
        pairs, v2_features, compact_features, clean_features, variant, query_metrics(v2b_diag)
    )
    task_side = task_side_sensitivity(pairs, v2_features, compact_features, clean_features, variant)

    ladder_metrics, ladder_diag, ladder_scores, ladder_weights = candidate_pool_ladder(
        pairs, v2_features, compact_features, clean_features, variant
    )
    fingerprint_summary, fingerprint_case_table = subject_fingerprint_cases(ladder_scores)
    hard_negative_summary = fingerprint_summary[
        fingerprint_summary["candidate_pool_condition"].eq(HARD_NEGATIVE_CONDITION)
        & fingerprint_summary["variant_name"].isin([V2_NAME, V2A_NAME, V2B_NAME])
    ].copy()
    hard_negative_cases = fingerprint_case_table[
        fingerprint_case_table["candidate_pool_condition"].eq(HARD_NEGATIVE_CONDITION)
        & fingerprint_case_table["variant_name"].isin([V2_NAME, V2A_NAME, V2B_NAME])
    ].copy()
    mechanism = mechanism_cases(ladder_scores)
    taxonomy = v2b_failure_taxonomy(v2_scores, v2_diag, v2a_diag, v2b_scores, v2b_diag)

    warnings = []
    primary = observed.set_index("variant_name", drop=False)
    if float(primary.loc[V2B_NAME, "top1"]) < float(primary.loc[V2A_NAME, "top1"]) - 0.03:
        warnings.append("v2B-gated primary top1 is lower than frozen v2A by more than 0.03.")
    if float(primary.loc[V2B_NAME, "mrr"]) < float(primary.loc[V2A_NAME, "mrr"]) - 0.03:
        warnings.append("v2B-gated primary MRR is lower than frozen v2A by more than 0.03.")
    strict = ladder_metrics[
        ladder_metrics["candidate_pool_condition"].eq("task_side_quality_strict")
        & ladder_metrics["variant_name"].eq(V2B_NAME)
    ]
    if not strict.empty and int(strict.iloc[0]["queries_below_min_distractors"]) > 0:
        warnings.append(
            "Strict task/side/quality pool has queries below MIN_DISTRACTORS; interpret strict-pool metrics cautiously."
        )
    all_medon = ladder_metrics[
        ladder_metrics["candidate_pool_condition"].eq(ALL_MEDON_CONDITION)
        & ladder_metrics["variant_name"].eq(V2B_NAME)
    ]
    if not all_medon.empty and float(all_medon.iloc[0]["top1"]) < 0.50:
        warnings.append("v2B-gated all-MedOn top1 remains below 0.50 under the hardest pool.")
    if not all_medon.empty and float(all_medon.iloc[0]["subject_fingerprint_gap"]) >= 0.10:
        warnings.append("v2B-gated all-MedOn subject-fingerprint gap remains >= 0.10.")
    if loso["collapse_flag"].astype(bool).any():
        subjects = ", ".join(loso.loc[loso["collapse_flag"].astype(bool), "removed_subject"].astype(str))
        warnings.append(f"LOSO collapse flag triggered for v2B-gated after removing: {subjects}")
    warnings.append("No independent external validation cohort was used in this audit.")
    warnings.append("v2B-gated is experimental; frozen v2A remains the locked reference pipeline.")

    summary = {
        "scientific_task": "paired_state_identifiability_with_hard_negative_and_fingerprint_audit",
        "selected_variant": {
            "name": variant.name,
            "description": variant.description,
            "gated_aperiodic": variant.gated_aperiodic,
            "top_k": TOP_K,
            "aperiodic_alpha": APERIODIC_ALPHA,
        },
        "validation": validation,
        "observed_metrics": observed.to_dict("records"),
        "random_label_null": random_summary.to_dict("records"),
        "matched_task_side_null": matched_summary.to_dict("records"),
        "bootstrap_ci": bootstrap_summary.to_dict("records"),
        "candidate_pool_ladder_metrics": ladder_metrics.to_dict("records"),
        "subject_fingerprint_summary": fingerprint_summary.to_dict("records"),
        "warnings": warnings,
        "external_oxford_mrc_dataset_used": False,
        "raw_fif_reread": False,
        "frozen_v2A_changed": False,
        "not_clinical_prediction_or_treatment": True,
    }

    paths = {
        "summary_json": write_json(summary, OUTPUT_DIR / "v2b_gated_audit_summary.json"),
        "observed_metrics": write_csv(observed, OUTPUT_DIR / "v2_v2A_v2B_observed_metrics.csv"),
        "random_summary": write_csv(random_summary, OUTPUT_DIR / "v2b_random_label_null_summary.csv"),
        "random_samples": write_csv(random_samples, OUTPUT_DIR / "v2b_random_label_null_samples.csv"),
        "random_delta_summary": write_csv(random_delta_summary, OUTPUT_DIR / "v2b_random_label_delta_summary.csv"),
        "matched_summary": write_csv(matched_summary, OUTPUT_DIR / "v2b_matched_task_side_null_summary.csv"),
        "matched_samples": write_csv(matched_samples, OUTPUT_DIR / "v2b_matched_task_side_null_samples.csv"),
        "matched_delta_summary": write_csv(matched_delta_summary, OUTPUT_DIR / "v2b_matched_task_side_delta_summary.csv"),
        "bootstrap_summary": write_csv(bootstrap_summary, OUTPUT_DIR / "v2b_bootstrap_ci_summary.csv"),
        "bootstrap_samples": write_csv(bootstrap_samples, OUTPUT_DIR / "v2b_bootstrap_samples.csv"),
        "loso": write_csv(loso, OUTPUT_DIR / "v2b_loso_sensitivity.csv"),
        "task_side": write_csv(task_side, OUTPUT_DIR / "v2b_task_side_sensitivity.csv"),
        "ladder_metrics": write_csv(ladder_metrics, OUTPUT_DIR / "v2b_candidate_pool_ladder_metrics.csv"),
        "ladder_diag": write_csv(ladder_diag, OUTPUT_DIR / "v2b_candidate_pool_ladder_query_diagnostics.csv"),
        "ladder_scores": write_csv(ladder_scores, OUTPUT_DIR / "v2b_candidate_pool_ladder_candidate_scores.csv"),
        "fingerprint_summary": write_csv(fingerprint_summary, OUTPUT_DIR / "v2b_subject_fingerprint_summary.csv"),
        "fingerprint_cases": write_csv(fingerprint_case_table, OUTPUT_DIR / "v2b_subject_fingerprint_cases.csv"),
        "hard_negative_summary": write_csv(hard_negative_summary, OUTPUT_DIR / "v2b_hard_negative_summary.csv"),
        "hard_negative_cases": write_csv(hard_negative_cases, OUTPUT_DIR / "v2b_hard_negative_cases.csv"),
        "mechanism_cases": write_csv(mechanism, OUTPUT_DIR / "v2a_to_v2b_mechanism_cases.csv"),
        "failure_taxonomy": write_csv(taxonomy, OUTPUT_DIR / "v2b_failure_taxonomy.csv"),
        "primary_weights": write_csv(primary_weights, OUTPUT_DIR / "v2b_primary_feature_weights_by_fold.csv"),
        "ladder_weights": write_csv(ladder_weights, OUTPUT_DIR / "v2b_ladder_feature_weights_by_fold.csv"),
    }
    paths["summary_md"] = write_report(to_builtin(summary), OUTPUT_DIR / "v2b_gated_audit_summary.md")

    plot_primary_metrics(observed, OUTPUT_DIR / "v2b_primary_top1_mrr.png")
    plot_ladder_metric(ladder_metrics, "top1", OUTPUT_DIR / "v2b_candidate_pool_ladder_top1.png")
    plot_ladder_metric(ladder_metrics, "mrr", OUTPUT_DIR / "v2b_candidate_pool_ladder_mrr.png")
    plot_subject_fingerprint_gap(ladder_metrics, OUTPUT_DIR / "v2b_subject_fingerprint_gap.png")
    plot_mechanism_counts(mechanism, OUTPUT_DIR / "v2a_to_v2b_mechanism_counts.png")

    v2a_primary = primary.loc[V2A_NAME]
    v2b_primary = primary.loc[V2B_NAME]
    all_medon_v2a = ladder_metrics[
        ladder_metrics["candidate_pool_condition"].eq(ALL_MEDON_CONDITION)
        & ladder_metrics["variant_name"].eq(V2A_NAME)
    ].iloc[0]
    all_medon_v2b = ladder_metrics[
        ladder_metrics["candidate_pool_condition"].eq(ALL_MEDON_CONDITION)
        & ladder_metrics["variant_name"].eq(V2B_NAME)
    ].iloc[0]
    hard_v2a = ladder_metrics[
        ladder_metrics["candidate_pool_condition"].eq(HARD_NEGATIVE_CONDITION)
        & ladder_metrics["variant_name"].eq(V2A_NAME)
    ].iloc[0]
    hard_v2b = ladder_metrics[
        ladder_metrics["candidate_pool_condition"].eq(HARD_NEGATIVE_CONDITION)
        & ladder_metrics["variant_name"].eq(V2B_NAME)
    ].iloc[0]

    print(f"complete_pairs: {validation['complete_pairs']}")
    print(f"subjects_with_complete_pairs: {validation['subjects_with_complete_pairs']}")
    print(f"Rest excluded: {validation['rest_excluded']}")
    print(f"sub-BYJoWR excluded: {validation['sub_BYJoWR_excluded']}")
    print(f"split files collapsed logically: {validation['split_files_collapsed_logically']}")
    print(f"top_k: {TOP_K}")
    print(f"aperiodic_alpha: {APERIODIC_ALPHA}")
    print(f"frozen v2A primary top1/MRR: {float(v2a_primary['top1']):.3f} / {float(v2a_primary['mrr']):.3f}")
    print(f"v2B-gated primary top1/MRR: {float(v2b_primary['top1']):.3f} / {float(v2b_primary['mrr']):.3f}")
    print(f"hard-negative v2A top1/MRR: {float(hard_v2a['top1']):.3f} / {float(hard_v2a['mrr']):.3f}")
    print(f"hard-negative v2B top1/MRR: {float(hard_v2b['top1']):.3f} / {float(hard_v2b['mrr']):.3f}")
    print(f"all-MedOn v2A top1/MRR: {float(all_medon_v2a['top1']):.3f} / {float(all_medon_v2a['mrr']):.3f}")
    print(f"all-MedOn v2B top1/MRR: {float(all_medon_v2b['top1']):.3f} / {float(all_medon_v2b['mrr']):.3f}")
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
