"""Experimental v2D k-reciprocal and quality-aware reranking for ds004998.

This is an exploratory study layer over cached frozen v2A, selected v2B-gated,
and v2C outputs. It does not change the frozen v2A/v2B pipelines, does not
read raw FIF files, does not use the Oxford/MRC external dataset, and does not
make clinical prediction, treatment, DBS, or causal medication-effect claims.
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

import matplotlib.pyplot as plt  # noqa: E402

from scripts.run_retrieval_distance_geometry_improvement import (  # noqa: E402
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
from scripts.run_v2c_deconfounded_transition_retrieval_experiment_17subjects_33pairs import (  # noqa: E402
    TRUE_PLUS_OTHER_CONDITION,
    V2C_CYCLE_NAME,
    V2C_DECONFOUNDED_NAME,
)


OUTPUT_DIR = Path("outputs/v2D_kreciprocal_quality_rerank_experiment_17subjects_33pairs")
V2B_AUDIT_DIR = Path("outputs/v2B_gated_scientific_audit_17subjects_33pairs")
DTSI_DIR = Path("outputs/dopaminergic_transition_specificity_index_17subjects_33pairs")
V2C_DIR = Path("outputs/v2C_deconfounded_transition_retrieval_experiment_17subjects_33pairs")

FORWARD_SCORES_PATH = V2B_AUDIT_DIR / "v2b_candidate_pool_ladder_candidate_scores.csv"
REVERSE_SCORES_PATH = DTSI_DIR / "reverse_medon_to_medoff_v2b_scores.csv"
V2C_SCORES_PATH = V2C_DIR / "v2c_candidate_scores.csv"

V2D_KRECIP_NAME = "v2D_kreciprocal_cycle"
V2D_KRECIP_QUALITY_NAME = "v2D_kreciprocal_quality_cycle"
V2D_KRECIP_QUALITY_DECONFOUNDED_NAME = "v2D_kreciprocal_quality_deconfounded"
V2D_QUALITY_DECONFOUNDED_NAME = "v2D_quality_deconfounded_cycle"

EVALUATED_CONDITIONS = [
    PRIMARY_CONDITION,
    HARD_NEGATIVE_CONDITION,
    ALL_MEDON_CONDITION,
    TRUE_PLUS_OTHER_CONDITION,
]
BASELINE_VARIANTS = [
    V2A_NAME,
    V2B_NAME,
    V2C_CYCLE_NAME,
    V2C_DECONFOUNDED_NAME,
]
V2D_VARIANT_NAMES = [
    V2D_KRECIP_NAME,
    V2D_KRECIP_QUALITY_NAME,
    V2D_KRECIP_QUALITY_DECONFOUNDED_NAME,
    V2D_QUALITY_DECONFOUNDED_NAME,
]

K_RECIPROCAL = TOP_K
CYCLE_WEIGHT = 1.0
K_RECIPROCAL_BONUS = 1.0
NEIGHBOR_JACCARD_BONUS = 1.0
QUALITY_MISMATCH_PENALTY = 2.0
WRONG_TASK_SIDE_PENALTY = 2.0
SAME_SUBJECT_WRONG_PENALTY = 2.0
BROAD_POOL_MIN_CANDIDATES = 20


@dataclass(frozen=True)
class V2DVariant:
    """A fixed exploratory v2D reranking variant."""

    name: str
    description: str
    use_kreciprocal: bool
    use_quality_penalty: bool
    use_deconfound_penalty: bool
    broad_pool_only: bool


V2D_VARIANTS = [
    V2DVariant(
        name=V2D_KRECIP_NAME,
        description="Forward+reverse cycle score plus k-reciprocal and neighborhood-overlap bonuses.",
        use_kreciprocal=True,
        use_quality_penalty=False,
        use_deconfound_penalty=False,
        broad_pool_only=False,
    ),
    V2DVariant(
        name=V2D_KRECIP_QUALITY_NAME,
        description="k-reciprocal cycle score with fixed quality-mismatch penalty.",
        use_kreciprocal=True,
        use_quality_penalty=True,
        use_deconfound_penalty=False,
        broad_pool_only=False,
    ),
    V2DVariant(
        name=V2D_KRECIP_QUALITY_DECONFOUNDED_NAME,
        description="Broad-pool k-reciprocal cycle score with fixed quality and confound penalties.",
        use_kreciprocal=True,
        use_quality_penalty=True,
        use_deconfound_penalty=True,
        broad_pool_only=True,
    ),
    V2DVariant(
        name=V2D_QUALITY_DECONFOUNDED_NAME,
        description="Broad-pool cycle score with fixed quality and confound penalties; k-reciprocal components recorded but not scored.",
        use_kreciprocal=False,
        use_quality_penalty=True,
        use_deconfound_penalty=True,
        broad_pool_only=True,
    ),
]


def write_text(lines: Iterable[str], path: Path) -> Path:
    """Write Markdown/text output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def as_bool(series: pd.Series) -> pd.Series:
    """Convert bool-like CSV columns to bool."""

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
    """Normalize score dtypes and bool-like columns."""

    out = scores.copy()
    for column in [
        "is_true_pair",
        "true_in_v2_top_k",
        "gated_aperiodic",
        "residualized",
        "same_subject_candidate",
        "same_task_side_candidate",
        "same_subject_wrong_task_side_candidate",
        "other_subject_same_task_side_candidate",
        "v2d_k_reciprocal",
        "v2d_quality_mismatch",
        "v2d_wrong_task_or_side",
        "v2d_same_subject_wrong",
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
        "reverse_score",
        "reverse_rank",
        "reverse_v2_score",
        "reverse_aperiodic_score",
        "v2d_forward_z",
        "v2d_reverse_z",
        "v2d_k_reciprocal_bonus",
        "v2d_neighbor_jaccard",
        "v2d_quality_penalty",
        "v2d_deconfound_penalty",
    ]:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def rank_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank candidate scores within condition, variant, and query."""

    ranked = scores.sort_values(["candidate_pool_condition", "variant_name", "query_pair_id", "score"]).copy()
    ranked["rank"] = ranked.groupby(
        ["candidate_pool_condition", "variant_name", "query_pair_id"], dropna=False
    ).cumcount() + 1
    return ranked


def reverse_score_lookup(reverse_scores: pd.DataFrame) -> pd.DataFrame:
    """Align reverse MedOn->MedOff v2B scores to forward query/candidate pairs."""

    focus = reverse_scores[
        reverse_scores["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & reverse_scores["variant_name"].astype(str).eq(V2B_NAME)
    ].copy()
    keep = ["query_pair_id", "candidate_pair_id", "score", "rank", "v2_score", "aperiodic_score"]
    return focus[keep].rename(
        columns={
            "query_pair_id": "candidate_pair_id",
            "candidate_pair_id": "query_pair_id",
            "score": "reverse_score",
            "rank": "reverse_rank",
            "v2_score": "reverse_v2_score",
            "aperiodic_score": "reverse_aperiodic_score",
        }
    )


def reverse_topk_sets(reverse_scores: pd.DataFrame) -> dict[str, set[str]]:
    """Return reverse top-k candidate-id sets keyed by reverse query id."""

    focus = reverse_scores[
        reverse_scores["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & reverse_scores["variant_name"].astype(str).eq(V2B_NAME)
    ].copy()
    out = {}
    for query_id, group in focus.groupby("query_pair_id", dropna=False):
        out[str(query_id)] = set(group.sort_values("rank").head(K_RECIPROCAL)["candidate_pair_id"].astype(str))
    return out


def neighbor_jaccard(forward_topk: set[str], reverse_topk: set[str]) -> float:
    """Jaccard overlap between forward and reverse top-k neighborhoods."""

    union = forward_topk | reverse_topk
    if not union:
        return 0.0
    return float(len(forward_topk & reverse_topk) / len(union))


def rerank_v2d_variant(
    forward_scores: pd.DataFrame,
    reverse_scores: pd.DataFrame,
    variant: V2DVariant,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rerank v2B scores with fixed k-reciprocal and quality-aware terms."""

    rev = reverse_score_lookup(reverse_scores)
    rev_topk = reverse_topk_sets(reverse_scores)
    focus = forward_scores[
        forward_scores["candidate_pool_condition"].astype(str).isin(EVALUATED_CONDITIONS)
        & forward_scores["variant_name"].astype(str).eq(V2B_NAME)
    ].copy()
    merged = focus.merge(rev, on=["query_pair_id", "candidate_pair_id"], how="left")
    rows: list[pd.DataFrame] = []
    component_rows: list[dict[str, object]] = []

    for (condition, query_id), group in merged.groupby(["candidate_pool_condition", "query_pair_id"], dropna=False):
        group = group.copy()
        candidate_count = int(pd.to_numeric(group["number_of_candidates"], errors="coerce").max())
        apply_rerank = (not variant.broad_pool_only) or candidate_count >= BROAD_POOL_MIN_CANDIDATES

        group["v2d_forward_z"] = 0.0
        group["v2d_reverse_z"] = 0.0
        group["v2d_k_reciprocal"] = False
        group["v2d_k_reciprocal_bonus"] = 0.0
        group["v2d_neighbor_jaccard"] = 0.0
        group["v2d_quality_mismatch"] = group["query_quality_flag"].astype(str).ne(group["candidate_quality_flag"].astype(str))
        group["v2d_quality_penalty"] = 0.0
        group["v2d_wrong_task_or_side"] = (
            (
                group["candidate_task_original"].astype(str).ne(group["query_task_original"].astype(str))
                | group["candidate_side"].astype(str).ne(group["query_side"].astype(str))
            )
            & ~group["is_true_pair"].astype(bool)
        )
        group["v2d_same_subject_wrong"] = (
            group["candidate_subject"].astype(str).eq(group["query_subject"].astype(str))
            & ~group["is_true_pair"].astype(bool)
        )
        group["v2d_deconfound_penalty"] = 0.0
        group["forward_score"] = pd.to_numeric(group["score"], errors="coerce")

        if not apply_rerank:
            group["score"] = group["forward_score"]
        else:
            forward_z = stable_zscore(group["score"])
            reverse_fill = float(pd.to_numeric(group["reverse_score"], errors="coerce").max())
            if not np.isfinite(reverse_fill):
                reverse_fill = float(pd.to_numeric(group["score"], errors="coerce").max())
            reverse_z = stable_zscore(group["reverse_score"].fillna(reverse_fill))
            forward_topk = set(group.sort_values("rank").head(K_RECIPROCAL)["candidate_pair_id"].astype(str))
            k_reciprocal_flags = []
            jaccards = []
            for _, candidate in group.iterrows():
                candidate_id = str(candidate["candidate_pair_id"])
                reverse_rank = candidate.get("reverse_rank", np.nan)
                k_reciprocal = bool(
                    float(candidate["rank"]) <= K_RECIPROCAL
                    and pd.notna(reverse_rank)
                    and float(reverse_rank) <= K_RECIPROCAL
                )
                k_reciprocal_flags.append(k_reciprocal)
                jaccards.append(neighbor_jaccard(forward_topk, rev_topk.get(candidate_id, set())))

            group["v2d_forward_z"] = forward_z
            group["v2d_reverse_z"] = CYCLE_WEIGHT * reverse_z
            group["v2d_k_reciprocal"] = k_reciprocal_flags
            group["v2d_neighbor_jaccard"] = jaccards
            if variant.use_kreciprocal:
                group["v2d_k_reciprocal_bonus"] = (
                    K_RECIPROCAL_BONUS * group["v2d_k_reciprocal"].astype(float)
                    + NEIGHBOR_JACCARD_BONUS * group["v2d_neighbor_jaccard"].astype(float)
                )
            if variant.use_quality_penalty:
                group["v2d_quality_penalty"] = QUALITY_MISMATCH_PENALTY * group["v2d_quality_mismatch"].astype(float)
            if variant.use_deconfound_penalty:
                group["v2d_deconfound_penalty"] = (
                    WRONG_TASK_SIDE_PENALTY * group["v2d_wrong_task_or_side"].astype(float)
                    + SAME_SUBJECT_WRONG_PENALTY * group["v2d_same_subject_wrong"].astype(float)
                )
            group["score"] = (
                group["v2d_forward_z"].astype(float)
                + group["v2d_reverse_z"].astype(float)
                - group["v2d_k_reciprocal_bonus"].astype(float)
                + group["v2d_quality_penalty"].astype(float)
                + group["v2d_deconfound_penalty"].astype(float)
            )

        group["variant_name"] = variant.name
        group["v2d_description"] = variant.description
        rows.append(group)
        true_row = group[group["is_true_pair"].astype(bool)].iloc[0]
        component_rows.append(
            {
                "candidate_pool_condition": condition,
                "variant_name": variant.name,
                "query_pair_id": query_id,
                "candidate_set_size": candidate_count,
                "rerank_applied": bool(apply_rerank),
                "true_pair_forward_rank": int(true_row["rank"]),
                "true_pair_reverse_rank": int(true_row.get("reverse_rank", 0)) if pd.notna(true_row.get("reverse_rank", np.nan)) else -1,
                "true_pair_k_reciprocal": bool(true_row.get("v2d_k_reciprocal", False)),
                "true_pair_neighbor_jaccard": float(true_row.get("v2d_neighbor_jaccard", np.nan)),
                "true_pair_quality_mismatch": bool(true_row.get("v2d_quality_mismatch", False)),
                "n_k_reciprocal_candidates": int(group["v2d_k_reciprocal"].astype(bool).sum()),
                "mean_neighbor_jaccard": float(pd.to_numeric(group["v2d_neighbor_jaccard"], errors="coerce").mean()),
                "n_quality_mismatch_candidates": int(group["v2d_quality_mismatch"].astype(bool).sum()),
                "n_wrong_task_or_side_candidates": int(group["v2d_wrong_task_or_side"].astype(bool).sum()),
                "n_same_subject_wrong_candidates": int(group["v2d_same_subject_wrong"].astype(bool).sum()),
            }
        )

    scores = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return rank_scores(scores), pd.DataFrame(component_rows)


def metrics_row(condition: str, variant_name: str, scores: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    """Build one metrics row and diagnostics table."""

    diagnostics = query_diagnostics_from_scores(scores)
    metrics = query_metrics(diagnostics)
    top = scores.sort_values("rank").groupby("query_pair_id", dropna=False).first().reset_index()
    if top.empty:
        top_same_subject_rate = float("nan")
        top_same_subject_wrong_rate = float("nan")
        top_same_task_side_rate = float("nan")
        top_wrong_task_side_rate = float("nan")
        top_quality_mismatch_rate = float("nan")
    else:
        top_same_subject = top["candidate_subject"].astype(str).eq(top["query_subject"].astype(str))
        top_same_subject_wrong = top_same_subject & ~top["is_true_pair"].astype(bool)
        top_same_task_side = (
            top["candidate_task_original"].astype(str).eq(top["query_task_original"].astype(str))
            & top["candidate_side"].astype(str).eq(top["query_side"].astype(str))
        )
        top_wrong_task_side = ~top_same_task_side & ~top["is_true_pair"].astype(bool)
        top_quality_mismatch = top["query_quality_flag"].astype(str).ne(top["candidate_quality_flag"].astype(str))
        top_same_subject_rate = float(top_same_subject.astype(float).mean())
        top_same_subject_wrong_rate = float(top_same_subject_wrong.astype(float).mean())
        top_same_task_side_rate = float(top_same_task_side.astype(float).mean())
        top_wrong_task_side_rate = float(top_wrong_task_side.astype(float).mean())
        top_quality_mismatch_rate = float(top_quality_mismatch.astype(float).mean())
    row = {
        "candidate_pool_condition": condition,
        "variant_name": variant_name,
        **metrics,
        "top1_same_subject_rate": top_same_subject_rate,
        "top1_same_subject_wrong_rate": top_same_subject_wrong_rate,
        "top1_same_task_side_rate": top_same_task_side_rate,
        "top1_wrong_task_or_side_rate": top_wrong_task_side_rate,
        "top1_quality_mismatch_rate": top_quality_mismatch_rate,
        "subject_fingerprint_gap": top_same_subject_rate - float(metrics["top1"])
        if np.isfinite(top_same_subject_rate) and np.isfinite(float(metrics["top1"]))
        else float("nan"),
    }
    diagnostics = diagnostics.copy()
    diagnostics.insert(0, "candidate_pool_condition", condition)
    return row, diagnostics


def build_metrics_tables(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build metrics and diagnostics for all loaded variants."""

    metric_rows: list[dict[str, object]] = []
    diagnostics_tables: list[pd.DataFrame] = []
    focus = scores[scores["candidate_pool_condition"].astype(str).isin(EVALUATED_CONDITIONS)].copy()
    for (condition, variant), subset in focus.groupby(["candidate_pool_condition", "variant_name"], dropna=False):
        row, diagnostics = metrics_row(str(condition), str(variant), subset)
        metric_rows.append(row)
        diagnostics_tables.append(diagnostics)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(diagnostics_tables, ignore_index=True) if diagnostics_tables else pd.DataFrame(),
    )


def comparison_to_baseline(metrics: pd.DataFrame, baseline_variant: str) -> pd.DataFrame:
    """Compare all non-baseline rows against a named baseline variant."""

    rows: list[dict[str, object]] = []
    baseline = metrics[metrics["variant_name"].astype(str).eq(baseline_variant)].set_index("candidate_pool_condition", drop=False)
    for _, row in metrics.iterrows():
        condition = str(row["candidate_pool_condition"])
        variant = str(row["variant_name"])
        if variant == baseline_variant or condition not in baseline.index:
            continue
        base = baseline.loc[condition]
        rows.append(
            {
                "baseline_variant": baseline_variant,
                "candidate_pool_condition": condition,
                "variant_name": variant,
                "delta_top1": float(row["top1"]) - float(base["top1"]),
                "delta_mrr": float(row["mrr"]) - float(base["mrr"]),
                "delta_failures": int(row["failures"]) - int(base["failures"]),
                "delta_subject_fingerprint_gap": float(row["subject_fingerprint_gap"])
                - float(base["subject_fingerprint_gap"]),
                "baseline_top1": float(base["top1"]),
                "variant_top1": float(row["top1"]),
                "baseline_mrr": float(base["mrr"]),
                "variant_mrr": float(row["mrr"]),
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
        local_samples: list[dict[str, object]] = []
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
            local_samples.append(
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
        samples = pd.DataFrame(local_samples)
        sample_rows.extend(local_samples)
        for metric in ["top1", "mrr"]:
            ci_low, ci_high = percentile_ci(samples[metric])
            summary_rows.append(
                {
                    "candidate_pool_condition": condition,
                    "variant_name": variant,
                    "metric": metric,
                    "observed": observed[metric],
                    "bootstrap_mean": float(pd.to_numeric(samples[metric], errors="coerce").mean()),
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
    """Return broad-pool failure cases for baseline and v2D variants."""

    focus = diagnostics[
        diagnostics["candidate_pool_condition"].astype(str).isin([ALL_MEDON_CONDITION, TRUE_PLUS_OTHER_CONDITION])
        & diagnostics["variant_name"].astype(str).isin([V2B_NAME, V2C_DECONFOUNDED_NAME, *V2D_VARIANT_NAMES])
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
        "top_ranked_candidate_quality",
        "failure_type",
        "retrieval_margin",
    ]
    return focus[[column for column in keep if column in focus.columns]].sort_values(
        ["candidate_pool_condition", "variant_name", "query_pair_id"]
    )


def plot_metric(metrics: pd.DataFrame, metric: str, path: Path) -> None:
    """Plot selected baseline and v2D metrics by condition."""

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    variants = [
        V2B_NAME,
        V2C_DECONFOUNDED_NAME,
        V2D_KRECIP_QUALITY_DECONFOUNDED_NAME,
        V2D_QUALITY_DECONFOUNDED_NAME,
    ]
    data = metrics[
        metrics["candidate_pool_condition"].astype(str).isin(EVALUATED_CONDITIONS)
        & metrics["variant_name"].astype(str).isin(variants)
    ].copy()
    if data.empty:
        ax.text(0.5, 0.5, "No metrics", ha="center", va="center")
        ax.set_axis_off()
    else:
        colors = {
            V2B_NAME: "#4C78A8",
            V2C_DECONFOUNDED_NAME: "#54A24B",
            V2D_KRECIP_QUALITY_DECONFOUNDED_NAME: "#F58518",
            V2D_QUALITY_DECONFOUNDED_NAME: "#B279A2",
        }
        x = np.arange(len(EVALUATED_CONDITIONS))
        width = 0.18
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(variants))
        for offset, variant in zip(offsets, variants, strict=False):
            values = []
            for condition in EVALUATED_CONDITIONS:
                row = data[data["candidate_pool_condition"].eq(condition) & data["variant_name"].eq(variant)]
                values.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + offset, values, width=width, label=variant, color=colors[variant])
        ax.set_xticks(x)
        ax.set_xticklabels(EVALUATED_CONDITIONS, rotation=25, ha="right")
        if metric in {"top1", "mrr"}:
            ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.set_title(f"Experimental v2D {metric}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_all_medon_delta(comparison: pd.DataFrame, path: Path) -> None:
    """Plot all-MedOn deltas against v2B."""

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    data = comparison[
        comparison["baseline_variant"].astype(str).eq(V2B_NAME)
        & comparison["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & comparison["variant_name"].astype(str).isin([V2C_DECONFOUNDED_NAME, *V2D_VARIANT_NAMES])
    ].copy()
    if data.empty:
        ax.text(0.5, 0.5, "No comparison rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.arange(len(data))
        ax.bar(x, data["delta_top1"].to_numpy(dtype=float), color="#72B7B2")
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(data["variant_name"], rotation=25, ha="right")
        ax.set_ylabel("Delta top1 vs v2B")
        ax.set_title("All-MedOn Delta vs v2B")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_summary_md(summary: dict[str, object], path: Path) -> Path:
    """Write a concise Markdown summary."""

    lines = [
        "# v2D k-reciprocal and quality-aware rerank experiment",
        "",
        "Scope: exploratory reranking over cached ds004998 outputs. Frozen v2A/v2B are unchanged.",
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
            f"top1={float(row['top1']):.3f}, MRR={float(row['mrr']):.3f}, failures={int(row['failures'])}"
        )
    lines.extend(["", "## Warnings"])
    lines.extend([f"- {warning}" for warning in summary["warnings"]])
    return write_text(lines, path)


def load_inputs() -> dict[str, object]:
    """Load cached inputs and validate the verified cohort."""

    missing = [str(path) for path in [FORWARD_SCORES_PATH, REVERSE_SCORES_PATH, V2C_SCORES_PATH] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required v2D inputs: " + "; ".join(missing))

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

    return {
        "validation": validation,
        "forward_scores": normalize_scores(read_csv_required(FORWARD_SCORES_PATH)),
        "reverse_scores": normalize_scores(read_csv_required(REVERSE_SCORES_PATH)),
        "v2c_scores": normalize_scores(read_csv_required(V2C_SCORES_PATH)),
    }


def run() -> dict[str, object]:
    """Run v2D experiment."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    validation = inputs["validation"]
    forward_scores = inputs["forward_scores"]
    reverse_scores = inputs["reverse_scores"]
    v2c_scores = inputs["v2c_scores"]

    v2d_score_tables: list[pd.DataFrame] = []
    component_tables: list[pd.DataFrame] = []
    for variant in V2D_VARIANTS:
        scores, components = rerank_v2d_variant(forward_scores, reverse_scores, variant)
        v2d_score_tables.append(scores)
        component_tables.append(components)

    baseline_v2a_v2b = forward_scores[
        forward_scores["candidate_pool_condition"].astype(str).isin(EVALUATED_CONDITIONS)
        & forward_scores["variant_name"].astype(str).isin([V2A_NAME, V2B_NAME])
    ].copy()
    baseline_v2c = v2c_scores[
        v2c_scores["candidate_pool_condition"].astype(str).isin(EVALUATED_CONDITIONS)
        & v2c_scores["variant_name"].astype(str).isin([V2C_CYCLE_NAME, V2C_DECONFOUNDED_NAME])
    ].copy()
    all_scores = pd.concat([baseline_v2a_v2b, baseline_v2c, *v2d_score_tables], ignore_index=True)
    all_scores = rank_scores(normalize_scores(all_scores))
    components = pd.concat(component_tables, ignore_index=True) if component_tables else pd.DataFrame()

    metrics, diagnostics = build_metrics_tables(all_scores)
    comparison_v2b = comparison_to_baseline(metrics, V2B_NAME)
    comparison_v2c = comparison_to_baseline(metrics, V2C_DECONFOUNDED_NAME)
    comparison = pd.concat([comparison_v2b, comparison_v2c], ignore_index=True)
    bootstrap_summary, bootstrap_samples = subject_bootstrap_ci(diagnostics)
    failures = failure_cases(diagnostics)

    key_metrics = metrics[
        metrics["candidate_pool_condition"].astype(str).isin([PRIMARY_CONDITION, HARD_NEGATIVE_CONDITION, ALL_MEDON_CONDITION])
        & metrics["variant_name"].astype(str).isin([V2B_NAME, V2C_DECONFOUNDED_NAME, *V2D_VARIANT_NAMES])
    ].sort_values(["candidate_pool_condition", "variant_name"])
    warnings = [
        "v2D is experimental and exploratory; frozen v2A/v2B and v2C outputs are unchanged.",
        "No Oxford/MRC external dataset and no raw FIF rereading were used.",
        "k-reciprocal, quality, and confound penalties are fixed audit settings, not a clinical model.",
        "Quality-aware reranking may reflect acquisition-quality structure; interpret as robustness control only.",
        "No clinical prediction, treatment recommendation, DBS optimization, or causal medication-effect claim is made.",
    ]
    summary = {
        "validation": validation,
        "v2d_fixed_settings": {
            "k_reciprocal": K_RECIPROCAL,
            "cycle_weight": CYCLE_WEIGHT,
            "k_reciprocal_bonus": K_RECIPROCAL_BONUS,
            "neighbor_jaccard_bonus": NEIGHBOR_JACCARD_BONUS,
            "quality_mismatch_penalty": QUALITY_MISMATCH_PENALTY,
            "wrong_task_side_penalty": WRONG_TASK_SIDE_PENALTY,
            "same_subject_wrong_penalty": SAME_SUBJECT_WRONG_PENALTY,
            "broad_pool_min_candidates": BROAD_POOL_MIN_CANDIDATES,
        },
        "variant_descriptions": [variant.__dict__ for variant in V2D_VARIANTS],
        "key_metrics": key_metrics.to_dict("records"),
        "all_medon_comparison_to_v2b": comparison_v2b[
            comparison_v2b["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        ].to_dict("records"),
        "warnings": warnings,
        "external_oxford_mrc_dataset_used": False,
        "raw_fif_reread": False,
        "frozen_v2A_changed": False,
        "v2B_logic_changed": False,
        "not_clinical_prediction_or_treatment": True,
    }

    paths = {
        "summary_json": write_json(summary, OUTPUT_DIR / "v2d_experiment_summary.json"),
        "metrics": write_csv(metrics, OUTPUT_DIR / "v2d_metrics.csv"),
        "candidate_scores": write_csv(all_scores, OUTPUT_DIR / "v2d_candidate_scores.csv"),
        "query_diagnostics": write_csv(diagnostics, OUTPUT_DIR / "v2d_query_diagnostics.csv"),
        "comparison": write_csv(comparison, OUTPUT_DIR / "v2d_comparison_to_baselines.csv"),
        "bootstrap_ci": write_csv(bootstrap_summary, OUTPUT_DIR / "v2d_bootstrap_ci.csv"),
        "bootstrap_samples": write_csv(bootstrap_samples, OUTPUT_DIR / "v2d_bootstrap_samples.csv"),
        "failure_cases": write_csv(failures, OUTPUT_DIR / "v2d_failure_cases.csv"),
        "kreciprocal_components": write_csv(components, OUTPUT_DIR / "v2d_kreciprocal_components.csv"),
    }
    paths["summary_md"] = write_summary_md(to_builtin(summary), OUTPUT_DIR / "v2d_experiment_summary.md")

    plot_metric(metrics, "top1", OUTPUT_DIR / "v2d_top1_by_condition.png")
    plot_metric(metrics, "mrr", OUTPUT_DIR / "v2d_mrr_by_condition.png")
    plot_all_medon_delta(comparison, OUTPUT_DIR / "v2d_all_medon_delta_top1_vs_v2b.png")

    v2b_all = metrics[
        metrics["candidate_pool_condition"].eq(ALL_MEDON_CONDITION) & metrics["variant_name"].eq(V2B_NAME)
    ].iloc[0]
    v2c_all = metrics[
        metrics["candidate_pool_condition"].eq(ALL_MEDON_CONDITION) & metrics["variant_name"].eq(V2C_DECONFOUNDED_NAME)
    ].iloc[0]
    v2d_all = metrics[
        metrics["candidate_pool_condition"].eq(ALL_MEDON_CONDITION)
        & metrics["variant_name"].eq(V2D_QUALITY_DECONFOUNDED_NAME)
    ].iloc[0]
    v2d_true_plus = metrics[
        metrics["candidate_pool_condition"].eq(TRUE_PLUS_OTHER_CONDITION)
        & metrics["variant_name"].eq(V2D_QUALITY_DECONFOUNDED_NAME)
    ].iloc[0]
    print(f"complete_pairs: {validation['complete_pairs']}")
    print(f"subjects_with_complete_pairs: {validation['subjects_with_complete_pairs']}")
    print(f"Rest excluded: {validation['rest_excluded']}")
    print(f"sub-BYJoWR excluded: {validation['sub_BYJoWR_excluded']}")
    print(f"split files collapsed logically: {validation['split_files_collapsed_logically']}")
    print(f"top_k: {TOP_K}")
    print(f"aperiodic_alpha: {APERIODIC_ALPHA}")
    print(f"v2B all-MedOn top1/MRR: {float(v2b_all['top1']):.3f}/{float(v2b_all['mrr']):.3f}")
    print(f"v2C deconfounded all-MedOn top1/MRR: {float(v2c_all['top1']):.3f}/{float(v2c_all['mrr']):.3f}")
    print(f"v2D quality-deconfounded all-MedOn top1/MRR: {float(v2d_all['top1']):.3f}/{float(v2d_all['mrr']):.3f}")
    print(
        "v2D quality-deconfounded true+all-other-subject top1/MRR: "
        f"{float(v2d_true_plus['top1']):.3f}/{float(v2d_true_plus['mrr']):.3f}"
    )
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
