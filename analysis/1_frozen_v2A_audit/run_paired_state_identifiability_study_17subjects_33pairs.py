"""Paired-state identifiability study for frozen v2A on ds004998.

Scientific task:
    Test whether MedOff queries identify their paired MedOn recordings above
    task/side-controlled alternatives, while quantifying how much the result is
    bounded by candidate-pool difficulty and subject-fingerprint confounds.

This script is a study layer around the frozen v2/v2A retrieval logic. It uses
cached frozen-v2A pair/features/manifests only. It does not read raw FIF files,
does not use Oxford/MRC external data, and does not tune or change scorer,
features, top_k, alpha, or reranking logic.
"""

from __future__ import annotations

import hashlib
import json
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

from scripts.run_retrieval_aperiodic_assisted_v2 import (  # noqa: E402
    AssistedVariantSpec,
    aper_distance_scores,
    evaluate_assisted_variant,
)
from scripts.run_retrieval_distance_geometry_improvement import (  # noqa: E402
    VariantSpec,
    evaluate_variant as evaluate_v2_variant,
    query_diagnostics_from_scores,
    to_builtin,
)
from scripts.run_retrieval_signal_level_compact_v3 import MIN_DISTRACTORS  # noqa: E402
from src.evaluation.predictive_rescue_analysis import build_base_candidate_sets  # noqa: E402

from scripts.run_v2a_scientific_audit_17subjects_33pairs import (  # noqa: E402
    APERIODIC_ALPHA,
    N_PERM,
    RANDOM_SEED,
    REQUIRED_CACHE_FILES,
    TOP_K,
    V2A_NAME,
    V2_NAME,
    base_feature_names,
    load_subspace_inventory,
    matched_task_side_null,
    normalize_cached_scores,
    observed_metrics_table,
    query_metrics,
    read_csv_required,
    read_json_required,
    rebuild_v2_scores_from_frozen_v2a,
    validate_cached_inputs,
    write_csv,
    write_json,
)


OUTPUT_DIR = Path("outputs/paired_state_identifiability_study_17subjects_33pairs")


def write_text(lines: Iterable[str], path: Path) -> Path:
    """Write a text/Markdown file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    """Return SHA256 hash for an input file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_row(query: pd.Series, candidate: pd.Series, is_true: bool, match_level: str) -> dict[str, object]:
    """Build one candidate-metadata row in the existing retrieval schema."""

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


def finalize_candidate_rows(rows: list[dict[str, object]], condition_name: str) -> pd.DataFrame:
    """Add candidate counts and condition labels to candidate rows."""

    data = pd.DataFrame(rows)
    if data.empty:
        return data
    counts = data.groupby("query_pair_id", dropna=False)["candidate_pair_id"].transform("count")
    data["number_of_candidates"] = counts.astype(int)
    data["candidate_pool_condition"] = condition_name
    return data


def deduplicate_candidates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep the first row for each candidate pair, preserving true-first order."""

    seen: set[str] = set()
    out = []
    for row in rows:
        candidate_id = str(row["candidate_pair_id"])
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        out.append(row)
    return out


def build_custom_candidate_pool(pairs: pd.DataFrame, condition_name: str) -> pd.DataFrame:
    """Build diagnostic candidate pools without changing frozen scoring logic."""

    pair_rows = [row for _, row in pairs.iterrows()]
    rows: list[dict[str, object]] = []
    original = build_base_candidate_sets(pairs, MIN_DISTRACTORS)
    original_by_query = {
        str(query_id): group.copy()
        for query_id, group in original.groupby("query_pair_id", dropna=False)
    }

    for _, query in pairs.iterrows():
        query_id = str(query["pair_id"])
        subject = str(query["subject"])
        task = str(query["task_original"])
        family = str(query["task_family"])
        side = str(query["side"])
        quality = str(query.get("quality_flag", "unknown"))
        query_rows: list[dict[str, object]] = [candidate_row(query, query, True, condition_name)]

        if condition_name == "original_frozen_matched_pool":
            cached = original_by_query[query_id].copy()
            for _, row in cached.iterrows():
                candidate = pairs.loc[pairs["pair_id"].astype(str).eq(str(row["candidate_pair_id"]))].iloc[0]
                query_rows.append(candidate_row(query, candidate, bool(row["is_true_pair"]), str(row["match_level"])))

        elif condition_name == "task_side_quality_ignored":
            pool = pairs[
                ~pairs["subject"].astype(str).eq(subject)
                & pairs["task_original"].astype(str).eq(task)
                & pairs["side"].astype(str).eq(side)
            ]
            for _, candidate in pool.iterrows():
                query_rows.append(candidate_row(query, candidate, False, "task_original_side_quality_ignored"))

        elif condition_name == "task_side_quality_strict":
            pool = pairs[
                ~pairs["subject"].astype(str).eq(subject)
                & pairs["task_original"].astype(str).eq(task)
                & pairs["side"].astype(str).eq(side)
                & pairs.get("quality_flag", pd.Series("unknown", index=pairs.index)).astype(str).eq(quality)
            ]
            for _, candidate in pool.iterrows():
                query_rows.append(candidate_row(query, candidate, False, "task_original_side_quality_strict"))

        elif condition_name == "original_plus_same_subject_wrong_task_side":
            cached = original_by_query[query_id].copy()
            for _, row in cached.iterrows():
                candidate = pairs.loc[pairs["pair_id"].astype(str).eq(str(row["candidate_pair_id"]))].iloc[0]
                query_rows.append(candidate_row(query, candidate, bool(row["is_true_pair"]), str(row["match_level"])))
            same_subject_wrong = pairs[
                pairs["subject"].astype(str).eq(subject)
                & ~pairs["pair_id"].astype(str).eq(query_id)
            ]
            for _, candidate in same_subject_wrong.iterrows():
                query_rows.append(candidate_row(query, candidate, False, "same_subject_wrong_task_side_added"))

        elif condition_name == "all_medon_candidates":
            for candidate in pair_rows:
                candidate_id = str(candidate["pair_id"])
                query_rows.append(candidate_row(query, candidate, candidate_id == query_id, "all_medon_candidates"))

        elif condition_name == "true_plus_all_other_subject_medon":
            pool = pairs[~pairs["subject"].astype(str).eq(subject)]
            for _, candidate in pool.iterrows():
                query_rows.append(candidate_row(query, candidate, False, "all_other_subject_medon"))

        elif condition_name == "task_family_side_quality_ignored":
            pool = pairs[
                ~pairs["subject"].astype(str).eq(subject)
                & pairs["task_family"].astype(str).eq(family)
                & pairs["side"].astype(str).eq(side)
            ]
            for _, candidate in pool.iterrows():
                query_rows.append(candidate_row(query, candidate, False, "task_family_side_quality_ignored"))

        else:
            raise ValueError(f"Unknown candidate-pool condition: {condition_name}")

        rows.extend(deduplicate_candidates(query_rows))
    return finalize_candidate_rows(rows, condition_name)


def evaluate_frozen_on_candidate_pool(
    pairs: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate frozen v2 and v2A on a diagnostic candidate pool."""

    v2_variant = VariantSpec(
        name=V2_NAME,
        category="frozen_v2_candidate_generator",
        score_kind="group_balanced",
        feature_space="full_28_plus_coupling",
        distance="cosine",
    )
    v2_scores, v2_diag = evaluate_v2_variant(pairs, candidate_pool, v2_features, v2_variant, clean_features)
    aper_scores = aper_distance_scores(pairs, candidate_pool, compact_features, "aperiodic_distance")
    v2a_spec = AssistedVariantSpec(
        name=V2A_NAME,
        mode="two_stage",
        top_k=TOP_K,
        alpha=APERIODIC_ALPHA,
        note="Frozen v2 top-5 plus compact aperiodic/residual rerank; no tuning.",
    )
    v2a_scores, v2a_diag = evaluate_assisted_variant(v2_scores, aper_scores, v2a_spec)
    return v2_scores, v2_diag, v2a_scores, v2a_diag


def add_pool_context_to_diag(diag: pd.DataFrame, candidate_pool: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Add pool-composition context to query diagnostics."""

    if diag.empty:
        return diag
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
    counts = (
        pool.groupby("query_pair_id", dropna=False)
        .agg(
            number_of_candidates=("candidate_pair_id", "size"),
            n_same_subject_wrong_distractors=("same_subject_wrong_distractor", "sum"),
            n_same_task_side_distractors=("same_task_side_distractor", "sum"),
        )
        .reset_index()
    )
    out = diag.merge(counts, on="query_pair_id", how="left", suffixes=("", "_pool"))
    out.insert(0, "candidate_pool_condition", condition)
    out["top1_same_subject"] = out["top_ranked_candidate_subject"].astype(str).eq(out["query_subject"].astype(str))
    out["top1_same_task_side"] = (
        out["top_ranked_candidate_task"].astype(str).eq(out["query_task"].astype(str))
        & out["top_ranked_candidate_side"].astype(str).eq(out["query_side"].astype(str))
    )
    out["top1_same_subject_wrong_task_or_side"] = (
        out["top1_same_subject"].astype(bool)
        & ~out["top_ranked_is_true_pair"].astype(bool)
        & (
            ~out["top_ranked_candidate_task"].astype(str).eq(out["query_task"].astype(str))
            | ~out["top_ranked_candidate_side"].astype(str).eq(out["query_side"].astype(str))
        )
    )
    return out


def candidate_pool_ladder(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run candidate-pool severity ladder for frozen v2/v2A."""

    conditions = [
        "original_frozen_matched_pool",
        "task_side_quality_ignored",
        "task_side_quality_strict",
        "task_family_side_quality_ignored",
        "original_plus_same_subject_wrong_task_side",
        "all_medon_candidates",
        "true_plus_all_other_subject_medon",
    ]
    metric_rows: list[dict[str, object]] = []
    diag_tables: list[pd.DataFrame] = []
    score_tables: list[pd.DataFrame] = []
    for condition in conditions:
        candidate_pool = build_custom_candidate_pool(pairs, condition)
        v2_scores, v2_diag, v2a_scores, v2a_diag = evaluate_frozen_on_candidate_pool(
            pairs, candidate_pool, v2_features, compact_features, clean_features
        )
        for scores, diag, variant in [(v2_scores, v2_diag, V2_NAME), (v2a_scores, v2a_diag, V2A_NAME)]:
            scores = scores.copy()
            if "candidate_pool_condition" in scores.columns:
                scores["candidate_pool_condition"] = condition
            else:
                scores.insert(0, "candidate_pool_condition", condition)
            score_tables.append(scores)
            enriched = add_pool_context_to_diag(diag, candidate_pool, condition)
            diag_tables.append(enriched)
            metrics = query_metrics(diag)
            metric_rows.append(
                {
                    "candidate_pool_condition": condition,
                    "variant_name": variant,
                    **metrics,
                    "candidate_set_size_mean": float(candidate_pool.groupby("query_pair_id").size().mean()),
                    "candidate_set_size_min": int(candidate_pool.groupby("query_pair_id").size().min()),
                    "candidate_set_size_max": int(candidate_pool.groupby("query_pair_id").size().max()),
                    "queries_below_min_distractors": int(
                        ((candidate_pool.groupby("query_pair_id").size() - 1) < MIN_DISTRACTORS).sum()
                    ),
                    "same_subject_wrong_distractors_mean": float(
                        enriched["n_same_subject_wrong_distractors"].mean()
                    )
                    if not enriched.empty
                    else float("nan"),
                    "top1_same_subject_rate": float(enriched["top1_same_subject"].astype(float).mean())
                    if not enriched.empty
                    else float("nan"),
                    "top1_same_task_side_rate": float(enriched["top1_same_task_side"].astype(float).mean())
                    if not enriched.empty
                    else float("nan"),
                    "top1_same_subject_wrong_task_or_side_rate": float(
                        enriched["top1_same_subject_wrong_task_or_side"].astype(float).mean()
                    )
                    if not enriched.empty
                    else float("nan"),
                    "top1_subject_fingerprint_gap": float(enriched["top1_same_subject"].astype(float).mean())
                    - float(metrics["top1"])
                    if not enriched.empty
                    else float("nan"),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    diagnostics = pd.concat(diag_tables, ignore_index=True) if diag_tables else pd.DataFrame()
    scores = pd.concat(score_tables, ignore_index=True) if score_tables else pd.DataFrame()
    return metrics, diagnostics, scores


def subject_fingerprint_cases(ladder_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify whether retrieval is identifying subject rather than paired state."""

    focus = ladder_scores[
        ladder_scores["candidate_pool_condition"].isin(
            ["all_medon_candidates", "original_plus_same_subject_wrong_task_side"]
        )
    ].copy()
    rows = []
    for (condition, variant, query_id), group in focus.groupby(
        ["candidate_pool_condition", "variant_name", "query_pair_id"], dropna=False
    ):
        group = group.sort_values("rank").copy()
        true = group[group["is_true_pair"].astype(bool)]
        if true.empty:
            continue
        top = group.iloc[0]
        true_row = true.iloc[0]
        same_subject = group[group["candidate_subject"].astype(str).eq(str(top["query_subject"]))]
        same_subject_wrong = same_subject[~same_subject["is_true_pair"].astype(bool)]
        best_same_subject_wrong = (
            same_subject_wrong.sort_values("rank").iloc[0] if not same_subject_wrong.empty else pd.Series(dtype=object)
        )
        best_same_subject_rank = int(same_subject["rank"].min()) if not same_subject.empty else np.nan
        best_wrong_rank = (
            int(best_same_subject_wrong.get("rank", np.nan)) if not best_same_subject_wrong.empty else np.nan
        )
        true_rank = int(true_row["rank"])
        rows.append(
            {
                "candidate_pool_condition": condition,
                "variant_name": variant,
                "query_pair_id": query_id,
                "query_subject": top["query_subject"],
                "query_task": top["query_task_original"],
                "query_side": top["query_side"],
                "candidate_set_size": int(top["number_of_candidates"]),
                "true_rank": true_rank,
                "top1_true_pair": bool(true_rank == 1),
                "top1_candidate_pair_id": top["candidate_pair_id"],
                "top1_candidate_subject": top["candidate_subject"],
                "top1_candidate_task": top["candidate_task_original"],
                "top1_candidate_side": top["candidate_side"],
                "top1_same_subject": bool(str(top["candidate_subject"]) == str(top["query_subject"])),
                "top1_same_subject_wrong_task_or_side": bool(
                    str(top["candidate_subject"]) == str(top["query_subject"])
                    and not bool(top["is_true_pair"])
                    and (
                        str(top["candidate_task_original"]) != str(top["query_task_original"])
                        or str(top["candidate_side"]) != str(top["query_side"])
                    )
                ),
                "same_subject_wrong_available": bool(not same_subject_wrong.empty),
                "best_same_subject_rank": best_same_subject_rank,
                "best_same_subject_wrong_rank": best_wrong_rank,
                "true_beats_same_subject_wrong": bool(
                    same_subject_wrong.empty or true_rank < int(best_same_subject_wrong.get("rank", 10**9))
                ),
                "best_same_subject_wrong_pair_id": best_same_subject_wrong.get("candidate_pair_id", ""),
                "best_same_subject_wrong_task": best_same_subject_wrong.get("candidate_task_original", ""),
                "best_same_subject_wrong_side": best_same_subject_wrong.get("candidate_side", ""),
            }
        )
    cases = pd.DataFrame(rows)
    if cases.empty:
        return pd.DataFrame(), cases
    summary = (
        cases.groupby(["candidate_pool_condition", "variant_name"], dropna=False)
        .agg(
            n_queries=("query_pair_id", "size"),
            top1_true_pair_rate=("top1_true_pair", "mean"),
            top1_same_subject_rate=("top1_same_subject", "mean"),
            top1_same_subject_wrong_task_or_side_rate=("top1_same_subject_wrong_task_or_side", "mean"),
            same_subject_wrong_available_rate=("same_subject_wrong_available", "mean"),
            true_beats_same_subject_wrong_rate=("true_beats_same_subject_wrong", "mean"),
            median_true_rank=("true_rank", "median"),
            median_best_same_subject_wrong_rank=("best_same_subject_wrong_rank", "median"),
        )
        .reset_index()
    )
    summary["subject_fingerprint_gap"] = summary["top1_same_subject_rate"] - summary["top1_true_pair_rate"]
    summary["confound_flag"] = summary["subject_fingerprint_gap"].astype(float) >= 0.10
    return summary, cases


def stratum_removal_sensitivity(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
) -> pd.DataFrame:
    """Remove one task/side stratum at a time and rerun frozen retrieval."""

    rows = []
    strata = sorted(pairs["task_original"].astype(str).unique())
    for stratum in strata:
        subset = pairs[~pairs["task_original"].astype(str).eq(stratum)].copy()
        if subset.empty:
            continue
        candidate_pool = build_base_candidate_sets(subset, MIN_DISTRACTORS)
        v2_scores, v2_diag, v2a_scores, v2a_diag = evaluate_frozen_on_candidate_pool(
            subset, candidate_pool, v2_features, compact_features, clean_features
        )
        v2_metrics = query_metrics(v2_diag)
        v2a_metrics = query_metrics(v2a_diag)
        rows.append(
            {
                "removed_task_side_stratum": stratum,
                "n_pairs_remaining": int(len(subset)),
                "n_subjects_remaining": int(subset["subject"].nunique()),
                "v2_top1": v2_metrics["top1"],
                "v2_mrr": v2_metrics["mrr"],
                "v2A_top1": v2a_metrics["top1"],
                "v2A_mrr": v2a_metrics["mrr"],
                "v2A_minus_v2_top1": float(v2a_metrics["top1"]) - float(v2_metrics["top1"]),
                "v2A_minus_v2_mrr": float(v2a_metrics["mrr"]) - float(v2_metrics["mrr"]),
            }
        )
    return pd.DataFrame(rows)


def plot_ladder(metrics: pd.DataFrame, metric: str, path: Path) -> None:
    """Plot candidate-pool ladder metric for v2/v2A."""

    fig, ax = plt.subplots(figsize=(11, 5.2))
    if metrics.empty:
        ax.text(0.5, 0.5, "No ladder metrics", ha="center", va="center")
        ax.set_axis_off()
    else:
        order = list(metrics["candidate_pool_condition"].drop_duplicates())
        variants = [V2_NAME, V2A_NAME]
        x = np.arange(len(order))
        width = 0.36
        colors = {V2_NAME: "#4C78A8", V2A_NAME: "#F58518"}
        for idx, variant in enumerate(variants):
            values = []
            for condition in order:
                row = metrics[
                    metrics["candidate_pool_condition"].eq(condition)
                    & metrics["variant_name"].eq(variant)
                ]
                values.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (idx - 0.5) * width, values, width=width, label=variant, color=colors[variant])
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.set_title(f"Candidate-Pool Severity Ladder: {metric}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_subject_gap(summary: pd.DataFrame, path: Path) -> None:
    """Plot subject-fingerprint gap."""

    fig, ax = plt.subplots(figsize=(9, 4.6))
    if summary.empty:
        ax.text(0.5, 0.5, "No subject-fingerprint rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        data = summary[summary["variant_name"].eq(V2A_NAME)].copy()
        x = np.arange(len(data))
        ax.bar(x, data["subject_fingerprint_gap"].to_numpy(dtype=float), color="#E45756")
        ax.axhline(0.10, color="#333333", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(data["candidate_pool_condition"], rotation=25, ha="right")
        ax.set_ylabel("Same-subject top1 minus true-pair top1")
        ax.set_title("Subject-Fingerprint Gap for Frozen v2A")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(summary: dict[str, object], path: Path) -> Path:
    """Write concise Markdown study report."""

    primary = pd.DataFrame(summary["primary_observed_metrics"])
    v2a = primary[primary["variant_name"].eq(V2A_NAME)].iloc[0]
    ladder = pd.DataFrame(summary["candidate_pool_ladder_v2A"])
    lines = [
        "# Paired-State Identifiability Study",
        "",
        "Scientific question: can frozen MedOff-to-MedOn retrieval identify the paired MedOn recording above task/side-controlled alternatives, and where does this break under harder candidate pools?",
        "",
        "## Frozen Primary Result",
        f"- complete_pairs: {summary['validation']['complete_pairs']}",
        f"- subjects_with_complete_pairs: {summary['validation']['subjects_with_complete_pairs']}",
        f"- v2A top1: {float(v2a['top1']):.3f}",
        f"- v2A MRR: {float(v2a['mrr']):.3f}",
        f"- top_k: {summary['validation']['top_k']}",
        f"- aperiodic_alpha: {summary['validation']['aperiodic_alpha']}",
        "",
        "## Candidate-Pool Ladder",
    ]
    for _, row in ladder.iterrows():
        lines.append(
            f"- {row['candidate_pool_condition']}: top1={float(row['top1']):.3f}, "
            f"MRR={float(row['mrr']):.3f}, same-subject gap={float(row['top1_subject_fingerprint_gap']):.3f}"
        )
    lines.extend(
        [
            "",
            "## Warnings",
        ]
    )
    warnings = summary.get("warnings", [])
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Boundary statement: this is paired-state identifiability evidence only; it is not clinical prediction, treatment recommendation, DBS optimization, or continuous medication-effect estimation.",
        ]
    )
    return write_text(lines, path)


def run() -> dict[str, object]:
    """Run the paired-state identifiability study."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
    compact_features = [str(feature) for feature in compact_inventory.get("selected_features", [])]

    validation = validate_cached_inputs(
        pairs, logical, usable, excluded, cohort_summary, frozen_summary, extraction_failure_log
    )
    all_features = base_feature_names(pairs)
    v2_features = [feature for feature in subspaces.get("v2_reference", []) if feature in all_features]
    compact_features = [feature for feature in compact_features if feature in all_features]
    clean_features = [feature for feature in subspaces.get("clean_stable_features", []) if feature in all_features]
    if not v2_features:
        raise AssertionError("Frozen v2_reference features are missing.")
    if not compact_features:
        raise AssertionError("Frozen compact v2A rerank features are missing.")

    v2a_scores = cached_v2a_scores.copy()
    v2a_diag = query_diagnostics_from_scores(v2a_scores)
    v2_scores = rebuild_v2_scores_from_frozen_v2a(v2a_scores)
    v2_diag = query_diagnostics_from_scores(v2_scores)
    observed = observed_metrics_table(v2_diag, v2a_diag)
    matched_summary, matched_samples = matched_task_side_null(
        {V2_NAME: v2_diag, V2A_NAME: v2a_diag},
        {V2_NAME: v2_scores, V2A_NAME: v2a_scores},
        N_PERM,
        RANDOM_SEED,
    )
    ladder_metrics, ladder_diag, ladder_scores = candidate_pool_ladder(
        pairs, v2_features, compact_features, clean_features
    )
    fingerprint_summary, fingerprint_cases = subject_fingerprint_cases(ladder_scores)
    stratum_sensitivity = stratum_removal_sensitivity(pairs, v2_features, compact_features, clean_features)

    input_hash_rows = [
        {
            "input_name": name,
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
        for name, path in REQUIRED_CACHE_FILES.items()
    ]
    input_hashes = pd.DataFrame(input_hash_rows)

    warnings = []
    v2a_primary = observed[observed["variant_name"].eq(V2A_NAME)].iloc[0]
    all_medon_v2a = ladder_metrics[
        ladder_metrics["candidate_pool_condition"].eq("all_medon_candidates")
        & ladder_metrics["variant_name"].eq(V2A_NAME)
    ]
    strict_v2a = ladder_metrics[
        ladder_metrics["candidate_pool_condition"].eq("task_side_quality_strict")
        & ladder_metrics["variant_name"].eq(V2A_NAME)
    ]
    if not strict_v2a.empty and int(strict_v2a.iloc[0]["queries_below_min_distractors"]) > 0:
        warnings.append(
            "Strict task/side/quality pool has queries below MIN_DISTRACTORS; interpret its inflated metrics cautiously."
        )
    if not all_medon_v2a.empty and float(all_medon_v2a.iloc[0]["top1_subject_fingerprint_gap"]) >= 0.10:
        warnings.append("All-MedOn candidate pool shows a subject-fingerprint gap >= 0.10 for v2A.")
    augmented_v2a = fingerprint_summary[
        fingerprint_summary["candidate_pool_condition"].eq("original_plus_same_subject_wrong_task_side")
        & fingerprint_summary["variant_name"].eq(V2A_NAME)
    ]
    if not augmented_v2a.empty and float(augmented_v2a.iloc[0]["true_beats_same_subject_wrong_rate"]) < 0.75:
        warnings.append("Same-subject wrong-task/side hard negatives materially limit paired-state specificity.")

    primary_v2a_ladder = ladder_metrics[ladder_metrics["variant_name"].eq(V2A_NAME)].copy()
    summary = {
        "scientific_task": "paired_state_identifiability_not_clinical_prediction",
        "validation": validation,
        "frozen_settings": {
            "top_k": TOP_K,
            "aperiodic_alpha": APERIODIC_ALPHA,
            "retrieval_logic_changed": False,
            "model_or_feature_tuning": False,
        },
        "primary_observed_metrics": observed.to_dict("records"),
        "primary_matched_task_side_null": matched_summary.to_dict("records"),
        "candidate_pool_ladder_v2A": primary_v2a_ladder.to_dict("records"),
        "subject_fingerprint_summary": fingerprint_summary.to_dict("records"),
        "input_hashes": input_hash_rows,
        "warnings": warnings,
        "external_oxford_mrc_dataset_used": False,
        "raw_fif_reread": False,
    }

    paths = {
        "summary_json": write_json(summary, OUTPUT_DIR / "identifiability_summary.json"),
        "primary_metrics": write_csv(observed, OUTPUT_DIR / "primary_observed_v2_v2A_metrics.csv"),
        "matched_null_summary": write_csv(
            matched_summary, OUTPUT_DIR / "primary_matched_task_side_null_summary.csv"
        ),
        "matched_null_samples": write_csv(
            matched_samples, OUTPUT_DIR / "primary_matched_task_side_null_samples.csv"
        ),
        "ladder_metrics": write_csv(ladder_metrics, OUTPUT_DIR / "candidate_pool_ladder_metrics.csv"),
        "ladder_diag": write_csv(ladder_diag, OUTPUT_DIR / "candidate_pool_ladder_query_diagnostics.csv"),
        "ladder_scores": write_csv(ladder_scores, OUTPUT_DIR / "candidate_pool_ladder_candidate_scores.csv"),
        "fingerprint_summary": write_csv(
            fingerprint_summary, OUTPUT_DIR / "subject_fingerprint_confound_summary.csv"
        ),
        "fingerprint_cases": write_csv(fingerprint_cases, OUTPUT_DIR / "subject_fingerprint_confound_cases.csv"),
        "stratum_sensitivity": write_csv(
            stratum_sensitivity, OUTPUT_DIR / "task_side_stratum_removal_sensitivity.csv"
        ),
        "input_hashes": write_csv(input_hashes, OUTPUT_DIR / "frozen_input_hashes.csv"),
    }
    paths["summary_md"] = write_report(summary, OUTPUT_DIR / "identifiability_summary.md")
    plot_ladder(ladder_metrics, "top1", OUTPUT_DIR / "candidate_pool_ladder_top1.png")
    plot_ladder(ladder_metrics, "mrr", OUTPUT_DIR / "candidate_pool_ladder_mrr.png")
    plot_subject_gap(fingerprint_summary, OUTPUT_DIR / "subject_fingerprint_gap.png")

    print(f"complete_pairs: {validation['complete_pairs']}")
    print(f"subjects_with_complete_pairs: {validation['subjects_with_complete_pairs']}")
    print(f"Rest excluded: {validation['rest_excluded']}")
    print(f"sub-BYJoWR excluded: {validation['sub_BYJoWR_excluded']}")
    print(f"split files collapsed logically: {validation['split_files_collapsed_logically']}")
    print(f"top_k: {TOP_K}")
    print(f"aperiodic_alpha: {APERIODIC_ALPHA}")
    print(f"primary v2A top1/MRR: {float(v2a_primary['top1']):.3f} / {float(v2a_primary['mrr']):.3f}")
    if not all_medon_v2a.empty:
        print(
            "all-MedOn v2A top1/MRR: "
            f"{float(all_medon_v2a.iloc[0]['top1']):.3f} / {float(all_medon_v2a.iloc[0]['mrr']):.3f}"
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
