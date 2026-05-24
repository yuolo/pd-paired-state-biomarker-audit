#!/usr/bin/env python3
"""Run an experimental montage-aware gated evidence OXF retrieval method.

The method decomposes transferability into query-specific evidence views:

1. Exact same-spacing bipolar evidence when the query has a valid same-spacing
   exact-contact ON/OFF pair.
2. Exact-contact evidence when exact labels match but spacing is unknown or
   segment-like.
3. Median-channel evidence as full-coverage fallback.

For each query, candidates comparable in the strongest valid view are ranked
lexicographically ahead of candidates that are unavailable in that view. This
is a fixed measurement-compatibility rule, not a learned weight and not tuned
on retrieval performance.

This OXF-only script does not alter frozen ds004998 v2A outputs and does not
change top_k or aperiodic_alpha.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from scripts import run_oxf_external_stn_retrieval_validation as oxf_base  # noqa: E402
from scripts import run_oxf_stn_physiology_locked_v2e as v2e  # noqa: E402
from scripts.run_v2a_scientific_audit_17subjects_33pairs import APERIODIC_ALPHA, RANDOM_SEED, TOP_K  # noqa: E402
from scripts.run_v2b_gated_scientific_audit_17subjects_33pairs import ALL_MEDON_CONDITION  # noqa: E402
from scripts.run_v2c_deconfounded_transition_retrieval_experiment_17subjects_33pairs import (  # noqa: E402
    V2C_DECONFOUNDED_NAME,
)
from scripts.run_v2d_kreciprocal_quality_rerank_experiment_17subjects_33pairs import (  # noqa: E402
    V2D_QUALITY_DECONFOUNDED_NAME,
)


OUTPUT_DIR = Path("outputs/oxf_montage_aware_gated_evidence_method")
MONTAGE_DIR = Path("outputs/oxf_montage_harmonization_audit")
V2E_DIR = Path("outputs/oxf_stn_physiology_locked_v2e")

MONTAGE_SCORES_PATH = MONTAGE_DIR / "montage_branch_candidate_scores.csv"
COVERAGE_CASES_PATH = MONTAGE_DIR / "montage_coverage_cases.csv"
V2E_SCORES_PATH = V2E_DIR / "v2e_candidate_scores.csv"

BASE_VARIANTS = [V2C_DECONFOUNDED_NAME, V2D_QUALITY_DECONFOUNDED_NAME]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return pd.read_csv(path)


def write_csv(data: pd.DataFrame, path: Path) -> str:
    data.to_csv(path, index=False)
    return str(path)


def write_json(data: dict[str, object], path: Path) -> str:
    path.write_text(json.dumps(oxf_base.to_builtin(data), indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def rank_percentile(frame: pd.DataFrame) -> pd.Series:
    rank = pd.to_numeric(frame["rank"], errors="coerce")
    size = pd.to_numeric(frame["number_of_candidates"], errors="coerce")
    return (rank - 1.0) / np.maximum(1.0, size - 1.0)


def focus_scores(scores: pd.DataFrame, feature_set: str | None, branch_id: str | None, variant: str) -> pd.DataFrame:
    out = scores[
        scores["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & scores["variant_name"].astype(str).eq(variant)
    ].copy()
    if feature_set is not None and "feature_set" in out:
        out = out[out["feature_set"].astype(str).eq(feature_set)].copy()
    if branch_id is not None and "branch_id" in out:
        out = out[out["branch_id"].astype(str).eq(branch_id)].copy()
    out["rank_percentile"] = rank_percentile(out)
    return out


def lookup_rank_percentile(scores: pd.DataFrame) -> dict[tuple[str, str], float]:
    return {
        (str(row["query_pair_id"]), str(row["candidate_pair_id"])): float(row["rank_percentile"])
        for _, row in scores.iterrows()
    }


def query_classes(coverage_cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in coverage_cases.iterrows():
        repair = str(row.get("repair_class", ""))
        if repair == "exact_same_spacing_available":
            evidence_view = "exact_same_spacing"
            priority = 0
        elif repair == "exact_available_unknown_or_segment_spacing":
            evidence_view = "exact_contact"
            priority = 1
        else:
            evidence_view = "median_fallback"
            priority = 2
        rows.append(
            {
                "pair_id": str(row["pair_id"]),
                "subject": row.get("subject", ""),
                "side": row.get("side", ""),
                "repair_class": repair,
                "gated_evidence_view": evidence_view,
                "evidence_priority": priority,
            }
        )
    return pd.DataFrame(rows)


def build_mage_scores(
    median_scores: pd.DataFrame,
    exact_scores: pd.DataFrame,
    same_spacing_scores: pd.DataFrame,
    classes: pd.DataFrame,
    base_variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_by_query = classes.set_index("pair_id", drop=False).to_dict("index")
    exact_lookup = lookup_rank_percentile(exact_scores)
    same_lookup = lookup_rank_percentile(same_spacing_scores)
    rows = []
    audit_rows = []
    for _, row in median_scores.iterrows():
        query_id = str(row["query_pair_id"])
        candidate_id = str(row["candidate_pair_id"])
        qclass = class_by_query.get(query_id, {})
        evidence_view = str(qclass.get("gated_evidence_view", "median_fallback"))
        median_pct = float(row["rank_percentile"])
        source = "median_fallback"
        priority = 0
        within_view_pct = median_pct
        if evidence_view == "exact_same_spacing":
            key = (query_id, candidate_id)
            if key in same_lookup:
                source = "exact_same_spacing"
                priority = 0
                within_view_pct = same_lookup[key]
            else:
                source = "median_after_same_spacing_unavailable"
                priority = 1
                within_view_pct = median_pct
        elif evidence_view == "exact_contact":
            key = (query_id, candidate_id)
            if key in exact_lookup:
                source = "exact_contact"
                priority = 0
                within_view_pct = exact_lookup[key]
            else:
                source = "median_after_exact_unavailable"
                priority = 1
                within_view_pct = median_pct
        # Lexicographic score: all compatible-view candidates precede fallback
        # candidates for that query; rank-percentile breaks ties within block.
        score = float(priority + within_view_pct)
        out = row.to_dict()
        out["score"] = score
        out["rank"] = np.nan
        out["variant_name"] = f"MAGE_{base_variant}"
        out["variant_category"] = "montage_aware_gated_evidence"
        out["feature_space"] = "query_specific_montage_evidence"
        out["score_kind"] = "lexicographic_view_priority_rank_percentile"
        out["distance"] = "rank_percentile"
        out["weight_scheme"] = "fixed_measurement_compatibility_no_learned_weights"
        out["residualized"] = False
        out["gated_evidence_view"] = evidence_view
        out["candidate_evidence_source"] = source
        out["evidence_priority_block"] = priority
        out["within_view_rank_percentile"] = within_view_pct
        rows.append(out)
        audit_rows.append(
            {
                "query_pair_id": query_id,
                "candidate_pair_id": candidate_id,
                "base_variant": base_variant,
                "query_evidence_view": evidence_view,
                "candidate_evidence_source": source,
                "evidence_priority_block": priority,
                "within_view_rank_percentile": within_view_pct,
                "score": score,
                "is_true_pair": bool(row["is_true_pair"]),
            }
        )
    scores = pd.DataFrame(rows)
    scores = v2e.rank_scores(v2e.normalize_scores(scores))
    return scores, pd.DataFrame(audit_rows)


def build_all_scores() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    montage_scores = read_csv(MONTAGE_SCORES_PATH)
    v2e_scores = read_csv(V2E_SCORES_PATH)
    coverage = read_csv(COVERAGE_CASES_PATH)
    classes = query_classes(coverage)
    all_tables = []
    audit_tables = []
    for variant in BASE_VARIANTS:
        median = focus_scores(v2e_scores, "median_channel_reference_with_bursts", None, variant)
        exact = focus_scores(v2e_scores, "off_beta_peak_contact_exact", None, variant)
        same = focus_scores(montage_scores, None, "exact_same_spacing_compact_residual_7", variant)
        mage, audit = build_mage_scores(median, exact, same, classes, variant)
        all_tables.append(mage)
        audit_tables.append(audit)
    return pd.concat(all_tables, ignore_index=True), pd.concat(audit_tables, ignore_index=True), classes


def evidence_view_summary(diagnostics: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    merged = diagnostics.merge(classes[["pair_id", "gated_evidence_view", "repair_class"]], left_on="query_pair_id", right_on="pair_id", how="left")
    rows = []
    for keys, group in merged.groupby(["variant_name", "gated_evidence_view", "repair_class"], dropna=False):
        variant, view, repair = keys
        rank = pd.to_numeric(group["true_medon_rank"], errors="coerce")
        rows.append(
            {
                "variant_name": variant,
                "gated_evidence_view": view,
                "repair_class": repair,
                "n_queries": int(len(group)),
                "top1": float(group["top_ranked_is_true_pair"].astype(bool).mean()),
                "mrr": float(pd.to_numeric(group["reciprocal_rank"], errors="coerce").mean()),
                "top3": float((rank <= 3).mean()),
                "top5": float((rank <= 5).mean()),
                "mean_true_rank": float(rank.mean()),
                "median_true_rank": float(rank.median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["variant_name", "gated_evidence_view", "repair_class"])


def focus_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[metrics["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)].copy().sort_values(
        ["top1", "mrr"], ascending=False
    )


def write_report(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# OXF Montage-Aware Gated Evidence Method",
        "",
        "Experimental OXF-only method. Fixed measurement-compatibility rule; no learned weights.",
        "",
        "## Key Results",
    ]
    for key, value in summary["key_results"].items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.3f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings"])
    for warning in summary["warnings"]:
        lines.append(f"- {warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    assert TOP_K == 5
    assert APERIODIC_ALPHA == 0.5

    scores, audit, classes = build_all_scores()
    metrics, diagnostics = v2e.build_metrics_tables(scores)
    focus = focus_metrics(metrics)
    view_summary = evidence_view_summary(diagnostics, classes)
    failures = v2e.failure_cases(diagnostics)
    bootstrap_ci, bootstrap_samples = v2e.subject_bootstrap_ci(diagnostics)
    random_summary, random_samples = oxf_base.random_label_null(diagnostics, scores, 5000, RANDOM_SEED)
    matched_summary, matched_samples = oxf_base.matched_side_null(diagnostics, scores, 5000, RANDOM_SEED)

    outputs = {
        "candidate_scores": write_csv(scores, OUTPUT_DIR / "mage_candidate_scores.csv"),
        "candidate_evidence_audit": write_csv(audit, OUTPUT_DIR / "mage_candidate_evidence_audit.csv"),
        "query_evidence_classes": write_csv(classes, OUTPUT_DIR / "mage_query_evidence_classes.csv"),
        "observed_metrics": write_csv(metrics, OUTPUT_DIR / "mage_observed_metrics.csv"),
        "focus_metrics": write_csv(focus, OUTPUT_DIR / "mage_focus_metrics.csv"),
        "query_diagnostics": write_csv(diagnostics, OUTPUT_DIR / "mage_query_diagnostics.csv"),
        "evidence_view_summary": write_csv(view_summary, OUTPUT_DIR / "mage_evidence_view_summary.csv"),
        "failure_cases": write_csv(failures, OUTPUT_DIR / "mage_failure_cases.csv"),
        "subject_bootstrap_ci": write_csv(bootstrap_ci, OUTPUT_DIR / "mage_subject_bootstrap_ci.csv"),
        "subject_bootstrap_samples": write_csv(bootstrap_samples, OUTPUT_DIR / "mage_subject_bootstrap_samples.csv"),
        "random_label_null_summary": write_csv(random_summary, OUTPUT_DIR / "mage_random_label_null_summary.csv"),
        "random_label_null_samples": write_csv(random_samples, OUTPUT_DIR / "mage_random_label_null_samples.csv"),
        "matched_side_null_summary": write_csv(matched_summary, OUTPUT_DIR / "mage_matched_side_null_summary.csv"),
        "matched_side_null_samples": write_csv(matched_samples, OUTPUT_DIR / "mage_matched_side_null_samples.csv"),
    }

    def variant_metric(name: str, metric: str) -> float:
        row = focus[focus["variant_name"].astype(str).eq(name)]
        if row.empty:
            return np.nan
        return float(row.iloc[0][metric])

    key_results = {
        "seed": RANDOM_SEED,
        "top_k": TOP_K,
        "aperiodic_alpha": APERIODIC_ALPHA,
        "n_queries": int(diagnostics["query_pair_id"].nunique()) if not diagnostics.empty else 0,
        "n_candidates_per_query": int(scores["number_of_candidates"].max()) if not scores.empty else 0,
        "MAGE_v2C_top1": variant_metric(f"MAGE_{V2C_DECONFOUNDED_NAME}", "top1"),
        "MAGE_v2C_mrr": variant_metric(f"MAGE_{V2C_DECONFOUNDED_NAME}", "mrr"),
        "MAGE_v2D_top1": variant_metric(f"MAGE_{V2D_QUALITY_DECONFOUNDED_NAME}", "top1"),
        "MAGE_v2D_mrr": variant_metric(f"MAGE_{V2D_QUALITY_DECONFOUNDED_NAME}", "mrr"),
    }
    warnings = [
        "MAGE is experimental and OXF-only; it is not a frozen v2A replacement.",
        "The method uses acquisition/montage evidence availability as a fixed compatibility rule.",
        "Lexicographic gating can improve exact-view queries but may encode montage-availability structure; report evidence-view strata.",
        "No model parameters, top_k, alpha, or base retrieval scores are tuned.",
        "Frozen ds004998 outputs are not modified.",
    ]
    summary = {"key_results": key_results, "outputs": outputs, "warnings": warnings}
    write_json(summary, OUTPUT_DIR / "mage_summary.json")
    write_report(summary, OUTPUT_DIR / "mage_summary.md")

    print("OXF MAGE method complete.")
    print(f"Output folder: {OUTPUT_DIR}")
    for key, value in key_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
