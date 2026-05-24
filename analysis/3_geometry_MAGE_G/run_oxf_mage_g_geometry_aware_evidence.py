#!/usr/bin/env python3
"""Run MAGE-G geometry-aware evidence retrieval on OXF all-pairs data.

MAGE-G is an experimental OXF-only extension of MAGE. It keeps the frozen
ds004998 pipeline untouched and does not tune model/scorer/top_k/alpha.

The scientific motivation is that STN-LFP recordings with different contact
spacing/source geometry should not be treated as exchangeable measurements.
The script therefore tests fixed geometry-aware rank transformations over
already cached OXF candidate scores:

1. MAGE_G_source_priority:
   Conservative source-compatible priority. Exact-contact candidates are
   compared first for exact-contact queries, and median-fallback candidates
   are compared first for median-fallback queries.

2. MAGE_G_source_span_soft:
   Exploratory soft geometry penalty. It keeps all candidates in one pool, but
   adds fixed penalties for broad montage-source mismatch and for contact
   center/span mismatch. Coefficients are fixed constants, not optimized.

3. MAGE_G_repair_class_stress:
   Fine repair-class priority, reported only as a leakage/confound stress test.
   Repair classes can encode recording-geometry metadata and must not be used
   as a primary algorithmic claim.
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
from scripts.run_oxf_raw_montage_reconstruction_audit import (  # noqa: E402
    BRANCH_EXACT_RECON_MEDIAN,
    OUTPUT_DIR as RAW_RECON_DIR,
)
from scripts.run_v2a_scientific_audit_17subjects_33pairs import (  # noqa: E402
    APERIODIC_ALPHA,
    RANDOM_SEED,
    TOP_K,
)
from scripts.run_v2b_gated_scientific_audit_17subjects_33pairs import ALL_MEDON_CONDITION  # noqa: E402
from scripts.run_v2c_deconfounded_transition_retrieval_experiment_17subjects_33pairs import (  # noqa: E402
    V2C_DECONFOUNDED_NAME,
)
from scripts.run_v2d_kreciprocal_quality_rerank_experiment_17subjects_33pairs import (  # noqa: E402
    V2D_QUALITY_DECONFOUNDED_NAME,
    build_metrics_tables,
    failure_cases,
    rank_scores,
    subject_bootstrap_ci,
)


OUTPUT_DIR = Path("outputs/oxf_mage_g_geometry_aware_evidence")
RAW_SCORES_PATH = RAW_RECON_DIR / "raw_reconstruction_candidate_scores.csv"
RAW_PAIRS_PATH = RAW_RECON_DIR / f"{BRANCH_EXACT_RECON_MEDIAN}_pairs.csv"
COVERAGE_CASES_PATH = Path("outputs/oxf_montage_harmonization_audit/montage_coverage_cases.csv")
CURRENT_MAGE_FOCUS_PATH = Path("outputs/oxf_montage_aware_gated_evidence_method/mage_focus_metrics.csv")
V2E_METRICS_PATH = Path("outputs/oxf_stn_physiology_locked_v2e/v2e_observed_metrics.csv")
MONTAGE_FOCUS_PATH = Path("outputs/oxf_montage_harmonization_audit/montage_branch_focus_metrics.csv")

BASE_VARIANTS = [
    oxf_base.V2A_NAME,
    V2C_DECONFOUNDED_NAME,
    V2D_QUALITY_DECONFOUNDED_NAME,
]

METHODS = [
    {
        "method_id": "MAGE_G_source_priority",
        "method_role": "cautious_sensitivity",
        "description": "Broad montage-source compatibility first; original rank fraction as within-block order.",
    },
    {
        "method_id": "MAGE_G_source_span_soft",
        "method_role": "exploratory_geometry_soft_penalty",
        "description": "Original rank fraction plus fixed source and contact-center/span geometry penalties.",
    },
    {
        "method_id": "MAGE_G_repair_class_stress",
        "method_role": "leakage_confound_stress_test",
        "description": "Fine repair-class priority; diagnostic only because repair class may fingerprint recording geometry.",
    },
]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    return str(path)


def write_json(data: dict[str, object], path: Path) -> str:
    path.write_text(json.dumps(oxf_base.to_builtin(data), indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def safe_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:  # noqa: BLE001
        return default
    return out if np.isfinite(out) else default


def rank_fraction(scores: pd.DataFrame) -> pd.Series:
    ranks = pd.to_numeric(scores["rank"], errors="coerce")
    counts = pd.to_numeric(scores["number_of_candidates"], errors="coerce").replace(1, np.nan)
    return ((ranks - 1.0) / (counts - 1.0)).fillna(1.0).clip(lower=0.0, upper=1.0)


def digit_center(value: object) -> float:
    digits = [int(char) for char in str(value) if char.isdigit()]
    return float(np.mean(digits)) if digits else np.nan


def normalized_distance(a: float, b: float, denom: float, missing_value: float = 0.5) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or denom <= 0:
        return missing_value
    return float(min(abs(a - b) / denom, 1.0))


def case_maps(coverage: pd.DataFrame) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for _, row in coverage.iterrows():
        pair_id = str(row["pair_id"])
        center = digit_center(row.get("off_contact_digits", ""))
        span = safe_float(row.get("off_contact_span"))
        rows[pair_id] = {
            "repair_class": str(row.get("repair_class", "")),
            "off_spacing_class": str(row.get("off_spacing_class", "")),
            "off_contact_digits": str(row.get("off_contact_digits", "")),
            "off_contact_center": center,
            "off_contact_span": span,
            "off_valid_bipolar_geometry": bool(row.get("off_valid_bipolar_geometry", False)),
        }
    return rows


def annotate_base_scores(scores: pd.DataFrame, pairs: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Attach broad montage source and fixed contact geometry to candidate rows."""

    pair_source = pairs.set_index("pair_id")["montage_hierarchy_source"].astype(str).to_dict()
    cases = case_maps(coverage)
    out = scores.copy()
    out["query_montage_source"] = out["query_pair_id"].astype(str).map(pair_source)
    out["candidate_montage_source"] = out["candidate_pair_id"].astype(str).map(pair_source)
    for side in ["query", "candidate"]:
        ids = out[f"{side}_pair_id"].astype(str)
        out[f"{side}_repair_class"] = ids.map(lambda value: str(cases.get(value, {}).get("repair_class", "")))
        out[f"{side}_spacing_class"] = ids.map(lambda value: str(cases.get(value, {}).get("off_spacing_class", "")))
        out[f"{side}_contact_digits"] = ids.map(lambda value: str(cases.get(value, {}).get("off_contact_digits", "")))
        out[f"{side}_contact_center"] = ids.map(lambda value: safe_float(cases.get(value, {}).get("off_contact_center")))
        out[f"{side}_contact_span"] = ids.map(lambda value: safe_float(cases.get(value, {}).get("off_contact_span")))
    out["source_mismatch"] = out["query_montage_source"].astype(str).ne(out["candidate_montage_source"].astype(str))
    out["repair_class_mismatch"] = out["query_repair_class"].astype(str).ne(out["candidate_repair_class"].astype(str))
    out["spacing_mismatch"] = out["query_spacing_class"].astype(str).ne(out["candidate_spacing_class"].astype(str))
    center_distance = [
        normalized_distance(q, c, 8.0)
        for q, c in zip(out["query_contact_center"].astype(float), out["candidate_contact_center"].astype(float), strict=False)
    ]
    span_distance = [
        normalized_distance(q, c, 4.0)
        for q, c in zip(out["query_contact_span"].astype(float), out["candidate_contact_span"].astype(float), strict=False)
    ]
    out["contact_center_distance"] = center_distance
    out["contact_span_distance"] = span_distance
    # Center is the more direct geometry proxy; span is down-weighted because
    # broad-vs-narrow spacing is already partly captured by montage source.
    out["geometry_distance"] = out["contact_center_distance"].astype(float) + 0.5 * out["contact_span_distance"].astype(float)
    out["rank_fraction"] = rank_fraction(out)
    return out


def method_scores(annotated: pd.DataFrame, method: dict[str, str]) -> pd.DataFrame:
    """Apply one fixed MAGE-G score transform."""

    out = annotated.copy()
    method_id = method["method_id"]
    if method_id == "MAGE_G_source_priority":
        out["score"] = out["source_mismatch"].astype(float) + 0.5 * out["rank_fraction"].astype(float)
        out["score_kind"] = "broad_source_lexicographic_priority"
        out["primary_interpretation"] = "cautious_source_compatible_sensitivity"
    elif method_id == "MAGE_G_source_span_soft":
        out["score"] = (
            out["rank_fraction"].astype(float)
            + 0.5 * out["source_mismatch"].astype(float)
            + 0.25 * out["geometry_distance"].astype(float)
        )
        out["score_kind"] = "rank_fraction_plus_fixed_geometry_penalty"
        out["primary_interpretation"] = "exploratory_soft_geometry_penalty"
    elif method_id == "MAGE_G_repair_class_stress":
        out["score"] = out["repair_class_mismatch"].astype(float) + 0.5 * out["rank_fraction"].astype(float)
        out["score_kind"] = "fine_repair_class_lexicographic_priority"
        out["primary_interpretation"] = "leakage_confound_stress_test"
    else:
        raise ValueError(f"Unsupported method: {method_id}")
    out["mage_g_method"] = method_id
    out["mage_g_method_role"] = method["method_role"]
    out["mage_g_description"] = method["description"]
    out["variant_name"] = method_id + "__" + out["variant_name"].astype(str)
    out["variant_category"] = "MAGE_G_geometry_aware_evidence"
    out["feature_space"] = "cached_oxf_stn_montage_geometry"
    out["distance"] = "rank_fraction_with_fixed_montage_geometry_terms"
    out["weight_scheme"] = "fixed_no_learned_weights_no_target_label_tuning"
    out["number_of_candidates"] = out.groupby(["candidate_pool_condition", "variant_name", "query_pair_id"])[
        "candidate_pair_id"
    ].transform("count")
    return rank_scores(out)


def build_mage_g_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_scores = read_csv(RAW_SCORES_PATH)
    pairs = read_csv(RAW_PAIRS_PATH)
    coverage = read_csv(COVERAGE_CASES_PATH)
    base = raw_scores[
        raw_scores["branch_id"].astype(str).eq(BRANCH_EXACT_RECON_MEDIAN)
        & raw_scores["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & raw_scores["variant_name"].astype(str).isin(BASE_VARIANTS)
    ].copy()
    annotated = annotate_base_scores(base, pairs, coverage)
    tables = [method_scores(annotated, method) for method in METHODS]
    scores = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    audit_cols = [
        "mage_g_method",
        "variant_name",
        "query_pair_id",
        "candidate_pair_id",
        "is_true_pair",
        "query_montage_source",
        "candidate_montage_source",
        "source_mismatch",
        "query_repair_class",
        "candidate_repair_class",
        "repair_class_mismatch",
        "query_spacing_class",
        "candidate_spacing_class",
        "spacing_mismatch",
        "query_contact_digits",
        "candidate_contact_digits",
        "query_contact_center",
        "candidate_contact_center",
        "query_contact_span",
        "candidate_contact_span",
        "contact_center_distance",
        "contact_span_distance",
        "geometry_distance",
        "rank_fraction",
        "score",
        "rank",
    ]
    return scores, scores[[col for col in audit_cols if col in scores.columns]].copy()


def focus_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    return metrics[metrics["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)].sort_values(
        ["top1", "mrr"], ascending=False
    )


def evidence_geometry_summary(diagnostics: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty or scores.empty:
        return pd.DataFrame()
    true_geometry = scores[scores["is_true_pair"].astype(bool)].copy()
    keep = [
        "variant_name",
        "query_pair_id",
        "mage_g_method",
        "query_montage_source",
        "query_repair_class",
        "query_spacing_class",
        "geometry_distance",
        "source_mismatch",
        "repair_class_mismatch",
    ]
    merged = diagnostics.merge(true_geometry[keep], on=["variant_name", "query_pair_id"], how="left")
    rows = []
    for keys, group in merged.groupby(
        ["variant_name", "mage_g_method", "query_montage_source", "query_repair_class", "query_spacing_class"],
        dropna=False,
    ):
        variant, method, source, repair, spacing = keys
        rank = pd.to_numeric(group["true_medon_rank"], errors="coerce")
        rows.append(
            {
                "variant_name": variant,
                "mage_g_method": method,
                "query_montage_source": source,
                "query_repair_class": repair,
                "query_spacing_class": spacing,
                "n_queries": int(len(group)),
                "top1": float(group["top_ranked_is_true_pair"].astype(bool).mean()),
                "mrr": float(pd.to_numeric(group["reciprocal_rank"], errors="coerce").mean()),
                "top3": float((rank <= 3).mean()),
                "top5": float((rank <= 5).mean()),
                "mean_true_rank": float(rank.mean()),
                "median_true_rank": float(rank.median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["variant_name", "query_montage_source", "query_repair_class"])


def failure_case_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Return explicit top-1 failure cases for MAGE-G diagnostics."""

    if diagnostics.empty:
        return pd.DataFrame()
    failed = diagnostics[~diagnostics["top_ranked_is_true_pair"].astype(bool)].copy()
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
    return failed[[col for col in keep if col in failed.columns]].reset_index(drop=True)


def optional_metric_rows() -> pd.DataFrame:
    rows = []
    if V2E_METRICS_PATH.exists():
        v2e_metrics = pd.read_csv(V2E_METRICS_PATH)
        for feature_set, variant, label in [
            ("median_channel_reference_with_bursts", V2D_QUALITY_DECONFOUNDED_NAME, "OXF_median_v2D"),
            ("off_beta_peak_contact_exact", V2D_QUALITY_DECONFOUNDED_NAME, "OXF_exact_contact_subset_v2D"),
        ]:
            match = v2e_metrics[
                v2e_metrics["feature_set"].astype(str).eq(feature_set)
                & v2e_metrics["variant_name"].astype(str).eq(variant)
                & v2e_metrics["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
            ]
            if not match.empty:
                row = match.iloc[0]
                rows.append(
                    {
                        "comparison_label": label,
                        "n_pairs": int(row["n_pairs"]),
                        "top1": float(row["top1"]),
                        "mrr": float(row["mrr"]),
                        "top5": float(row["top5"]),
                    }
                )
    if MONTAGE_FOCUS_PATH.exists():
        montage = pd.read_csv(MONTAGE_FOCUS_PATH)
        for branch, variant, label in [
            (
                "all_pairs_exact_else_median_compact_residual_7",
                V2D_QUALITY_DECONFOUNDED_NAME,
                "all_pairs_exact_else_median_v2D",
            ),
            ("exact_same_spacing_compact_residual_7", V2D_QUALITY_DECONFOUNDED_NAME, "exact_same_spacing_subset_v2D"),
        ]:
            match = montage[
                montage["branch_id"].astype(str).eq(branch)
                & montage["variant_name"].astype(str).eq(variant)
                & montage["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
            ]
            if not match.empty:
                row = match.iloc[0]
                rows.append(
                    {
                        "comparison_label": label,
                        "n_pairs": int(row["n_pairs"]),
                        "top1": float(row["top1"]),
                        "mrr": float(row["mrr"]),
                        "top5": float(row["top5"]),
                    }
                )
    if CURRENT_MAGE_FOCUS_PATH.exists():
        mage = pd.read_csv(CURRENT_MAGE_FOCUS_PATH)
        for variant, label in [
            (f"MAGE_{V2D_QUALITY_DECONFOUNDED_NAME}", "current_MAGE_v2D"),
            (f"MAGE_{V2C_DECONFOUNDED_NAME}", "current_MAGE_v2C"),
        ]:
            match = mage[mage["variant_name"].astype(str).eq(variant)]
            if not match.empty:
                row = match.iloc[0]
                rows.append(
                    {
                        "comparison_label": label,
                        "n_pairs": int(row["n_pairs"]),
                        "top1": float(row["top1"]),
                        "mrr": float(row["mrr"]),
                        "top5": float(row["top5"]),
                    }
                )
    return pd.DataFrame(rows)


def add_mage_g_comparisons(comparison: pd.DataFrame, focus: pd.DataFrame) -> pd.DataFrame:
    rows = comparison.to_dict("records") if not comparison.empty else []
    for _, row in focus.iterrows():
        rows.append(
            {
                "comparison_label": str(row["variant_name"]),
                "n_pairs": int(row["n_pairs"]),
                "top1": float(row["top1"]),
                "mrr": float(row["mrr"]),
                "top5": float(row["top5"]),
            }
        )
    return pd.DataFrame(rows)


def metric_value(focus: pd.DataFrame, method: str, base_variant: str, metric: str) -> float:
    variant = f"{method}__{base_variant}"
    rows = focus[focus["variant_name"].astype(str).eq(variant)]
    return safe_float(rows.iloc[0].get(metric)) if not rows.empty else np.nan


def write_report(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# OXF MAGE-G Geometry-Aware Evidence",
        "",
        "Experimental OXF-only extension. Frozen ds004998 v2A is not modified.",
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

    scores, geometry_audit = build_mage_g_scores()
    metrics, diagnostics = build_metrics_tables(scores)
    focus = focus_metrics(metrics)
    failures = failure_case_table(diagnostics)
    if failures.empty:
        failures = failure_cases(diagnostics)
    bootstrap_ci, bootstrap_samples = subject_bootstrap_ci(diagnostics)
    random_summary, random_samples = oxf_base.random_label_null(diagnostics, scores, 5000, RANDOM_SEED)
    matched_summary, matched_samples = oxf_base.matched_side_null(diagnostics, scores, 5000, RANDOM_SEED)
    geometry_summary = evidence_geometry_summary(diagnostics, scores)
    comparison = add_mage_g_comparisons(optional_metric_rows(), focus)

    outputs = {
        "candidate_scores": write_csv(scores, OUTPUT_DIR / "mage_g_candidate_scores.csv"),
        "geometry_audit": write_csv(geometry_audit, OUTPUT_DIR / "mage_g_geometry_audit.csv"),
        "observed_metrics": write_csv(metrics, OUTPUT_DIR / "mage_g_observed_metrics.csv"),
        "focus_metrics": write_csv(focus, OUTPUT_DIR / "mage_g_focus_metrics.csv"),
        "query_diagnostics": write_csv(diagnostics, OUTPUT_DIR / "mage_g_query_diagnostics.csv"),
        "failure_cases": write_csv(failures, OUTPUT_DIR / "mage_g_failure_cases.csv"),
        "subject_bootstrap_ci": write_csv(bootstrap_ci, OUTPUT_DIR / "mage_g_subject_bootstrap_ci.csv"),
        "subject_bootstrap_samples": write_csv(bootstrap_samples, OUTPUT_DIR / "mage_g_subject_bootstrap_samples.csv"),
        "random_label_null_summary": write_csv(random_summary, OUTPUT_DIR / "mage_g_random_label_null_summary.csv"),
        "random_label_null_samples": write_csv(random_samples, OUTPUT_DIR / "mage_g_random_label_null_samples.csv"),
        "matched_side_null_summary": write_csv(matched_summary, OUTPUT_DIR / "mage_g_matched_side_null_summary.csv"),
        "matched_side_null_samples": write_csv(matched_samples, OUTPUT_DIR / "mage_g_matched_side_null_samples.csv"),
        "geometry_summary": write_csv(geometry_summary, OUTPUT_DIR / "mage_g_geometry_summary.csv"),
        "method_comparison": write_csv(comparison, OUTPUT_DIR / "mage_g_method_comparison.csv"),
    }

    key_results = {
        "seed": RANDOM_SEED,
        "top_k": TOP_K,
        "aperiodic_alpha": APERIODIC_ALPHA,
        "n_queries": int(diagnostics["query_pair_id"].nunique()) if not diagnostics.empty else 0,
        "n_candidates_per_query": int(scores["number_of_candidates"].max()) if not scores.empty else 0,
        "source_priority_v2D_top1": metric_value(
            focus,
            "MAGE_G_source_priority",
            V2D_QUALITY_DECONFOUNDED_NAME,
            "top1",
        ),
        "source_priority_v2D_mrr": metric_value(
            focus,
            "MAGE_G_source_priority",
            V2D_QUALITY_DECONFOUNDED_NAME,
            "mrr",
        ),
        "source_span_soft_v2D_top1": metric_value(
            focus,
            "MAGE_G_source_span_soft",
            V2D_QUALITY_DECONFOUNDED_NAME,
            "top1",
        ),
        "source_span_soft_v2D_mrr": metric_value(
            focus,
            "MAGE_G_source_span_soft",
            V2D_QUALITY_DECONFOUNDED_NAME,
            "mrr",
        ),
        "repair_class_stress_v2D_top1": metric_value(
            focus,
            "MAGE_G_repair_class_stress",
            V2D_QUALITY_DECONFOUNDED_NAME,
            "top1",
        ),
        "repair_class_stress_v2D_mrr": metric_value(
            focus,
            "MAGE_G_repair_class_stress",
            V2D_QUALITY_DECONFOUNDED_NAME,
            "mrr",
        ),
    }
    warnings = [
        "MAGE-G is experimental and OXF-only; it is not a frozen ds004998 v2A replacement.",
        "MAGE_G_source_priority is the cautious source-compatible branch.",
        "MAGE_G_source_span_soft is exploratory; fixed geometry penalties were not learned, but this branch should be validated externally before becoming primary.",
        "MAGE_G_repair_class_stress is a confound/leakage stress test and must not be presented as an algorithmic improvement.",
        "No model parameters, top_k, alpha, base features, or base retrieval logic are changed.",
    ]
    summary = {"key_results": key_results, "outputs": outputs, "warnings": warnings}
    outputs["summary_json"] = write_json(summary, OUTPUT_DIR / "mage_g_summary.json")
    write_report(summary, OUTPUT_DIR / "mage_g_summary.md")

    print("OXF MAGE-G geometry-aware evidence complete.")
    print(f"Output folder: {OUTPUT_DIR}")
    for key, value in key_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
