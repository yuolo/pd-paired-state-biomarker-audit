#!/usr/bin/env python3
"""Validate whether OXF MAGE-G results are reproducible and robust.

This script is an audit layer. It does not change MAGE-G scores or any frozen
ds004998 outputs. It checks:

- exact reproducibility of saved MAGE-G candidate scores;
- paired deltas against current MAGE and median/all-pairs baselines;
- subject-aware bootstrap CIs for those deltas;
- LOSO sensitivity and collapse flags;
- geometry-metadata permutation controls;
- score-component contribution and top-1 change taxonomy.
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
from scripts import run_oxf_mage_g_geometry_aware_evidence as mage_g  # noqa: E402
from scripts.run_v2a_scientific_audit_17subjects_33pairs import (  # noqa: E402
    APERIODIC_ALPHA,
    RANDOM_SEED,
    TOP_K,
    query_metrics,
)
from scripts.run_v2b_gated_scientific_audit_17subjects_33pairs import ALL_MEDON_CONDITION  # noqa: E402
from scripts.run_v2c_deconfounded_transition_retrieval_experiment_17subjects_33pairs import (  # noqa: E402
    V2C_DECONFOUNDED_NAME,
)
from scripts.run_v2d_kreciprocal_quality_rerank_experiment_17subjects_33pairs import (  # noqa: E402
    V2D_QUALITY_DECONFOUNDED_NAME,
    build_metrics_tables,
    rank_scores,
)


OUTPUT_DIR = Path("outputs/oxf_mage_g_validity_audit")
MAGE_G_DIR = Path("outputs/oxf_mage_g_geometry_aware_evidence")
MAGE_DIR = Path("outputs/oxf_montage_aware_gated_evidence_method")
RAW_RECON_DIR = Path("outputs/oxf_raw_montage_reconstruction_audit")
V2E_DIR = Path("outputs/oxf_stn_physiology_locked_v2e")

TARGET_VARIANT = f"MAGE_G_source_span_soft__{V2D_QUALITY_DECONFOUNDED_NAME}"
SOURCE_PRIORITY_VARIANT = f"MAGE_G_source_priority__{V2D_QUALITY_DECONFOUNDED_NAME}"
REPAIR_STRESS_VARIANT = f"MAGE_G_repair_class_stress__{V2D_QUALITY_DECONFOUNDED_NAME}"
CURRENT_MAGE_VARIANT = f"MAGE_{V2D_QUALITY_DECONFOUNDED_NAME}"
MEDIAN_BASELINE_VARIANT = V2D_QUALITY_DECONFOUNDED_NAME
ALL_PAIRS_BASELINE_VARIANT = V2D_QUALITY_DECONFOUNDED_NAME
N_BOOT = 5000
N_PERM = 5000


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


def percentile_ci(values: pd.Series | np.ndarray) -> tuple[float, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    return float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))


def focus_scores(scores: pd.DataFrame, variant: str, feature_set: str | None = None, branch_id: str | None = None) -> pd.DataFrame:
    out = scores[
        scores["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & scores["variant_name"].astype(str).eq(variant)
    ].copy()
    if feature_set is not None and "feature_set" in out:
        out = out[out["feature_set"].astype(str).eq(feature_set)].copy()
    if branch_id is not None and "branch_id" in out:
        out = out[out["branch_id"].astype(str).eq(branch_id)].copy()
    return out


def focus_diag(diagnostics: pd.DataFrame, variant: str) -> pd.DataFrame:
    return diagnostics[
        diagnostics["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & diagnostics["variant_name"].astype(str).eq(variant)
    ].copy()


def diagnostics_from_scores(scores: pd.DataFrame) -> pd.DataFrame:
    _metrics, diagnostics = build_metrics_tables(scores)
    return diagnostics


def reproducibility_check(saved_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rebuilt_scores, _audit = mage_g.build_mage_g_scores()
    key = ["candidate_pool_condition", "variant_name", "query_pair_id", "candidate_pair_id"]
    saved = saved_scores[key + ["score", "rank"]].copy()
    rebuilt = rebuilt_scores[key + ["score", "rank"]].copy()
    merged = saved.merge(rebuilt, on=key, how="outer", suffixes=("_saved", "_rebuilt"), indicator=True)
    merged["score_abs_diff"] = (
        pd.to_numeric(merged["score_saved"], errors="coerce") - pd.to_numeric(merged["score_rebuilt"], errors="coerce")
    ).abs()
    merged["rank_equal"] = pd.to_numeric(merged["rank_saved"], errors="coerce").eq(
        pd.to_numeric(merged["rank_rebuilt"], errors="coerce")
    )
    summary = pd.DataFrame(
        [
            {
                "check": "rebuild_saved_mage_g_scores",
                "saved_rows": int(len(saved)),
                "rebuilt_rows": int(len(rebuilt)),
                "merged_rows": int(len(merged)),
                "left_only_rows": int((merged["_merge"] == "left_only").sum()),
                "right_only_rows": int((merged["_merge"] == "right_only").sum()),
                "max_score_abs_diff": float(pd.to_numeric(merged["score_abs_diff"], errors="coerce").max()),
                "rank_mismatch_rows": int((~merged["rank_equal"].fillna(False)).sum()),
                "pass": bool(
                    (merged["_merge"] == "both").all()
                    and float(pd.to_numeric(merged["score_abs_diff"], errors="coerce").max()) < 1e-12
                    and bool(merged["rank_equal"].all())
                ),
            }
        ]
    )
    mismatches = merged[
        (merged["_merge"] != "both")
        | (pd.to_numeric(merged["score_abs_diff"], errors="coerce") >= 1e-12)
        | (~merged["rank_equal"].fillna(False))
    ].copy()
    return summary, mismatches


def paired_query_table(
    target: pd.DataFrame,
    baselines: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    target_keep = target[
        [
            "query_pair_id",
            "query_subject",
            "query_side",
            "true_medon_rank",
            "top_ranked_is_true_pair",
            "reciprocal_rank",
            "percentile_rank",
            "failure_type",
        ]
    ].rename(
        columns={
            "true_medon_rank": "target_rank",
            "top_ranked_is_true_pair": "target_success",
            "reciprocal_rank": "target_rr",
            "percentile_rank": "target_percentile",
            "failure_type": "target_failure_type",
        }
    )
    merged = target_keep.copy()
    for label, diag in baselines.items():
        base = diag[
            [
                "query_pair_id",
                "true_medon_rank",
                "top_ranked_is_true_pair",
                "reciprocal_rank",
                "percentile_rank",
                "failure_type",
            ]
        ].rename(
            columns={
                "true_medon_rank": f"{label}_rank",
                "top_ranked_is_true_pair": f"{label}_success",
                "reciprocal_rank": f"{label}_rr",
                "percentile_rank": f"{label}_percentile",
                "failure_type": f"{label}_failure_type",
            }
        )
        merged = merged.merge(base, on="query_pair_id", how="left")
        rows.append(label)
    for label in rows:
        merged[f"delta_top1_vs_{label}"] = merged["target_success"].astype(float) - merged[f"{label}_success"].astype(float)
        merged[f"delta_mrr_vs_{label}"] = pd.to_numeric(merged["target_rr"], errors="coerce") - pd.to_numeric(
            merged[f"{label}_rr"], errors="coerce"
        )
        merged[f"rank_improved_vs_{label}"] = pd.to_numeric(merged["target_rank"], errors="coerce") < pd.to_numeric(
            merged[f"{label}_rank"], errors="coerce"
        )
        merged[f"rank_worsened_vs_{label}"] = pd.to_numeric(merged["target_rank"], errors="coerce") > pd.to_numeric(
            merged[f"{label}_rank"], errors="coerce"
        )
    return merged


def paired_delta_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = [
        col.replace("delta_top1_vs_", "")
        for col in paired.columns
        if col.startswith("delta_top1_vs_")
    ]
    for label in labels:
        rows.append(
            {
                "comparison": f"target_vs_{label}",
                "n_queries": int(len(paired)),
                "delta_top1": float(pd.to_numeric(paired[f"delta_top1_vs_{label}"], errors="coerce").mean()),
                "delta_mrr": float(pd.to_numeric(paired[f"delta_mrr_vs_{label}"], errors="coerce").mean()),
                "n_rank_improved": int(paired[f"rank_improved_vs_{label}"].astype(bool).sum()),
                "n_rank_worsened": int(paired[f"rank_worsened_vs_{label}"].astype(bool).sum()),
                "n_top1_gain": int((paired[f"delta_top1_vs_{label}"] > 0).sum()),
                "n_top1_loss": int((paired[f"delta_top1_vs_{label}"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def subject_bootstrap_delta(paired: pd.DataFrame, seed: int = RANDOM_SEED, n_boot: int = N_BOOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    subjects = sorted(paired["query_subject"].astype(str).unique())
    labels = [col.replace("delta_top1_vs_", "") for col in paired.columns if col.startswith("delta_top1_vs_")]
    samples = []
    for idx in range(n_boot):
        draw = rng.choice(subjects, size=len(subjects), replace=True)
        rows = pd.concat([paired[paired["query_subject"].astype(str).eq(str(subject))] for subject in draw], ignore_index=True)
        sample = {"bootstrap_index": int(idx), "seed": seed, "n_subjects_drawn": int(len(draw)), "n_queries": int(len(rows))}
        for label in labels:
            sample[f"delta_top1_vs_{label}"] = float(pd.to_numeric(rows[f"delta_top1_vs_{label}"], errors="coerce").mean())
            sample[f"delta_mrr_vs_{label}"] = float(pd.to_numeric(rows[f"delta_mrr_vs_{label}"], errors="coerce").mean())
        samples.append(sample)
    sample_df = pd.DataFrame(samples)
    summary_rows = []
    for label in labels:
        for metric in ["delta_top1", "delta_mrr"]:
            column = f"{metric}_vs_{label}"
            low, high = percentile_ci(sample_df[column])
            observed = float(pd.to_numeric(paired[column], errors="coerce").mean())
            summary_rows.append(
                {
                    "comparison": f"target_vs_{label}",
                    "metric": metric,
                    "observed": observed,
                    "bootstrap_mean": float(pd.to_numeric(sample_df[column], errors="coerce").mean()),
                    "ci_lower_95": low,
                    "ci_upper_95": high,
                    "ci_excludes_zero_positive": bool(low > 0),
                    "n_bootstrap": n_boot,
                    "seed": seed,
                    "bootstrap_unit": "subject",
                }
            )
    return pd.DataFrame(summary_rows), sample_df


def loso_target(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full = query_metrics(diagnostics)
    for subject in sorted(diagnostics["query_subject"].astype(str).unique()):
        reduced = diagnostics[~diagnostics["query_subject"].astype(str).eq(subject)].copy()
        metrics = query_metrics(reduced)
        rows.append(
            {
                "left_out_subject": subject,
                "n_pairs_remaining": metrics["n_pairs"],
                "top1": metrics["top1"],
                "mrr": metrics["mrr"],
                "delta_top1_vs_full": float(metrics["top1"]) - float(full["top1"]),
                "delta_mrr_vs_full": float(metrics["mrr"]) - float(full["mrr"]),
                "collapse_flag": bool(float(metrics["top1"]) < 0.45 or float(metrics["mrr"]) < 0.55),
            }
        )
    return pd.DataFrame(rows)


def geometry_permutation_scores(base_scores: pd.DataFrame, n_perm: int = N_PERM, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Permute query-side geometry metadata and rerun the MAGE-G soft formula."""

    rng = np.random.default_rng(seed)
    target_base = base_scores[
        base_scores["variant_name"].astype(str).eq(TARGET_VARIANT)
        & base_scores["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
    ].copy()
    if target_base.empty:
        raise AssertionError(f"No rows found for geometry permutation target: {TARGET_VARIANT}")
    query_meta_cols = [
        "query_montage_source",
        "query_repair_class",
        "query_spacing_class",
        "query_contact_digits",
        "query_contact_center",
        "query_contact_span",
    ]
    query_meta = target_base.groupby("query_pair_id", dropna=False)[query_meta_cols].first().reset_index()
    query_ids = query_meta["query_pair_id"].astype(str).tolist()
    sample_rows = []
    all_rows = []
    for perm_idx in range(n_perm):
        shuffled = query_meta.copy()
        shuffled_values = shuffled[query_meta_cols].to_numpy(dtype=object).copy()
        rng.shuffle(shuffled_values, axis=0)
        shuffled.loc[:, query_meta_cols] = shuffled_values
        meta_map = shuffled.set_index("query_pair_id")[query_meta_cols].to_dict("index")
        perm = target_base.copy()
        for col in query_meta_cols:
            perm[col] = perm["query_pair_id"].astype(str).map(lambda value, c=col: meta_map.get(value, {}).get(c, np.nan))
        perm["source_mismatch"] = perm["query_montage_source"].astype(str).ne(perm["candidate_montage_source"].astype(str))
        center_dist = [
            mage_g.normalized_distance(q, c, 8.0)
            for q, c in zip(perm["query_contact_center"].astype(float), perm["candidate_contact_center"].astype(float), strict=False)
        ]
        span_dist = [
            mage_g.normalized_distance(q, c, 4.0)
            for q, c in zip(perm["query_contact_span"].astype(float), perm["candidate_contact_span"].astype(float), strict=False)
        ]
        perm["contact_center_distance"] = center_dist
        perm["contact_span_distance"] = span_dist
        perm["geometry_distance"] = perm["contact_center_distance"].astype(float) + 0.5 * perm["contact_span_distance"].astype(float)
        perm["score"] = (
            perm["rank_fraction"].astype(float)
            + 0.5 * perm["source_mismatch"].astype(float)
            + 0.25 * perm["geometry_distance"].astype(float)
        )
        perm["variant_name"] = f"geometry_permutation_{perm_idx}"
        perm = rank_scores(perm)
        _metrics, diag = build_metrics_tables(perm)
        observed = query_metrics(diag)
        row = {
            "permutation_index": int(perm_idx),
            "top1": observed["top1"],
            "mrr": observed["mrr"],
            "percentile_rank": observed["percentile_rank"],
            "n_queries": int(len(query_ids)),
            "seed": seed,
            "null_type": "query_geometry_metadata_permutation",
        }
        sample_rows.append(row)
        if perm_idx < 5:
            all_rows.append(perm.assign(permutation_index=perm_idx))
    samples = pd.DataFrame(sample_rows)
    observed_scores = base_scores[base_scores["variant_name"].astype(str).eq(TARGET_VARIANT)].copy()
    _m, observed_diag = build_metrics_tables(observed_scores)
    observed = query_metrics(observed_diag)
    summary_rows = []
    for metric in ["top1", "mrr", "percentile_rank"]:
        vals = pd.to_numeric(samples[metric], errors="coerce").dropna().to_numpy(dtype=float)
        obs = float(observed[metric])
        if len(vals) == 0:
            empirical_p = np.nan
            null_mean = np.nan
            null_sd = np.nan
        else:
            empirical_p = float((np.sum(vals >= obs) + 1.0) / (len(vals) + 1.0))
            null_mean = float(np.nanmean(vals))
            null_sd = float(np.nanstd(vals, ddof=1))
        summary_rows.append(
            {
                "metric": metric,
                "observed": obs,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "empirical_p_greater_equal": empirical_p,
                "n_perm": int(len(vals)),
                "seed": seed,
                "null_type": "query_geometry_metadata_permutation",
            }
        )
    return pd.DataFrame(summary_rows), samples


def component_contribution(scores: pd.DataFrame) -> pd.DataFrame:
    target = scores[scores["variant_name"].astype(str).eq(TARGET_VARIANT)].copy()
    rows = []
    for is_true, group in target.groupby("is_true_pair", dropna=False):
        rows.append(
            {
                "row_type": "true_pair" if bool(is_true) else "distractor",
                "n_rows": int(len(group)),
                "mean_rank_fraction": float(pd.to_numeric(group["rank_fraction"], errors="coerce").mean()),
                "mean_source_mismatch": float(group["source_mismatch"].astype(float).mean()),
                "mean_geometry_distance": float(pd.to_numeric(group["geometry_distance"], errors="coerce").mean()),
                "mean_score": float(pd.to_numeric(group["score"], errors="coerce").mean()),
                "median_rank": float(pd.to_numeric(group["rank"], errors="coerce").median()),
            }
        )
    top = target.sort_values("rank").groupby("query_pair_id", dropna=False).first().reset_index()
    rows.append(
        {
            "row_type": "top1_candidate",
            "n_rows": int(len(top)),
            "mean_rank_fraction": float(pd.to_numeric(top["rank_fraction"], errors="coerce").mean()),
            "mean_source_mismatch": float(top["source_mismatch"].astype(float).mean()),
            "mean_geometry_distance": float(pd.to_numeric(top["geometry_distance"], errors="coerce").mean()),
            "mean_score": float(pd.to_numeric(top["score"], errors="coerce").mean()),
            "median_rank": float(pd.to_numeric(top["rank"], errors="coerce").median()),
        }
    )
    return pd.DataFrame(rows)


def validation_decision(
    target_metrics: dict[str, object],
    delta_summary: pd.DataFrame,
    delta_bootstrap: pd.DataFrame,
    loso: pd.DataFrame,
    geometry_null: pd.DataFrame,
    reproducibility: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    top1 = float(target_metrics["top1"])
    mrr = float(target_metrics["mrr"])
    rows.append({"criterion": "target_top1_at_least_0_55", "pass": bool(top1 >= 0.55), "value": top1})
    rows.append({"criterion": "target_mrr_at_least_0_65", "pass": bool(mrr >= 0.65), "value": mrr})
    for comparison in ["target_vs_current_mage", "target_vs_median_v2d", "target_vs_all_pairs_exact_else_median"]:
        row = delta_summary[delta_summary["comparison"].astype(str).eq(comparison)]
        if not row.empty:
            rows.append(
                {
                    "criterion": f"{comparison}_delta_top1_nonnegative",
                    "pass": bool(float(row.iloc[0]["delta_top1"]) >= 0.0),
                    "value": float(row.iloc[0]["delta_top1"]),
                }
            )
    mrr_ci = delta_bootstrap[
        delta_bootstrap["comparison"].astype(str).eq("target_vs_current_mage")
        & delta_bootstrap["metric"].astype(str).eq("delta_mrr")
    ]
    if not mrr_ci.empty:
        rows.append(
            {
                "criterion": "delta_mrr_vs_current_mage_ci_not_strongly_negative",
                "pass": bool(float(mrr_ci.iloc[0]["ci_upper_95"]) > 0.0),
                "value": float(mrr_ci.iloc[0]["observed"]),
            }
        )
    rows.append(
        {
            "criterion": "no_loso_collapse_below_0_45_top1_or_0_55_mrr",
            "pass": bool(not loso["collapse_flag"].astype(bool).any()),
            "value": int(loso["collapse_flag"].astype(bool).sum()),
        }
    )
    geom_top1 = geometry_null[geometry_null["metric"].astype(str).eq("top1")]
    if not geom_top1.empty:
        geom_p = safe_float(geom_top1.iloc[0]["empirical_p_greater_equal"])
        rows.append(
            {
                "criterion": "geometry_permutation_top1_p_below_0_05",
                "pass": bool(np.isfinite(geom_p) and geom_p < 0.05),
                "value": geom_p,
            }
        )
    rows.append(
        {
            "criterion": "saved_scores_rebuild_exactly",
            "pass": bool(reproducibility.iloc[0]["pass"]) if not reproducibility.empty else False,
            "value": safe_float(reproducibility.iloc[0].get("max_score_abs_diff")) if not reproducibility.empty else np.nan,
        }
    )
    return pd.DataFrame(rows)


def write_report(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# OXF MAGE-G Validity Audit",
        "",
        "Audit only. No retrieval outputs or frozen ds004998 files are modified.",
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

    mage_g_scores = read_csv(MAGE_G_DIR / "mage_g_candidate_scores.csv")
    mage_g_diag = read_csv(MAGE_G_DIR / "mage_g_query_diagnostics.csv")
    current_mage_scores = read_csv(MAGE_DIR / "mage_candidate_scores.csv")
    raw_scores = read_csv(RAW_RECON_DIR / "raw_reconstruction_candidate_scores.csv")
    v2e_scores = read_csv(V2E_DIR / "v2e_candidate_scores.csv")

    target_scores = focus_scores(mage_g_scores, TARGET_VARIANT)
    target_diag = focus_diag(mage_g_diag, TARGET_VARIANT)
    source_priority_diag = focus_diag(mage_g_diag, SOURCE_PRIORITY_VARIANT)
    repair_stress_diag = focus_diag(mage_g_diag, REPAIR_STRESS_VARIANT)

    current_mage_diag = diagnostics_from_scores(focus_scores(current_mage_scores, CURRENT_MAGE_VARIANT))
    median_diag = diagnostics_from_scores(
        focus_scores(v2e_scores, MEDIAN_BASELINE_VARIANT, feature_set="median_channel_reference_with_bursts")
    )
    all_pairs_diag = diagnostics_from_scores(
        focus_scores(
            raw_scores,
            ALL_PAIRS_BASELINE_VARIANT,
            branch_id="all_pairs_exact_raw_reconstruct_else_median_compact_residual_7",
        )
    )

    paired = paired_query_table(
        target_diag,
        {
            "current_mage": current_mage_diag,
            "median_v2d": median_diag,
            "all_pairs_exact_else_median": all_pairs_diag,
            "source_priority": source_priority_diag,
            "repair_stress": repair_stress_diag,
        },
    )
    delta_summary = paired_delta_summary(paired)
    delta_bootstrap, delta_bootstrap_samples = subject_bootstrap_delta(paired)
    loso = loso_target(target_diag)
    reproducibility_summary, reproducibility_mismatches = reproducibility_check(mage_g_scores)
    geometry_null_summary, geometry_null_samples = geometry_permutation_scores(mage_g_scores)
    components = component_contribution(mage_g_scores)
    target_metrics = query_metrics(target_diag)
    decision = validation_decision(
        target_metrics,
        delta_summary,
        delta_bootstrap,
        loso,
        geometry_null_summary,
        reproducibility_summary,
    )

    outputs = {
        "paired_query_deltas": write_csv(paired, OUTPUT_DIR / "mage_g_validity_paired_query_deltas.csv"),
        "paired_delta_summary": write_csv(delta_summary, OUTPUT_DIR / "mage_g_validity_paired_delta_summary.csv"),
        "paired_delta_bootstrap_ci": write_csv(delta_bootstrap, OUTPUT_DIR / "mage_g_validity_paired_delta_bootstrap_ci.csv"),
        "paired_delta_bootstrap_samples": write_csv(
            delta_bootstrap_samples,
            OUTPUT_DIR / "mage_g_validity_paired_delta_bootstrap_samples.csv",
        ),
        "loso_sensitivity": write_csv(loso, OUTPUT_DIR / "mage_g_validity_loso_sensitivity.csv"),
        "reproducibility_summary": write_csv(
            reproducibility_summary,
            OUTPUT_DIR / "mage_g_validity_reproducibility_summary.csv",
        ),
        "reproducibility_mismatches": write_csv(
            reproducibility_mismatches,
            OUTPUT_DIR / "mage_g_validity_reproducibility_mismatches.csv",
        ),
        "geometry_permutation_null_summary": write_csv(
            geometry_null_summary,
            OUTPUT_DIR / "mage_g_validity_geometry_permutation_null_summary.csv",
        ),
        "geometry_permutation_null_samples": write_csv(
            geometry_null_samples,
            OUTPUT_DIR / "mage_g_validity_geometry_permutation_null_samples.csv",
        ),
        "component_contribution": write_csv(components, OUTPUT_DIR / "mage_g_validity_component_contribution.csv"),
        "validation_decision": write_csv(decision, OUTPUT_DIR / "mage_g_validity_decision.csv"),
    }

    key_results = {
        "seed": RANDOM_SEED,
        "top_k": TOP_K,
        "aperiodic_alpha": APERIODIC_ALPHA,
        "target_variant": TARGET_VARIANT,
        "target_top1": float(target_metrics["top1"]),
        "target_mrr": float(target_metrics["mrr"]),
        "target_top5": float(target_metrics["top5"]),
        "delta_top1_vs_current_mage": safe_float(
            delta_summary.loc[delta_summary["comparison"].eq("target_vs_current_mage"), "delta_top1"].iloc[0]
        ),
        "delta_mrr_vs_current_mage": safe_float(
            delta_summary.loc[delta_summary["comparison"].eq("target_vs_current_mage"), "delta_mrr"].iloc[0]
        ),
        "delta_top1_vs_median_v2d": safe_float(
            delta_summary.loc[delta_summary["comparison"].eq("target_vs_median_v2d"), "delta_top1"].iloc[0]
        ),
        "delta_mrr_vs_median_v2d": safe_float(
            delta_summary.loc[delta_summary["comparison"].eq("target_vs_median_v2d"), "delta_mrr"].iloc[0]
        ),
        "loso_collapse_count": int(loso["collapse_flag"].astype(bool).sum()),
        "geometry_permutation_top1_p": safe_float(
            geometry_null_summary.loc[geometry_null_summary["metric"].eq("top1"), "empirical_p_greater_equal"].iloc[0]
        ),
        "reproducibility_pass": bool(reproducibility_summary.iloc[0]["pass"]),
        "decision_pass_count": int(decision["pass"].astype(bool).sum()),
        "decision_total_count": int(len(decision)),
    }
    warnings = [
        "This audit validates reproducibility and robustness; it does not make MAGE-G a frozen primary method.",
        "MAGE_G_source_span_soft remains exploratory because geometry penalty constants were introduced after OXF inspection.",
        "Repair-class stress is treated as a confound/leakage check, not an algorithmic improvement.",
        "Geometry permutation tests whether the observed geometry assignment matters; it is not an external validation cohort.",
    ]
    summary = {"key_results": key_results, "outputs": outputs, "warnings": warnings}
    outputs["summary_json"] = write_json(summary, OUTPUT_DIR / "mage_g_validity_summary.json")
    write_report(summary, OUTPUT_DIR / "mage_g_validity_summary.md")

    print("OXF MAGE-G validity audit complete.")
    print(f"Output folder: {OUTPUT_DIR}")
    for key, value in key_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
