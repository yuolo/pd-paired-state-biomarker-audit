#!/usr/bin/env python3
"""Reviewer-facing robustness checks for the v2D/MAGE-G journal package.

This script is intentionally downstream-only. It reads cached OXF/ds004998
outputs and writes an additional audit package:

1. subject-LOSO stability of the OXF-trained MAGE-G logistic ranker;
2. subject-clustered bootstrap CIs for the OXF ladder and reverse-transfer
   numbers used in the manuscript;
3. an SVG histogram of geometry-permutation null outcomes;
4. direct numeric-repair versus no-numeric comparison under the same median
   fallback policy.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_oxf_frozen_reverse_transfer_to_ds004998 as reverse  # noqa: E402


OUTPUT_DIR = Path("outputs/journal_mage_g_robustness_audit")
FIGURES_DIR = OUTPUT_DIR / "figures"
MANUSCRIPT_FIGURES_DIR = Path("manuscript/figures")
MANUSCRIPT_TABLES_DIR = Path("manuscript/tables")

RANDOM_SEED = 42
N_BOOT = 5000
ALL_MEDON_CONDITION = "all_medon_candidates"
V2D_VARIANT = "v2D_quality_deconfounded_cycle"
MAGE_G_VARIANT = "MAGE_G_source_span_soft__v2D_quality_deconfounded_cycle"

OXF_EXTERNAL_DIAG = Path("outputs/oxf_external_stn_retrieval_validation/oxf_query_diagnostics.csv")
OXF_RAW_RECON_DIAG = Path("outputs/oxf_raw_montage_reconstruction_audit/raw_reconstruction_query_diagnostics.csv")
OXF_MAGE_DIAG = Path("outputs/oxf_montage_aware_gated_evidence_method/mage_query_diagnostics.csv")
OXF_MAGE_G_DIAG = Path("outputs/oxf_mage_g_geometry_aware_evidence/mage_g_query_diagnostics.csv")
DS_STN_DIAG = Path("outputs/ds004998_stn_only_cross_dataset_branch/ds004998_stn_only_query_diagnostics.csv")
REVERSE_DS_DIAG = Path("outputs/oxf_frozen_reverse_transfer_to_ds004998/ds004998_oxf_frozen_query_diagnostics.csv")
REVERSE_OXF_TRAIN_DIAG = Path("outputs/oxf_frozen_reverse_transfer_to_ds004998/oxf_train_fit_query_diagnostics.csv")
GEOMETRY_PERM_SAMPLES = Path("outputs/oxf_mage_g_validity_audit/mage_g_validity_geometry_permutation_null_samples.csv")
GEOMETRY_PERM_SUMMARY = Path("outputs/oxf_mage_g_validity_audit/mage_g_validity_geometry_permutation_null_summary.csv")
MAGE_G_PAIRED_DELTA_BOOTSTRAP_CI = Path("outputs/oxf_mage_g_validity_audit/mage_g_validity_paired_delta_bootstrap_ci.csv")
MAGE_G_PAIRED_DELTA_SUMMARY = Path("outputs/oxf_mage_g_validity_audit/mage_g_validity_paired_delta_summary.csv")

RAW_NO_NUMERIC_BRANCH = "all_pairs_exact_raw_reconstruct_else_median_compact_residual_7"
RAW_NUMERIC_BRANCH = "all_pairs_exact_raw_reconstruct_numeric_else_median_compact_residual_7"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def to_builtin(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(data: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(data), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(data: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return path


def write_text(lines: Iterable[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y"})


def filter_diagnostics(path: Path, **filters: str) -> pd.DataFrame:
    data = read_csv(path)
    out = data.copy()
    for column, value in filters.items():
        if value is None:
            continue
        if column not in out:
            raise KeyError(f"{path} has no column {column}")
        out = out[out[column].astype(str).eq(str(value))].copy()
    if out.empty:
        raise AssertionError(f"No diagnostic rows in {path} for filters={filters}")
    return out.reset_index(drop=True)


def query_metrics(diagnostics: pd.DataFrame) -> dict[str, float]:
    rank = pd.to_numeric(diagnostics["true_medon_rank"], errors="coerce")
    success = bool_series(diagnostics["top_ranked_is_true_pair"])
    return {
        "n_pairs": float(len(diagnostics)),
        "top1": float(success.mean()),
        "mrr": float(pd.to_numeric(diagnostics["reciprocal_rank"], errors="coerce").mean()),
        "top3": float((rank <= 3).mean()),
        "top5": float((rank <= 5).mean()),
        "percentile_rank": float(pd.to_numeric(diagnostics.get("percentile_rank", np.nan), errors="coerce").mean()),
        "failures": float((~success).sum()),
    }


def vector_cosine(left: pd.Series, right: pd.Series) -> float:
    common = [feature for feature in left.index if feature in right.index]
    if not common:
        return np.nan
    a = left[common].to_numpy(dtype=float)
    b = right[common].to_numpy(dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else np.nan


def vector_pearson(left: pd.Series, right: pd.Series) -> float:
    common = [feature for feature in left.index if feature in right.index]
    if len(common) < 2:
        return np.nan
    a = left[common].to_numpy(dtype=float)
    b = right[common].to_numpy(dtype=float)
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def sign_agreement(left: pd.Series, right: pd.Series) -> float:
    common = [feature for feature in left.index if feature in right.index]
    if not common:
        return np.nan
    a = np.sign(left[common].to_numpy(dtype=float))
    b = np.sign(right[common].to_numpy(dtype=float))
    return float(np.mean(a == b))


def run_loso_coefficient_stability() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one OXF MAGE-G logistic ranker per held-out subject."""

    scores = reverse.add_rank_fraction(reverse.filter_scores(reverse.OXF_MAGE_G_SCORES, MAGE_G_VARIANT))
    full_ranker = reverse.fit_ranker(
        "MAGE_G_source_span_soft_full",
        "OXF MAGE-G source/span soft v2D all-MedOn",
        scores,
        reverse.MAGE_G_FEATURES,
    )
    full_coef = pd.Series(full_ranker.coefficients, dtype=float)
    geometry_features = [feature for feature in reverse.MAGE_G_GEOMETRY_FEATURES if feature in full_coef.index]

    fold_rows = []
    coefficient_rows = []
    subjects = sorted(scores["query_subject"].astype(str).unique())
    for heldout in subjects:
        query_subject = scores["query_subject"].astype(str)
        candidate_subject = scores["candidate_subject"].astype(str) if "candidate_subject" in scores else pd.Series("", index=scores.index)
        train = scores[query_subject.ne(heldout) & candidate_subject.ne(heldout)].copy()
        test = scores[query_subject.eq(heldout)].copy()
        ranker = reverse.fit_ranker(
            f"MAGE_G_source_span_soft_loso_{heldout}",
            "OXF MAGE-G source/span soft v2D all-MedOn LOSO",
            train,
            reverse.MAGE_G_FEATURES,
        )
        scored = reverse.apply_ranker(test, ranker, f"LOSO_MAGE_G_source_span_soft__heldout_{heldout}", "oxf_subject_loso")
        diag = reverse.query_diagnostics(scored)
        metrics = query_metrics(diag)
        coef = pd.Series(ranker.coefficients, dtype=float)
        fold_rows.append(
            {
                "heldout_subject": heldout,
                "train_subjects": int(train["query_subject"].astype(str).nunique()),
                "test_subjects": int(test["query_subject"].astype(str).nunique()),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_queries": int(train["query_pair_id"].nunique()),
                "test_queries": int(test["query_pair_id"].nunique()),
                "test_top1": metrics["top1"],
                "test_mrr": metrics["mrr"],
                "test_top5": metrics["top5"],
                "test_failures": metrics["failures"],
                "coef_cosine_vs_full_all": vector_cosine(coef, full_coef),
                "coef_pearson_vs_full_all": vector_pearson(coef, full_coef),
                "coef_sign_agreement_vs_full_all": sign_agreement(coef, full_coef),
                "coef_cosine_vs_full_geometry": vector_cosine(coef[geometry_features], full_coef[geometry_features]),
                "coef_pearson_vs_full_geometry": vector_pearson(coef[geometry_features], full_coef[geometry_features]),
                "coef_sign_agreement_vs_full_geometry": sign_agreement(coef[geometry_features], full_coef[geometry_features]),
                "intercept": float(ranker.intercept),
                "full_intercept": float(full_ranker.intercept),
                "intercept_delta_vs_full": float(ranker.intercept - full_ranker.intercept),
            }
        )
        for feature in reverse.MAGE_G_FEATURES:
            coefficient_rows.append(
                {
                    "heldout_subject": heldout,
                    "feature": feature,
                    "feature_group": "geometry" if feature in geometry_features else "base_ranker",
                    "coefficient": float(coef[feature]),
                    "full_coefficient": float(full_coef[feature]),
                    "delta_vs_full": float(coef[feature] - full_coef[feature]),
                    "abs_delta_vs_full": float(abs(coef[feature] - full_coef[feature])),
                    "same_sign_as_full": bool(np.sign(coef[feature]) == np.sign(full_coef[feature])),
                    "train_mean": float(ranker.mean[feature]),
                    "train_scale": float(ranker.scale[feature]),
                    "train_fill_value": float(ranker.train_fill_values[feature]),
                }
            )

    coefficients = pd.DataFrame(coefficient_rows)
    stability_rows = []
    for feature, group in coefficients.groupby("feature", sort=False):
        vals = pd.to_numeric(group["coefficient"], errors="coerce")
        full = float(group["full_coefficient"].iloc[0])
        std = float(vals.std(ddof=1))
        mean = float(vals.mean())
        stability_rows.append(
            {
                "feature": feature,
                "feature_group": str(group["feature_group"].iloc[0]),
                "full_coefficient": full,
                "loso_mean_coefficient": mean,
                "loso_sd_coefficient": std,
                "loso_min_coefficient": float(vals.min()),
                "loso_max_coefficient": float(vals.max()),
                "mean_abs_delta_vs_full": float(pd.to_numeric(group["abs_delta_vs_full"], errors="coerce").mean()),
                "sign_agreement_vs_full": float(group["same_sign_as_full"].astype(bool).mean()),
                "coefficient_cv_abs": float(std / abs(mean)) if abs(mean) > 1e-12 else np.nan,
                "n_folds": int(len(group)),
            }
        )
    return pd.DataFrame(fold_rows), coefficients, pd.DataFrame(stability_rows)


def bootstrap_one(
    label: str,
    experiment: str,
    dataset: str,
    diagnostics: pd.DataFrame,
    source_file: Path,
    n_boot: int = N_BOOT,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(RANDOM_SEED)
    observed = query_metrics(diagnostics)
    subjects = sorted(diagnostics["query_subject"].astype(str).unique())
    sample_rows = []
    for index in range(n_boot):
        draw = rng.choice(subjects, size=len(subjects), replace=True)
        sample = pd.concat(
            [diagnostics[diagnostics["query_subject"].astype(str).eq(subject)] for subject in draw],
            ignore_index=True,
        )
        metrics = query_metrics(sample)
        sample_rows.append(
            {
                "bootstrap_index": int(index),
                "label": label,
                "experiment": experiment,
                "dataset": dataset,
                "top1": metrics["top1"],
                "mrr": metrics["mrr"],
                "top3": metrics["top3"],
                "top5": metrics["top5"],
                "n_query_rows": int(len(sample)),
                "seed": RANDOM_SEED,
                "bootstrap_unit": "subject",
            }
        )
    sample_df = pd.DataFrame(sample_rows)
    summary_rows = []
    for metric in ["top1", "mrr", "top3", "top5"]:
        values = pd.to_numeric(sample_df[metric], errors="coerce").dropna().to_numpy(dtype=float)
        summary_rows.append(
            {
                "label": label,
                "experiment": experiment,
                "dataset": dataset,
                "metric": metric,
                "observed": float(observed[metric]),
                "ci_lower_95": float(np.percentile(values, 2.5)),
                "ci_upper_95": float(np.percentile(values, 97.5)),
                "n_subjects": int(len(subjects)),
                "n_queries": int(len(diagnostics)),
                "n_bootstrap": int(n_boot),
                "seed": RANDOM_SEED,
                "bootstrap_unit": "subject",
                "source_file": str(source_file),
            }
        )
    return summary_rows, sample_rows


def run_subject_clustered_bootstrap() -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = [
        {
            "label": "OXF_locked_STN_only_v2D",
            "experiment": "oxf_ladder",
            "dataset": "OXF",
            "path": OXF_EXTERNAL_DIAG,
            "filters": {"candidate_pool_condition": ALL_MEDON_CONDITION, "variant_name": V2D_VARIANT},
        },
        {
            "label": "OXF_exact_raw_else_median_v2D",
            "experiment": "oxf_ladder",
            "dataset": "OXF",
            "path": OXF_RAW_RECON_DIAG,
            "filters": {
                "branch_id": RAW_NO_NUMERIC_BRANCH,
                "candidate_pool_condition": ALL_MEDON_CONDITION,
                "variant_name": V2D_VARIANT,
            },
        },
        {
            "label": "OXF_current_MAGE_v2D",
            "experiment": "oxf_ladder",
            "dataset": "OXF",
            "path": OXF_MAGE_DIAG,
            "filters": {"candidate_pool_condition": ALL_MEDON_CONDITION, "variant_name": f"MAGE_{V2D_VARIANT}"},
        },
        {
            "label": "OXF_MAGE_G_source_span_soft_v2D",
            "experiment": "oxf_ladder",
            "dataset": "OXF",
            "path": OXF_MAGE_G_DIAG,
            "filters": {"candidate_pool_condition": ALL_MEDON_CONDITION, "variant_name": MAGE_G_VARIANT},
        },
        {
            "label": "reverse_reference_ds_native_STN_v2D",
            "experiment": "reverse_transfer",
            "dataset": "ds004998",
            "path": DS_STN_DIAG,
            "filters": {"candidate_pool_condition": ALL_MEDON_CONDITION, "variant_name": V2D_VARIANT},
        },
        {
            "label": "reverse_OXF_train_fit_no_MAGE_G",
            "experiment": "reverse_transfer",
            "dataset": "OXF",
            "path": REVERSE_OXF_TRAIN_DIAG,
            "filters": {"candidate_pool_condition": ALL_MEDON_CONDITION, "variant_name": "OXF_train_fit_no_MAGE_G"},
        },
        {
            "label": "reverse_OXF_train_fit_MAGE_G",
            "experiment": "reverse_transfer",
            "dataset": "OXF",
            "path": REVERSE_OXF_TRAIN_DIAG,
            "filters": {"candidate_pool_condition": ALL_MEDON_CONDITION, "variant_name": "OXF_train_fit_MAGE_G_source_span_soft"},
        },
        {
            "label": "reverse_ds_frozen_no_MAGE_G",
            "experiment": "reverse_transfer",
            "dataset": "ds004998",
            "path": REVERSE_DS_DIAG,
            "filters": {
                "candidate_pool_condition": ALL_MEDON_CONDITION,
                "variant_name": "OXF_frozen_no_MAGE_G__v2D_quality_deconfounded_cycle",
            },
        },
        {
            "label": "reverse_ds_frozen_MAGE_G",
            "experiment": "reverse_transfer",
            "dataset": "ds004998",
            "path": REVERSE_DS_DIAG,
            "filters": {
                "candidate_pool_condition": ALL_MEDON_CONDITION,
                "variant_name": "OXF_frozen_MAGE_G_source_span_soft__v2D_quality_deconfounded_cycle",
            },
        },
    ]
    summary_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    for spec in specs:
        diagnostics = filter_diagnostics(spec["path"], **spec["filters"])
        summary, samples = bootstrap_one(
            spec["label"],
            spec["experiment"],
            spec["dataset"],
            diagnostics,
            spec["path"],
        )
        summary_rows.extend(summary)
        sample_rows.extend(samples)
    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def paired_delta_bootstrap(left: pd.DataFrame, right: pd.DataFrame, n_boot: int = N_BOOT) -> dict[str, object]:
    paired = left.merge(
        right,
        on="query_pair_id",
        suffixes=("_left", "_right"),
        how="inner",
    )
    paired["query_subject"] = paired["query_subject_left"].astype(str)
    subjects = sorted(paired["query_subject"].unique())
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for index in range(n_boot):
        draw = rng.choice(subjects, size=len(subjects), replace=True)
        sample = pd.concat([paired[paired["query_subject"].eq(subject)] for subject in draw], ignore_index=True)
        left_metrics = query_metrics(
            sample.rename(
                columns={
                    "true_medon_rank_left": "true_medon_rank",
                    "top_ranked_is_true_pair_left": "top_ranked_is_true_pair",
                    "reciprocal_rank_left": "reciprocal_rank",
                    "percentile_rank_left": "percentile_rank",
                }
            )
        )
        right_metrics = query_metrics(
            sample.rename(
                columns={
                    "true_medon_rank_right": "true_medon_rank",
                    "top_ranked_is_true_pair_right": "top_ranked_is_true_pair",
                    "reciprocal_rank_right": "reciprocal_rank",
                    "percentile_rank_right": "percentile_rank",
                }
            )
        )
        rows.append(
            {
                "bootstrap_index": int(index),
                "delta_top1_right_minus_left": right_metrics["top1"] - left_metrics["top1"],
                "delta_mrr_right_minus_left": right_metrics["mrr"] - left_metrics["mrr"],
                "delta_top5_right_minus_left": right_metrics["top5"] - left_metrics["top5"],
            }
        )
    samples = pd.DataFrame(rows)
    return {
        "delta_top1_ci_lower_95": float(np.percentile(samples["delta_top1_right_minus_left"], 2.5)),
        "delta_top1_ci_upper_95": float(np.percentile(samples["delta_top1_right_minus_left"], 97.5)),
        "delta_mrr_ci_lower_95": float(np.percentile(samples["delta_mrr_right_minus_left"], 2.5)),
        "delta_mrr_ci_upper_95": float(np.percentile(samples["delta_mrr_right_minus_left"], 97.5)),
        "delta_top5_ci_lower_95": float(np.percentile(samples["delta_top5_right_minus_left"], 2.5)),
        "delta_top5_ci_upper_95": float(np.percentile(samples["delta_top5_right_minus_left"], 97.5)),
        "n_subjects": int(len(subjects)),
        "n_paired_queries": int(len(paired)),
    }


def run_numeric_repair_same_fallback_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = [
        "v2A_top5_aperiodic_rerank",
        "v2C_deconfounded_cycle",
        V2D_VARIANT,
    ]
    summary_rows = []
    delta_rows = []
    data = read_csv(OXF_RAW_RECON_DIAG)
    for variant in variants:
        no_numeric = data[
            data["branch_id"].astype(str).eq(RAW_NO_NUMERIC_BRANCH)
            & data["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
            & data["variant_name"].astype(str).eq(variant)
        ].copy()
        numeric = data[
            data["branch_id"].astype(str).eq(RAW_NUMERIC_BRANCH)
            & data["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
            & data["variant_name"].astype(str).eq(variant)
        ].copy()
        if no_numeric.empty or numeric.empty:
            continue
        no_metrics = query_metrics(no_numeric)
        numeric_metrics = query_metrics(numeric)
        ci = paired_delta_bootstrap(no_numeric, numeric)
        paired = no_numeric.merge(
            numeric,
            on="query_pair_id",
            suffixes=("_no_numeric", "_numeric"),
            how="inner",
        )
        paired["no_numeric_top1"] = bool_series(paired["top_ranked_is_true_pair_no_numeric"])
        paired["numeric_top1"] = bool_series(paired["top_ranked_is_true_pair_numeric"])
        paired["rank_no_numeric"] = pd.to_numeric(paired["true_medon_rank_no_numeric"], errors="coerce")
        paired["rank_numeric"] = pd.to_numeric(paired["true_medon_rank_numeric"], errors="coerce")
        paired["numeric_rank_delta"] = paired["rank_numeric"] - paired["rank_no_numeric"]
        paired["numeric_rank_improved"] = paired["numeric_rank_delta"] < 0
        paired["numeric_rank_worsened"] = paired["numeric_rank_delta"] > 0
        paired["numeric_top1_gain"] = paired["numeric_top1"] & ~paired["no_numeric_top1"]
        paired["numeric_top1_loss"] = paired["no_numeric_top1"] & ~paired["numeric_top1"]
        summary_rows.append(
            {
                "comparison": "numeric_repair_vs_no_numeric_same_median_fallback",
                "candidate_pool_condition": ALL_MEDON_CONDITION,
                "variant_name": variant,
                "no_numeric_branch": RAW_NO_NUMERIC_BRANCH,
                "numeric_branch": RAW_NUMERIC_BRANCH,
                "n_pairs": int(no_metrics["n_pairs"]),
                "no_numeric_top1": no_metrics["top1"],
                "numeric_top1": numeric_metrics["top1"],
                "delta_top1_numeric_minus_no_numeric": numeric_metrics["top1"] - no_metrics["top1"],
                "delta_top1_ci_lower_95": ci["delta_top1_ci_lower_95"],
                "delta_top1_ci_upper_95": ci["delta_top1_ci_upper_95"],
                "no_numeric_mrr": no_metrics["mrr"],
                "numeric_mrr": numeric_metrics["mrr"],
                "delta_mrr_numeric_minus_no_numeric": numeric_metrics["mrr"] - no_metrics["mrr"],
                "delta_mrr_ci_lower_95": ci["delta_mrr_ci_lower_95"],
                "delta_mrr_ci_upper_95": ci["delta_mrr_ci_upper_95"],
                "no_numeric_top5": no_metrics["top5"],
                "numeric_top5": numeric_metrics["top5"],
                "delta_top5_numeric_minus_no_numeric": numeric_metrics["top5"] - no_metrics["top5"],
                "numeric_rank_improved": int(paired["numeric_rank_improved"].sum()),
                "numeric_rank_worsened": int(paired["numeric_rank_worsened"].sum()),
                "numeric_top1_gain": int(paired["numeric_top1_gain"].sum()),
                "numeric_top1_loss": int(paired["numeric_top1_loss"].sum()),
                "n_subjects": ci["n_subjects"],
            }
        )
        for _, row in paired.iterrows():
            delta_rows.append(
                {
                    "variant_name": variant,
                    "query_pair_id": row["query_pair_id"],
                    "query_subject": row["query_subject_no_numeric"],
                    "rank_no_numeric": row["rank_no_numeric"],
                    "rank_numeric": row["rank_numeric"],
                    "numeric_rank_delta": row["numeric_rank_delta"],
                    "no_numeric_top1": bool(row["no_numeric_top1"]),
                    "numeric_top1": bool(row["numeric_top1"]),
                    "numeric_top1_gain": bool(row["numeric_top1_gain"]),
                    "numeric_top1_loss": bool(row["numeric_top1_loss"]),
                    "top_candidate_no_numeric": row.get("top_ranked_candidate_pair_id_no_numeric", ""),
                    "top_candidate_numeric": row.get("top_ranked_candidate_pair_id_numeric", ""),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(delta_rows)


def run_median_to_mage_g_paired_delta_ci() -> pd.DataFrame:
    """Extract paired subject-bootstrap CI for OXF median-v2D to MAGE-G."""

    ci = read_csv(MAGE_G_PAIRED_DELTA_BOOTSTRAP_CI)
    summary = read_csv(MAGE_G_PAIRED_DELTA_SUMMARY)
    ci = ci[ci["comparison"].astype(str).eq("target_vs_median_v2d")].copy()
    summary = summary[summary["comparison"].astype(str).eq("target_vs_median_v2d")].copy()
    if ci.empty:
        raise AssertionError("Missing target_vs_median_v2d paired bootstrap CI")
    summary_map = summary.set_index("comparison").to_dict("index") if not summary.empty else {}
    rows = []
    for _, row in ci.iterrows():
        comparison_summary = summary_map.get(str(row["comparison"]), {})
        rows.append(
            {
                "contrast": "MAGE_G_source_span_soft_v2D_minus_OXF_median_v2D",
                "comparison": row["comparison"],
                "metric": row["metric"],
                "observed_delta": float(row["observed"]),
                "bootstrap_mean_delta": float(row["bootstrap_mean"]),
                "ci_lower_95": float(row["ci_lower_95"]),
                "ci_upper_95": float(row["ci_upper_95"]),
                "ci_excludes_zero_positive": str(row["ci_excludes_zero_positive"]).strip().lower() == "true",
                "n_bootstrap": int(row["n_bootstrap"]),
                "bootstrap_unit": row["bootstrap_unit"],
                "seed": int(row["seed"]),
                "n_queries": int(comparison_summary.get("n_queries", 30)),
                "n_rank_improved": int(comparison_summary.get("n_rank_improved", 0)),
                "n_rank_worsened": int(comparison_summary.get("n_rank_worsened", 0)),
                "n_top1_gain": int(comparison_summary.get("n_top1_gain", 0)),
                "n_top1_loss": int(comparison_summary.get("n_top1_loss", 0)),
                "source_file": str(MAGE_G_PAIRED_DELTA_BOOTSTRAP_CI),
            }
        )
    return pd.DataFrame(rows)


def histogram_bins(values: np.ndarray, observed: float, n_bins: int = 18) -> tuple[np.ndarray, np.ndarray]:
    low = min(float(np.nanmin(values)), observed)
    high = max(float(np.nanmax(values)), observed)
    if math.isclose(low, high):
        low -= 0.5
        high += 0.5
    padding = (high - low) * 0.04
    bins = np.linspace(low - padding, high + padding, n_bins + 1)
    counts, edges = np.histogram(values, bins=bins)
    return counts, edges


def make_geometry_permutation_svg(samples: pd.DataFrame, summary: pd.DataFrame, path: Path) -> Path:
    width = 1060
    height = 420
    panel_width = 430
    panel_height = 250
    margin_left = 70
    margin_top = 80
    gap = 70
    colors = {
        "bar": "#6f9ed6",
        "axis": "#333333",
        "observed": "#c43c39",
        "text": "#222222",
        "grid": "#dddddd",
    }
    metric_titles = [("top1", "Top-1"), ("mrr", "MRR")]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="32" text-anchor="middle" font-family="Arial" font-size="20" fill="{colors["text"]}">MAGE-G geometry permutation null distribution</text>',
        f'<text x="{width / 2:.1f}" y="55" text-anchor="middle" font-family="Arial" font-size="12" fill="{colors["text"]}">Query geometry metadata permuted 5000 times; red line is observed MAGE-G source-span-soft</text>',
    ]
    for panel_idx, (metric, title) in enumerate(metric_titles):
        x0 = margin_left + panel_idx * (panel_width + gap)
        y0 = margin_top
        values = pd.to_numeric(samples[metric], errors="coerce").dropna().to_numpy(dtype=float)
        obs_row = summary[summary["metric"].astype(str).eq(metric)]
        observed = float(obs_row["observed"].iloc[0])
        p_value = float(obs_row["empirical_p_greater_equal"].iloc[0])
        counts, edges = histogram_bins(values, observed)
        max_count = max(int(counts.max()), 1)
        svg.append(f'<text x="{x0 + panel_width / 2:.1f}" y="{y0 - 24}" text-anchor="middle" font-family="Arial" font-size="16" fill="{colors["text"]}">{title}</text>')
        svg.append(f'<text x="{x0 + panel_width / 2:.1f}" y="{y0 - 7}" text-anchor="middle" font-family="Arial" font-size="12" fill="{colors["text"]}">observed={observed:.3f}, empirical p={p_value:.4f}</text>')
        for tick in range(5):
            yy = y0 + panel_height - tick * panel_height / 4
            svg.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + panel_width}" y2="{yy:.1f}" stroke="{colors["grid"]}" stroke-width="1"/>')
        for idx, count in enumerate(counts):
            left = x0 + (edges[idx] - edges[0]) / (edges[-1] - edges[0]) * panel_width
            right = x0 + (edges[idx + 1] - edges[0]) / (edges[-1] - edges[0]) * panel_width
            bar_height = (count / max_count) * panel_height
            y = y0 + panel_height - bar_height
            svg.append(
                f'<rect x="{left + 1:.1f}" y="{y:.1f}" width="{max(right - left - 2, 1):.1f}" height="{bar_height:.1f}" fill="{colors["bar"]}" opacity="0.85"/>'
            )
        obs_x = x0 + (observed - edges[0]) / (edges[-1] - edges[0]) * panel_width
        svg.append(f'<line x1="{obs_x:.1f}" y1="{y0}" x2="{obs_x:.1f}" y2="{y0 + panel_height}" stroke="{colors["observed"]}" stroke-width="3"/>')
        svg.append(f'<line x1="{x0}" y1="{y0 + panel_height}" x2="{x0 + panel_width}" y2="{y0 + panel_height}" stroke="{colors["axis"]}" stroke-width="1.5"/>')
        svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + panel_height}" stroke="{colors["axis"]}" stroke-width="1.5"/>')
        for tick in np.linspace(edges[0], edges[-1], 5):
            xx = x0 + (tick - edges[0]) / (edges[-1] - edges[0]) * panel_width
            svg.append(f'<line x1="{xx:.1f}" y1="{y0 + panel_height}" x2="{xx:.1f}" y2="{y0 + panel_height + 5}" stroke="{colors["axis"]}" stroke-width="1"/>')
            svg.append(f'<text x="{xx:.1f}" y="{y0 + panel_height + 22}" text-anchor="middle" font-family="Arial" font-size="11" fill="{colors["text"]}">{tick:.2f}</text>')
        svg.append(f'<text x="{x0 + panel_width / 2:.1f}" y="{y0 + panel_height + 48}" text-anchor="middle" font-family="Arial" font-size="12" fill="{colors["text"]}">permuted {title}</text>')
        svg.append(f'<text x="{x0 - 38}" y="{y0 + panel_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 {x0 - 38} {y0 + panel_height / 2:.1f})" font-family="Arial" font-size="12" fill="{colors["text"]}">count</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return path


def write_summary(
    loso_folds: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    numeric_comparison: pd.DataFrame,
    median_to_mage_g_delta: pd.DataFrame,
    geometry_summary: pd.DataFrame,
    path: Path,
) -> Path:
    top1_rows = bootstrap_ci[bootstrap_ci["metric"].astype(str).eq("top1")].copy()
    row_map = top1_rows.set_index("label")
    loso_cos = pd.to_numeric(loso_folds["coef_cosine_vs_full_all"], errors="coerce")
    loso_geom_cos = pd.to_numeric(loso_folds["coef_cosine_vs_full_geometry"], errors="coerce")
    mage_loso_top1 = pd.to_numeric(loso_folds["test_top1"], errors="coerce")
    geom_top1 = geometry_summary[geometry_summary["metric"].astype(str).eq("top1")].iloc[0]
    v2d_numeric = numeric_comparison[numeric_comparison["variant_name"].astype(str).eq(V2D_VARIANT)].iloc[0]
    paired_top1 = median_to_mage_g_delta[median_to_mage_g_delta["metric"].astype(str).eq("delta_top1")].iloc[0]
    paired_mrr = median_to_mage_g_delta[median_to_mage_g_delta["metric"].astype(str).eq("delta_mrr")].iloc[0]

    def ci_line(label: str) -> str:
        row = row_map.loc[label]
        return f"{label}: {row['observed']:.3f} [{row['ci_lower_95']:.3f}, {row['ci_upper_95']:.3f}]"

    lines = [
        "# Journal MAGE-G Robustness Audit",
        "",
        "Downstream audit package for reviewer-facing robustness checks.",
        "",
        "## LOSO Coefficient Stability",
        f"- folds: {len(loso_folds)} subject-held-out folds",
        f"- all-feature cosine vs full model: median={loso_cos.median():.3f}, min={loso_cos.min():.3f}, max={loso_cos.max():.3f}",
        f"- geometry-feature cosine vs full model: median={loso_geom_cos.median():.3f}, min={loso_geom_cos.min():.3f}, max={loso_geom_cos.max():.3f}",
        f"- LOSO held-out top1: mean={mage_loso_top1.mean():.3f}, min={mage_loso_top1.min():.3f}, max={mage_loso_top1.max():.3f}",
        "",
        "## Subject-Clustered CI Highlights",
        f"- {ci_line('OXF_locked_STN_only_v2D')}",
        f"- {ci_line('OXF_exact_raw_else_median_v2D')}",
        f"- {ci_line('OXF_current_MAGE_v2D')}",
        f"- {ci_line('OXF_MAGE_G_source_span_soft_v2D')}",
        f"- {ci_line('reverse_ds_frozen_no_MAGE_G')}",
        f"- {ci_line('reverse_ds_frozen_MAGE_G')}",
        "",
        "## Paired Median-to-MAGE-G Delta",
        f"- top1 delta: {float(paired_top1['observed_delta']):.3f} "
        f"[{float(paired_top1['ci_lower_95']):.3f}, {float(paired_top1['ci_upper_95']):.3f}]",
        f"- MRR delta: {float(paired_mrr['observed_delta']):.3f} "
        f"[{float(paired_mrr['ci_lower_95']):.3f}, {float(paired_mrr['ci_upper_95']):.3f}]",
        "",
        "## Geometry Permutation",
        f"- observed top1: {float(geom_top1['observed']):.3f}",
        f"- null mean top1: {float(geom_top1['null_mean']):.3f}",
        f"- empirical p >= observed: {float(geom_top1['empirical_p_greater_equal']):.4f}",
        "",
        "## Numeric Repair Same-Fallback Check",
        f"- v2D no numeric top1: {float(v2d_numeric['no_numeric_top1']):.3f}",
        f"- v2D numeric top1: {float(v2d_numeric['numeric_top1']):.3f}",
        f"- v2D delta numeric - no numeric: {float(v2d_numeric['delta_top1_numeric_minus_no_numeric']):.3f} "
        f"[{float(v2d_numeric['delta_top1_ci_lower_95']):.3f}, {float(v2d_numeric['delta_top1_ci_upper_95']):.3f}]",
    ]
    return write_text(lines, path)


def run() -> dict[str, object]:
    ensure_dirs()

    loso_folds, loso_coefficients, loso_stability = run_loso_coefficient_stability()
    bootstrap_ci, bootstrap_samples = run_subject_clustered_bootstrap()
    numeric_comparison, numeric_deltas = run_numeric_repair_same_fallback_comparison()
    median_to_mage_g_delta = run_median_to_mage_g_paired_delta_ci()

    geometry_samples = read_csv(GEOMETRY_PERM_SAMPLES)
    geometry_summary = read_csv(GEOMETRY_PERM_SUMMARY)
    geometry_svg = make_geometry_permutation_svg(
        geometry_samples,
        geometry_summary,
        FIGURES_DIR / "geometry_permutation_distribution.svg",
    )
    manuscript_geometry_svg = make_geometry_permutation_svg(
        geometry_samples,
        geometry_summary,
        MANUSCRIPT_FIGURES_DIR / "geometry_permutation_distribution.svg",
    )

    paths = {
        "loso_fold_metrics": write_csv(loso_folds, OUTPUT_DIR / "loso_mage_g_fold_metrics.csv"),
        "loso_coefficients": write_csv(loso_coefficients, OUTPUT_DIR / "loso_mage_g_coefficients.csv"),
        "loso_coefficient_stability": write_csv(
            loso_stability,
            OUTPUT_DIR / "loso_mage_g_coefficient_stability_summary.csv",
        ),
        "subject_clustered_bootstrap_ci": write_csv(
            bootstrap_ci,
            OUTPUT_DIR / "subject_clustered_bootstrap_ci_summary.csv",
        ),
        "subject_clustered_bootstrap_samples": write_csv(
            bootstrap_samples,
            OUTPUT_DIR / "subject_clustered_bootstrap_samples.csv",
        ),
        "numeric_repair_comparison": write_csv(
            numeric_comparison,
            OUTPUT_DIR / "numeric_repair_same_median_fallback_comparison.csv",
        ),
        "numeric_repair_query_deltas": write_csv(
            numeric_deltas,
            OUTPUT_DIR / "numeric_repair_same_median_fallback_query_deltas.csv",
        ),
        "median_to_mage_g_paired_delta_ci": write_csv(
            median_to_mage_g_delta,
            OUTPUT_DIR / "median_to_mage_g_paired_delta_ci.csv",
        ),
        "manuscript_median_to_mage_g_paired_delta_ci": write_csv(
            median_to_mage_g_delta,
            MANUSCRIPT_TABLES_DIR / "journal_median_to_mage_g_paired_delta_ci.csv",
        ),
        "geometry_permutation_svg": geometry_svg,
        "manuscript_geometry_permutation_svg": manuscript_geometry_svg,
        "summary_md": write_summary(
            loso_folds,
            bootstrap_ci,
            numeric_comparison,
            median_to_mage_g_delta,
            geometry_summary,
            OUTPUT_DIR / "journal_mage_g_robustness_audit_summary.md",
        ),
    }
    summary = {
        "outputs": {key: str(path) for key, path in paths.items()},
        "n_bootstrap": N_BOOT,
        "seed": RANDOM_SEED,
        "loso_subject_folds": int(len(loso_folds)),
    }
    paths["summary_json"] = write_json(summary, OUTPUT_DIR / "journal_mage_g_robustness_audit_summary.json")
    print(f"output folder: {OUTPUT_DIR}")
    for key, path in paths.items():
        print(f"{key}: {path}")
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
