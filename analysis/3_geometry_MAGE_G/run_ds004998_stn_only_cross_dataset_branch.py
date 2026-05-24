"""Run a ds004998 STN-only common-layer retrieval branch.

This script tests whether the parts shared with OXF -- STN band power and STN
compact aperiodic/residual features -- are sufficient to support the retrieval
effect inside ds004998. It reads cached 18-subject ds004998 outputs only and
does not modify frozen v2A, features, scorer, top_k, alpha, or raw extraction.

This is a diagnostic cross-dataset branch, not clinical prediction, treatment
recommendation, DBS optimization, or causal medication-effect estimation.
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
)
from scripts.run_retrieval_distance_geometry_improvement import to_builtin  # noqa: E402
from scripts.run_v2a_scientific_audit_17subjects_33pairs import (  # noqa: E402
    APERIODIC_ALPHA,
    N_BOOT,
    N_PERM,
    RANDOM_SEED,
    TOP_K,
    V2A_NAME,
    V2_NAME,
    percentile_ci,
    query_metrics,
    read_csv_required,
    read_json_required,
)
from scripts.run_v2b_gated_scientific_audit_17subjects_33pairs import (  # noqa: E402
    ALL_MEDON_CONDITION,
    HARD_NEGATIVE_CONDITION,
    PRIMARY_CONDITION,
    V2B_NAME,
)
from scripts.run_v2b_hard_negative_metric_study_17subjects_33pairs import (  # noqa: E402
    V2B_VARIANTS,
    evaluate_v2b_variant,
)
from scripts.run_v2c_deconfounded_transition_retrieval_experiment_17subjects_33pairs import (  # noqa: E402
    TRUE_PLUS_OTHER_CONDITION,
    V2C_CYCLE_NAME,
    V2C_DECONFOUNDED_NAME,
    cycle_rerank_scores,
)
from scripts.run_v2d_kreciprocal_quality_rerank_experiment_17subjects_33pairs import (  # noqa: E402
    V2D_QUALITY_DECONFOUNDED_NAME,
    V2D_VARIANTS,
    build_metrics_tables,
    comparison_to_baseline,
    failure_cases,
    normalize_scores,
    rank_scores,
    rerank_v2d_variant,
    subject_bootstrap_ci,
)


OUTPUT_DIR = Path("outputs/ds004998_stn_only_cross_dataset_branch")
FROZEN_DIR = Path("outputs/v2A_frozen_18subjects_34pairs")
COHORT_DIR = Path("outputs/ds004998_full_cohort_completeness_audit_18subjects_34pairs")

EXPECTED_PAIRS = 34
EXPECTED_SUBJECTS = 18
EVALUATED_CONDITIONS = [
    PRIMARY_CONDITION,
    HARD_NEGATIVE_CONDITION,
    ALL_MEDON_CONDITION,
    TRUE_PLUS_OTHER_CONDITION,
]

STN_POWER_FEATURES = [
    "stn_alpha_power",
    "stn_low_beta_power",
    "stn_high_beta_power",
    "stn_broad_beta_power",
    "stn_gamma_power",
]
STN_COMPACT_FEATURES = [
    "stn_aperiodic_slope",
    "stn_aperiodic_offset",
    "stn_low_beta_residual_power",
    "stn_high_beta_residual_power",
    "stn_broad_beta_residual_power",
    "stn_beta_peak_amplitude",
    "stn_beta_peak_frequency",
]
STN_ONLY_FEATURES = [*STN_POWER_FEATURES, *STN_COMPACT_FEATURES]
COMMON_FEATURE_MAPPING = [
    ("stn_alpha_power", "stn_alpha_power", "stn_alpha_log_power", "STN_power"),
    ("stn_low_beta_power", "stn_low_beta_power", "stn_low_beta_log_power", "STN_power"),
    ("stn_high_beta_power", "stn_high_beta_power", "stn_high_beta_log_power", "STN_power"),
    ("stn_broad_beta_power", "stn_broad_beta_power", "stn_broad_beta_log_power", "STN_power"),
    ("stn_gamma_power", "stn_gamma_power", "stn_gamma_log_power", "STN_power"),
    ("aperiodic_slope", "stn_aperiodic_slope", "aperiodic_slope", "compact_aperiodic_residual"),
    ("aperiodic_offset", "stn_aperiodic_offset", "aperiodic_offset", "compact_aperiodic_residual"),
    ("low_beta_residual_power", "stn_low_beta_residual_power", "low_beta_residual_power", "compact_aperiodic_residual"),
    ("high_beta_residual_power", "stn_high_beta_residual_power", "high_beta_residual_power", "compact_aperiodic_residual"),
    ("broad_beta_residual_power", "stn_broad_beta_residual_power", "broad_beta_residual_power", "compact_aperiodic_residual"),
    ("beta_peak_amplitude", "stn_beta_peak_amplitude", "beta_peak_amplitude", "compact_aperiodic_residual"),
    ("beta_peak_frequency", "stn_beta_peak_frequency", "beta_peak_frequency", "compact_aperiodic_residual"),
]


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
    """Write text output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def selected_v2b_variant():
    """Return selected v2B variant."""

    for variant in V2B_VARIANTS:
        if variant.name == V2B_NAME:
            return variant
    raise AssertionError(f"Selected v2B variant missing: {V2B_NAME}")


def available_features(pairs: pd.DataFrame, candidates: list[str]) -> tuple[list[str], pd.DataFrame]:
    """Return available finite features and inventory."""

    rows = []
    available = []
    for feature in candidates:
        cols = [f"x_{feature}", f"medon_{feature}", f"y_{feature}"]
        present = all(col in pairs.columns for col in cols)
        finite_fraction = 0.0
        if present:
            values = pairs[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            finite_fraction = float(np.isfinite(values).mean()) if values.size else 0.0
        usable = bool(present and math.isclose(finite_fraction, 1.0))
        if usable:
            available.append(feature)
        rows.append(
            {
                "feature": feature,
                "feature_family": "STN_power" if feature in STN_POWER_FEATURES else "compact_aperiodic_residual",
                "present": present,
                "finite_value_fraction": finite_fraction,
                "usable_for_retrieval": usable,
            }
        )
    return available, pd.DataFrame(rows)


def reverse_pairs(pairs: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Swap OFF and ON vectors for ON-to-OFF retrieval."""

    out = pairs.copy()
    for feature in features:
        x_col = f"x_{feature}"
        m_col = f"medon_{feature}"
        if x_col in out and m_col in out:
            temp = out[x_col].copy()
            out[x_col] = out[m_col]
            out[m_col] = temp
            out[f"y_{feature}"] = out[m_col]
    return out


def add_condition(data: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Add candidate-pool condition."""

    out = data.copy()
    out["candidate_pool_condition"] = condition
    return out


def evaluate_forward_scores(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate v2/v2A/v2B on ds004998 STN-only feature space."""

    score_tables = []
    pool_tables = []
    for condition in EVALUATED_CONDITIONS:
        pool = build_custom_candidate_pool(pairs, condition)
        pool_tables.append(pool)
        v2_scores, _v2_diag, v2a_scores, _v2a_diag = evaluate_frozen_on_candidate_pool(
            pairs,
            pool,
            v2_features,
            compact_features,
            compact_features,
        )
        v2b_scores, _v2b_diag, _weights = evaluate_v2b_variant(
            pairs,
            pool,
            v2_features,
            compact_features,
            compact_features,
            selected_v2b_variant(),
        )
        score_tables.extend(
            [
                add_condition(v2_scores, condition),
                add_condition(v2a_scores, condition),
                add_condition(v2b_scores, condition),
            ]
        )
    return normalize_scores(pd.concat(score_tables, ignore_index=True)), pd.concat(pool_tables, ignore_index=True)


def evaluate_reverse_v2b(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
) -> pd.DataFrame:
    """Evaluate selected v2B in reverse ON-to-OFF direction."""

    reversed_pairs = reverse_pairs(pairs, v2_features)
    pool = build_custom_candidate_pool(reversed_pairs, ALL_MEDON_CONDITION)
    scores, _diag, _weights = evaluate_v2b_variant(
        reversed_pairs,
        pool,
        v2_features,
        compact_features,
        compact_features,
        selected_v2b_variant(),
    )
    return add_condition(normalize_scores(scores), ALL_MEDON_CONDITION)


def random_label_null(
    diagnostics: pd.DataFrame,
    scores: pd.DataFrame,
    n_perm: int = N_PERM,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random-label null preserving each query candidate-set size."""

    rng = np.random.default_rng(seed)
    summary_rows = []
    sample_rows = []
    for (condition, variant), diag in diagnostics.groupby(["candidate_pool_condition", "variant_name"], dropna=False):
        subset_scores = scores[
            scores["candidate_pool_condition"].astype(str).eq(str(condition))
            & scores["variant_name"].astype(str).eq(str(variant))
        ]
        sizes = subset_scores.groupby("query_pair_id", dropna=False)["candidate_pair_id"].size().to_numpy(dtype=int)
        if len(sizes) == 0:
            continue
        observed = query_metrics(diag)
        local_samples = []
        for idx in range(n_perm):
            ranks = np.asarray([rng.integers(1, size + 1) for size in sizes], dtype=float)
            local_samples.append(
                {
                    "permutation_index": int(idx),
                    "candidate_pool_condition": condition,
                    "variant_name": variant,
                    "top1": float(np.mean(ranks == 1)),
                    "mrr": float(np.mean(1.0 / ranks)),
                    "percentile_rank": float(np.mean(1.0 - ((ranks - 1.0) / np.maximum(1.0, sizes - 1.0)))),
                    "seed": seed,
                    "null_type": "random_label",
                }
            )
        samples = pd.DataFrame(local_samples)
        sample_rows.extend(local_samples)
        for metric in ["top1", "mrr", "percentile_rank"]:
            null_values = samples[metric].to_numpy(dtype=float)
            obs = float(observed[metric])
            summary_rows.append(
                {
                    "candidate_pool_condition": condition,
                    "variant_name": variant,
                    "metric": metric,
                    "observed": obs,
                    "null_mean": float(np.nanmean(null_values)),
                    "null_sd": float(np.nanstd(null_values, ddof=1)),
                    "empirical_p_greater_equal": float((np.sum(null_values >= obs) + 1.0) / (len(null_values) + 1.0)),
                    "n_perm": int(len(null_values)),
                    "seed": seed,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def subject_bootstrap(diagnostics: pd.DataFrame, n_boot: int = N_BOOT, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subject-aware bootstrap for top1/MRR."""

    rng = np.random.default_rng(seed)
    summary_rows = []
    sample_rows = []
    for (condition, variant), subset in diagnostics.groupby(["candidate_pool_condition", "variant_name"], dropna=False):
        subjects = sorted(subset["query_subject"].astype(str).unique())
        if not subjects:
            continue
        stats = {}
        for subject in subjects:
            rows = subset[subset["query_subject"].astype(str).eq(subject)]
            stats[subject] = {
                "n": int(len(rows)),
                "top1": float(rows["top_ranked_is_true_pair"].astype(float).sum()),
                "mrr": float(pd.to_numeric(rows["reciprocal_rank"], errors="coerce").sum()),
            }
        observed = query_metrics(subset)
        local_samples = []
        for idx in range(n_boot):
            draw = rng.choice(subjects, size=len(subjects), replace=True)
            n = 0
            top1 = 0.0
            mrr = 0.0
            for subject in draw:
                item = stats[str(subject)]
                n += int(item["n"])
                top1 += float(item["top1"])
                mrr += float(item["mrr"])
            local_samples.append(
                {
                    "bootstrap_index": int(idx),
                    "candidate_pool_condition": condition,
                    "variant_name": variant,
                    "top1": float(top1 / n) if n else np.nan,
                    "mrr": float(mrr / n) if n else np.nan,
                    "n_query_rows": int(n),
                    "seed": seed,
                    "bootstrap_unit": "subject",
                }
            )
        samples = pd.DataFrame(local_samples)
        sample_rows.extend(local_samples)
        for metric in ["top1", "mrr"]:
            low, high = percentile_ci(samples[metric])
            summary_rows.append(
                {
                    "candidate_pool_condition": condition,
                    "variant_name": variant,
                    "metric": metric,
                    "observed": float(observed[metric]),
                    "ci_lower_95": low,
                    "ci_upper_95": high,
                    "n_boot": n_boot,
                    "seed": seed,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def loso_sensitivity(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Drop one subject at a time."""

    rows = []
    for (condition, variant), subset in diagnostics.groupby(["candidate_pool_condition", "variant_name"], dropna=False):
        full = query_metrics(subset)
        for subject in sorted(subset["query_subject"].astype(str).unique()):
            reduced = subset[~subset["query_subject"].astype(str).eq(subject)]
            metrics = query_metrics(reduced)
            rows.append(
                {
                    "candidate_pool_condition": condition,
                    "variant_name": variant,
                    "left_out_subject": subject,
                    "n_pairs_remaining": metrics["n_pairs"],
                    "top1": metrics["top1"],
                    "mrr": metrics["mrr"],
                    "delta_top1_vs_full": float(metrics["top1"]) - float(full["top1"]),
                    "delta_mrr_vs_full": float(metrics["mrr"]) - float(full["mrr"]),
                    "collapse_flag": bool(float(metrics["top1"]) < 0.25 or float(metrics["mrr"]) < 0.35),
                }
            )
    return pd.DataFrame(rows)


def hard_negative_cases(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate same-subject wrong-task/side hard negatives."""

    rows = []
    focus = scores[scores["candidate_pool_condition"].astype(str).isin([HARD_NEGATIVE_CONDITION, ALL_MEDON_CONDITION])]
    for (condition, variant, query_id), group in focus.groupby(["candidate_pool_condition", "variant_name", "query_pair_id"], dropna=False):
        same_subject_wrong = group[
            group["candidate_subject"].astype(str).eq(group["query_subject"].astype(str))
            & ~group["is_true_pair"].astype(bool)
        ].copy()
        if same_subject_wrong.empty:
            continue
        true = group[group["is_true_pair"].astype(bool)].iloc[0]
        best = same_subject_wrong.sort_values("rank").iloc[0]
        rank_among = 1 + int((same_subject_wrong["score"].to_numpy(dtype=float) < float(true["score"])).sum())
        rows.append(
            {
                "candidate_pool_condition": condition,
                "variant_name": variant,
                "query_pair_id": query_id,
                "query_subject": true["query_subject"],
                "query_task": true["query_task_original"],
                "query_side": true["query_side"],
                "true_rank_full_pool": int(true["rank"]),
                "true_rank_among_same_subject_candidates": int(rank_among),
                "true_beats_same_subject_hard_negative": bool(rank_among == 1),
                "best_same_subject_candidate_pair_id": best["candidate_pair_id"],
                "best_same_subject_candidate_task": best["candidate_task_original"],
                "best_same_subject_candidate_side": best["candidate_side"],
                "best_same_subject_rank_full_pool": int(best["rank"]),
            }
        )
    cases = pd.DataFrame(rows)
    summary_rows = []
    if not cases.empty:
        for (condition, variant), subset in cases.groupby(["candidate_pool_condition", "variant_name"], dropna=False):
            summary_rows.append(
                {
                    "candidate_pool_condition": condition,
                    "variant_name": variant,
                    "queries_with_same_subject_hard_negative": int(len(subset)),
                    "success_rate": float(subset["true_beats_same_subject_hard_negative"].astype(float).mean()),
                    "failure_count": int((~subset["true_beats_same_subject_hard_negative"].astype(bool)).sum()),
                }
            )
    return pd.DataFrame(summary_rows), cases


def feature_mapping() -> pd.DataFrame:
    """Write canonical common-feature mapping."""

    return pd.DataFrame(
        [
            {
                "canonical_feature": canonical,
                "ds004998_feature": ds_feature,
                "oxf_feature": oxf_feature,
                "feature_family": family,
                "included_in_ds004998_stn_only": ds_feature in STN_ONLY_FEATURES,
            }
            for canonical, ds_feature, oxf_feature, family in COMMON_FEATURE_MAPPING
        ]
    )


def transition_effects(pairs: pd.DataFrame) -> pd.DataFrame:
    """Compute OFF->ON transition effects for STN-only features."""

    rows = []
    for feature in STN_ONLY_FEATURES:
        delta = pd.to_numeric(pairs[f"medon_{feature}"], errors="coerce") - pd.to_numeric(pairs[f"x_{feature}"], errors="coerce")
        delta = delta.dropna().to_numpy(dtype=float)
        std = float(np.nanstd(delta, ddof=1)) if len(delta) > 1 else np.nan
        dz = float(np.nanmean(delta) / std) if np.isfinite(std) and std > 1e-12 else np.nan
        rows.append(
            {
                "feature": feature,
                "feature_family": "STN_power" if feature in STN_POWER_FEATURES else "compact_aperiodic_residual",
                "n": int(len(delta)),
                "mean_delta": float(np.nanmean(delta)) if len(delta) else np.nan,
                "median_delta": float(np.nanmedian(delta)) if len(delta) else np.nan,
                "cohens_dz": dz,
                "dominant_direction": "ON_greater" if len(delta) and np.nanmean(delta) > 0 else "OFF_greater",
                "sign_consistency": float(max((delta > 0).mean(), (delta < 0).mean())) if len(delta) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_metrics(metrics: pd.DataFrame, path: Path) -> None:
    """Plot all-MedOn top1/MRR."""

    focus = metrics[
        metrics["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & metrics["variant_name"].astype(str).isin([V2A_NAME, V2B_NAME, V2C_DECONFOUNDED_NAME, V2D_QUALITY_DECONFOUNDED_NAME])
    ].copy()
    if focus.empty:
        return
    pivot = focus.pivot_table(index="variant_name", values=["top1", "mrr"], aggfunc="first")
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.45, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.55, color="black", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Metric")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(summary: dict[str, object], path: Path) -> Path:
    """Write Markdown report."""

    lines = [
        "# ds004998 STN-Only Cross-Dataset Branch",
        "",
        "Scope: cached 18-subject ds004998 STN-only common-layer diagnostic.",
        "",
        "## Validation",
    ]
    for key, value in summary["validation"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Key Metrics"])
    for row in summary["key_metrics"]:
        lines.append(
            f"- {row['candidate_pool_condition']} / {row['variant_name']}: "
            f"top1={float(row['top1']):.3f}, MRR={float(row['mrr']):.3f}, failures={int(row['failures'])}"
        )
    lines.extend(["", "## Warnings"])
    for warning in summary["warnings"]:
        lines.append(f"- {warning}")
    return write_text(lines, path)


def run() -> dict[str, object]:
    """Run the ds004998 STN-only common-layer branch."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = read_csv_required(FROZEN_DIR / "frozen_v2a_pairs.csv")
    summary = read_json_required(FROZEN_DIR / "frozen_v2a_summary.json")
    failure_log = read_csv_required(FROZEN_DIR / "tables" / "extraction_failure_log.csv")
    excluded = read_csv_required(COHORT_DIR / "excluded_subjects.csv")
    if len(pairs) != EXPECTED_PAIRS:
        raise AssertionError(f"complete_pairs={len(pairs)}, expected={EXPECTED_PAIRS}")
    if pairs["subject"].nunique() != EXPECTED_SUBJECTS:
        raise AssertionError(f"subjects={pairs['subject'].nunique()}, expected={EXPECTED_SUBJECTS}")
    if not failure_log.empty:
        raise AssertionError("Extraction failure log is not empty.")
    if int(summary.get("top_k", -1)) != TOP_K:
        raise AssertionError("top_k changed.")
    if not math.isclose(float(summary.get("aperiodic_alpha", np.nan)), float(APERIODIC_ALPHA)):
        raise AssertionError("aperiodic_alpha changed.")

    v2_features, inventory = available_features(pairs, STN_ONLY_FEATURES)
    compact_features = [feature for feature in STN_COMPACT_FEATURES if feature in v2_features]
    if not v2_features or not compact_features:
        raise AssertionError("STN-only features are missing.")

    forward_scores, candidate_pools = evaluate_forward_scores(pairs, v2_features, compact_features)
    reverse_scores = evaluate_reverse_v2b(pairs, v2_features, compact_features)
    v2c_cycle = cycle_rerank_scores(forward_scores, reverse_scores, V2C_CYCLE_NAME, deconfounded=False)
    v2c_deconfounded = cycle_rerank_scores(forward_scores, reverse_scores, V2C_DECONFOUNDED_NAME, deconfounded=True)
    v2d_tables = []
    component_tables = []
    for variant in V2D_VARIANTS:
        scores, components = rerank_v2d_variant(forward_scores, reverse_scores, variant)
        v2d_tables.append(scores)
        component_tables.append(components)
    all_scores = pd.concat([forward_scores, v2c_cycle, v2c_deconfounded, *v2d_tables], ignore_index=True)
    all_scores = rank_scores(normalize_scores(all_scores))
    components = pd.concat(component_tables, ignore_index=True) if component_tables else pd.DataFrame()
    metrics, diagnostics = build_metrics_tables(all_scores)
    comparison_v2a = comparison_to_baseline(metrics, V2A_NAME)
    comparison_v2b = comparison_to_baseline(metrics, V2B_NAME)
    bootstrap_ci, bootstrap_samples = subject_bootstrap(diagnostics)
    random_summary, random_samples = random_label_null(diagnostics, all_scores)
    loso = loso_sensitivity(diagnostics)
    hard_summary, hard_cases = hard_negative_cases(all_scores)
    failures = failure_cases(diagnostics)
    transitions = transition_effects(pairs)
    mapping = feature_mapping()

    key_metrics = metrics[
        metrics["candidate_pool_condition"].astype(str).isin([PRIMARY_CONDITION, HARD_NEGATIVE_CONDITION, ALL_MEDON_CONDITION])
        & metrics["variant_name"].astype(str).isin([V2A_NAME, V2B_NAME, V2C_DECONFOUNDED_NAME, V2D_QUALITY_DECONFOUNDED_NAME])
    ].sort_values(["candidate_pool_condition", "variant_name"])
    validation = {
        "complete_pairs": int(len(pairs)),
        "subjects": int(pairs["subject"].nunique()),
        "top_k": TOP_K,
        "aperiodic_alpha": APERIODIC_ALPHA,
        "stn_only_feature_count": int(len(v2_features)),
        "compact_feature_count": int(len(compact_features)),
        "extraction_failure_log_rows": int(len(failure_log)),
        "sub_BYJoWR_excluded": bool("sub-BYJoWR" in set(excluded.get("subject", pd.Series(dtype=str)).astype(str))),
    }
    warnings = [
        "This is a diagnostic STN-only branch; frozen ds004998 full v2A outputs are unchanged.",
        "MEG, motor, and cortico-STN coupling features are intentionally excluded.",
        "Feature units differ from OXF for raw/log power; retrieval uses train-fold scaling, but effect sizes are within-dataset only.",
        "No clinical prediction, treatment, DBS optimization, or causal medication-effect claim is made.",
    ]
    out_summary = {
        "validation": validation,
        "key_metrics": key_metrics.to_dict("records"),
        "warnings": warnings,
        "claim_boundary": {
            "stn_only_cross_dataset_branch": True,
            "frozen_v2A_changed": False,
            "parameter_tuning": False,
            "clinical_prediction_or_treatment_claim": False,
        },
    }

    paths = {
        "summary_json": write_json(out_summary, OUTPUT_DIR / "ds004998_stn_only_summary.json"),
        "summary_md": write_report(out_summary, OUTPUT_DIR / "ds004998_stn_only_summary.md"),
        "feature_inventory": write_csv(inventory, OUTPUT_DIR / "ds004998_stn_only_feature_inventory.csv"),
        "canonical_mapping": write_csv(mapping, OUTPUT_DIR / "ds004998_oxf_canonical_stn_feature_mapping.csv"),
        "transition_effects": write_csv(transitions, OUTPUT_DIR / "ds004998_stn_only_transition_effects.csv"),
        "candidate_pools": write_csv(candidate_pools, OUTPUT_DIR / "ds004998_stn_only_candidate_pools.csv"),
        "candidate_scores": write_csv(all_scores, OUTPUT_DIR / "ds004998_stn_only_candidate_scores.csv"),
        "forward_scores": write_csv(forward_scores, OUTPUT_DIR / "ds004998_stn_only_forward_scores.csv"),
        "reverse_scores": write_csv(reverse_scores, OUTPUT_DIR / "ds004998_stn_only_reverse_v2b_scores.csv"),
        "metrics": write_csv(metrics, OUTPUT_DIR / "ds004998_stn_only_observed_metrics.csv"),
        "diagnostics": write_csv(diagnostics, OUTPUT_DIR / "ds004998_stn_only_query_diagnostics.csv"),
        "comparison_to_v2a": write_csv(comparison_v2a, OUTPUT_DIR / "ds004998_stn_only_comparison_to_v2a.csv"),
        "comparison_to_v2b": write_csv(comparison_v2b, OUTPUT_DIR / "ds004998_stn_only_comparison_to_v2b.csv"),
        "bootstrap_ci": write_csv(bootstrap_ci, OUTPUT_DIR / "ds004998_stn_only_subject_bootstrap_ci.csv"),
        "bootstrap_samples": write_csv(bootstrap_samples, OUTPUT_DIR / "ds004998_stn_only_subject_bootstrap_samples.csv"),
        "random_label_null_summary": write_csv(random_summary, OUTPUT_DIR / "ds004998_stn_only_random_label_null_summary.csv"),
        "random_label_null_samples": write_csv(random_samples, OUTPUT_DIR / "ds004998_stn_only_random_label_null_samples.csv"),
        "loso": write_csv(loso, OUTPUT_DIR / "ds004998_stn_only_loso_sensitivity.csv"),
        "hard_negative_summary": write_csv(hard_summary, OUTPUT_DIR / "ds004998_stn_only_hard_negative_summary.csv"),
        "hard_negative_cases": write_csv(hard_cases, OUTPUT_DIR / "ds004998_stn_only_hard_negative_cases.csv"),
        "failure_cases": write_csv(failures, OUTPUT_DIR / "ds004998_stn_only_failure_cases.csv"),
        "v2d_components": write_csv(components, OUTPUT_DIR / "ds004998_stn_only_v2d_components.csv"),
    }
    plot_metrics(metrics, OUTPUT_DIR / "figures" / "ds004998_stn_only_all_medon_top1_mrr.png")

    print(f"complete_pairs: {validation['complete_pairs']}")
    print(f"subjects: {validation['subjects']}")
    print(f"stn_only_feature_count: {validation['stn_only_feature_count']}")
    print(f"compact_feature_count: {validation['compact_feature_count']}")
    print(f"top_k: {TOP_K}")
    print(f"aperiodic_alpha: {APERIODIC_ALPHA}")
    for condition in [PRIMARY_CONDITION, HARD_NEGATIVE_CONDITION, ALL_MEDON_CONDITION]:
        for variant in [V2A_NAME, V2B_NAME, V2C_DECONFOUNDED_NAME, V2D_QUALITY_DECONFOUNDED_NAME]:
            row = metrics[
                metrics["candidate_pool_condition"].astype(str).eq(condition)
                & metrics["variant_name"].astype(str).eq(variant)
            ]
            if row.empty:
                continue
            item = row.iloc[0]
            print(
                f"{condition} / {variant}: "
                f"top1={float(item['top1']):.3f}, MRR={float(item['mrr']):.3f}, failures={int(item['failures'])}"
            )
    print(f"output folder: {OUTPUT_DIR}")
    print("warnings:")
    for warning in warnings:
        print(f"- {warning}")
    return {"paths": {key: str(path) for key, path in paths.items()}, "summary": out_summary}


def main() -> None:
    """Entry point."""

    run()


if __name__ == "__main__":
    main()
