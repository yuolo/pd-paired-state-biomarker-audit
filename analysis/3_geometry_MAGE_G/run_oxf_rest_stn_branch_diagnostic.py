#!/usr/bin/env python3
"""Diagnostic OXF Rest-STN branch analyses.

This script tests pre-defined, physiology-motivated OXF feature branches and
contact-coverage failure modes. It reads cached OXF feature tables only. It
does not tune top_k, alpha, scorer logic, or ds004998 frozen outputs.
"""

from __future__ import annotations

import json
import os
import re
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
from scripts.run_v2a_scientific_audit_17subjects_33pairs import (  # noqa: E402
    APERIODIC_ALPHA,
    RANDOM_SEED,
    TOP_K,
)
from scripts.run_v2b_gated_scientific_audit_17subjects_33pairs import ALL_MEDON_CONDITION, V2B_NAME  # noqa: E402
from scripts.run_v2c_deconfounded_transition_retrieval_experiment_17subjects_33pairs import (  # noqa: E402
    V2C_DECONFOUNDED_NAME,
)
from scripts.run_v2d_kreciprocal_quality_rerank_experiment_17subjects_33pairs import (  # noqa: E402
    V2D_QUALITY_DECONFOUNDED_NAME,
)


OUTPUT_DIR = Path("outputs/oxf_rest_stn_branch_diagnostic")
V2E_DIR = Path("outputs/oxf_stn_physiology_locked_v2e")
DEEP_DIVE_DIR = Path("outputs/cross_dataset_discrepancy_deep_dive")

PAIRS_PATH = V2E_DIR / "v2e_pairs.csv"
CONTACT_SELECTION_PATH = V2E_DIR / "v2e_contact_selection.csv"
ALIGNMENT_PATH = DEEP_DIVE_DIR / "feature_transition_alignment.csv"

FOCUS_SOURCE_FEATURE_SETS = [
    "median_channel_reference_with_bursts",
    "off_beta_peak_contact_exact",
    "off_beta_peak_contact_numeric",
]

FOCUS_VARIANTS = [
    oxf_base.V2A_NAME,
    V2B_NAME,
    V2C_DECONFOUNDED_NAME,
    V2D_QUALITY_DECONFOUNDED_NAME,
]

VARIANT_LABELS = {
    oxf_base.V2A_NAME: "v2A",
    V2B_NAME: "v2B",
    V2C_DECONFOUNDED_NAME: "v2C",
    V2D_QUALITY_DECONFOUNDED_NAME: "v2D",
}

COMMON_POWER = [
    "stn_alpha_log_power",
    "stn_low_beta_log_power",
    "stn_high_beta_log_power",
    "stn_broad_beta_log_power",
    "stn_gamma_log_power",
]

COMMON_COMPACT = [
    "aperiodic_slope",
    "aperiodic_offset",
    "low_beta_residual_power",
    "high_beta_residual_power",
    "broad_beta_residual_power",
    "beta_peak_amplitude",
    "beta_peak_frequency",
]

BETA_RESIDUAL_CORE = [
    "low_beta_residual_power",
    "high_beta_residual_power",
    "broad_beta_residual_power",
]

APERIODIC_ONLY = [
    "aperiodic_slope",
    "aperiodic_offset",
]

BETA_PEAK_PAIR = [
    "beta_peak_amplitude",
    "beta_peak_frequency",
]

BETA_BURSTS = list(v2e.BURST_FEATURES)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    return str(path)


def write_json(obj: dict, path: Path) -> str:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def available_features(pairs: pd.DataFrame, features: list[str]) -> list[str]:
    out = []
    for feature in features:
        x_col = f"x_{feature}"
        y_col = f"medon_{feature}"
        if x_col in pairs.columns and y_col in pairs.columns:
            vals = pairs[[x_col, y_col]].apply(pd.to_numeric, errors="coerce")
            if vals.notna().all(axis=1).sum() >= 3:
                out.append(feature)
    return out


def alignment_feature_lists() -> dict[str, list[str]]:
    align = read_csv(ALIGNMENT_PATH)
    oxf_by_canonical = dict(zip(align["canonical_feature"].astype(str), align["oxf_feature"].astype(str), strict=False))
    strong = align[
        ~align["oxf_exact_weak_transition"].astype(bool)
    ]["oxf_feature"].astype(str).tolist()
    direction_agree = align[
        align["ds_oxf_exact_direction_agree"].astype(bool)
    ]["oxf_feature"].astype(str).tolist()
    aligned_strong = align[
        align["ds_oxf_exact_direction_agree"].astype(bool)
        & ~align["oxf_exact_weak_transition"].astype(bool)
    ]["oxf_feature"].astype(str).tolist()
    compact_aligned = [f for f in direction_agree if f in COMMON_COMPACT]
    return {
        "direction_agree_common": direction_agree,
        "strong_exact_transition_common": strong,
        "aligned_strong_common": aligned_strong,
        "compact_direction_agree": compact_aligned,
        "mapping_oxf_features": list(oxf_by_canonical.values()),
    }


def branch_specs() -> pd.DataFrame:
    lists = alignment_feature_lists()
    specs = [
        {
            "branch_feature_group": "common_all_12",
            "rationale": "All ds004998/OXF shared STN features.",
            "features": [*COMMON_POWER, *COMMON_COMPACT],
        },
        {
            "branch_feature_group": "common_stn_power_5",
            "rationale": "Raw/log STN band-power family only.",
            "features": COMMON_POWER,
        },
        {
            "branch_feature_group": "common_compact_residual_7",
            "rationale": "Compact aperiodic/residual/peak family only.",
            "features": COMMON_COMPACT,
        },
        {
            "branch_feature_group": "direction_agree_common",
            "rationale": "Shared features whose OFF->ON direction agrees between ds004998 and OXF exact-contact.",
            "features": lists["direction_agree_common"],
        },
        {
            "branch_feature_group": "strong_exact_transition_common",
            "rationale": "Shared features with non-weak OXF exact-contact transition effect.",
            "features": lists["strong_exact_transition_common"],
        },
        {
            "branch_feature_group": "aligned_strong_common",
            "rationale": "Shared features with both cross-dataset direction agreement and non-weak exact-contact transition.",
            "features": lists["aligned_strong_common"],
        },
        {
            "branch_feature_group": "compact_direction_agree",
            "rationale": "Compact features with cross-dataset transition direction agreement.",
            "features": lists["compact_direction_agree"],
        },
        {
            "branch_feature_group": "beta_residual_core_3",
            "rationale": "Low/high/broad beta residual power core.",
            "features": BETA_RESIDUAL_CORE,
        },
        {
            "branch_feature_group": "aperiodic_only_2",
            "rationale": "Aperiodic slope and offset only.",
            "features": APERIODIC_ONLY,
        },
        {
            "branch_feature_group": "beta_peak_pair_2",
            "rationale": "Beta peak amplitude and frequency only.",
            "features": BETA_PEAK_PAIR,
        },
        {
            "branch_feature_group": "beta_bursts_only_10",
            "rationale": "Pre-defined beta burst features only.",
            "features": BETA_BURSTS,
        },
        {
            "branch_feature_group": "common_all_plus_beta_bursts",
            "rationale": "Shared STN features plus pre-defined beta bursts.",
            "features": [*COMMON_POWER, *COMMON_COMPACT, *BETA_BURSTS],
        },
    ]
    rows = []
    for spec in specs:
        features = list(dict.fromkeys(spec["features"]))
        compact = [f for f in features if f in oxf_base.COMPACT_FEATURES or f in BETA_BURSTS]
        if not compact:
            compact = features.copy()
        rows.append(
            {
                "branch_feature_group": spec["branch_feature_group"],
                "rationale": spec["rationale"],
                "features": ";".join(features),
                "rerank_features": ";".join(compact),
                "n_declared_features": len(features),
                "n_declared_rerank_features": len(compact),
                "diagnostic_note": "Rerank features equal branch features when no compact features are present."
                if set(compact) == set(features) and not any(f in oxf_base.COMPACT_FEATURES for f in features)
                else "Uses compact/burst subset for reranking.",
            }
        )
    return pd.DataFrame(rows)


def add_branch_columns(tables: dict[str, pd.DataFrame], source_feature_set: str, branch: pd.Series) -> dict[str, pd.DataFrame]:
    out = {}
    for name, table in tables.items():
        if table.empty:
            out[name] = table
            continue
        frame = table.copy()
        frame.insert(0, "branch_feature_group", branch["branch_feature_group"])
        frame.insert(0, "source_feature_set", source_feature_set)
        out[name] = frame
    return out


def evaluate_one_feature_set_fast(
    pairs: pd.DataFrame,
    feature_set: str,
    usable_features: list[str],
    rerank_features: list[str],
) -> dict[str, pd.DataFrame]:
    """Evaluate retrieval variants while skipping expensive bootstrap for branch matrix."""

    if pairs.empty or not usable_features or not rerank_features or pairs["subject"].nunique() < 3:
        empty = pd.DataFrame()
        return {
            "scores": empty,
            "metrics": empty,
            "diagnostics": empty,
            "components": empty,
            "failure_cases": empty,
            "hard_summary": empty,
            "hard_cases": empty,
        }
    clean_features = rerank_features
    forward_scores, _candidate_pools = oxf_base.evaluate_forward_scores(
        pairs,
        usable_features,
        rerank_features,
        clean_features,
    )
    reverse_scores = oxf_base.evaluate_reverse_v2b(pairs, usable_features, rerank_features, clean_features)
    v2c_cycle = v2e.cycle_rerank_scores(
        forward_scores,
        reverse_scores,
        v2e.V2C_CYCLE_NAME,
        deconfounded=False,
    )
    v2c_deconfounded = v2e.cycle_rerank_scores(
        forward_scores,
        reverse_scores,
        V2C_DECONFOUNDED_NAME,
        deconfounded=True,
    )
    v2d_tables = []
    component_tables = []
    for variant in v2e.V2D_VARIANTS:
        scores, components = v2e.rerank_v2d_variant(forward_scores, reverse_scores, variant)
        v2d_tables.append(scores)
        component_tables.append(components)
    scores = pd.concat([forward_scores, v2c_cycle, v2c_deconfounded, *v2d_tables], ignore_index=True)
    scores = v2e.rank_scores(v2e.normalize_scores(scores))
    metrics, diagnostics = v2e.build_metrics_tables(scores)
    components = pd.concat(component_tables, ignore_index=True) if component_tables else pd.DataFrame()
    hard_summary, hard_cases = oxf_base.hard_negative_cases(scores)
    failures = v2e.failure_cases(diagnostics)
    tables = {
        "scores": scores,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "components": components,
        "failure_cases": failures,
        "hard_summary": hard_summary,
        "hard_cases": hard_cases,
    }
    for table in tables.values():
        if not table.empty:
            table.insert(0, "feature_set", feature_set)
    return tables


def evaluate_branches() -> dict[str, pd.DataFrame]:
    pairs_all = read_csv(PAIRS_PATH)
    specs = branch_specs()
    collected: dict[str, list[pd.DataFrame]] = {
        "scores": [],
        "metrics": [],
        "diagnostics": [],
        "components": [],
        "bootstrap_ci": [],
        "failure_cases": [],
        "hard_summary": [],
        "hard_cases": [],
    }
    availability_rows = []
    for source_feature_set in FOCUS_SOURCE_FEATURE_SETS:
        pairs = pairs_all[pairs_all["feature_set"].astype(str).eq(source_feature_set)].copy()
        if pairs.empty:
            continue
        for _, branch in specs.iterrows():
            features = [f for f in str(branch["features"]).split(";") if f]
            rerank_features = [f for f in str(branch["rerank_features"]).split(";") if f]
            usable = available_features(pairs, features)
            rerank = [f for f in rerank_features if f in usable]
            if not rerank and usable:
                rerank = usable.copy()
            availability_rows.append(
                {
                    "source_feature_set": source_feature_set,
                    "branch_feature_group": branch["branch_feature_group"],
                    "n_pairs_source": int(len(pairs)),
                    "n_subjects_source": int(pairs["subject"].nunique()),
                    "n_declared_features": len(features),
                    "n_usable_features": len(usable),
                    "n_rerank_features": len(rerank),
                    "usable_features": ";".join(usable),
                    "rerank_features": ";".join(rerank),
                    "skipped": bool(len(usable) == 0 or len(rerank) == 0),
                }
            )
            if len(usable) == 0 or len(rerank) == 0:
                continue
            result = evaluate_one_feature_set_fast(
                pairs=pairs,
                feature_set=f"{source_feature_set}__{branch['branch_feature_group']}",
                usable_features=usable,
                rerank_features=rerank,
            )
            result = add_branch_columns(result, source_feature_set, branch)
            for key in collected:
                if key in result and not result[key].empty:
                    collected[key].append(result[key])
    return {
        key: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for key, frames in collected.items()
    } | {"branch_availability": pd.DataFrame(availability_rows), "branch_specs": specs}


def focus_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    out = metrics[
        metrics["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & metrics["variant_name"].astype(str).isin(FOCUS_VARIANTS)
    ].copy()
    out["variant_label"] = out["variant_name"].map(VARIANT_LABELS)
    return out.sort_values(["top1", "mrr"], ascending=False).reset_index(drop=True)


def annotate_metric_eligibility(best: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    if best.empty:
        return best
    cols = [
        "source_feature_set",
        "branch_feature_group",
        "n_declared_features",
        "n_usable_features",
        "n_rerank_features",
        "usable_features",
        "rerank_features",
        "skipped",
    ]
    out = best.merge(availability[cols], on=["source_feature_set", "branch_feature_group"], how="left")
    out["feature_completeness"] = (
        pd.to_numeric(out["n_usable_features"], errors="coerce")
        / pd.to_numeric(out["n_declared_features"], errors="coerce")
    )
    out["complete_declared_branch"] = out["n_usable_features"].eq(out["n_declared_features"])
    out["exact_contact_primary_source"] = out["source_feature_set"].astype(str).eq("off_beta_peak_contact_exact")
    out["numeric_contact_sensitivity_only"] = out["source_feature_set"].astype(str).eq("off_beta_peak_contact_numeric")
    out["primary_eligible"] = (
        out["exact_contact_primary_source"]
        & out["complete_declared_branch"]
        & out["variant_label"].astype(str).isin(["v2C", "v2D"])
    )
    out["interpretation_warning"] = np.select(
        [
            out["numeric_contact_sensitivity_only"],
            ~out["complete_declared_branch"],
            out["source_feature_set"].astype(str).eq("median_channel_reference_with_bursts"),
        ],
        [
            "numeric contact sensitivity only; do not use as primary without independent montage verification",
            "incomplete declared feature group; available-feature subset may not match branch name",
            "median channel reference; contact averaging can dilute physiological contact specificity",
        ],
        default="primary-eligible if exact-contact and pre-specified",
    )
    return out.sort_values(["top1", "mrr"], ascending=False).reset_index(drop=True)


def failure_composition(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    focus = diagnostics[
        diagnostics["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & diagnostics["variant_name"].astype(str).isin(FOCUS_VARIANTS)
    ].copy()
    focus["variant_label"] = focus["variant_name"].map(VARIANT_LABELS)
    rows = []
    for keys, group in focus.groupby(["source_feature_set", "branch_feature_group", "variant_label", "variant_name"], dropna=False):
        total = len(group)
        for failure_type, count in group["failure_type"].fillna("unknown").value_counts().items():
            rows.append(
                {
                    "source_feature_set": keys[0],
                    "branch_feature_group": keys[1],
                    "variant_label": keys[2],
                    "variant_name": keys[3],
                    "failure_type": failure_type,
                    "count": int(count),
                    "rate": float(count / total) if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def query_margin_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    focus = diagnostics[
        diagnostics["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & diagnostics["variant_name"].astype(str).isin(FOCUS_VARIANTS)
    ].copy()
    focus["variant_label"] = focus["variant_name"].map(VARIANT_LABELS)
    rows = []
    for keys, group in focus.groupby(["source_feature_set", "branch_feature_group", "variant_label", "variant_name"], dropna=False):
        rank = pd.to_numeric(group["true_medon_rank"], errors="coerce")
        margin = pd.to_numeric(group["retrieval_margin"], errors="coerce")
        rows.append(
            {
                "source_feature_set": keys[0],
                "branch_feature_group": keys[1],
                "variant_label": keys[2],
                "variant_name": keys[3],
                "n_queries": int(len(group)),
                "top1": float(group["top_ranked_is_true_pair"].astype(bool).mean()),
                "mrr": float(pd.to_numeric(group["reciprocal_rank"], errors="coerce").mean()),
                "top5": float((rank <= 5).mean()),
                "mean_true_rank": float(rank.mean()),
                "median_true_rank": float(rank.median()),
                "mean_margin": float(margin.mean()),
                "median_margin": float(margin.median()),
                "negative_margin_rate": float((margin < 0).mean()),
                "true_rank_gt5_rate": float((rank > 5).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["top1", "mrr"], ascending=False)


def normalize_token(token: object) -> str:
    if pd.isna(token):
        return ""
    text = str(token).strip()
    digits = re.findall(r"\d+", text)
    if not digits:
        return text
    return "".join(digits)


def token_digits(token: object) -> set[str]:
    return set(normalize_token(token))


def contact_coverage_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    contact = read_csv(CONTACT_SELECTION_PATH)
    exact = contact[contact["feature_set"].astype(str).eq("off_beta_peak_contact_exact")].copy()
    numeric = contact[contact["feature_set"].astype(str).eq("off_beta_peak_contact_numeric")].copy()
    exact_key = exact.set_index(["subject", "side"])
    numeric_key = numeric.set_index(["subject", "side"])
    rows = []
    for key, exact_row in exact_key.iterrows():
        num_row = numeric_key.loc[key] if key in numeric_key.index else pd.Series(dtype=object)
        available = str(exact_row.get("available_on_channels", ""))
        available_tokens = [item for item in available.split(";") if item]
        off_token = exact_row.get("off_contact_token", "")
        off_digits = token_digits(off_token)
        available_digits = set().union(*(token_digits(tok) for tok in available_tokens)) if available_tokens else set()
        exact_paired = str(exact_row.get("selection_status")) == "paired"
        numeric_paired = str(num_row.get("selection_status")) == "paired" if not num_row.empty else False
        if exact_paired:
            repair_class = "exact_available"
        elif numeric_paired:
            repair_class = "numeric_only_recovered"
        elif off_digits and off_digits.issubset(available_digits):
            repair_class = "constituent_contacts_available_only"
        elif available_tokens:
            repair_class = "no_exact_or_numeric_contact"
        else:
            repair_class = "no_on_channel_inventory"
        rows.append(
            {
                "subject": key[0],
                "side": key[1],
                "exact_status": exact_row.get("selection_status"),
                "numeric_status": num_row.get("selection_status") if not num_row.empty else "",
                "off_exact_token": exact_row.get("off_contact_token"),
                "off_numeric_token": num_row.get("off_contact_token") if not num_row.empty else "",
                "exact_on_token": exact_row.get("on_contact_token"),
                "numeric_on_token": num_row.get("on_contact_token") if not num_row.empty else "",
                "off_selected_channel": exact_row.get("off_selected_channel"),
                "exact_on_selected_channel": exact_row.get("on_selected_channel"),
                "numeric_on_selected_channel": num_row.get("on_selected_channel") if not num_row.empty else "",
                "available_on_channels": available,
                "exact_paired": exact_paired,
                "numeric_paired": numeric_paired,
                "repair_class": repair_class,
                "scientific_action": contact_action(repair_class),
            }
        )
    cases = pd.DataFrame(rows)
    summary = (
        cases.groupby("repair_class", dropna=False)
        .agg(n_pairs=("subject", "size"), n_subjects=("subject", "nunique"))
        .reset_index()
    )
    summary["rate"] = summary["n_pairs"] / len(cases) if len(cases) else np.nan
    return summary.sort_values("n_pairs", ascending=False), cases.sort_values(["repair_class", "subject", "side"])


def contact_action(repair_class: str) -> str:
    if repair_class == "exact_available":
        return "Use in primary exact-contact subset."
    if repair_class == "numeric_only_recovered":
        return "Use as sensitivity only unless channel nomenclature can be verified from metadata."
    if repair_class == "constituent_contacts_available_only":
        return "Do not reconstruct bipolar contact without raw montage verification."
    if repair_class == "no_exact_or_numeric_contact":
        return "Leave out of exact-contact primary subset; report as contact coverage loss."
    return "Unavailable without additional metadata."


def branch_recommendations(best: pd.DataFrame, contact_summary: pd.DataFrame) -> pd.DataFrame:
    def best_row(source: str, variant_label: str) -> pd.Series:
        frame = best[best["source_feature_set"].astype(str).eq(source) & best["variant_label"].astype(str).eq(variant_label)]
        if frame.empty:
            return pd.Series(dtype=object)
        return frame.sort_values(["top1", "mrr"], ascending=False).iloc[0]

    exact_v2c = best_row("off_beta_peak_contact_exact", "v2C")
    exact_v2d = best_row("off_beta_peak_contact_exact", "v2D")
    median_v2d = best_row("median_channel_reference_with_bursts", "v2D")
    numeric_v2d = best_row("off_beta_peak_contact_numeric", "v2D")
    numeric_warning = str(numeric_v2d.get("interpretation_warning", "numeric contact sensitivity only"))
    numeric_only = contact_summary[contact_summary["repair_class"].eq("numeric_only_recovered")]
    numeric_only_n = int(numeric_only.iloc[0]["n_pairs"]) if not numeric_only.empty else 0
    rows = [
        {
            "priority": 1,
            "change": "Use exact-contact OXF as the primary external STN branch.",
            "evidence": f"Best exact-contact branch v2C: {exact_v2c.get('branch_feature_group', 'NA')} top1={float(exact_v2c.get('top1', np.nan)):.3f}, MRR={float(exact_v2c.get('mrr', np.nan)):.3f}; v2D best top1={float(exact_v2d.get('top1', np.nan)):.3f}.",
            "guardrail": "OFF-only contact selection; exact ON token matching; report smaller n explicitly.",
        },
        {
            "priority": 2,
            "change": "Treat numeric contact recovery as sensitivity, not primary.",
            "evidence": f"Numeric-only recovery adds {numeric_only_n} pair(s); best numeric v2D branch top1={float(numeric_v2d.get('top1', np.nan)):.3f}, but warning: {numeric_warning}.",
            "guardrail": "Do not merge exact and numeric subsets unless contact nomenclature is independently verified.",
        },
        {
            "priority": 3,
            "change": "Prioritize compact/direction-aligned STN branches for interpretation, but keep all-common as reference.",
            "evidence": f"Median v2D best branch top1={float(median_v2d.get('top1', np.nan)):.3f}; exact-contact branches determine whether the feature-family gap is contact-driven or physiology-driven.",
            "guardrail": "Do not choose branch by retrieval score for a primary claim. Use branch audit to motivate a pre-specified next run.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(summary: dict, recommendations: pd.DataFrame, path: Path) -> None:
    lines = [
        "# OXF Rest-STN Branch Diagnostic",
        "",
        "Diagnostic only: cached OXF features, fixed top_k=5, alpha=0.5, no ds004998 pipeline changes.",
        "",
        "## Key Numbers",
    ]
    for key, value in summary["key_numbers"].items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.3f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Recommendations"])
    for _, row in recommendations.sort_values("priority").iterrows():
        lines.append(f"{int(row['priority'])}. {row['change']}")
        lines.append(f"   Evidence: {row['evidence']}")
        lines.append(f"   Guardrail: {row['guardrail']}")
    lines.extend(["", "## Warnings"])
    for warning in summary["warnings"]:
        lines.append(f"- {warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    assert TOP_K == 5
    assert APERIODIC_ALPHA == 0.5

    results = evaluate_branches()
    metrics = results["metrics"]
    diagnostics = results["diagnostics"]
    best = annotate_metric_eligibility(focus_metrics(metrics), results["branch_availability"])
    failures = failure_composition(diagnostics)
    margins = query_margin_summary(diagnostics)
    contact_summary, contact_cases = contact_coverage_audit()
    recommendations = branch_recommendations(best, contact_summary)

    outputs = {
        "branch_specs": write_csv(results["branch_specs"], OUTPUT_DIR / "branch_feature_specs.csv"),
        "branch_availability": write_csv(results["branch_availability"], OUTPUT_DIR / "branch_feature_availability.csv"),
        "branch_metrics_all": write_csv(metrics, OUTPUT_DIR / "branch_metrics_all.csv"),
        "branch_metrics_focus": write_csv(best, OUTPUT_DIR / "branch_metrics_focus.csv"),
        "branch_primary_eligible_metrics": write_csv(
            best[best["primary_eligible"].astype(bool)].copy(),
            OUTPUT_DIR / "branch_primary_eligible_metrics.csv",
        ),
        "branch_query_diagnostics": write_csv(diagnostics, OUTPUT_DIR / "branch_query_diagnostics.csv"),
        "branch_failure_composition": write_csv(failures, OUTPUT_DIR / "branch_failure_composition.csv"),
        "branch_query_margin_summary": write_csv(margins, OUTPUT_DIR / "branch_query_margin_summary.csv"),
        "branch_bootstrap_ci": write_csv(results["bootstrap_ci"], OUTPUT_DIR / "branch_bootstrap_ci.csv"),
        "branch_failure_cases": write_csv(results["failure_cases"], OUTPUT_DIR / "branch_failure_cases.csv"),
        "branch_hard_negative_summary": write_csv(results["hard_summary"], OUTPUT_DIR / "branch_hard_negative_summary.csv"),
        "branch_hard_negative_cases": write_csv(results["hard_cases"], OUTPUT_DIR / "branch_hard_negative_cases.csv"),
        "contact_coverage_summary": write_csv(contact_summary, OUTPUT_DIR / "contact_coverage_summary.csv"),
        "contact_coverage_cases": write_csv(contact_cases, OUTPUT_DIR / "contact_coverage_cases.csv"),
        "recommended_next_changes": write_csv(recommendations, OUTPUT_DIR / "recommended_next_changes.csv"),
    }

    key_numbers: dict[str, object] = {
        "seed": RANDOM_SEED,
        "top_k": TOP_K,
        "aperiodic_alpha": APERIODIC_ALPHA,
        "n_source_feature_sets": len(FOCUS_SOURCE_FEATURE_SETS),
        "n_branch_feature_groups": int(results["branch_specs"]["branch_feature_group"].nunique()),
    }
    if not best.empty:
        top = best.sort_values(["top1", "mrr"], ascending=False).iloc[0]
        key_numbers.update(
            {
                "best_any_top1_source_feature_set": str(top["source_feature_set"]),
                "best_any_top1_branch_feature_group": str(top["branch_feature_group"]),
                "best_any_top1_variant": str(top["variant_name"]),
                "best_any_top1": float(top["top1"]),
                "best_any_mrr_at_best_top1": float(top["mrr"]),
                "best_any_n_pairs": int(top["n_pairs"]),
                "best_any_interpretation_warning": str(top["interpretation_warning"]),
            }
        )
        eligible = best[best["primary_eligible"].astype(bool)].copy()
        if not eligible.empty:
            row = eligible.sort_values(["top1", "mrr"], ascending=False).iloc[0]
            key_numbers.update(
                {
                    "best_primary_eligible_top1_source_feature_set": str(row["source_feature_set"]),
                    "best_primary_eligible_top1_branch_feature_group": str(row["branch_feature_group"]),
                    "best_primary_eligible_top1_variant": str(row["variant_name"]),
                    "best_primary_eligible_top1": float(row["top1"]),
                    "best_primary_eligible_mrr": float(row["mrr"]),
                    "best_primary_eligible_n_pairs": int(row["n_pairs"]),
                }
            )
        for source in FOCUS_SOURCE_FEATURE_SETS:
            source_best = best[best["source_feature_set"].astype(str).eq(source)]
            if not source_best.empty:
                row = source_best.sort_values(["top1", "mrr"], ascending=False).iloc[0]
                key_numbers[f"{source}_best_top1"] = float(row["top1"])
                key_numbers[f"{source}_best_branch"] = str(row["branch_feature_group"])
                key_numbers[f"{source}_best_variant"] = str(row["variant_name"])

    warnings = [
        "This is a diagnostic feature-branch audit, not a tuned replacement for frozen v2A.",
        "Some branches use their own family as rerank features when compact features are absent; those rows are diagnostic only.",
        "Exact-contact OXF has reduced sample size and must be reported separately from full OXF median analysis.",
        "Numeric contact matching can add pairs but is not primary without independent montage/channel nomenclature verification.",
    ]
    summary = {
        "key_numbers": key_numbers,
        "outputs": outputs,
        "warnings": warnings,
    }
    write_json(summary, OUTPUT_DIR / "oxf_rest_stn_branch_diagnostic_summary.json")
    write_report(summary, recommendations, OUTPUT_DIR / "oxf_rest_stn_branch_diagnostic_summary.md")

    print("OXF Rest-STN branch diagnostic complete.")
    print(f"Output folder: {OUTPUT_DIR}")
    for key, value in key_numbers.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    print("Top branch rows:")
    for _, row in best.head(8).iterrows():
        print(
            f"- {row['source_feature_set']} / {row['branch_feature_group']} / "
            f"{row['variant_name']}: top1={float(row['top1']):.3f}, MRR={float(row['mrr']):.3f}, "
            f"n={int(row['n_pairs'])}"
        )


if __name__ == "__main__":
    main()
