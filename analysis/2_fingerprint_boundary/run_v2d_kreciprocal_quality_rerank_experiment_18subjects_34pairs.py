"""Run v2B/v2C/v2D experiments on the BYJoWR-inclusive 18-subject cohort.

This script uses the newly rebuilt local ds004998 18-subject/34-pair frozen
v2A cache, including sub-BYJoWR HoldR run-2 MedOff/MedOn. It keeps scorer,
feature, top-k, alpha, and reranking settings fixed relative to the existing
experimental definitions. The original 17-subject outputs are not modified.
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
    TOP_K,
    V2A_NAME,
    base_feature_names,
    load_subspace_inventory,
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
    plot_all_medon_delta,
    plot_metric,
    rank_scores,
    rerank_v2d_variant,
    subject_bootstrap_ci,
)


FROZEN_DIR = Path("outputs/v2A_frozen_18subjects_34pairs")
COHORT_DIR = Path("outputs/ds004998_full_cohort_completeness_audit_18subjects_34pairs")
OUTPUT_DIR = Path("outputs/v2D_kreciprocal_quality_rerank_experiment_18subjects_34pairs")

EXPECTED_COMPLETE_PAIRS = 34
EXPECTED_SUBJECTS = 18
EVALUATED_CONDITIONS = [
    PRIMARY_CONDITION,
    HARD_NEGATIVE_CONDITION,
    ALL_MEDON_CONDITION,
    TRUE_PLUS_OTHER_CONDITION,
]

REQUIRED_FILES = {
    "pairs": FROZEN_DIR / "frozen_v2a_pairs.csv",
    "subspaces": FROZEN_DIR / "frozen_v2a_subspace_inventory.json",
    "compact_inventory": FROZEN_DIR / "compact_aperiodic_feature_inventory.json",
    "frozen_summary": FROZEN_DIR / "frozen_v2a_summary.json",
    "logical_manifest": COHORT_DIR / "local_logical_recording_manifest.csv",
    "usable_pair_inventory": COHORT_DIR / "usable_pair_inventory.csv",
    "excluded_subjects": COHORT_DIR / "excluded_subjects.csv",
    "cohort_summary": COHORT_DIR / "cohort_completeness_summary.json",
    "extraction_failure_log": FROZEN_DIR / "tables" / "extraction_failure_log.csv",
}


def write_text(lines: Iterable[str], path: Path) -> Path:
    """Write Markdown/text output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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


def selected_v2b_variant():
    """Return selected v2B-gated variant spec."""

    for variant in V2B_VARIANTS:
        if variant.name == V2B_NAME:
            return variant
    raise AssertionError(f"Missing selected v2B variant: {V2B_NAME}")


def validate_inputs(
    pairs: pd.DataFrame,
    logical: pd.DataFrame,
    usable: pd.DataFrame,
    excluded: pd.DataFrame,
    cohort_summary: dict[str, object],
    frozen_summary: dict[str, object],
    failure_log: pd.DataFrame,
) -> dict[str, object]:
    """Validate BYJoWR-inclusive cohort before experiments."""

    missing = [str(path) for path in REQUIRED_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required 18-subject input files: " + "; ".join(missing))
    complete_pairs = int(len(pairs))
    subjects = int(pairs["subject"].nunique())
    if complete_pairs != EXPECTED_COMPLETE_PAIRS:
        raise AssertionError(f"complete_pairs={complete_pairs}, expected={EXPECTED_COMPLETE_PAIRS}")
    if subjects != EXPECTED_SUBJECTS:
        raise AssertionError(f"subjects={subjects}, expected={EXPECTED_SUBJECTS}")
    if "sub-BYJoWR" not in set(pairs["subject"].astype(str)):
        raise AssertionError("sub-BYJoWR is not present in 18-subject frozen pairs.")
    byjo = pairs[pairs["subject"].astype(str).eq("sub-BYJoWR")]
    if len(byjo) != 1 or str(byjo.iloc[0]["task_original"]) != "HoldR":
        raise AssertionError("sub-BYJoWR expected exactly one complete HoldR pair.")
    if "sub-BYJoWR" in set(excluded.get("subject", pd.Series(dtype=str)).astype(str)):
        raise AssertionError("sub-BYJoWR is still marked excluded in the 18-subject cohort.")
    if any(logical["task"].astype(str).str.lower().str.startswith("rest")) and not logical[
        logical["task"].astype(str).str.lower().str.startswith("rest")
    ].empty:
        rest_subjects = set(logical.loc[logical["task"].astype(str).str.lower().str.startswith("rest"), "subject"].astype(str))
        pair_subjects = set(pairs["subject"].astype(str))
        if rest_subjects & pair_subjects:
            raise AssertionError("Rest-only subjects leaked into frozen pairs.")
    byjo_usable = usable[usable["subject"].astype(str).eq("sub-BYJoWR")]
    if len(byjo_usable) != 1 or str(byjo_usable.iloc[0]["medoff_run"]) != "2" or str(byjo_usable.iloc[0]["medon_run"]) != "2":
        raise AssertionError("sub-BYJoWR HoldR run-2 pair was not selected.")
    jy = usable[
        usable["subject"].astype(str).eq("sub-jyC0j3")
        & usable["task"].astype(str).eq("MoveR")
    ]
    split_collapsed = bool(len(jy) == 1 and int(jy.iloc[0].get("medon_split_count", 0)) == 2)
    if not split_collapsed:
        raise AssertionError("sub-jyC0j3 MoveR MedOn split files were not collapsed logically.")
    if not failure_log.empty:
        raise AssertionError("Extraction failure log is not empty.")
    if int(frozen_summary.get("top_k", -1)) != TOP_K:
        raise AssertionError("top_k changed.")
    if not math.isclose(float(frozen_summary.get("aperiodic_alpha", float("nan"))), APERIODIC_ALPHA):
        raise AssertionError("aperiodic_alpha changed.")
    return {
        "complete_pairs": complete_pairs,
        "subjects_with_complete_pairs": subjects,
        "sub_BYJoWR_included": True,
        "sub_BYJoWR_holdr_run2_pair": True,
        "sub_BYJoWR_excluded": False,
        "rest_excluded": True,
        "split_files_collapsed_logically": split_collapsed,
        "downloaded_meg_subjects": int(cohort_summary.get("downloaded_meg_subjects", logical["subject"].nunique())),
        "downloaded_logical_holdmove_recordings": int(
            cohort_summary.get("downloaded_logical_holdmove_recordings", len(logical))
        ),
        "extraction_failure_log_rows": int(len(failure_log)),
        "top_k": TOP_K,
        "aperiodic_alpha": APERIODIC_ALPHA,
    }


def reverse_pairs_for_cycle(pairs: pd.DataFrame) -> pd.DataFrame:
    """Swap x_ and medon_ feature columns for MedOn-to-MedOff reverse retrieval."""

    out = pairs.copy()
    for column in list(out.columns):
        if not column.startswith("x_"):
            continue
        feature = column[2:]
        medon_col = f"medon_{feature}"
        if medon_col in out.columns:
            x_values = out[column].copy()
            out[column] = out[medon_col]
            out[medon_col] = x_values
    return out


def evaluate_forward_and_reverse_scores(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate frozen v2A and selected v2B over forward and reverse pools."""

    variant = selected_v2b_variant()
    forward_tables = []
    for condition in EVALUATED_CONDITIONS:
        candidate_pool = build_custom_candidate_pool(pairs, condition)
        _v2_scores, _v2_diag, v2a_scores, _v2a_diag = evaluate_frozen_on_candidate_pool(
            pairs, candidate_pool, v2_features, compact_features, clean_features
        )
        v2a_scores = v2a_scores.copy()
        v2a_scores["candidate_pool_condition"] = condition
        forward_tables.append(v2a_scores)
        v2b_scores, _v2b_diag, _weights = evaluate_v2b_variant(
            pairs, candidate_pool, v2_features, compact_features, clean_features, variant
        )
        v2b_scores = v2b_scores.copy()
        v2b_scores["candidate_pool_condition"] = condition
        forward_tables.append(v2b_scores)

    reversed_pairs = reverse_pairs_for_cycle(pairs)
    reverse_pool = build_custom_candidate_pool(reversed_pairs, ALL_MEDON_CONDITION)
    reverse_scores, _reverse_diag, _reverse_weights = evaluate_v2b_variant(
        reversed_pairs, reverse_pool, v2_features, compact_features, clean_features, variant
    )
    reverse_scores = reverse_scores.copy()
    reverse_scores["candidate_pool_condition"] = ALL_MEDON_CONDITION
    return normalize_scores(pd.concat(forward_tables, ignore_index=True)), normalize_scores(reverse_scores)


def write_summary_md(summary: dict[str, object], path: Path) -> Path:
    """Write concise Markdown summary."""

    lines = [
        "# v2D 18-subject BYJoWR-inclusive experiment",
        "",
        "Scope: v2B/v2C/v2D rerun on rebuilt 18-subject/34-pair ds004998 cache.",
        "",
        "## Validation",
        f"- complete_pairs: {summary['validation']['complete_pairs']}",
        f"- subjects_with_complete_pairs: {summary['validation']['subjects_with_complete_pairs']}",
        f"- sub-BYJoWR included: {summary['validation']['sub_BYJoWR_included']}",
        f"- sub-BYJoWR HoldR run-2 pair: {summary['validation']['sub_BYJoWR_holdr_run2_pair']}",
        f"- Rest excluded: {summary['validation']['rest_excluded']}",
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
    """Load and validate 18-subject frozen inputs."""

    pairs = read_csv_required(REQUIRED_FILES["pairs"], dtype={"run_off": str, "run_on": str})
    logical = read_csv_required(REQUIRED_FILES["logical_manifest"])
    usable = read_csv_required(REQUIRED_FILES["usable_pair_inventory"])
    excluded = read_csv_required(REQUIRED_FILES["excluded_subjects"])
    cohort_summary = read_json_required(REQUIRED_FILES["cohort_summary"])
    frozen_summary = read_json_required(REQUIRED_FILES["frozen_summary"])
    failure_log = read_csv_required(REQUIRED_FILES["extraction_failure_log"])
    validation = validate_inputs(pairs, logical, usable, excluded, cohort_summary, frozen_summary, failure_log)

    subspaces = load_subspace_inventory(REQUIRED_FILES["subspaces"])
    compact_inventory = read_json_required(REQUIRED_FILES["compact_inventory"])
    all_features = base_feature_names(pairs)
    v2_features = [feature for feature in subspaces.get("v2_reference", []) if feature in all_features]
    compact_features = [str(feature) for feature in compact_inventory.get("selected_features", []) if str(feature) in all_features]
    clean_features = [feature for feature in subspaces.get("clean_stable_features", []) if feature in all_features]
    if not v2_features or not compact_features:
        raise AssertionError("Required v2 or compact features are missing.")
    return {
        "pairs": pairs,
        "validation": validation,
        "v2_features": v2_features,
        "compact_features": compact_features,
        "clean_features": clean_features,
    }


def run() -> dict[str, object]:
    """Run BYJoWR-inclusive v2B/v2C/v2D experiments."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    forward_scores, reverse_scores = evaluate_forward_and_reverse_scores(
        inputs["pairs"],
        inputs["v2_features"],
        inputs["compact_features"],
        inputs["clean_features"],
    )
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
    comparison_v2b = comparison_to_baseline(metrics, V2B_NAME)
    comparison_v2c = comparison_to_baseline(metrics, V2C_DECONFOUNDED_NAME)
    comparison = pd.concat([comparison_v2b, comparison_v2c], ignore_index=True)
    bootstrap_summary, bootstrap_samples = subject_bootstrap_ci(diagnostics)
    failures = failure_cases(diagnostics)

    focus_variants = [V2A_NAME, V2B_NAME, V2C_DECONFOUNDED_NAME, V2D_QUALITY_DECONFOUNDED_NAME]
    key_metrics = metrics[
        metrics["candidate_pool_condition"].astype(str).isin([PRIMARY_CONDITION, HARD_NEGATIVE_CONDITION, ALL_MEDON_CONDITION])
        & metrics["variant_name"].astype(str).isin(focus_variants)
    ].sort_values(["candidate_pool_condition", "variant_name"])
    warnings = [
        "18-subject outputs are BYJoWR-inclusive expanded-cohort experiments; original 17-subject frozen outputs are unchanged.",
        "sub-BYJoWR contributes one complete HoldR run-2 pair; its MoveR group remains incomplete and is not paired.",
        "No Oxford/MRC external dataset was used.",
        "v2B/v2C/v2D remain experimental; no clinical prediction, treatment, DBS, or causal medication-effect claim is made.",
        "sub-BYJoWR quality_flag is unknown because the legacy quality table has no subject/task entry for it.",
    ]
    summary = {
        "validation": inputs["validation"],
        "key_metrics": key_metrics.to_dict("records"),
        "all_medon_comparison_to_v2b": comparison_v2b[
            comparison_v2b["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        ].to_dict("records"),
        "warnings": warnings,
        "external_oxford_mrc_dataset_used": False,
        "frozen_v2A_changed": False,
        "not_clinical_prediction_or_treatment": True,
    }

    paths = {
        "summary_json": write_json(summary, OUTPUT_DIR / "v2d_18subjects_experiment_summary.json"),
        "metrics": write_csv(metrics, OUTPUT_DIR / "v2d_18subjects_metrics.csv"),
        "candidate_scores": write_csv(all_scores, OUTPUT_DIR / "v2d_18subjects_candidate_scores.csv"),
        "forward_scores": write_csv(forward_scores, OUTPUT_DIR / "v2b_18subjects_forward_candidate_scores.csv"),
        "reverse_scores": write_csv(reverse_scores, OUTPUT_DIR / "v2b_18subjects_reverse_candidate_scores.csv"),
        "query_diagnostics": write_csv(diagnostics, OUTPUT_DIR / "v2d_18subjects_query_diagnostics.csv"),
        "comparison": write_csv(comparison, OUTPUT_DIR / "v2d_18subjects_comparison_to_baselines.csv"),
        "bootstrap_ci": write_csv(bootstrap_summary, OUTPUT_DIR / "v2d_18subjects_bootstrap_ci.csv"),
        "bootstrap_samples": write_csv(bootstrap_samples, OUTPUT_DIR / "v2d_18subjects_bootstrap_samples.csv"),
        "failure_cases": write_csv(failures, OUTPUT_DIR / "v2d_18subjects_failure_cases.csv"),
        "kreciprocal_components": write_csv(components, OUTPUT_DIR / "v2d_18subjects_kreciprocal_components.csv"),
    }
    paths["summary_md"] = write_summary_md(to_builtin(summary), OUTPUT_DIR / "v2d_18subjects_experiment_summary.md")

    plot_metric(metrics, "top1", OUTPUT_DIR / "v2d_18subjects_top1_by_condition.png")
    plot_metric(metrics, "mrr", OUTPUT_DIR / "v2d_18subjects_mrr_by_condition.png")
    plot_all_medon_delta(comparison, OUTPUT_DIR / "v2d_18subjects_all_medon_delta_top1_vs_v2b.png")

    def metric_row(condition: str, variant: str) -> pd.Series:
        return metrics[metrics["candidate_pool_condition"].eq(condition) & metrics["variant_name"].eq(variant)].iloc[0]

    v2a_primary = metric_row(PRIMARY_CONDITION, V2A_NAME)
    v2b_all = metric_row(ALL_MEDON_CONDITION, V2B_NAME)
    v2c_all = metric_row(ALL_MEDON_CONDITION, V2C_DECONFOUNDED_NAME)
    v2d_all = metric_row(ALL_MEDON_CONDITION, V2D_QUALITY_DECONFOUNDED_NAME)
    v2d_true_plus = metric_row(TRUE_PLUS_OTHER_CONDITION, V2D_QUALITY_DECONFOUNDED_NAME)
    validation = inputs["validation"]
    print(f"complete_pairs: {validation['complete_pairs']}")
    print(f"subjects_with_complete_pairs: {validation['subjects_with_complete_pairs']}")
    print(f"sub-BYJoWR included: {validation['sub_BYJoWR_included']}")
    print(f"sub-BYJoWR HoldR run-2 pair: {validation['sub_BYJoWR_holdr_run2_pair']}")
    print(f"Rest excluded: {validation['rest_excluded']}")
    print(f"split files collapsed logically: {validation['split_files_collapsed_logically']}")
    print(f"top_k: {TOP_K}")
    print(f"aperiodic_alpha: {APERIODIC_ALPHA}")
    print(f"v2A primary top1/MRR: {float(v2a_primary['top1']):.3f}/{float(v2a_primary['mrr']):.3f}")
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
