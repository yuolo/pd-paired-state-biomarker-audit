"""Run frozen v2A retrieval on the local 17-subject/33-pair ds004998 cohort.

This script fixes only local ds004998 discovery, split-file handling, cohort
filtering, and manifest generation around the frozen v2A scorer. It does not
change feature definitions, scorer logic, top-k, alpha, thresholds, metrics,
or retrieval parameters. Rest recordings are excluded from the main Hold/Move
paired-state retrieval run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_retrieval_aperiodic_assisted_v2 import (  # noqa: E402
    AssistedVariantSpec,
    aper_distance_scores,
    evaluate_assisted_variant,
    same_subject_assisted,
)
from scripts.run_retrieval_distance_geometry_improvement import (  # noqa: E402
    VariantSpec,
    evaluate_variant as evaluate_v2_variant,
    query_change_log_for_variant,
    same_subject_hard_negative as same_subject_hard_negative_v2,
    summarize_diagnostics,
    to_builtin,
)
from scripts.run_retrieval_signal_level_compact_v3 import (  # noqa: E402
    COMPACT_APERIODIC_FEATURES,
    MIN_DISTRACTORS,
    build_compact_subspaces,
    build_paired_examples_all_features,
    merge_compact_features,
    signal_segment,
    spectral_params_from_data,
    task_side_columns,
)
from src.data_loading.load_ds004998_bids import (  # noqa: E402
    VALID_HOLDMOVE_TASKS,
    build_holdmove_usable_pair_inventory,
    build_logical_recording_manifest,
    inspect_recording,
    load_lfp_montage_mapping,
    read_raw_header,
    write_inspection_outputs,
)
from src.evaluation.predictive_rescue_analysis import (  # noqa: E402
    build_base_candidate_sets,
    read_quality_table,
    read_subspace_definitions,
)
from src.pathology_model.build_real_state_vectors import write_real_state_outputs  # noqa: E402
from src.preprocessing.channel_selection import (  # noqa: E402
    meg_picks,
    motor_cortical_proxy_picks,
    stn_lfp_picks,
)
from src.preprocessing.meg_lfp_preprocessing import (  # noqa: E402
    build_extraction_failure_log,
    empty_feature_table,
    empty_timing_table,
    extract_recording_features_timed,
)

RANDOM_SEED = 42
DATA_ROOT = Path(os.environ.get("DS004998_ROOT", "data/raw/ds004998"))
AUDIT_DIR = Path("outputs/ds004998_full_cohort_completeness_audit")
OUTPUT_DIR = Path("outputs/v2A_frozen_17subjects_33pairs")
TABLES_DIR = OUTPUT_DIR / "tables"
EXPECTED_COMPLETE_PAIRS = 33
EXPECTED_SUBJECTS_WITH_COMPLETE_PAIRS = 17
V2_SCORER_NAME = "group_balanced_cosine_full28_plus_coupling"
V2A_SCORER_NAME = "v2A_top5_aperiodic_rerank"
TOP_K = 5
APERIODIC_ALPHA = 0.5
DO_NOT_DOWNLOAD_DATA = True
EXPECTED_COMPLETE_GROUPS = {
    ("sub-0cGdk9", "HoldL"),
    ("sub-0cGdk9", "MoveL"),
    ("sub-2IU8mi", "HoldL"),
    ("sub-2IU8mi", "MoveL"),
    ("sub-2IhVOz", "HoldR"),
    ("sub-2IhVOz", "MoveR"),
    ("sub-AB2PeX", "HoldL"),
    ("sub-AB2PeX", "MoveL"),
    ("sub-AbzsOg", "HoldL"),
    ("sub-AbzsOg", "MoveL"),
    ("sub-FIyfdR", "HoldR"),
    ("sub-FIyfdR", "MoveR"),
    ("sub-FYbcap", "HoldL"),
    ("sub-FYbcap", "MoveL"),
    ("sub-PuPVlx", "HoldL"),
    ("sub-PuPVlx", "MoveL"),
    ("sub-QZTsn6", "HoldL"),
    ("sub-QZTsn6", "MoveL"),
    ("sub-VopvKx", "HoldR"),
    ("sub-VopvKx", "MoveR"),
    ("sub-dCsWjQ", "HoldL"),
    ("sub-dCsWjQ", "MoveL"),
    ("sub-gNX5yb", "HoldL"),
    ("sub-gNX5yb", "MoveL"),
    ("sub-hnetKS", "HoldL"),
    ("sub-hnetKS", "MoveL"),
    ("sub-i4oK0F", "HoldL"),
    ("sub-i4oK0F", "MoveL"),
    ("sub-iDpl28", "HoldR"),
    ("sub-jyC0j3", "HoldR"),
    ("sub-jyC0j3", "MoveR"),
    ("sub-oLNpHd", "HoldL"),
    ("sub-oLNpHd", "MoveL"),
}

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))


def write_json(data: dict[str, object], path: Path) -> Path:
    """Write JSON with parent directory creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(data), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write CSV with parent directory creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def inspect_selected_recordings(paths: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inspect only selected canonical logical recordings."""

    recording_rows: list[dict[str, object]] = []
    channel_tables: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []
    for path in paths:
        recording, channels, qc = inspect_recording(path)
        recording_rows.append(recording)
        if not channels.empty:
            channel_tables.append(channels)
        qc_rows.append(qc)
    recordings = pd.DataFrame(recording_rows)
    channels = pd.concat(channel_tables, ignore_index=True) if channel_tables else pd.DataFrame()
    qc = pd.DataFrame(qc_rows)
    return recordings, channels, qc


def extract_features_for_paths(
    paths: list[str],
    max_duration_sec: float,
    max_windows: int,
    n_jobs: int,
    parallel_backend: str,
    recording_batch_size: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract existing v2 feature windows for selected canonical recordings."""

    feature_tables: list[pd.DataFrame] = []
    qc_error_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    n_jobs = max(1, int(n_jobs or 1))
    backend = str(parallel_backend or "sequential").lower()

    def handle_result(result: tuple[pd.DataFrame, dict[str, object] | None, dict[str, object]]) -> None:
        features_one, error_record, timing = result
        timing_rows.append(timing)
        if not features_one.empty:
            feature_tables.append(features_one)
        if error_record is not None:
            qc_error_rows.append(error_record)

    if n_jobs == 1 or backend == "sequential":
        for path in paths:
            handle_result(extract_recording_features_timed(path, max_duration_sec=max_duration_sec, max_windows=max_windows))
    elif backend in {"process", "thread"}:
        executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
        batch_size = max(1, int(recording_batch_size or len(paths) or 1))
        for batch_start in range(0, len(paths), batch_size):
            batch = paths[batch_start : batch_start + batch_size]
            try:
                executor = executor_cls(max_workers=n_jobs)
            except PermissionError:
                executor = ThreadPoolExecutor(max_workers=n_jobs)
            with executor:
                futures = [
                    executor.submit(
                        extract_recording_features_timed,
                        path,
                        max_duration_sec,
                        max_windows,
                    )
                    for path in batch
                ]
                for future in as_completed(futures):
                    handle_result(future.result())
    else:
        raise ValueError(f"Unknown parallel_backend: {parallel_backend}")

    features = pd.concat(feature_tables, ignore_index=True) if feature_tables else empty_feature_table()
    qc_errors = pd.DataFrame(qc_error_rows)
    timing = pd.DataFrame(timing_rows) if timing_rows else empty_timing_table()
    return features, qc_errors, timing


def compact_aperiodic_for_recordings(recordings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute the existing compact aperiodic/residual v2A feature family."""

    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for _, rec in recordings.iterrows():
        path = Path(str(rec["file_path"]))
        try:
            raw = read_raw_header(path)
            mapping = load_lfp_montage_mapping(path)
            sfreq = float(raw.info["sfreq"])
            stn_data = signal_segment(raw, stn_lfp_picks(raw, lfp_mapping=mapping))
            motor_data = signal_segment(raw, motor_cortical_proxy_picks(raw))
            meg_data = signal_segment(raw, meg_picks(raw)[:64])
            task = str(rec.get("task", "unknown"))
            base = {
                "subject": rec.get("subject", ""),
                "session": rec.get("session", ""),
                "task": task,
                "medication": rec.get("medication", ""),
                "run": str(rec.get("run", "")),
                "condition": rec.get("condition", task),
                "task_original": task,
                "task_family": "Hold" if task.startswith("Hold") else "Move" if task.startswith("Move") else task,
                "side": "L" if task.endswith("L") else "R" if task.endswith("R") else "none",
                "file_path": str(path),
            }
            row = dict(base)
            row.update(spectral_params_from_data(stn_data, sfreq, "stn"))
            row.update(spectral_params_from_data(motor_data, sfreq, "motor"))
            row.update(spectral_params_from_data(meg_data, sfreq, "meg"))
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            errors.append({"file_path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    full = pd.DataFrame(rows)
    selected = [feature for feature in COMPACT_APERIODIC_FEATURES if feature in full.columns]
    id_cols = ["subject", "session", "task", "medication", "run", "condition", "task_original", "task_family", "side", "file_path"]
    compact = full[[*id_cols, *selected]].copy() if selected else full[id_cols].copy() if not full.empty else pd.DataFrame(columns=id_cols)
    inventory = {
        "available": bool(selected),
        "selected_features": selected,
        "missing_features": [feature for feature in COMPACT_APERIODIC_FEATURES if feature not in selected],
        "n_selected_features": int(len(selected)),
        "method": "frozen_v2A_compact_aperiodic_residual_features_from_signal_level_welch_psd",
        "n_state_rows": int(len(compact)),
        "extraction_errors": errors,
    }
    return compact, inventory


def validate_pair_counts(
    pairs: pd.DataFrame,
    allow_count_mismatch: bool,
    expected_pairs: int,
    expected_subjects: int,
) -> dict[str, object]:
    """Validate complete-pair inventory against frozen expected counts."""

    actual_pairs = int(len(pairs))
    actual_subjects = int(pairs["subject"].nunique()) if not pairs.empty else 0
    actual_groups = set(zip(pairs["subject"].astype(str), pairs["task"].astype(str), strict=False))
    missing_groups = sorted(EXPECTED_COMPLETE_GROUPS - actual_groups)
    unexpected_groups = sorted(actual_groups - EXPECTED_COMPLETE_GROUPS)
    summary = {
        "complete_pairs": actual_pairs,
        "subjects_with_complete_pairs": actual_subjects,
        "expected_complete_pairs": expected_pairs,
        "expected_subjects_with_complete_pairs": expected_subjects,
        "missing_expected_groups": [f"{subject},{task}" for subject, task in missing_groups],
        "unexpected_groups": [f"{subject},{task}" for subject, task in unexpected_groups],
        "allow_count_mismatch": bool(allow_count_mismatch),
    }
    mismatches = []
    if actual_pairs != expected_pairs:
        mismatches.append(f"complete_pairs={actual_pairs} expected={expected_pairs}")
    if actual_subjects != expected_subjects:
        mismatches.append(f"subjects_with_complete_pairs={actual_subjects} expected={expected_subjects}")
    if missing_groups or unexpected_groups:
        mismatches.append("complete group list differs from expected inventory")
    if mismatches and not allow_count_mismatch:
        raise SystemExit("Cohort sanity check failed: " + "; ".join(mismatches) + ". Use --allow-count-mismatch only for explicit diagnostics.")
    return summary


def evaluate_frozen_v2a(
    compact_state_vectors: pd.DataFrame,
    quality: pd.DataFrame,
    subspaces: dict[str, list[str]],
    compact_aperiodic_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Evaluate frozen v2 and v2A without changing retrieval logic."""

    pairs, pair_log = build_paired_examples_all_features(compact_state_vectors, quality)
    base_candidates = build_base_candidate_sets(pairs, MIN_DISTRACTORS)
    clean_features = [feature for feature in subspaces.get("clean_stable_features", []) if f"x_{feature}" in pairs.columns]
    v2_features = subspaces.get("v2_reference", [])
    v2_variant = VariantSpec(
        name="v2_reference",
        category="frozen_v2_candidate_generator",
        score_kind="group_balanced",
        feature_space="full_28_plus_coupling",
        distance="cosine",
    )
    v2_scores, v2_diag = evaluate_v2_variant(pairs, base_candidates, v2_features, v2_variant, clean_features)
    _, v2_hard_summary = same_subject_hard_negative_v2(pairs, v2_features, v2_variant, clean_features)
    aper_distance = aper_distance_scores(pairs, base_candidates, compact_aperiodic_features, "aperiodic_distance")
    v2a_spec = AssistedVariantSpec(
        name=V2A_SCORER_NAME,
        mode="two_stage",
        top_k=TOP_K,
        alpha=APERIODIC_ALPHA,
        note="Frozen v2 top-5 plus compact aperiodic/residual rerank; no tuning.",
    )
    v2a_scores, v2a_diag = evaluate_assisted_variant(v2_scores, aper_distance, v2a_spec)
    _, v2a_hard_summary = same_subject_assisted(pairs, v2_features, compact_aperiodic_features, clean_features, v2a_spec)
    v2_row = {
        "scorer": "v2_reference",
        "n_pairs": int(len(v2_diag)),
        **summarize_diagnostics(v2_diag),
        "same_subject_hard_negative_success": v2_hard_summary.get("true_beats_all_rate", np.nan),
    }
    v2a_row = {
        "scorer": V2A_SCORER_NAME,
        "n_pairs": int(len(v2a_diag)),
        **summarize_diagnostics(v2a_diag),
        "same_subject_hard_negative_success": v2a_hard_summary.get("true_beats_all_rate", np.nan),
    }
    change = query_change_log_for_variant(v2_diag, v2a_diag, V2A_SCORER_NAME)
    summary = {
        "v2": v2_row,
        "v2A": v2a_row,
        "fail_to_success_vs_v2": int(change["change_type"].eq("fail_to_success").sum()) if not change.empty else 0,
        "success_to_failure_vs_v2": int(change["change_type"].eq("success_to_failure").sum()) if not change.empty else 0,
        "top_k": TOP_K,
        "aperiodic_alpha": APERIODIC_ALPHA,
        "retrieval_logic_changed": False,
    }
    return pairs, pair_log, v2a_scores, v2a_diag, pd.DataFrame([v2_row, v2a_row]), summary, change


def run(args: argparse.Namespace) -> dict[str, Path]:
    """Run completeness audit, selected feature extraction, and frozen v2A."""

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    logical = build_logical_recording_manifest(args.data_root, require_existing=True)
    usable_pairs, incomplete, excluded = build_holdmove_usable_pair_inventory(logical, VALID_HOLDMOVE_TASKS)
    cohort_summary = validate_pair_counts(
        usable_pairs,
        args.allow_count_mismatch,
        args.expected_pairs,
        args.expected_subjects,
    )

    downloaded_subjects = int(logical["subject"].nunique()) if not logical.empty else 0
    holdmove_logical = logical[logical["task"].isin(VALID_HOLDMOVE_TASKS) & logical["acq"].isin(["MedOff", "MedOn"])].copy()
    audit_summary = {
        **cohort_summary,
        "downloaded_meg_subjects": downloaded_subjects,
        "downloaded_logical_holdmove_recordings": int(len(holdmove_logical)),
        "excluded_rest_only_subjects": excluded.loc[excluded["reason"].eq("rest_only_excluded_from_main_holdmove_v2a"), "subject"].astype(str).tolist(),
        "excluded_incomplete_holdmove_subjects": excluded.loc[excluded["reason"].eq("incomplete_holdmove_medoff_medon_groups"), "subject"].astype(str).tolist(),
        "do_not_download_data": DO_NOT_DOWNLOAD_DATA,
    }
    write_csv(logical, AUDIT_DIR / "local_logical_recording_manifest.csv")
    write_csv(usable_pairs, AUDIT_DIR / "usable_pair_inventory.csv")
    write_csv(incomplete, AUDIT_DIR / "incomplete_groups.csv")
    write_csv(excluded, AUDIT_DIR / "excluded_subjects.csv")
    write_json(audit_summary, AUDIT_DIR / "cohort_completeness_summary.json")

    selected_paths = sorted(
        set(usable_pairs["medoff_canonical_path"].astype(str)).union(set(usable_pairs["medon_canonical_path"].astype(str)))
    )
    recordings, channels, qc = inspect_selected_recordings(selected_paths)
    inspect_paths = write_inspection_outputs(recordings, channels, qc, TABLES_DIR)
    features, qc_feature_errors, timing = extract_features_for_paths(
        selected_paths,
        args.max_duration_sec,
        args.max_windows,
        args.n_jobs,
        args.parallel_backend,
        args.recording_batch_size,
    )
    if not qc_feature_errors.empty:
        qc = pd.concat([qc, qc_feature_errors], ignore_index=True)
    state_paths = write_real_state_outputs(features, TABLES_DIR)
    failure_log = build_extraction_failure_log(recordings, qc, features)
    write_csv(failure_log, TABLES_DIR / "extraction_failure_log.csv")
    write_csv(timing, TABLES_DIR / "extraction_timing_by_recording.csv")

    state_vectors = task_side_columns(pd.read_csv(state_paths["state_vectors_enhanced_coupling"], dtype={"run": str}))
    compact_aper, compact_aper_inventory = compact_aperiodic_for_recordings(recordings)
    write_csv(compact_aper, OUTPUT_DIR / "compact_aperiodic_features.csv")
    write_json(compact_aper_inventory, OUTPUT_DIR / "compact_aperiodic_feature_inventory.json")
    compact_state = merge_compact_features(state_vectors, [compact_aper])
    write_csv(compact_state, OUTPUT_DIR / "compact_v2a_state_vectors.csv")

    quality = read_quality_table(args.quality_table, [])
    existing_subspaces = read_subspace_definitions(args.subspace_definitions, compact_state, [])
    subspaces = build_compact_subspaces(existing_subspaces, compact_state, [], compact_aper_inventory.get("selected_features", []))
    write_json({"subspaces": subspaces}, OUTPUT_DIR / "frozen_v2a_subspace_inventory.json")

    pairs, pair_log, v2a_scores, v2a_diag, results, v2a_summary, change = evaluate_frozen_v2a(
        compact_state,
        quality,
        subspaces,
        list(compact_aper_inventory.get("selected_features", [])),
    )
    if int(len(pairs)) != args.expected_pairs and not args.allow_count_mismatch:
        raise SystemExit(f"Frozen v2A pair construction produced {len(pairs)} pairs; expected {args.expected_pairs}.")

    paths = {
        "logical_manifest": AUDIT_DIR / "local_logical_recording_manifest.csv",
        "usable_pairs": AUDIT_DIR / "usable_pair_inventory.csv",
        "incomplete": AUDIT_DIR / "incomplete_groups.csv",
        "excluded": AUDIT_DIR / "excluded_subjects.csv",
        "recording_manifest": inspect_paths["recording_manifest"],
        "state_vectors": OUTPUT_DIR / "compact_v2a_state_vectors.csv",
        "v2a_pairs": write_csv(pairs, OUTPUT_DIR / "frozen_v2a_pairs.csv"),
        "pair_log": write_csv(pair_log, OUTPUT_DIR / "frozen_v2a_pairing_log.csv"),
        "v2a_scores": write_csv(v2a_scores, OUTPUT_DIR / "frozen_v2a_candidate_scores.csv"),
        "v2a_diagnostics": write_csv(v2a_diag, OUTPUT_DIR / "frozen_v2a_query_diagnostics.csv"),
        "v2a_results": write_csv(results, OUTPUT_DIR / "frozen_v2a_results.csv"),
        "v2a_change_log": write_csv(change, OUTPUT_DIR / "frozen_v2a_query_change_log_vs_v2.csv"),
        "v2a_summary": write_json({**audit_summary, **v2a_summary}, OUTPUT_DIR / "frozen_v2a_summary.json"),
    }

    rest_only = ", ".join(audit_summary["excluded_rest_only_subjects"])
    incomplete_subjects = ", ".join(audit_summary["excluded_incomplete_holdmove_subjects"])
    v2a_row = v2a_summary["v2A"]
    print(f"downloaded MEG subjects: {downloaded_subjects}")
    print(f"downloaded logical Hold/Move recordings: {audit_summary['downloaded_logical_holdmove_recordings']}")
    print(f"usable Hold/Move subjects: {cohort_summary['subjects_with_complete_pairs']}")
    print(f"complete Hold/Move MedOff-MedOn pairs: {cohort_summary['complete_pairs']}")
    print(f"excluded Rest-only subjects: {rest_only}")
    print(f"excluded incomplete Hold/Move subject: {incomplete_subjects}")
    print(f"frozen v2A rerun completed on {int(v2a_row['n_pairs'])} pairs")
    print(f"frozen v2A top1/MRR: {float(v2a_row['top1']):.3f} / {float(v2a_row['mrr']):.3f}")
    print(f"output path: {OUTPUT_DIR}")
    return paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--quality-table", default="outputs/tables/real_recording_quality.csv")
    parser.add_argument("--subspace-definitions", default="outputs/tables/subspace_definitions.csv")
    parser.add_argument("--expected-pairs", type=int, default=EXPECTED_COMPLETE_PAIRS)
    parser.add_argument("--expected-subjects", type=int, default=EXPECTED_SUBJECTS_WITH_COMPLETE_PAIRS)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--max-duration-sec", type=float, default=120.0)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--parallel-backend", choices=["process", "thread", "sequential"], default="thread")
    parser.add_argument("--recording-batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Entry point."""

    run(parse_args())


if __name__ == "__main__":
    main()
