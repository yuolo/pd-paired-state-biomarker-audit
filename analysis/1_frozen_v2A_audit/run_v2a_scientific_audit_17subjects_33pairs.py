"""Scientific audit analyses for frozen v2A on ds004998 Hold/Move pairs.

This script consumes the cached frozen-v2A 17-subject/33-pair outputs and
adds audit controls, diagnostic feature-family ablations, and true-vs-wrong
transition contrasts. It does not read raw FIF files, does not use external
Oxford/MRC data, and does not change the frozen retrieval settings.
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
    query_diagnostics_from_scores,
    same_subject_hard_negative,
    summarize_diagnostics,
    to_builtin,
)
from scripts.run_retrieval_signal_level_compact_v3 import MIN_DISTRACTORS  # noqa: E402
from src.evaluation.predictive_rescue_analysis import (  # noqa: E402
    build_base_candidate_sets,
    fit_train_scaler,
    transform,
)

try:  # pragma: no cover - optional dependency in some lightweight installs.
    from scipy.stats import wilcoxon
except Exception:  # noqa: BLE001
    wilcoxon = None


RANDOM_SEED = 42
N_PERM = 5000
N_BOOT = 5000
TOP_K = 5
APERIODIC_ALPHA = 0.5

FROZEN_DIR = Path("outputs/v2A_frozen_17subjects_33pairs")
COHORT_DIR = Path("outputs/ds004998_full_cohort_completeness_audit")
OUTPUT_DIR = Path("outputs/v2A_scientific_audit_17subjects_33pairs")

EXPECTED_COMPLETE_PAIRS = 33
EXPECTED_SUBJECTS_WITH_COMPLETE_PAIRS = 17
EXPECTED_DOWNLOADED_SUBJECTS = 20
EXPECTED_LOGICAL_HOLDMOVE_RECORDINGS = 68
V2_NAME = "v2_reference"
V2A_NAME = "v2A_top5_aperiodic_rerank"

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

REQUIRED_CACHE_FILES = {
    "pairs": FROZEN_DIR / "frozen_v2a_pairs.csv",
    "candidate_scores": FROZEN_DIR / "frozen_v2a_candidate_scores.csv",
    "subspaces": FROZEN_DIR / "frozen_v2a_subspace_inventory.json",
    "compact_inventory": FROZEN_DIR / "compact_aperiodic_feature_inventory.json",
    "frozen_summary": FROZEN_DIR / "frozen_v2a_summary.json",
    "logical_manifest": COHORT_DIR / "local_logical_recording_manifest.csv",
    "usable_pair_inventory": COHORT_DIR / "usable_pair_inventory.csv",
    "excluded_subjects": COHORT_DIR / "excluded_subjects.csv",
    "cohort_summary": COHORT_DIR / "cohort_completeness_summary.json",
    "extraction_failure_log": FROZEN_DIR / "tables" / "extraction_failure_log.csv",
}


def write_json(data: dict[str, object], path: Path) -> Path:
    """Write JSON with stable conversion of numpy/pandas scalars."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(data), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(data: pd.DataFrame, path: Path) -> Path:
    """Write a CSV with parent directory creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return path


def write_text(lines: Iterable[str], path: Path) -> Path:
    """Write a text or Markdown report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_json_required(path: Path) -> dict[str, object]:
    """Read a required JSON file."""

    if not path.exists():
        raise FileNotFoundError(f"Required cached file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_required(path: Path, **kwargs: object) -> pd.DataFrame:
    """Read a required CSV file."""

    if not path.exists():
        raise FileNotFoundError(f"Required cached file is missing: {path}")
    return pd.read_csv(path, **kwargs)


def as_bool(series: pd.Series) -> pd.Series:
    """Convert CSV bool-like values to bool."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def empirical_p_larger(observed: float, null_values: pd.Series) -> float:
    """One-sided empirical p-value for larger-is-better metrics."""

    values = pd.to_numeric(null_values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(observed):
        return float("nan")
    return float((1 + np.sum(values >= observed)) / (1 + len(values)))


def percentile_ci(values: pd.Series | np.ndarray, low: float = 2.5, high: float = 97.5) -> tuple[float, float]:
    """Return percentile CI with finite-value filtering."""

    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, low)), float(np.percentile(arr, high))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance with a stable zero-vector fallback."""

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12 or not np.isfinite(denom):
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def rank_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank candidate scores within query and variant."""

    ranked = scores.sort_values(["variant_name", "query_pair_id", "score"], ascending=True).copy()
    ranked["rank"] = ranked.groupby(["variant_name", "query_pair_id"], dropna=False).cumcount() + 1
    return ranked


def base_feature_names(pairs: pd.DataFrame) -> list[str]:
    """Return feature names present as x_, medon_, and y_ pair columns."""

    features = []
    for column in pairs.columns:
        if not column.startswith("x_"):
            continue
        feature = column[2:]
        if f"medon_{feature}" in pairs.columns and f"y_{feature}" in pairs.columns:
            features.append(feature)
    return features


def pair_value(row: pd.Series, prefix: str, features: list[str]) -> np.ndarray:
    """Extract a feature vector from one pair row."""

    return row[[f"{prefix}{feature}" for feature in features]].to_numpy(dtype=float)


def rebuild_v2_scores_from_frozen_v2a(cached_scores: pd.DataFrame) -> pd.DataFrame:
    """Recover the frozen v2 candidate-generator scores from cached v2A rows."""

    if "v2_score" not in cached_scores.columns:
        raise ValueError("frozen_v2a_candidate_scores.csv does not contain v2_score.")
    v2 = cached_scores.copy()
    v2["variant_name"] = V2_NAME
    v2["score"] = pd.to_numeric(v2["v2_score"], errors="coerce")
    for column in ["is_true_pair", "residualized", "true_in_v2_top_k"]:
        if column in v2:
            v2[column] = as_bool(v2[column])
    v2["variant_category"] = "frozen_v2_candidate_generator"
    v2["score_kind"] = "group_balanced"
    v2["distance"] = "cosine"
    v2["feature_space"] = "full_28_plus_coupling"
    v2 = v2.drop(columns=["rank"], errors="ignore")
    return rank_scores(v2)


def normalize_cached_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Normalize cached candidate score dtypes."""

    out = scores.copy()
    for column in ["is_true_pair", "residualized", "true_in_v2_top_k"]:
        if column in out:
            out[column] = as_bool(out[column])
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce").astype(int)
    return out


def query_metrics(diagnostics: pd.DataFrame) -> dict[str, object]:
    """Compute top-k, rank, and failure metrics from query diagnostics."""

    if diagnostics.empty:
        return {
            "n_pairs": 0,
            "top1": float("nan"),
            "mrr": float("nan"),
            "top3": float("nan"),
            "top5": float("nan"),
            "percentile_rank": float("nan"),
            "mean_true_rank": float("nan"),
            "median_true_rank": float("nan"),
            "failures": 0,
            "retrieval_margin": float("nan"),
        }
    ranks = pd.to_numeric(diagnostics["true_medon_rank"], errors="coerce")
    success = diagnostics["top_ranked_is_true_pair"].astype(bool)
    return {
        "n_pairs": int(len(diagnostics)),
        "top1": float(success.mean()),
        "mrr": float(pd.to_numeric(diagnostics["reciprocal_rank"], errors="coerce").mean()),
        "top3": float((ranks <= 3).mean()),
        "top5": float((ranks <= 5).mean()),
        "percentile_rank": float(pd.to_numeric(diagnostics["percentile_rank"], errors="coerce").mean()),
        "mean_true_rank": float(ranks.mean()),
        "median_true_rank": float(ranks.median()),
        "failures": int((~success).sum()),
        "retrieval_margin": float(pd.to_numeric(diagnostics["retrieval_margin"], errors="coerce").mean()),
    }


def metrics_from_rank_records(records: list[dict[str, float]]) -> dict[str, float]:
    """Summarize metrics from records with rank and candidate count."""

    frame = pd.DataFrame(records)
    if frame.empty:
        return {"top1": float("nan"), "mrr": float("nan"), "percentile_rank": float("nan")}
    return {
        "top1": float(frame["top1"].mean()),
        "mrr": float(frame["mrr"].mean()),
        "percentile_rank": float(frame["percentile_rank"].mean()),
    }


def build_score_lookup(candidate_scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group candidate scores by query id."""

    return {str(query_id): group.copy() for query_id, group in candidate_scores.groupby("query_pair_id", dropna=False)}


def validate_cached_inputs(
    pairs: pd.DataFrame,
    logical: pd.DataFrame,
    usable: pd.DataFrame,
    excluded: pd.DataFrame,
    cohort_summary: dict[str, object],
    frozen_summary: dict[str, object],
    extraction_failure_log: pd.DataFrame,
) -> dict[str, object]:
    """Run pre-analysis assertions for the verified expanded cohort."""

    missing = [str(path) for path in REQUIRED_CACHE_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required cached frozen/cohort files: " + "; ".join(missing))

    complete_pairs = int(len(pairs))
    subjects = int(pairs["subject"].nunique())
    groups = set(zip(pairs["subject"].astype(str), pairs["task_original"].astype(str), strict=False))
    missing_groups = sorted(EXPECTED_COMPLETE_GROUPS - groups)
    unexpected_groups = sorted(groups - EXPECTED_COMPLETE_GROUPS)

    if complete_pairs != EXPECTED_COMPLETE_PAIRS:
        raise AssertionError(f"complete_pairs={complete_pairs}, expected {EXPECTED_COMPLETE_PAIRS}")
    if subjects != EXPECTED_SUBJECTS_WITH_COMPLETE_PAIRS:
        raise AssertionError(f"subjects_with_complete_pairs={subjects}, expected {EXPECTED_SUBJECTS_WITH_COMPLETE_PAIRS}")
    if missing_groups or unexpected_groups:
        raise AssertionError(f"Complete group inventory mismatch: missing={missing_groups}, unexpected={unexpected_groups}")
    if pairs["task_original"].astype(str).str.contains("Rest", case=False, na=False).any():
        raise AssertionError("Rest rows are present in frozen pairs.")
    if usable["task"].astype(str).str.contains("Rest", case=False, na=False).any():
        raise AssertionError("Rest rows are present in usable Hold/Move inventory.")

    excluded_subjects = set(excluded.get("subject", pd.Series(dtype=str)).astype(str))
    if "sub-BYJoWR" not in excluded_subjects:
        raise AssertionError("sub-BYJoWR is not marked excluded.")
    rest_only = set(
        excluded.loc[
            excluded.get("reason", pd.Series(dtype=str)).astype(str).eq("rest_only_excluded_from_main_holdmove_v2a"),
            "subject",
        ].astype(str)
    )
    if not {"sub-6m9kB5", "sub-8RgPiG"}.issubset(rest_only):
        raise AssertionError("Rest-only exclusions are incomplete.")

    holdmove = logical[logical["task"].astype(str).isin(["HoldL", "HoldR", "MoveL", "MoveR"])]
    holdmove = holdmove[holdmove["acq"].astype(str).isin(["MedOff", "MedOn"])]
    if int(len(holdmove)) != EXPECTED_LOGICAL_HOLDMOVE_RECORDINGS:
        raise AssertionError(
            f"logical Hold/Move recordings={len(holdmove)}, expected {EXPECTED_LOGICAL_HOLDMOVE_RECORDINGS}"
        )
    if int(logical["subject"].nunique()) != EXPECTED_DOWNLOADED_SUBJECTS:
        raise AssertionError(f"downloaded MEG subjects={logical['subject'].nunique()}, expected {EXPECTED_DOWNLOADED_SUBJECTS}")

    jy = logical[
        logical["subject"].astype(str).eq("sub-jyC0j3")
        & logical["task"].astype(str).eq("MoveR")
        & logical["acq"].astype(str).eq("MedOn")
    ]
    if len(jy) != 1 or int(jy.iloc[0].get("physical_file_count", 0)) != 2:
        raise AssertionError("sub-jyC0j3 MoveR MedOn split files were not collapsed into one logical recording.")
    jy_pair = usable[usable["subject"].astype(str).eq("sub-jyC0j3") & usable["task"].astype(str).eq("MoveR")]
    if len(jy_pair) != 1 or int(jy_pair.iloc[0].get("medon_split_count", 0)) != 2:
        raise AssertionError("sub-jyC0j3 MoveR MedOn split count is not preserved in usable pair inventory.")

    if TOP_K != 5:
        raise AssertionError(f"top_k changed: {TOP_K}")
    if not math.isclose(APERIODIC_ALPHA, 0.5):
        raise AssertionError(f"aperiodic_alpha changed: {APERIODIC_ALPHA}")
    if int(frozen_summary.get("top_k", -1)) != TOP_K:
        raise AssertionError("Cached frozen summary top_k does not match TOP_K.")
    if not math.isclose(float(frozen_summary.get("aperiodic_alpha", float("nan"))), APERIODIC_ALPHA):
        raise AssertionError("Cached frozen summary aperiodic_alpha does not match APERIODIC_ALPHA.")
    if not extraction_failure_log.empty:
        raise AssertionError("Extraction failure log is not 0 rows.")

    return {
        "complete_pairs": complete_pairs,
        "subjects_with_complete_pairs": subjects,
        "downloaded_meg_subjects": int(logical["subject"].nunique()),
        "downloaded_logical_holdmove_recordings": int(len(holdmove)),
        "rest_excluded": True,
        "sub_BYJoWR_excluded": True,
        "split_files_collapsed_logically": True,
        "sub_jyC0j3_MoveR_MedOn_physical_files": 2,
        "extraction_failure_log_rows": int(len(extraction_failure_log)),
        "top_k": TOP_K,
        "aperiodic_alpha": APERIODIC_ALPHA,
        "cohort_summary_complete_pairs": int(cohort_summary.get("complete_pairs", complete_pairs)),
        "cohort_summary_subjects": int(cohort_summary.get("subjects_with_complete_pairs", subjects)),
        "external_dataset_used": False,
    }


def observed_metrics_table(v2_diag: pd.DataFrame, v2a_diag: pd.DataFrame) -> pd.DataFrame:
    """Build observed v2/v2A metric table."""

    change = query_change_log_for_variant(v2_diag, v2a_diag, V2A_NAME)
    fail_to_success = int(change["change_type"].eq("fail_to_success").sum()) if not change.empty else 0
    success_to_failure = int(change["change_type"].eq("success_to_failure").sum()) if not change.empty else 0
    rows = []
    for name, diag in [(V2_NAME, v2_diag), (V2A_NAME, v2a_diag)]:
        row = {"variant_name": name, **query_metrics(diag)}
        failures = diag[~diag["top_ranked_is_true_pair"].astype(bool)]
        counts = failures["failure_type"].fillna("unknown").astype(str).value_counts().sort_index().to_dict()
        row["failure_type_counts_json"] = json.dumps({str(k): int(v) for k, v in counts.items()}, sort_keys=True)
        for failure_type in [
            "other_subject_same_task_side",
            "other_subject_wrong_task",
            "quality_related",
            "same_subject_wrong_side",
            "same_subject_wrong_task",
            "unclear",
        ]:
            row[f"failure_type_{failure_type}"] = int(counts.get(failure_type, 0))
        row["v2_failures"] = int((~v2_diag["top_ranked_is_true_pair"].astype(bool)).sum())
        row["v2A_failures"] = int((~v2a_diag["top_ranked_is_true_pair"].astype(bool)).sum())
        row["v2_failure_to_v2A_success"] = fail_to_success
        row["v2_success_to_v2A_failure"] = success_to_failure
        rows.append(row)
    return pd.DataFrame(rows)


def random_label_null(
    diagnostics_by_variant: dict[str, pd.DataFrame],
    scores_by_variant: dict[str, pd.DataFrame],
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random-label null over global MedOn labels."""

    rng = np.random.default_rng(seed)
    sample_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for variant, diagnostics in diagnostics_by_variant.items():
        scores = scores_by_variant[variant]
        query_ids = diagnostics["query_pair_id"].astype(str).tolist()
        all_pair_ids = sorted(scores["candidate_pair_id"].astype(str).unique())
        grouped = build_score_lookup(scores)
        for permutation_index in range(n_perm):
            shuffled = rng.permutation(all_pair_ids)
            records: list[dict[str, float]] = []
            for idx, query_id in enumerate(query_ids):
                group = grouped[query_id]
                pseudo_id = str(shuffled[idx])
                pseudo = group[group["candidate_pair_id"].astype(str).eq(pseudo_id)]
                n_candidates = float(group["number_of_candidates"].iloc[0])
                if pseudo.empty:
                    rank = n_candidates + 1.0
                else:
                    rank = float(pseudo.iloc[0]["rank"])
                denom = max(1.0, n_candidates - 1.0)
                records.append(
                    {
                        "top1": float(rank == 1.0),
                        "mrr": 1.0 / rank,
                        "percentile_rank": max(0.0, 1.0 - ((rank - 1.0) / denom)),
                    }
                )
            metrics = metrics_from_rank_records(records)
            sample_rows.append({"variant_name": variant, "permutation_index": permutation_index, **metrics})
        null = pd.DataFrame([row for row in sample_rows if row["variant_name"] == variant])
        observed = query_metrics(diagnostics)
        summary_rows.append(
            {
                "variant_name": variant,
                "n_permutations": int(n_perm),
                "seed": int(seed),
                "observed_top1": observed["top1"],
                "null_mean_top1": float(null["top1"].mean()),
                "empirical_p_top1": empirical_p_larger(float(observed["top1"]), null["top1"]),
                "observed_mrr": observed["mrr"],
                "null_mean_mrr": float(null["mrr"].mean()),
                "empirical_p_mrr": empirical_p_larger(float(observed["mrr"]), null["mrr"]),
                "observed_percentile_rank": observed["percentile_rank"],
                "null_mean_percentile_rank": float(null["percentile_rank"].mean()),
                "empirical_p_percentile_rank": empirical_p_larger(
                    float(observed["percentile_rank"]), null["percentile_rank"]
                ),
                "null_method": "global random MedOn label shuffle; absent labels assigned unretrieved rank",
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def eligible_matched_null_candidates(group: pd.DataFrame) -> tuple[list[str], str]:
    """Return matched null candidate ids, excluding the true pair."""

    query_task = str(group["query_task_original"].iloc[0])
    query_family = str(group["query_task_family"].iloc[0])
    query_side = str(group["query_side"].iloc[0])
    distractors = group[~group["is_true_pair"].astype(bool)].copy()
    if distractors.empty:
        return [], "unavailable_no_distractors"
    exact = distractors[
        distractors["candidate_task_original"].astype(str).eq(query_task)
        & distractors["candidate_side"].astype(str).eq(query_side)
    ]
    if not exact.empty:
        return exact["candidate_pair_id"].astype(str).tolist(), "task_original_side"
    family = distractors[
        distractors["candidate_task_family"].astype(str).eq(query_family)
        & distractors["candidate_side"].astype(str).eq(query_side)
    ]
    if not family.empty:
        return family["candidate_pair_id"].astype(str).tolist(), "task_family_side"
    return distractors["candidate_pair_id"].astype(str).tolist(), "candidate_pool_fallback"


def matched_task_side_null(
    diagnostics_by_variant: dict[str, pd.DataFrame],
    scores_by_variant: dict[str, pd.DataFrame],
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matched task/side null that excludes the true pair as pseudo-positive."""

    rng = np.random.default_rng(seed)
    sample_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for variant, diagnostics in diagnostics_by_variant.items():
        scores = scores_by_variant[variant]
        grouped = build_score_lookup(scores)
        query_ids = diagnostics["query_pair_id"].astype(str).tolist()
        eligibles: dict[str, tuple[list[str], str]] = {
            query_id: eligible_matched_null_candidates(grouped[query_id]) for query_id in query_ids
        }
        level_counts = pd.Series([level for _, level in eligibles.values()]).value_counts().to_dict()
        unavailable = [query_id for query_id, (ids, _) in eligibles.items() if not ids]
        for permutation_index in range(n_perm):
            records: list[dict[str, float]] = []
            for query_id in query_ids:
                group = grouped[query_id]
                ids, _level = eligibles[query_id]
                n_candidates = float(group["number_of_candidates"].iloc[0])
                if not ids:
                    rank = n_candidates + 1.0
                else:
                    pseudo_id = str(rng.choice(ids))
                    pseudo = group[group["candidate_pair_id"].astype(str).eq(pseudo_id)]
                    rank = float(pseudo.iloc[0]["rank"]) if not pseudo.empty else n_candidates + 1.0
                denom = max(1.0, n_candidates - 1.0)
                records.append(
                    {
                        "top1": float(rank == 1.0),
                        "mrr": 1.0 / rank,
                        "percentile_rank": max(0.0, 1.0 - ((rank - 1.0) / denom)),
                    }
                )
            metrics = metrics_from_rank_records(records)
            sample_rows.append({"variant_name": variant, "permutation_index": permutation_index, **metrics})
        null = pd.DataFrame([row for row in sample_rows if row["variant_name"] == variant])
        observed = query_metrics(diagnostics)
        summary_rows.append(
            {
                "variant_name": variant,
                "n_permutations": int(n_perm),
                "seed": int(seed),
                "observed_top1": observed["top1"],
                "null_mean_top1": float(null["top1"].mean()),
                "empirical_p_top1": empirical_p_larger(float(observed["top1"]), null["top1"]),
                "observed_mrr": observed["mrr"],
                "null_mean_mrr": float(null["mrr"].mean()),
                "empirical_p_mrr": empirical_p_larger(float(observed["mrr"]), null["mrr"]),
                "observed_percentile_rank": observed["percentile_rank"],
                "null_mean_percentile_rank": float(null["percentile_rank"].mean()),
                "unavailable_queries": int(len(unavailable)),
                "eligible_level_counts_json": json.dumps({str(k): int(v) for k, v in level_counts.items()}, sort_keys=True),
                "null_method": "within candidate pool matched task/side pseudo-positive; true pair excluded",
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def subject_bootstrap_samples(
    v2_diag: pd.DataFrame,
    v2a_diag: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subject-aware bootstrap for v2A metrics and v2A-v2 deltas."""

    rng = np.random.default_rng(seed)
    subjects = sorted(v2a_diag["query_subject"].astype(str).unique())
    v2_by_subject = {subject: v2_diag[v2_diag["query_subject"].astype(str).eq(subject)].copy() for subject in subjects}
    v2a_by_subject = {subject: v2a_diag[v2a_diag["query_subject"].astype(str).eq(subject)].copy() for subject in subjects}
    sample_rows = []
    for bootstrap_index in range(n_boot):
        draw = rng.choice(subjects, size=len(subjects), replace=True)
        boot_v2 = pd.concat([v2_by_subject[str(subject)] for subject in draw], ignore_index=True)
        boot_v2a = pd.concat([v2a_by_subject[str(subject)] for subject in draw], ignore_index=True)
        v2_metrics = query_metrics(boot_v2)
        v2a_metrics = query_metrics(boot_v2a)
        sample_rows.append(
            {
                "bootstrap_index": bootstrap_index,
                "n_subjects_sampled": len(draw),
                "n_query_rows": int(len(boot_v2a)),
                "v2_top1": v2_metrics["top1"],
                "v2_mrr": v2_metrics["mrr"],
                "v2_percentile_rank": v2_metrics["percentile_rank"],
                "v2A_top1": v2a_metrics["top1"],
                "v2A_mrr": v2a_metrics["mrr"],
                "v2A_percentile_rank": v2a_metrics["percentile_rank"],
                "delta_top1_v2A_minus_v2": float(v2a_metrics["top1"]) - float(v2_metrics["top1"]),
                "delta_mrr_v2A_minus_v2": float(v2a_metrics["mrr"]) - float(v2_metrics["mrr"]),
                "delta_percentile_rank_v2A_minus_v2": float(v2a_metrics["percentile_rank"])
                - float(v2_metrics["percentile_rank"]),
            }
        )
    samples = pd.DataFrame(sample_rows)
    observed_v2 = query_metrics(v2_diag)
    observed_v2a = query_metrics(v2a_diag)
    summary_rows = []
    metric_observed = {
        "v2A_top1": observed_v2a["top1"],
        "v2A_mrr": observed_v2a["mrr"],
        "v2A_percentile_rank": observed_v2a["percentile_rank"],
        "delta_top1_v2A_minus_v2": float(observed_v2a["top1"]) - float(observed_v2["top1"]),
        "delta_mrr_v2A_minus_v2": float(observed_v2a["mrr"]) - float(observed_v2["mrr"]),
        "delta_percentile_rank_v2A_minus_v2": float(observed_v2a["percentile_rank"])
        - float(observed_v2["percentile_rank"]),
    }
    for metric, observed in metric_observed.items():
        ci_low, ci_high = percentile_ci(samples[metric])
        summary_rows.append(
            {
                "metric": metric,
                "observed": observed,
                "bootstrap_mean": float(samples[metric].mean()),
                "ci_lower_95": ci_low,
                "ci_upper_95": ci_high,
                "n_bootstrap": int(n_boot),
                "seed": int(seed),
                "bootstrap_unit": "subject",
            }
        )
    return pd.DataFrame(summary_rows), samples


def evaluate_frozen_subset(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute frozen v2 and v2A on a provided subset."""

    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    base_candidates = build_base_candidate_sets(pairs, MIN_DISTRACTORS)
    v2_variant = VariantSpec(
        name=V2_NAME,
        category="frozen_v2_candidate_generator",
        score_kind="group_balanced",
        feature_space="full_28_plus_coupling",
        distance="cosine",
    )
    v2_scores, v2_diag = evaluate_v2_variant(pairs, base_candidates, v2_features, v2_variant, clean_features)
    aper_scores = aper_distance_scores(pairs, base_candidates, compact_features, "aperiodic_distance")
    v2a_spec = AssistedVariantSpec(
        name=V2A_NAME,
        mode="two_stage",
        top_k=TOP_K,
        alpha=APERIODIC_ALPHA,
        note="Frozen v2 top-5 plus compact aperiodic/residual rerank; no tuning.",
    )
    v2a_scores, v2a_diag = evaluate_assisted_variant(v2_scores, aper_scores, v2a_spec)
    return v2_scores, v2_diag, v2a_scores, v2a_diag


def loso_sensitivity(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
    observed_v2a: dict[str, object],
) -> pd.DataFrame:
    """Remove one subject at a time and recompute v2/v2A metrics."""

    rows = []
    collapse_rule = "v2A_top1_drop>=0.25 or v2A_mrr_drop>=0.20 or v2A_top1<=0.50"
    observed_top1 = float(observed_v2a["top1"])
    observed_mrr = float(observed_v2a["mrr"])
    for subject in sorted(pairs["subject"].astype(str).unique()):
        subset = pairs[~pairs["subject"].astype(str).eq(subject)].copy()
        _v2_scores, v2_diag, _v2a_scores, v2a_diag = evaluate_frozen_subset(
            subset, v2_features, compact_features, clean_features
        )
        v2_metrics = query_metrics(v2_diag)
        v2a_metrics = query_metrics(v2a_diag)
        top1_drop = observed_top1 - float(v2a_metrics["top1"])
        mrr_drop = observed_mrr - float(v2a_metrics["mrr"])
        rows.append(
            {
                "removed_subject": subject,
                "n_pairs_remaining": int(len(subset)),
                "n_subjects_remaining": int(subset["subject"].nunique()),
                "v2_top1": v2_metrics["top1"],
                "v2_mrr": v2_metrics["mrr"],
                "v2A_top1": v2a_metrics["top1"],
                "v2A_mrr": v2a_metrics["mrr"],
                "v2A_minus_v2_top1": float(v2a_metrics["top1"]) - float(v2_metrics["top1"]),
                "v2A_minus_v2_mrr": float(v2a_metrics["mrr"]) - float(v2_metrics["mrr"]),
                "v2A_top1_drop_from_observed": top1_drop,
                "v2A_mrr_drop_from_observed": mrr_drop,
                "collapse_flag": bool(top1_drop >= 0.25 or mrr_drop >= 0.20 or float(v2a_metrics["top1"]) <= 0.50),
                "collapse_rule": collapse_rule,
            }
        )
    return pd.DataFrame(rows)


def task_side_sensitivity(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
) -> pd.DataFrame:
    """Evaluate Hold, Move, Left, and Right subsets."""

    subsets = {
        "Hold_only": pairs[pairs["task_family"].astype(str).eq("Hold")].copy(),
        "Move_only": pairs[pairs["task_family"].astype(str).eq("Move")].copy(),
        "Left_only": pairs[pairs["side"].astype(str).eq("L")].copy(),
        "Right_only": pairs[pairs["side"].astype(str).eq("R")].copy(),
    }
    rows = []
    for label, subset in subsets.items():
        if subset.empty:
            rows.append({"subset": label, "status": "skipped_empty", "n_pairs": 0, "n_subjects": 0})
            continue
        _v2_scores, v2_diag, _v2a_scores, v2a_diag = evaluate_frozen_subset(
            subset, v2_features, compact_features, clean_features
        )
        v2_metrics = query_metrics(v2_diag)
        v2a_metrics = query_metrics(v2a_diag)
        rows.append(
            {
                "subset": label,
                "status": "ok",
                "n_pairs": int(len(subset)),
                "n_subjects": int(subset["subject"].nunique()),
                "v2_top1": v2_metrics["top1"],
                "v2_mrr": v2_metrics["mrr"],
                "v2A_top1": v2a_metrics["top1"],
                "v2A_mrr": v2a_metrics["mrr"],
                "v2A_minus_v2_top1": float(v2a_metrics["top1"]) - float(v2_metrics["top1"]),
                "v2A_minus_v2_mrr": float(v2a_metrics["mrr"]) - float(v2_metrics["mrr"]),
            }
        )
    return pd.DataFrame(rows)


def hard_negative_outputs(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run same-subject hard-negative controls for v2 and v2A."""

    v2_variant = VariantSpec(
        name=V2_NAME,
        category="frozen_v2_candidate_generator",
        score_kind="group_balanced",
        feature_space="full_28_plus_coupling",
        distance="cosine",
    )
    v2_cases, v2_summary = same_subject_hard_negative(pairs, v2_features, v2_variant, clean_features)
    v2a_spec = AssistedVariantSpec(
        name=V2A_NAME,
        mode="two_stage",
        top_k=TOP_K,
        alpha=APERIODIC_ALPHA,
        note="Frozen v2 top-5 plus compact aperiodic/residual rerank; no tuning.",
    )
    v2a_cases, v2a_summary = same_subject_assisted(pairs, v2_features, compact_features, clean_features, v2a_spec)
    case_tables = []
    for table, variant in [(v2_cases, V2_NAME), (v2a_cases, V2A_NAME)]:
        if table.empty:
            continue
        table = table.copy()
        table["variant_name"] = variant
        table["hard_negative_failure"] = ~table["true_beats_all_same_subject_negatives"].astype(bool)
        case_tables.append(table)
    cases = pd.concat(case_tables, ignore_index=True) if case_tables else pd.DataFrame()
    summary_rows = []
    for variant, summary in [(V2_NAME, v2_summary), (V2A_NAME, v2a_summary)]:
        n_cases = int(summary.get("queries_with_same_subject_hard_negatives", 0) or 0)
        summary_rows.append(
            {
                "variant_name": variant,
                "status": "available" if n_cases else "unavailable",
                "queries_with_same_subject_hard_negatives": n_cases,
                "success_rate": summary.get("true_beats_all_rate", float("nan")),
                "mean_true_rank": summary.get("mean_true_rank", float("nan")),
                "top_k": TOP_K if variant == V2A_NAME else float("nan"),
                "aperiodic_alpha": APERIODIC_ALPHA if variant == V2A_NAME else float("nan"),
            }
        )
    return pd.DataFrame(summary_rows), cases


def failure_taxonomy(v2_scores: pd.DataFrame, v2_diag: pd.DataFrame, v2a_scores: pd.DataFrame, v2a_diag: pd.DataFrame) -> pd.DataFrame:
    """Create per-query failure taxonomy for frozen v2/v2A."""

    v2_by_query = build_score_lookup(v2_scores)
    v2a_by_query = build_score_lookup(v2a_scores)
    v2d = v2_diag.set_index("query_pair_id", drop=False)
    v2ad = v2a_diag.set_index("query_pair_id", drop=False)
    rows = []
    query_ids = sorted(set(v2d.index).union(set(v2ad.index)))
    for query_id in query_ids:
        v2_row = v2d.loc[query_id] if query_id in v2d.index else pd.Series(dtype=object)
        v2a_row = v2ad.loc[query_id] if query_id in v2ad.index else pd.Series(dtype=object)
        v2_group = v2_by_query.get(str(query_id), pd.DataFrame())
        true_v2 = v2_group[v2_group.get("is_true_pair", pd.Series(dtype=bool)).astype(bool)] if not v2_group.empty else pd.DataFrame()
        candidate_pool_issue = bool(true_v2.empty)
        v2_rank = int(v2_row.get("true_medon_rank", 10**9)) if "true_medon_rank" in v2_row else 10**9
        v2a_success = bool(v2a_row.get("top_ranked_is_true_pair", False))
        true_in_v2_top5 = bool((not candidate_pool_issue) and v2_rank <= TOP_K)
        if candidate_pool_issue:
            audit_stage = "candidate-pool issue"
        elif v2a_success:
            audit_stage = "success"
        elif not true_in_v2_top5:
            audit_stage = "stage1 unrecoverable"
        else:
            audit_stage = "stage2 rerank failure"

        top_v2a = v2a_group = v2a_by_query.get(str(query_id), pd.DataFrame())
        top_v2a_row = top_v2a.sort_values("rank").iloc[0] if not top_v2a.empty else pd.Series(dtype=object)
        wrong_subject = str(top_v2a_row.get("candidate_subject", "")) if not bool(top_v2a_row.get("is_true_pair", False)) else ""
        wrong_task = str(top_v2a_row.get("candidate_task_original", "")) if wrong_subject else ""
        wrong_side = str(top_v2a_row.get("candidate_side", "")) if wrong_subject else ""
        query_subject = str(v2a_row.get("query_subject", v2_row.get("query_subject", "")))
        query_task = str(v2a_row.get("query_task", v2_row.get("query_task", "")))
        query_side = str(v2a_row.get("query_side", v2_row.get("query_side", "")))
        rows.append(
            {
                "query_pair_id": query_id,
                "query_subject": query_subject,
                "query_task": query_task,
                "query_side": query_side,
                "query_quality_flag": v2a_row.get("query_quality", v2_row.get("query_quality", "")),
                "true_medon_quality_flag": v2a_row.get("true_medon_quality", v2_row.get("true_medon_quality", "")),
                "v2_true_rank": v2_row.get("true_medon_rank", float("nan")),
                "v2A_true_rank": v2a_row.get("true_medon_rank", float("nan")),
                "v2_top1_candidate_pair_id": v2_row.get("top_ranked_candidate_pair_id", ""),
                "v2_top1_candidate_subject": v2_row.get("top_ranked_candidate_subject", ""),
                "v2_top1_candidate_task": v2_row.get("top_ranked_candidate_task", ""),
                "v2_top1_candidate_side": v2_row.get("top_ranked_candidate_side", ""),
                "v2A_top1_candidate_pair_id": v2a_row.get("top_ranked_candidate_pair_id", ""),
                "v2A_top1_candidate_subject": v2a_row.get("top_ranked_candidate_subject", ""),
                "v2A_top1_candidate_task": v2a_row.get("top_ranked_candidate_task", ""),
                "v2A_top1_candidate_side": v2a_row.get("top_ranked_candidate_side", ""),
                "true_in_v2_top5": true_in_v2_top5,
                "audit_failure_stage": audit_stage,
                "v2_success": bool(v2_row.get("top_ranked_is_true_pair", False)),
                "v2A_success": v2a_success,
                "v2_failure_type": v2_row.get("failure_type", ""),
                "v2A_failure_type": v2a_row.get("failure_type", ""),
                "wrong_candidate_subject": wrong_subject,
                "wrong_candidate_task": wrong_task,
                "wrong_candidate_side": wrong_side,
                "wrong_candidate_quality_flag": top_v2a_row.get("candidate_quality_flag", "") if wrong_subject else "",
                "wrong_same_subject": bool(wrong_subject and wrong_subject == query_subject),
                "wrong_same_task": bool(wrong_task and wrong_task == query_task),
                "wrong_same_side": bool(wrong_side and wrong_side == query_side),
                "wrong_task_or_side": bool(wrong_subject and (wrong_task != query_task or wrong_side != query_side)),
                "candidate_set_size": v2a_row.get("candidate_set_size", v2_row.get("candidate_set_size", float("nan"))),
                "candidate_pool_issue": candidate_pool_issue,
            }
        )
    return pd.DataFrame(rows)


def load_subspace_inventory(path: Path) -> dict[str, list[str]]:
    """Read frozen subspace inventory."""

    data = read_json_required(path).get("subspaces", {})
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[str]] = {}
    for name, values in data.items():
        if isinstance(values, list):
            out[str(name)] = [str(value) for value in values]
    return out


def infer_feature_groups(
    all_features: list[str],
    subspaces: dict[str, list[str]],
    compact_features: list[str],
) -> tuple[dict[str, list[str]], pd.DataFrame, list[str]]:
    """Infer requested diagnostic feature families from existing feature columns."""

    available = set(all_features)

    def keep(features: Iterable[str]) -> list[str]:
        return [feature for feature in features if feature in available]

    def has_beta_gamma(feature: str) -> bool:
        text = feature.lower()
        return "beta" in text or "gamma" in text

    groups = {
        "STN_features": keep([feature for feature in all_features if feature.lower().startswith("stn_")]),
        "compact_aperiodic_residual": keep(
            compact_features
            or [
                feature
                for feature in all_features
                if any(token in feature.lower() for token in ["aperiodic", "residual", "peak"])
            ]
        ),
        "coupling": keep(
            subspaces.get(
                "coupling",
                [feature for feature in all_features if "coupling" in feature.lower() or "coherence" in feature.lower()],
            )
        ),
        "raw_beta_gamma_power": keep(
            [
                feature
                for feature in all_features
                if has_beta_gamma(feature)
                and feature.lower().endswith("_power")
                and "residual" not in feature.lower()
                and "coupling" not in feature.lower()
            ]
        ),
        "MEG_motor_power_features": keep(
            [
                feature
                for feature in all_features
                if (feature.lower().startswith("meg_") or feature.lower().startswith("motor_"))
                and feature.lower().endswith("_power")
                and "residual" not in feature.lower()
                and "coupling" not in feature.lower()
            ]
        ),
        "all_v2_features": keep(subspaces.get("v2_reference", [])),
        "v2A_compact_rerank_features": keep(compact_features),
    }
    patterns = {
        "STN_features": "feature starts with stn_",
        "compact_aperiodic_residual": "compact inventory features or aperiodic/residual/peak pattern",
        "coupling": "frozen coupling subspace or coupling/coherence pattern",
        "raw_beta_gamma_power": "beta/gamma raw *_power excluding residual and coupling",
        "MEG_motor_power_features": "meg_/motor_ raw *_power excluding residual and coupling",
        "all_v2_features": "frozen v2_reference subspace",
        "v2A_compact_rerank_features": "frozen compact aperiodic/residual rerank inventory",
    }
    rows = []
    warnings = []
    for group, features in groups.items():
        if not features:
            warnings.append(f"Feature group {group} has zero available columns; skipped.")
            rows.append(
                {
                    "feature_group": group,
                    "feature": "",
                    "status": "skipped_zero_columns",
                    "n_features_in_group": 0,
                    "mapping_rule": patterns[group],
                }
            )
            continue
        for feature in features:
            rows.append(
                {
                    "feature_group": group,
                    "feature": feature,
                    "status": "available",
                    "n_features_in_group": int(len(features)),
                    "mapping_rule": patterns[group],
                }
            )
    return groups, pd.DataFrame(rows), warnings


def ablation_scorer_for_group(group: str) -> str:
    """Return diagnostic scorer type for a feature-family ablation."""

    if group in {"compact_aperiodic_residual", "v2A_compact_rerank_features"}:
        return "standard"
    if group == "all_v2_features":
        return "group_balanced"
    return "group_balanced"


def bootstrap_ci_for_diagnostics(
    diagnostics: pd.DataFrame,
    n_boot: int,
    seed: int,
    group_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subject-aware bootstrap CI for top1 and MRR from diagnostics."""

    if diagnostics.empty:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    subjects = sorted(diagnostics["query_subject"].astype(str).unique())
    by_subject = {subject: diagnostics[diagnostics["query_subject"].astype(str).eq(subject)].copy() for subject in subjects}
    sample_rows = []
    for bootstrap_index in range(n_boot):
        draw = rng.choice(subjects, size=len(subjects), replace=True)
        sample = pd.concat([by_subject[str(subject)] for subject in draw], ignore_index=True)
        metrics = query_metrics(sample)
        sample_rows.append(
            {
                "feature_group": group_name,
                "bootstrap_index": bootstrap_index,
                "top1": metrics["top1"],
                "mrr": metrics["mrr"],
            }
        )
    samples = pd.DataFrame(sample_rows)
    summary_rows = []
    observed = query_metrics(diagnostics)
    for metric in ["top1", "mrr"]:
        ci_low, ci_high = percentile_ci(samples[metric])
        summary_rows.append(
            {
                "feature_group": group_name,
                "metric": metric,
                "observed": observed[metric],
                "bootstrap_mean": float(samples[metric].mean()),
                "ci_lower_95": ci_low,
                "ci_upper_95": ci_high,
                "n_bootstrap": int(n_boot),
                "seed": int(seed),
                "bootstrap_unit": "subject",
            }
        )
    return pd.DataFrame(summary_rows), samples


def feature_family_ablation(
    pairs: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    clean_features: list[str],
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate diagnostic feature-family retrieval ablations."""

    base_candidates = build_base_candidate_sets(pairs, MIN_DISTRACTORS)
    metric_rows = []
    failure_rows = []
    per_query_rows = []
    ci_tables = []
    for group_name, features in feature_groups.items():
        if not features:
            metric_rows.append({"feature_group": group_name, "status": "skipped_zero_columns", "n_features": 0})
            continue
        scorer = ablation_scorer_for_group(group_name)
        variant = VariantSpec(
            name=f"ablation_{group_name}",
            category="feature_family_ablation",
            score_kind=scorer,
            feature_space=group_name,
            distance="cosine",
            note="Diagnostic feature-family ablation; does not replace frozen v2A.",
        )
        scores, diagnostics = evaluate_v2_variant(pairs, base_candidates, features, variant, clean_features)
        hard_cases, hard_summary = same_subject_hard_negative(pairs, features, variant, clean_features)
        metrics = query_metrics(diagnostics)
        failure_subset = diagnostics[~diagnostics["top_ranked_is_true_pair"].astype(bool)]
        counts = failure_subset["failure_type"].fillna("unknown").astype(str).value_counts().sort_index().to_dict()
        metric_rows.append(
            {
                "feature_group": group_name,
                "status": "ok",
                "n_features": int(len(features)),
                "scorer": scorer,
                **metrics,
                "same_subject_hard_negative_success": hard_summary.get("true_beats_all_rate", float("nan")),
                "other_subject_same_task_side_failure_count": int(counts.get("other_subject_same_task_side", 0)),
                "quality_related_failure_count": int(counts.get("quality_related", 0)),
            }
        )
        for failure_type, count in counts.items():
            failure_rows.append(
                {
                    "feature_group": group_name,
                    "failure_type": str(failure_type),
                    "count": int(count),
                    "n_failures": int(len(failure_subset)),
                    "n_pairs": int(len(diagnostics)),
                }
            )
        per_query = diagnostics.copy()
        per_query.insert(0, "feature_group", group_name)
        per_query.insert(1, "n_features", int(len(features)))
        per_query_rows.append(per_query)
        ci, _samples = bootstrap_ci_for_diagnostics(diagnostics, n_boot, seed, group_name)
        if not ci.empty:
            ci_tables.append(ci)
    metrics_table = pd.DataFrame(metric_rows)
    failure_table = pd.DataFrame(failure_rows)
    per_query_table = pd.concat(per_query_rows, ignore_index=True) if per_query_rows else pd.DataFrame()
    ci_table = pd.concat(ci_tables, ignore_index=True) if ci_tables else pd.DataFrame()
    return metrics_table, ci_table, failure_table, per_query_table


def transition_distance_records(
    pairs: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> pd.DataFrame:
    """Build true-vs-wrong transition distance rows by feature group."""

    pair_lookup = pairs.set_index("pair_id", drop=False)
    grouped_scores = build_score_lookup(candidate_scores)
    rows = []
    for group_name, features in feature_groups.items():
        if not features:
            continue
        context_by_subject = {}
        for subject in sorted(pairs["subject"].astype(str).unique()):
            train = pairs[~pairs["subject"].astype(str).eq(subject)].copy()
            mean, std = fit_train_scaler(train, features)
            context_by_subject[subject] = (mean, std)
        for _, query in pairs.iterrows():
            query_id = str(query["pair_id"])
            subject = str(query["subject"])
            if query_id not in grouped_scores:
                continue
            mean, std = context_by_subject[subject]
            qz = transform(pair_value(query, "x_", features), mean, std)
            true_z = transform(pair_value(query, "medon_", features), mean, std)
            wrong_scores = grouped_scores[query_id][~grouped_scores[query_id]["is_true_pair"].astype(bool)].copy()
            if wrong_scores.empty:
                continue
            wrong_records = []
            for _, meta in wrong_scores.iterrows():
                candidate = pair_lookup.loc[str(meta["candidate_pair_id"])]
                candidate_z = transform(pair_value(candidate, "medon_", features), mean, std)
                wrong_records.append(
                    {
                        "candidate_pair_id": str(meta["candidate_pair_id"]),
                        "candidate_subject": str(meta["candidate_subject"]),
                        "candidate_task": str(meta["candidate_task_original"]),
                        "candidate_side": str(meta["candidate_side"]),
                        "candidate_quality": str(meta.get("candidate_quality_flag", "")),
                        "cosine": cosine_distance(qz, candidate_z),
                        "zscored_euclidean": float(np.linalg.norm(candidate_z - qz)),
                    }
                )
            true_distances = {
                "cosine": cosine_distance(qz, true_z),
                "zscored_euclidean": float(np.linalg.norm(true_z - qz)),
            }
            for metric in ["cosine", "zscored_euclidean"]:
                wrong_values = np.asarray([record[metric] for record in wrong_records], dtype=float)
                finite_mask = np.isfinite(wrong_values)
                if not finite_mask.any():
                    continue
                finite_wrong = wrong_values[finite_mask]
                finite_records = [record for record, keep in zip(wrong_records, finite_mask, strict=False) if keep]
                best_idx = int(np.argmin(finite_wrong))
                best = finite_records[best_idx]
                true_distance = float(true_distances[metric])
                best_wrong = float(finite_wrong[best_idx])
                rank = 1 + int(np.sum(finite_wrong < true_distance))
                rows.append(
                    {
                        "query_pair_id": query_id,
                        "query_subject": subject,
                        "query_task": query["task_original"],
                        "query_side": query["side"],
                        "feature_group": group_name,
                        "distance_metric": metric,
                        "n_features": int(len(features)),
                        "n_wrong_candidates": int(len(finite_wrong)),
                        "true_distance": true_distance,
                        "best_wrong_distance": best_wrong,
                        "mean_wrong_distance": float(np.mean(finite_wrong)),
                        "wrong_closer_than_true": bool(best_wrong < true_distance),
                        "true_rank_within_group": int(rank),
                        "margin": best_wrong - true_distance,
                        "best_wrong_pair_id": best["candidate_pair_id"],
                        "best_wrong_subject": best["candidate_subject"],
                        "best_wrong_task": best["candidate_task"],
                        "best_wrong_side": best["candidate_side"],
                        "best_wrong_quality": best["candidate_quality"],
                    }
                )
    return pd.DataFrame(rows)


def sign_flip_pvalue(values: pd.Series, n_perm: int, seed: int) -> float:
    """One-sided sign-flip p-value for mean margin > 0."""

    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return float("nan")
    observed = float(np.mean(arr))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(arr), replace=True)
        null.append(float(np.mean(arr * signs)))
    return float((1 + np.sum(np.asarray(null) >= observed)) / (1 + n_perm))


def transition_summary_tables(
    transition_rows: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize transition margins and bootstrap margin CIs."""

    if transition_rows.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    summary_rows = []
    ci_rows = []
    rng = np.random.default_rng(seed)
    for (group, metric), subset in transition_rows.groupby(["feature_group", "distance_metric"], dropna=False):
        margins = pd.to_numeric(subset["margin"], errors="coerce").dropna()
        wilcoxon_p = float("nan")
        if wilcoxon is not None and len(margins) > 0:
            try:
                wilcoxon_p = float(wilcoxon(margins, alternative="greater").pvalue)
            except Exception:  # noqa: BLE001
                wilcoxon_p = float("nan")
        summary_rows.append(
            {
                "feature_group": group,
                "distance_metric": metric,
                "n_queries": int(subset["query_pair_id"].nunique()),
                "wrong_closer_than_true_rate": float(subset["wrong_closer_than_true"].astype(float).mean()),
                "mean_margin": float(margins.mean()) if len(margins) else float("nan"),
                "median_margin": float(margins.median()) if len(margins) else float("nan"),
                "mean_true_distance": float(pd.to_numeric(subset["true_distance"], errors="coerce").mean()),
                "mean_best_wrong_distance": float(pd.to_numeric(subset["best_wrong_distance"], errors="coerce").mean()),
                "mean_wrong_distance": float(pd.to_numeric(subset["mean_wrong_distance"], errors="coerce").mean()),
                "mean_true_rank_within_group": float(pd.to_numeric(subset["true_rank_within_group"], errors="coerce").mean()),
                "sign_flip_p_mean_margin": sign_flip_pvalue(subset["margin"], N_PERM, seed),
                "wilcoxon_p_mean_margin_greater_than_zero": wilcoxon_p,
            }
        )
        subjects = sorted(subset["query_subject"].astype(str).unique())
        by_subject = {subject: subset[subset["query_subject"].astype(str).eq(subject)].copy() for subject in subjects}
        boot_means = []
        for bootstrap_index in range(n_boot):
            draw = rng.choice(subjects, size=len(subjects), replace=True)
            sampled = pd.concat([by_subject[str(subject)] for subject in draw], ignore_index=True)
            boot_means.append(float(pd.to_numeric(sampled["margin"], errors="coerce").mean()))
        ci_low, ci_high = percentile_ci(boot_means)
        ci_rows.append(
            {
                "feature_group": group,
                "distance_metric": metric,
                "metric": "mean_margin",
                "observed": float(margins.mean()) if len(margins) else float("nan"),
                "bootstrap_mean": float(np.nanmean(boot_means)) if boot_means else float("nan"),
                "ci_lower_95": ci_low,
                "ci_upper_95": ci_high,
                "n_bootstrap": int(n_boot),
                "seed": int(seed),
                "bootstrap_unit": "subject",
            }
        )
    failures = transition_rows[transition_rows["wrong_closer_than_true"].astype(bool)].copy()
    return pd.DataFrame(summary_rows), pd.DataFrame(ci_rows), failures


def plot_feature_metric(metrics: pd.DataFrame, ci: pd.DataFrame, metric: str, path: Path) -> None:
    """Plot feature-family top1 or MRR bars."""

    ok = metrics[metrics["status"].astype(str).eq("ok")].copy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if ok.empty:
        ax.text(0.5, 0.5, "No available feature groups", ha="center", va="center")
        ax.set_axis_off()
    else:
        ok = ok.sort_values(metric, ascending=False)
        x = np.arange(len(ok))
        values = ok[metric].to_numpy(dtype=float)
        ci_metric = ci[ci["metric"].astype(str).eq(metric)].set_index("feature_group")
        yerr = None
        lows = []
        highs = []
        for _, row in ok.iterrows():
            group = str(row["feature_group"])
            if group in ci_metric.index:
                ci_row = ci_metric.loc[group]
                lows.append(max(0.0, float(row[metric]) - float(ci_row["ci_lower_95"])))
                highs.append(max(0.0, float(ci_row["ci_upper_95"]) - float(row[metric])))
            else:
                lows.append(0.0)
                highs.append(0.0)
        yerr = np.vstack([lows, highs])
        ax.bar(x, values, yerr=yerr, capsize=3, color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(ok["feature_group"], rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.set_title(f"Feature-Family {metric.upper()}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_failure_stacked(failures: pd.DataFrame, path: Path) -> None:
    """Plot stacked failure counts by feature family."""

    fig, ax = plt.subplots(figsize=(10, 4.8))
    if failures.empty:
        ax.text(0.5, 0.5, "No feature-family failures", ha="center", va="center")
        ax.set_axis_off()
    else:
        pivot = failures.pivot_table(
            index="feature_group", columns="failure_type", values="count", aggfunc="sum", fill_value=0
        )
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        bottom = np.zeros(len(pivot), dtype=float)
        x = np.arange(len(pivot))
        palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
        for idx, column in enumerate(pivot.columns):
            values = pivot[column].to_numpy(dtype=float)
            ax.bar(x, values, bottom=bottom, label=str(column), color=palette[idx % len(palette)])
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=35, ha="right")
        ax.set_ylabel("Failure count")
        ax.legend(fontsize=8)
        ax.set_title("Feature-Family Failure Types")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_transition_bars(summary: pd.DataFrame, value_col: str, path: Path, ylabel: str, title: str) -> None:
    """Plot transition summary bars with distance metric as hue."""

    fig, ax = plt.subplots(figsize=(10, 4.8))
    if summary.empty:
        ax.text(0.5, 0.5, "No transition rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        groups = list(summary["feature_group"].astype(str).drop_duplicates())
        metrics = list(summary["distance_metric"].astype(str).drop_duplicates())
        width = 0.8 / max(1, len(metrics))
        x = np.arange(len(groups))
        palette = ["#4C78A8", "#F58518", "#54A24B"]
        for idx, metric in enumerate(metrics):
            vals = []
            for group in groups:
                row = summary[summary["feature_group"].astype(str).eq(group) & summary["distance_metric"].astype(str).eq(metric)]
                vals.append(float(row[value_col].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (idx - (len(metrics) - 1) / 2) * width, vals, width=width, label=metric, color=palette[idx % len(palette)])
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_true_rank_distribution(rows: pd.DataFrame, path: Path) -> None:
    """Plot true rank distributions by feature group."""

    fig, ax = plt.subplots(figsize=(10, 4.8))
    if rows.empty:
        ax.text(0.5, 0.5, "No transition rank rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        subset = rows[rows["distance_metric"].astype(str).eq("zscored_euclidean")].copy()
        if subset.empty:
            subset = rows.copy()
        groups = list(subset["feature_group"].astype(str).drop_duplicates())
        data = [
            pd.to_numeric(subset.loc[subset["feature_group"].astype(str).eq(group), "true_rank_within_group"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
            for group in groups
        ]
        ax.boxplot(data, tick_labels=groups, showmeans=True)
        ax.set_xticklabels(groups, rotation=35, ha="right")
        ax.set_ylabel("True rank within candidate pool")
        ax.set_title("True Rank Distribution by Feature Group")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_audit_markdown(summary: dict[str, object], warnings: list[str], path: Path) -> Path:
    """Write compact Markdown audit summary."""

    observed = summary["observed_metrics"]
    lines = [
        "# v2A Scientific Audit",
        "",
        "Scope: frozen v2/v2A paired-state retrieval audit on verified ds004998 Hold/Move MedOff-MedOn pairs only.",
        "",
        "## Cohort Checks",
        f"- complete_pairs: {summary['validation']['complete_pairs']}",
        f"- subjects_with_complete_pairs: {summary['validation']['subjects_with_complete_pairs']}",
        f"- downloaded_logical_holdmove_recordings: {summary['validation']['downloaded_logical_holdmove_recordings']}",
        f"- Rest excluded: {summary['validation']['rest_excluded']}",
        f"- sub-BYJoWR excluded: {summary['validation']['sub_BYJoWR_excluded']}",
        f"- split files collapsed logically: {summary['validation']['split_files_collapsed_logically']}",
        f"- top_k: {summary['validation']['top_k']}",
        f"- aperiodic_alpha: {summary['validation']['aperiodic_alpha']}",
        "",
        "## Observed Metrics",
    ]
    for row in observed:
        lines.append(
            f"- {row['variant_name']}: top1={float(row['top1']):.3f}, "
            f"MRR={float(row['mrr']):.3f}, top3={float(row['top3']):.3f}, top5={float(row['top5']):.3f}, "
            f"failures={int(row['failures'])}"
        )
    lines.extend(["", "## Warnings"])
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- none")
    return write_text(lines, path)


def run() -> dict[str, object]:
    """Run the full scientific audit."""

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

    v2a_scores = rank_scores(cached_v2a_scores.drop(columns=["rank"], errors="ignore"))
    v2a_diag = query_diagnostics_from_scores(v2a_scores)
    v2_scores = rebuild_v2_scores_from_frozen_v2a(v2a_scores)
    v2_diag = query_diagnostics_from_scores(v2_scores)
    observed = observed_metrics_table(v2_diag, v2a_diag)

    all_features = base_feature_names(pairs)
    v2_features = [feature for feature in subspaces.get("v2_reference", []) if feature in all_features]
    compact_features = [feature for feature in compact_features if feature in all_features]
    clean_features = [feature for feature in subspaces.get("clean_stable_features", []) if feature in all_features]
    if not v2_features:
        raise AssertionError("Frozen v2_reference feature list is empty.")
    if not compact_features:
        raise AssertionError("Frozen compact aperiodic/residual rerank feature list is empty.")

    diagnostics_by_variant = {V2_NAME: v2_diag, V2A_NAME: v2a_diag}
    scores_by_variant = {V2_NAME: v2_scores, V2A_NAME: v2a_scores}

    random_summary, random_samples = random_label_null(diagnostics_by_variant, scores_by_variant, N_PERM, RANDOM_SEED)
    matched_summary, matched_samples = matched_task_side_null(diagnostics_by_variant, scores_by_variant, N_PERM, RANDOM_SEED)
    bootstrap_summary, bootstrap_samples = subject_bootstrap_samples(v2_diag, v2a_diag, N_BOOT, RANDOM_SEED)
    loso = loso_sensitivity(pairs, v2_features, compact_features, clean_features, query_metrics(v2a_diag))
    task_side = task_side_sensitivity(pairs, v2_features, compact_features, clean_features)
    hard_summary, hard_cases = hard_negative_outputs(pairs, v2_features, compact_features, clean_features)
    taxonomy = failure_taxonomy(v2_scores, v2_diag, v2a_scores, v2a_diag)

    feature_groups, mapping, feature_warnings = infer_feature_groups(all_features, subspaces, compact_features)
    ablation_metrics, ablation_ci, ablation_failures, ablation_ranks = feature_family_ablation(
        pairs, feature_groups, clean_features, N_BOOT, RANDOM_SEED
    )
    transition_rows = transition_distance_records(pairs, v2a_scores, feature_groups)
    transition_summary, transition_ci, transition_failures = transition_summary_tables(
        transition_rows, N_BOOT, RANDOM_SEED
    )

    warnings = list(feature_warnings)
    v2a_metric = observed[observed["variant_name"].eq(V2A_NAME)].iloc[0]
    if abs(float(v2a_metric["top1"]) - 0.848) > 0.02:
        warnings.append(f"v2A top1 sanity check outside expected neighborhood: {float(v2a_metric['top1']):.3f}")
    if abs(float(v2a_metric["mrr"]) - 0.893) > 0.02:
        warnings.append(f"v2A MRR sanity check outside expected neighborhood: {float(v2a_metric['mrr']):.3f}")
    if loso["collapse_flag"].astype(bool).any():
        subjects = ", ".join(loso.loc[loso["collapse_flag"].astype(bool), "removed_subject"].astype(str))
        warnings.append(f"LOSO collapse flag triggered for: {subjects}")

    paths = {
        "audit_summary_json": write_json(
            {
                "validation": validation,
                "observed_metrics": observed.to_dict("records"),
                "random_label_null": random_summary.to_dict("records"),
                "matched_task_side_null": matched_summary.to_dict("records"),
                "bootstrap_ci": bootstrap_summary.to_dict("records"),
                "warnings": warnings,
                "n_feature_groups_available": int(
                    mapping.loc[mapping["status"].astype(str).eq("available"), "feature_group"].nunique()
                )
                if not mapping.empty
                else 0,
                "n_transition_rows": int(len(transition_rows)),
                "not_clinical_prediction_or_treatment": True,
                "external_oxford_mrc_dataset_used": False,
            },
            OUTPUT_DIR / "audit_summary.json",
        ),
        "observed_metrics": write_csv(observed, OUTPUT_DIR / "observed_v2_v2A_metrics.csv"),
        "random_summary": write_csv(random_summary, OUTPUT_DIR / "random_label_null_summary.csv"),
        "random_samples": write_csv(random_samples, OUTPUT_DIR / "random_label_null_samples.csv"),
        "matched_summary": write_csv(matched_summary, OUTPUT_DIR / "matched_task_side_null_summary.csv"),
        "matched_samples": write_csv(matched_samples, OUTPUT_DIR / "matched_task_side_null_samples.csv"),
        "bootstrap_summary": write_csv(bootstrap_summary, OUTPUT_DIR / "bootstrap_ci_summary.csv"),
        "bootstrap_samples": write_csv(bootstrap_samples, OUTPUT_DIR / "bootstrap_samples.csv"),
        "loso": write_csv(loso, OUTPUT_DIR / "loso_sensitivity.csv"),
        "task_side": write_csv(task_side, OUTPUT_DIR / "task_side_sensitivity.csv"),
        "hard_summary": write_csv(hard_summary, OUTPUT_DIR / "hard_negative_summary.csv"),
        "hard_cases": write_csv(hard_cases, OUTPUT_DIR / "hard_negative_cases.csv"),
        "taxonomy": write_csv(taxonomy, OUTPUT_DIR / "failure_taxonomy.csv"),
        "mapping": write_csv(mapping, OUTPUT_DIR / "feature_group_column_mapping.csv"),
        "ablation_metrics": write_csv(ablation_metrics, OUTPUT_DIR / "feature_family_ablation_metrics.csv"),
        "ablation_ci": write_csv(ablation_ci, OUTPUT_DIR / "feature_family_ablation_bootstrap_ci.csv"),
        "ablation_failures": write_csv(ablation_failures, OUTPUT_DIR / "feature_family_ablation_failure_counts.csv"),
        "ablation_ranks": write_csv(ablation_ranks, OUTPUT_DIR / "feature_family_ablation_per_query_ranks.csv"),
        "transition_rows": write_csv(transition_rows, OUTPUT_DIR / "transition_true_vs_wrong_per_query.csv"),
        "transition_summary": write_csv(transition_summary, OUTPUT_DIR / "transition_group_summary.csv"),
        "transition_ci": write_csv(transition_ci, OUTPUT_DIR / "transition_margin_bootstrap_ci.csv"),
        "transition_failures": write_csv(transition_failures, OUTPUT_DIR / "transition_failure_cases.csv"),
    }
    summary_data = read_json_required(OUTPUT_DIR / "audit_summary.json")
    paths["audit_summary_md"] = write_audit_markdown(summary_data, warnings, OUTPUT_DIR / "audit_summary.md")

    plot_feature_metric(ablation_metrics, ablation_ci, "top1", OUTPUT_DIR / "feature_family_top1_bar.png")
    plot_feature_metric(ablation_metrics, ablation_ci, "mrr", OUTPUT_DIR / "feature_family_mrr_bar.png")
    plot_failure_stacked(ablation_failures, OUTPUT_DIR / "feature_family_failure_type_stacked_bar.png")
    plot_transition_bars(
        transition_summary,
        "mean_margin",
        OUTPUT_DIR / "transition_margin_by_feature_group.png",
        "Mean margin",
        "Transition Margin by Feature Group",
    )
    plot_transition_bars(
        transition_summary,
        "wrong_closer_than_true_rate",
        OUTPUT_DIR / "wrong_closer_than_true_rate_by_group.png",
        "Wrong closer than true rate",
        "Wrong-Closer Rate by Feature Group",
    )
    plot_true_rank_distribution(transition_rows, OUTPUT_DIR / "true_rank_distribution_by_group.png")

    print(f"complete_pairs: {validation['complete_pairs']}")
    print(f"subjects_with_complete_pairs: {validation['subjects_with_complete_pairs']}")
    print(f"Rest excluded: {validation['rest_excluded']}")
    print(f"sub-BYJoWR excluded: {validation['sub_BYJoWR_excluded']}")
    print(f"split files collapsed logically: {validation['split_files_collapsed_logically']}")
    print(f"top_k: {TOP_K}")
    print(f"aperiodic_alpha: {APERIODIC_ALPHA}")
    print(f"v2A top1/MRR: {float(v2a_metric['top1']):.3f} / {float(v2a_metric['mrr']):.3f}")
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
