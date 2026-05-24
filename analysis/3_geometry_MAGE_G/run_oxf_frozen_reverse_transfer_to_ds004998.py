#!/usr/bin/env python3
"""Train frozen OXF rankers and test them on ds004998.

This reverse-transfer experiment asks the opposite of the current OXF external
validation path: learn a small rank-calibration layer on OXF candidate scores,
freeze it, and apply it once to cached ds004998 STN-only candidate scores.

Two frozen branches are compared:

1. no_MAGE_G:
   Trained on OXF raw-reconstruction v2D candidate scores.
2. MAGE_G_source_span_soft:
   Trained on OXF MAGE-G source/span soft candidate scores. For ds004998, the
   frozen candidate scores are augmented with side-specific STN contact geometry
   reconstructed from the frozen ds004998 channel manifest before applying the
   OXF-frozen ranker.

The script reads cached outputs only. It does not modify frozen ds004998 v2A,
raw extraction, feature tables, top_k, aperiodic_alpha, or MAGE-G outputs.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
except Exception as exc:  # noqa: BLE001
    LogisticRegression = None
    SKLEARN_IMPORT_ERROR = str(exc)
else:
    SKLEARN_IMPORT_ERROR = ""


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_DIR = Path("outputs/oxf_frozen_reverse_transfer_to_ds004998")

ALL_MEDON_CONDITION = "all_medon_candidates"
RANDOM_SEED = 42
TOP_K = 5
APERIODIC_ALPHA = 0.5

V2D_VARIANT = "v2D_quality_deconfounded_cycle"
MAGE_G_VARIANT = "MAGE_G_source_span_soft__v2D_quality_deconfounded_cycle"
RAW_RECON_BRANCH = "all_pairs_exact_raw_reconstruct_else_median_compact_residual_7"

OXF_RAW_SCORES = Path("outputs/oxf_raw_montage_reconstruction_audit/raw_reconstruction_candidate_scores.csv")
OXF_MAGE_G_SCORES = Path("outputs/oxf_mage_g_geometry_aware_evidence/mage_g_candidate_scores.csv")
DS_STN_SCORES = Path("outputs/ds004998_stn_only_cross_dataset_branch/ds004998_stn_only_candidate_scores.csv")
DS_STN_METRICS = Path("outputs/ds004998_stn_only_cross_dataset_branch/ds004998_stn_only_observed_metrics.csv")
DS_FROZEN_PAIRS = Path("outputs/v2A_frozen_18subjects_34pairs/frozen_v2a_pairs.csv")
DS_CHANNEL_MANIFEST = Path("outputs/v2A_frozen_18subjects_34pairs/tables/ds004998_channel_manifest.csv")

COMMON_SCORE_FEATURES = [
    "score",
    "rank_fraction",
    "forward_score",
    "reverse_score",
    "reverse_rank_fraction",
    "reverse_v2_score",
    "reverse_aperiodic_score",
    "v2d_forward_z",
    "v2d_reverse_z",
    "v2d_k_reciprocal",
    "v2d_k_reciprocal_bonus",
    "v2d_neighbor_jaccard",
    "v2d_quality_mismatch",
    "v2d_quality_penalty",
    "v2d_wrong_task_or_side",
    "v2d_same_subject_wrong",
    "v2d_deconfound_penalty",
]

MAGE_G_GEOMETRY_FEATURES = [
    "source_mismatch",
    "spacing_mismatch",
    "contact_center_distance",
    "contact_span_distance",
    "geometry_distance",
]

MAGE_G_FEATURES = [*COMMON_SCORE_FEATURES, *MAGE_G_GEOMETRY_FEATURES]

NEUTRAL_GEOMETRY_DEFAULTS = {
    "source_mismatch": 0.0,
    "spacing_mismatch": 0.0,
    "contact_center_distance": 0.0,
    "contact_span_distance": 0.0,
    "geometry_distance": 0.0,
}

MISSING_GEOMETRY_DISTANCE_DEFAULT = 0.5


@dataclass(frozen=True)
class FrozenRanker:
    """Frozen rank-calibration model learned on OXF."""

    label: str
    train_source: str
    feature_columns: list[str]
    train_fill_values: dict[str, float]
    mean: dict[str, float]
    scale: dict[str, float]
    coefficients: dict[str, float]
    intercept: float
    train_rows: int
    train_queries: int
    train_positive_rows: int


def ensure_dirs() -> None:
    """Create the output folder."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def to_builtin(value: object) -> object:
    """Convert numpy/pandas scalars into JSON-safe values."""

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
    """Write JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(data), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(data: pd.DataFrame, path: Path) -> Path:
    """Write CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return path


def write_text(lines: Iterable[str], path: Path) -> Path:
    """Write text/Markdown."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV."""

    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def filter_scores(path: Path, variant: str, branch_id: str | None = None) -> pd.DataFrame:
    """Read and filter a candidate-score table."""

    data = read_csv(path)
    out = data[
        data["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & data["variant_name"].astype(str).eq(variant)
    ].copy()
    if branch_id is not None and "branch_id" in out:
        out = out[out["branch_id"].astype(str).eq(branch_id)].copy()
    if out.empty:
        raise AssertionError(f"No rows for variant={variant} in {path}")
    return out.reset_index(drop=True)


def add_rank_fraction(data: pd.DataFrame) -> pd.DataFrame:
    """Attach rank-fraction features used by the OXF frozen rankers."""

    out = data.copy()
    ranks = pd.to_numeric(out["rank"], errors="coerce")
    counts = pd.to_numeric(out["number_of_candidates"], errors="coerce").replace(1, np.nan)
    if "rank_fraction" not in out:
        out["rank_fraction"] = ((ranks - 1.0) / (counts - 1.0)).fillna(1.0).clip(0.0, 1.0)
    if "reverse_rank" in out and "reverse_rank_fraction" not in out:
        reverse = pd.to_numeric(out["reverse_rank"], errors="coerce")
        out["reverse_rank_fraction"] = ((reverse - 1.0) / (counts - 1.0)).fillna(1.0).clip(0.0, 1.0)
    return out


def bool_series(values: pd.Series, default: bool = False) -> pd.Series:
    """Parse bool-ish columns from cached CSVs."""

    if values.dtype == bool:
        return values.fillna(default)
    text = values.astype(str).str.strip().str.lower()
    parsed = text.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "y": True,
            "false": False,
            "0": False,
            "no": False,
            "n": False,
        }
    )
    return parsed.fillna(default).astype(bool)


def side_name(value: object) -> str:
    """Convert compact pair-side labels into manifest side labels."""

    text = str(value).strip().lower()
    if text in {"l", "left"}:
        return "left"
    if text in {"r", "right"}:
        return "right"
    return text


def contact_indices_text(indices: list[int]) -> str:
    """Serialize contact indices for compact CSV audit columns."""

    return " ".join(str(index) for index in indices)


def spacing_class(indices: list[int]) -> str:
    """Coarse side-montage spacing class for ds004998 contact sets."""

    if not indices:
        return "unknown"
    span = max(indices) - min(indices) if len(indices) > 1 else 0
    if len(indices) >= 7 or span >= 6:
        return "extended_8_contact"
    if len(indices) >= 3 and span <= 3:
        return "compact_4_contact"
    if span <= 3:
        return "compact_partial_contact"
    return "intermediate_or_sparse_contact"


def montage_source_class(all_indices: list[int]) -> str:
    """Broad montage source class used for ds source-mismatch penalties."""

    if not all_indices:
        return "ds004998_side_montage_unknown"
    span = max(all_indices) - min(all_indices) if len(all_indices) > 1 else 0
    if len(all_indices) >= 7 or span >= 6:
        return "ds004998_side_montage_8contact"
    if len(all_indices) >= 4 or span >= 3:
        return "ds004998_side_montage_4contact"
    return "ds004998_side_montage_partial"


def contact_geometry_from_rows(rows: pd.DataFrame) -> dict[str, object]:
    """Summarize usable contact geometry from manifest rows."""

    if rows.empty:
        return {
            "all_contact_indices": "",
            "all_contact_count": 0,
            "contact_indices": "",
            "contact_count": 0,
            "contact_center": np.nan,
            "contact_span": np.nan,
            "spacing_class": "unknown",
            "montage_source": "ds004998_side_montage_unknown",
            "repair_class": "ds_side_contacts_missing",
            "bad_contact_count": 0,
            "used_all_contacts_because_all_bad": False,
        }

    work = rows.copy()
    work["_contact_index"] = pd.to_numeric(work["lfp_contact_index"], errors="coerce")
    work = work[work["_contact_index"].notna()].copy()
    if work.empty:
        return {
            "all_contact_indices": "",
            "all_contact_count": 0,
            "contact_indices": "",
            "contact_count": 0,
            "contact_center": np.nan,
            "contact_span": np.nan,
            "spacing_class": "unknown",
            "montage_source": "ds004998_side_montage_unknown",
            "repair_class": "ds_side_contacts_missing",
            "bad_contact_count": 0,
            "used_all_contacts_because_all_bad": False,
        }

    work["_contact_index"] = work["_contact_index"].round().astype(int)
    all_indices = sorted(work["_contact_index"].dropna().astype(int).unique().tolist())
    bad_mask = bool_series(work["is_bad"], default=False) if "is_bad" in work else pd.Series(False, index=work.index)
    good_indices = sorted(work.loc[~bad_mask, "_contact_index"].dropna().astype(int).unique().tolist())
    used_all_contacts = False
    if not good_indices and all_indices:
        good_indices = all_indices
        used_all_contacts = True

    center = float(np.mean(good_indices)) if good_indices else np.nan
    span = float(max(good_indices) - min(good_indices)) if len(good_indices) > 1 else (0.0 if good_indices else np.nan)
    bad_contact_count = max(0, len(all_indices) - len(good_indices)) if not used_all_contacts else len(all_indices)
    if not good_indices:
        repair_class = "ds_side_contacts_missing"
    elif used_all_contacts:
        repair_class = "ds_side_contacts_all_bad_fallback"
    elif bad_contact_count > 0:
        repair_class = "ds_side_contacts_bad_filtered"
    else:
        repair_class = "ds_side_contacts_good"

    return {
        "all_contact_indices": contact_indices_text(all_indices),
        "all_contact_count": int(len(all_indices)),
        "contact_indices": contact_indices_text(good_indices),
        "contact_count": int(len(good_indices)),
        "contact_center": center,
        "contact_span": span,
        "spacing_class": spacing_class(good_indices),
        "montage_source": montage_source_class(all_indices),
        "repair_class": repair_class,
        "bad_contact_count": int(bad_contact_count),
        "used_all_contacts_because_all_bad": bool(used_all_contacts),
    }


def ds_manifest_rows(
    manifest: pd.DataFrame,
    subject: object,
    task: object,
    run_number: object,
    side: object,
    medication: str,
) -> tuple[pd.DataFrame, str]:
    """Select STN side-contact manifest rows for one ds recording."""

    target_run = pd.to_numeric(pd.Series([run_number]), errors="coerce").iloc[0]
    if not np.isfinite(target_run):
        return manifest.iloc[0:0].copy(), f"{medication}: missing_run"

    lfp_mask = bool_series(manifest["is_lfp_stn"], default=False)
    run_values = pd.to_numeric(manifest["run"], errors="coerce")
    base = manifest[
        manifest["subject"].astype(str).eq(str(subject))
        & manifest["task"].astype(str).eq(str(task))
        & np.isclose(run_values, float(target_run), equal_nan=False)
        & lfp_mask
        & manifest["lfp_side"].astype(str).str.lower().eq(side_name(side))
    ].copy()
    if base.empty:
        return base, f"{medication}: no_manifest_rows"

    token = f"acq-{medication}"
    med = base[base["file_path"].astype(str).str.contains(token, case=False, na=False)].copy()
    if med.empty:
        return base, f"{medication}: no_{token}_path_match_used_run_task_side"
    return med, f"{medication}: matched_{token}"


def build_ds_pair_contact_geometry(pairs: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Build ds004998 pair-level side-contact geometry from the frozen manifest."""

    rows = []
    for _, pair in pairs.iterrows():
        side = side_name(pair.get("side", ""))
        off_rows, off_note = ds_manifest_rows(
            manifest,
            pair.get("subject"),
            pair.get("task_original"),
            pair.get("run_off"),
            side,
            "MedOff",
        )
        on_rows, on_note = ds_manifest_rows(
            manifest,
            pair.get("subject"),
            pair.get("task_original"),
            pair.get("run_on"),
            side,
            "MedOn",
        )
        off_geo = contact_geometry_from_rows(off_rows)
        on_geo = contact_geometry_from_rows(on_rows)
        primary_source = "off" if int(off_geo["contact_count"]) > 0 else "on"
        primary = off_geo if primary_source == "off" else on_geo
        rows.append(
            {
                "pair_id": str(pair["pair_id"]),
                "subject": str(pair.get("subject", "")),
                "session": str(pair.get("session", "")),
                "task_original": str(pair.get("task_original", "")),
                "task_family": str(pair.get("task_family", "")),
                "side": str(pair.get("side", "")),
                "manifest_side": side,
                "run_off": pair.get("run_off"),
                "run_on": pair.get("run_on"),
                "off_contact_indices": off_geo["contact_indices"],
                "off_all_contact_indices": off_geo["all_contact_indices"],
                "off_contact_count": off_geo["contact_count"],
                "off_all_contact_count": off_geo["all_contact_count"],
                "off_contact_center": off_geo["contact_center"],
                "off_contact_span": off_geo["contact_span"],
                "off_spacing_class": off_geo["spacing_class"],
                "off_montage_source": off_geo["montage_source"],
                "off_repair_class": off_geo["repair_class"],
                "off_bad_contact_count": off_geo["bad_contact_count"],
                "on_contact_indices": on_geo["contact_indices"],
                "on_all_contact_indices": on_geo["all_contact_indices"],
                "on_contact_count": on_geo["contact_count"],
                "on_all_contact_count": on_geo["all_contact_count"],
                "on_contact_center": on_geo["contact_center"],
                "on_contact_span": on_geo["contact_span"],
                "on_spacing_class": on_geo["spacing_class"],
                "on_montage_source": on_geo["montage_source"],
                "on_repair_class": on_geo["repair_class"],
                "on_bad_contact_count": on_geo["bad_contact_count"],
                "primary_geometry_source": primary_source,
                "primary_contact_indices": primary["contact_indices"],
                "primary_contact_count": primary["contact_count"],
                "primary_contact_center": primary["contact_center"],
                "primary_contact_span": primary["contact_span"],
                "primary_spacing_class": primary["spacing_class"],
                "primary_montage_source": primary["montage_source"],
                "primary_repair_class": primary["repair_class"],
                "geometry_available": int(primary["contact_count"]) > 0,
                "geometry_note": f"{off_note}; {on_note}",
            }
        )
    return pd.DataFrame(rows)


def normalized_distance_series(
    left: pd.Series,
    right: pd.Series,
    denom: float,
    missing_value: float = MISSING_GEOMETRY_DISTANCE_DEFAULT,
) -> pd.Series:
    """Normalize pairwise distances with a fixed missing-geometry fallback."""

    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    distance = ((left_num - right_num).abs() / denom).clip(upper=1.0)
    distance = distance.where(left_num.notna() & right_num.notna(), missing_value)
    return distance.fillna(missing_value).astype(float)


def neutral_ds_mage_g_transport_view(ds_base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fallback ds004998 view used when no contact geometry is available."""

    out = add_rank_fraction(ds_base)
    rows = []
    for feature, default in NEUTRAL_GEOMETRY_DEFAULTS.items():
        was_present = feature in out
        if not was_present:
            out[feature] = default
        rows.append(
            {
                "feature": feature,
                "present_in_ds_cached_scores": was_present,
                "applied_default": default,
                "policy": "neutral_missing_geometry_on_ds004998",
            }
        )
    out["score"] = (
        pd.to_numeric(out["rank_fraction"], errors="coerce").fillna(1.0)
        + 0.5 * pd.to_numeric(out["source_mismatch"], errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(out["geometry_distance"], errors="coerce").fillna(0.0)
    )
    out = rerank_by_score(
        out,
        score_column="score",
        rank_column="rank",
        tie_break_columns=["query_pair_id", "score", "rank_fraction", "candidate_pair_id"],
    )
    return out, pd.DataFrame(rows)


def make_ds_mage_g_transport_view(
    ds_base: pd.DataFrame,
    pair_geometry: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the ds004998 view used by the frozen MAGE-G ranker."""

    if pair_geometry is None or pair_geometry.empty:
        return neutral_ds_mage_g_transport_view(ds_base)

    out = add_rank_fraction(ds_base)
    geometry = pair_geometry.set_index("pair_id", drop=False)
    for side in ["query", "candidate"]:
        ids = out[f"{side}_pair_id"].astype(str)
        out[f"{side}_montage_source"] = ids.map(geometry["primary_montage_source"]).fillna(
            "ds004998_side_montage_unknown"
        )
        out[f"{side}_repair_class"] = ids.map(geometry["primary_repair_class"]).fillna("ds_side_contacts_missing")
        out[f"{side}_spacing_class"] = ids.map(geometry["primary_spacing_class"]).fillna("unknown")
        out[f"{side}_contact_digits"] = ids.map(geometry["primary_contact_indices"]).fillna("")
        out[f"{side}_contact_center"] = pd.to_numeric(ids.map(geometry["primary_contact_center"]), errors="coerce")
        out[f"{side}_contact_span"] = pd.to_numeric(ids.map(geometry["primary_contact_span"]), errors="coerce")
        out[f"{side}_geometry_available"] = ids.map(geometry["geometry_available"]).fillna(False).astype(bool)

    out["source_mismatch"] = out["query_montage_source"].astype(str).ne(out["candidate_montage_source"].astype(str))
    out["repair_class_mismatch"] = out["query_repair_class"].astype(str).ne(out["candidate_repair_class"].astype(str))
    out["spacing_mismatch"] = out["query_spacing_class"].astype(str).ne(out["candidate_spacing_class"].astype(str))
    out["contact_center_distance"] = normalized_distance_series(
        out["query_contact_center"],
        out["candidate_contact_center"],
        8.0,
    )
    out["contact_span_distance"] = normalized_distance_series(
        out["query_contact_span"],
        out["candidate_contact_span"],
        4.0,
    )
    out["geometry_distance"] = out["contact_center_distance"].astype(float) + 0.5 * out[
        "contact_span_distance"
    ].astype(float)
    rows = []
    for feature in MAGE_G_GEOMETRY_FEATURES:
        rows.append(
            {
                "feature": feature,
                "present_in_ds_cached_scores": feature in ds_base,
                "applied_default": (
                    MISSING_GEOMETRY_DISTANCE_DEFAULT
                    if feature in {"contact_center_distance", "contact_span_distance"}
                    else "computed"
                ),
                "policy": "computed_from_ds004998_channel_manifest_pair_contact_geometry",
            }
        )
    out["score"] = (
        pd.to_numeric(out["rank_fraction"], errors="coerce").fillna(1.0)
        + 0.5 * pd.to_numeric(out["source_mismatch"], errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(out["geometry_distance"], errors="coerce").fillna(0.0)
    )
    out = rerank_by_score(
        out,
        score_column="score",
        rank_column="rank",
        tie_break_columns=["query_pair_id", "score", "rank_fraction", "candidate_pair_id"],
    )
    return out, pd.DataFrame(rows)


def numeric_series(data: pd.DataFrame, column: str, default: float) -> pd.Series:
    """Return a numeric feature series with robust bool/string conversion."""

    if column not in data:
        return pd.Series(default, index=data.index, dtype=float)
    values = data[column]
    if values.dtype == bool:
        return values.astype(float)
    mapped = values.replace(
        {
            True: 1.0,
            False: 0.0,
            "True": 1.0,
            "False": 0.0,
            "true": 1.0,
            "false": 0.0,
        }
    )
    return pd.to_numeric(mapped, errors="coerce").astype(float)


def feature_matrix(
    data: pd.DataFrame,
    feature_columns: list[str],
    fill_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build a numeric feature matrix."""

    fill_values = fill_values or {}
    columns = {}
    for feature in feature_columns:
        default = float(fill_values.get(feature, NEUTRAL_GEOMETRY_DEFAULTS.get(feature, np.nan)))
        columns[feature] = numeric_series(data, feature, default)
    matrix = pd.DataFrame(columns, index=data.index)
    if fill_values:
        for feature in feature_columns:
            matrix[feature] = matrix[feature].fillna(float(fill_values.get(feature, 0.0)))
    return matrix


def fit_ranker(label: str, train_source: str, train: pd.DataFrame, feature_columns: list[str]) -> FrozenRanker:
    """Fit a balanced logistic ranker on OXF candidate rows."""

    if LogisticRegression is None:
        raise RuntimeError(f"scikit-learn is required for this script: {SKLEARN_IMPORT_ERROR}")

    train = add_rank_fraction(train)
    raw = feature_matrix(train, feature_columns)
    fill_values = {
        feature: float(raw[feature].median()) if raw[feature].notna().any() else 0.0
        for feature in feature_columns
    }
    matrix = raw.fillna(fill_values)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    X = ((matrix - mean) / scale).to_numpy(dtype=float)
    y = train["is_true_pair"].astype(bool).to_numpy(dtype=int)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )
    model.fit(X, y)
    return FrozenRanker(
        label=label,
        train_source=train_source,
        feature_columns=feature_columns,
        train_fill_values={feature: float(fill_values[feature]) for feature in feature_columns},
        mean={feature: float(mean[feature]) for feature in feature_columns},
        scale={feature: float(scale[feature]) for feature in feature_columns},
        coefficients={feature: float(coef) for feature, coef in zip(feature_columns, model.coef_[0], strict=True)},
        intercept=float(model.intercept_[0]),
        train_rows=int(len(train)),
        train_queries=int(train["query_pair_id"].nunique()),
        train_positive_rows=int(train["is_true_pair"].astype(bool).sum()),
    )


def apply_ranker(data: pd.DataFrame, ranker: FrozenRanker, output_variant: str, split_label: str) -> pd.DataFrame:
    """Apply a frozen ranker to candidate rows."""

    data = add_rank_fraction(data)
    matrix = feature_matrix(data, ranker.feature_columns, ranker.train_fill_values)
    mean = pd.Series(ranker.mean)
    scale = pd.Series(ranker.scale).replace(0.0, 1.0)
    coef = pd.Series(ranker.coefficients)
    X = (matrix[ranker.feature_columns] - mean[ranker.feature_columns]) / scale[ranker.feature_columns]
    linear = X.to_numpy(dtype=float) @ coef[ranker.feature_columns].to_numpy(dtype=float) + ranker.intercept
    probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -50.0, 50.0)))
    out = data.copy()
    out["oxf_frozen_linear_score"] = linear
    out["oxf_frozen_true_probability"] = probability
    out["score"] = -probability
    out["variant_name"] = output_variant
    out["variant_category"] = "oxf_frozen_reverse_transfer"
    out["feature_space"] = "common_stn_candidate_score_components"
    out["score_kind"] = "frozen_oxf_balanced_logistic_ranker"
    out["weight_scheme"] = f"trained_on_oxf_frozen::{ranker.label}"
    out["reverse_transfer_split"] = split_label
    out["candidate_pool_condition"] = ALL_MEDON_CONDITION
    out = rerank_by_score(
        out,
        score_column="score",
        rank_column="rank",
        tie_break_columns=["query_pair_id", "score", "rank_fraction", "candidate_pair_id"],
    )
    return out


def rerank_by_score(
    data: pd.DataFrame,
    score_column: str,
    rank_column: str,
    tie_break_columns: list[str],
) -> pd.DataFrame:
    """Rank each query by a score column; lower is better."""

    out = data.copy()
    out["_original_order"] = np.arange(len(out))
    sort_cols = [col for col in tie_break_columns if col in out]
    if "_original_order" not in sort_cols:
        sort_cols.append("_original_order")
    ascending = [True for _ in sort_cols]
    ranked_parts = []
    for _, group in out.sort_values(sort_cols, ascending=ascending).groupby("query_pair_id", dropna=False, sort=False):
        part = group.copy()
        part[rank_column] = np.arange(1, len(part) + 1, dtype=int)
        part["number_of_candidates"] = int(len(part))
        ranked_parts.append(part)
    ranked = pd.concat(ranked_parts, ignore_index=True) if ranked_parts else out
    return ranked.drop(columns=["_original_order"], errors="ignore")


def query_diagnostics(scores: pd.DataFrame) -> pd.DataFrame:
    """Build query-level diagnostics from candidate scores."""

    rows = []
    for (condition, variant, query_id), group in scores.groupby(
        ["candidate_pool_condition", "variant_name", "query_pair_id"],
        dropna=False,
    ):
        ranked = group.sort_values(["rank", "score", "candidate_pair_id"]).copy()
        true_rows = ranked[ranked["is_true_pair"].astype(bool)]
        if true_rows.empty:
            continue
        true = true_rows.iloc[0]
        top = ranked.iloc[0]
        true_rank = int(true["rank"])
        n_candidates = int(len(ranked))
        top_true = bool(top["is_true_pair"])
        same_subject_wrong = (
            str(top.get("candidate_subject", "")) == str(true.get("query_subject", ""))
            and not top_true
        )
        same_task_side = (
            str(top.get("candidate_task_family", top.get("candidate_task_original", "")))
            == str(true.get("query_task_family", true.get("query_task_original", "")))
            and str(top.get("candidate_side", "")) == str(true.get("query_side", ""))
        )
        if top_true:
            failure_type = "success"
        elif same_subject_wrong:
            failure_type = "same_subject_wrong_candidate"
        elif not same_task_side:
            failure_type = "wrong_task_or_side"
        else:
            failure_type = "other_subject_same_task_side"
        rows.append(
            {
                "candidate_pool_condition": condition,
                "variant_name": variant,
                "query_pair_id": query_id,
                "query_subject": true.get("query_subject", ""),
                "query_task": true.get("query_task_original", ""),
                "query_task_family": true.get("query_task_family", ""),
                "query_side": true.get("query_side", ""),
                "true_medon_rank": true_rank,
                "top_ranked_is_true_pair": top_true,
                "reciprocal_rank": float(1.0 / true_rank),
                "percentile_rank": float(1.0 - ((true_rank - 1.0) / max(1.0, n_candidates - 1.0))),
                "number_of_candidates": n_candidates,
                "top_ranked_candidate_pair_id": top.get("candidate_pair_id", ""),
                "top_ranked_candidate_subject": top.get("candidate_subject", ""),
                "top_ranked_candidate_task": top.get("candidate_task_original", ""),
                "top_ranked_candidate_task_family": top.get("candidate_task_family", ""),
                "top_ranked_candidate_side": top.get("candidate_side", ""),
                "failure_type": failure_type,
            }
        )
    return pd.DataFrame(rows)


def metrics_from_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Compute retrieval metrics."""

    rows = []
    for (condition, variant), group in diagnostics.groupby(["candidate_pool_condition", "variant_name"], dropna=False):
        rank = pd.to_numeric(group["true_medon_rank"], errors="coerce")
        success = group["top_ranked_is_true_pair"].astype(bool)
        top_subject = group["top_ranked_candidate_subject"].astype(str)
        query_subject = group["query_subject"].astype(str)
        top_task = group["top_ranked_candidate_task_family"].astype(str)
        query_task = group["query_task_family"].astype(str)
        top_side = group["top_ranked_candidate_side"].astype(str)
        query_side = group["query_side"].astype(str)
        rows.append(
            {
                "candidate_pool_condition": condition,
                "variant_name": variant,
                "n_pairs": int(len(group)),
                "top1": float(success.mean()),
                "mrr": float(pd.to_numeric(group["reciprocal_rank"], errors="coerce").mean()),
                "top3": float((rank <= 3).mean()),
                "top5": float((rank <= 5).mean()),
                "percentile_rank": float(pd.to_numeric(group["percentile_rank"], errors="coerce").mean()),
                "mean_true_rank": float(rank.mean()),
                "median_true_rank": float(rank.median()),
                "failures": int((~success).sum()),
                "top1_same_subject_rate": float(top_subject.eq(query_subject).mean()),
                "top1_same_subject_wrong_rate": float((top_subject.eq(query_subject) & ~success).mean()),
                "top1_same_task_side_rate": float((top_task.eq(query_task) & top_side.eq(query_side)).mean()),
                "top1_wrong_task_or_side_rate": float((~(top_task.eq(query_task) & top_side.eq(query_side))).mean()),
            }
        )
    return pd.DataFrame(rows)


def subject_bootstrap(diagnostics: pd.DataFrame, n_boot: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subject-aware bootstrap for ds test metrics."""

    rng = np.random.default_rng(RANDOM_SEED)
    summary_rows = []
    sample_rows = []
    for (condition, variant), subset in diagnostics.groupby(["candidate_pool_condition", "variant_name"], dropna=False):
        subjects = sorted(subset["query_subject"].astype(str).unique())
        observed = query_metrics_dict(subset)
        for idx in range(n_boot):
            draw = rng.choice(subjects, size=len(subjects), replace=True)
            rows = pd.concat([subset[subset["query_subject"].astype(str).eq(subject)] for subject in draw], ignore_index=True)
            sample_rows.append(
                {
                    "bootstrap_index": int(idx),
                    "candidate_pool_condition": condition,
                    "variant_name": variant,
                    "top1": float(rows["top_ranked_is_true_pair"].astype(bool).mean()),
                    "mrr": float(pd.to_numeric(rows["reciprocal_rank"], errors="coerce").mean()),
                    "n_query_rows": int(len(rows)),
                    "seed": RANDOM_SEED,
                    "bootstrap_unit": "subject",
                }
            )
        samples = pd.DataFrame([row for row in sample_rows if row["variant_name"] == variant])
        for metric in ["top1", "mrr"]:
            vals = pd.to_numeric(samples[metric], errors="coerce").dropna().to_numpy(dtype=float)
            summary_rows.append(
                {
                    "candidate_pool_condition": condition,
                    "variant_name": variant,
                    "metric": metric,
                    "observed": float(observed[metric]),
                    "ci_lower_95": float(np.percentile(vals, 2.5)) if len(vals) else np.nan,
                    "ci_upper_95": float(np.percentile(vals, 97.5)) if len(vals) else np.nan,
                    "n_boot": n_boot,
                    "seed": RANDOM_SEED,
                    "bootstrap_unit": "subject",
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def query_metrics_dict(diagnostics: pd.DataFrame) -> dict[str, float]:
    """Small metric helper for a single diagnostic subset."""

    rank = pd.to_numeric(diagnostics["true_medon_rank"], errors="coerce")
    return {
        "n_pairs": float(len(diagnostics)),
        "top1": float(diagnostics["top_ranked_is_true_pair"].astype(bool).mean()),
        "mrr": float(pd.to_numeric(diagnostics["reciprocal_rank"], errors="coerce").mean()),
        "top5": float((rank <= 5).mean()),
    }


def model_table(rankers: list[FrozenRanker]) -> pd.DataFrame:
    """Flatten frozen coefficients."""

    rows = []
    for ranker in rankers:
        for feature in ranker.feature_columns:
            rows.append(
                {
                    "ranker": ranker.label,
                    "train_source": ranker.train_source,
                    "feature": feature,
                    "coefficient": ranker.coefficients[feature],
                    "train_mean": ranker.mean[feature],
                    "train_scale": ranker.scale[feature],
                    "train_fill_value": ranker.train_fill_values[feature],
                    "intercept": ranker.intercept,
                    "train_rows": ranker.train_rows,
                    "train_queries": ranker.train_queries,
                    "train_positive_rows": ranker.train_positive_rows,
                }
            )
    return pd.DataFrame(rows)


def compare_metrics(test_metrics: pd.DataFrame, native_reference: pd.DataFrame) -> pd.DataFrame:
    """Compare frozen transfer metrics with the ds native reference row."""

    rows = []
    ref = native_reference[
        native_reference["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & native_reference["variant_name"].astype(str).eq(V2D_VARIANT)
    ]
    ref_top1 = float(ref.iloc[0]["top1"]) if not ref.empty else np.nan
    ref_mrr = float(ref.iloc[0]["mrr"]) if not ref.empty else np.nan
    for _, row in test_metrics.iterrows():
        rows.append(
            {
                "comparison": f"{row['variant_name']}_vs_ds_native_{V2D_VARIANT}",
                "variant_name": row["variant_name"],
                "top1": float(row["top1"]),
                "mrr": float(row["mrr"]),
                "native_ds_v2d_top1": ref_top1,
                "native_ds_v2d_mrr": ref_mrr,
                "delta_top1_vs_native_ds_v2d": float(row["top1"]) - ref_top1 if np.isfinite(ref_top1) else np.nan,
                "delta_mrr_vs_native_ds_v2d": float(row["mrr"]) - ref_mrr if np.isfinite(ref_mrr) else np.nan,
            }
        )
    pivot = test_metrics.set_index("variant_name")
    base_name = "OXF_frozen_no_MAGE_G__v2D_quality_deconfounded_cycle"
    mage_name = "OXF_frozen_MAGE_G_source_span_soft__v2D_quality_deconfounded_cycle"
    if base_name in pivot.index and mage_name in pivot.index:
        rows.append(
            {
                "comparison": "frozen_MAGE_G_vs_frozen_no_MAGE_G_on_ds",
                "variant_name": mage_name,
                "top1": float(pivot.loc[mage_name, "top1"]),
                "mrr": float(pivot.loc[mage_name, "mrr"]),
                "native_ds_v2d_top1": ref_top1,
                "native_ds_v2d_mrr": ref_mrr,
                "delta_top1_vs_native_ds_v2d": float(pivot.loc[mage_name, "top1"]) - ref_top1 if np.isfinite(ref_top1) else np.nan,
                "delta_mrr_vs_native_ds_v2d": float(pivot.loc[mage_name, "mrr"]) - ref_mrr if np.isfinite(ref_mrr) else np.nan,
                "delta_top1_vs_frozen_no_MAGE_G": float(pivot.loc[mage_name, "top1"]) - float(pivot.loc[base_name, "top1"]),
                "delta_mrr_vs_frozen_no_MAGE_G": float(pivot.loc[mage_name, "mrr"]) - float(pivot.loc[base_name, "mrr"]),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: dict[str, object], path: Path) -> Path:
    """Write a concise Markdown report."""

    lines = [
        "# OXF-Frozen Reverse Transfer to ds004998",
        "",
        "Scope: OXF-trained frozen rankers tested once on cached ds004998 STN-only candidate scores.",
        "",
        "## Key Results",
    ]
    for key, value in summary["key_results"].items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.3f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Geometry Transport"])
    for row in summary["geometry_transport"]:
        lines.append(
            f"- {row['feature']}: present_in_ds={row['present_in_ds_cached_scores']}, "
            f"default={row['applied_default']}, policy={row['policy']}"
        )
    lines.extend(["", "## Warnings"])
    for warning in summary["warnings"]:
        lines.append(f"- {warning}")
    return write_text(lines, path)


def run() -> dict[str, object]:
    """Run the reverse-transfer experiment."""

    ensure_dirs()
    oxf_base = add_rank_fraction(filter_scores(OXF_RAW_SCORES, V2D_VARIANT, RAW_RECON_BRANCH))
    oxf_mage_g = add_rank_fraction(filter_scores(OXF_MAGE_G_SCORES, MAGE_G_VARIANT))
    ds_base = add_rank_fraction(filter_scores(DS_STN_SCORES, V2D_VARIANT))
    ds_pairs = read_csv(DS_FROZEN_PAIRS)
    ds_manifest = read_csv(DS_CHANNEL_MANIFEST)
    ds_pair_geometry = build_ds_pair_contact_geometry(ds_pairs, ds_manifest)
    ds_mage_g_view, geometry_transport = make_ds_mage_g_transport_view(ds_base, ds_pair_geometry)

    no_mage_ranker = fit_ranker(
        "no_MAGE_G",
        "OXF raw reconstruction v2D all-MedOn",
        oxf_base,
        COMMON_SCORE_FEATURES,
    )
    mage_g_ranker = fit_ranker(
        "MAGE_G_source_span_soft",
        "OXF MAGE-G source/span soft v2D all-MedOn",
        oxf_mage_g,
        MAGE_G_FEATURES,
    )
    rankers = [no_mage_ranker, mage_g_ranker]

    oxf_train_scores = pd.concat(
        [
            apply_ranker(oxf_base, no_mage_ranker, "OXF_train_fit_no_MAGE_G", "oxf_train_fit"),
            apply_ranker(oxf_mage_g, mage_g_ranker, "OXF_train_fit_MAGE_G_source_span_soft", "oxf_train_fit"),
        ],
        ignore_index=True,
    )
    oxf_train_diag = query_diagnostics(oxf_train_scores)
    oxf_train_metrics = metrics_from_diagnostics(oxf_train_diag)

    ds_test_scores = pd.concat(
        [
            apply_ranker(
                ds_base,
                no_mage_ranker,
                "OXF_frozen_no_MAGE_G__v2D_quality_deconfounded_cycle",
                "ds004998_test",
            ),
            apply_ranker(
                ds_mage_g_view,
                mage_g_ranker,
                "OXF_frozen_MAGE_G_source_span_soft__v2D_quality_deconfounded_cycle",
                "ds004998_test",
            ),
        ],
        ignore_index=True,
    )
    ds_diag = query_diagnostics(ds_test_scores)
    ds_metrics = metrics_from_diagnostics(ds_diag)
    bootstrap_ci, bootstrap_samples = subject_bootstrap(ds_diag)
    native_reference = read_csv(DS_STN_METRICS)
    comparison = compare_metrics(ds_metrics, native_reference)

    metrics_index = ds_metrics.set_index("variant_name")
    no_mage_name = "OXF_frozen_no_MAGE_G__v2D_quality_deconfounded_cycle"
    mage_name = "OXF_frozen_MAGE_G_source_span_soft__v2D_quality_deconfounded_cycle"
    native = native_reference[
        native_reference["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & native_reference["variant_name"].astype(str).eq(V2D_VARIANT)
    ]
    native_top1 = float(native.iloc[0]["top1"]) if not native.empty else np.nan
    native_mrr = float(native.iloc[0]["mrr"]) if not native.empty else np.nan
    no_mage_top1 = float(metrics_index.loc[no_mage_name, "top1"])
    no_mage_mrr = float(metrics_index.loc[no_mage_name, "mrr"])
    mage_top1 = float(metrics_index.loc[mage_name, "top1"])
    mage_mrr = float(metrics_index.loc[mage_name, "mrr"])
    geometry_available_pairs = int(ds_pair_geometry["geometry_available"].astype(bool).sum())
    geometry_missing_pairs = int((~ds_pair_geometry["geometry_available"].astype(bool)).sum())
    geometry_features_computed_count = int(
        geometry_transport["policy"].astype(str).str.contains("computed_from_ds004998").sum()
    )
    ds_geometry_rows_with_missing_distance = int(
        (
            pd.to_numeric(ds_mage_g_view["contact_center_distance"], errors="coerce").eq(
                MISSING_GEOMETRY_DISTANCE_DEFAULT
            )
            & (
                ds_mage_g_view["query_contact_center"].isna()
                | ds_mage_g_view["candidate_contact_center"].isna()
            )
        ).sum()
    )
    warnings = [
        "This is an exploratory reverse-transfer experiment; it is not a replacement for frozen ds004998 v2A.",
        "The OXF-trained logistic layer is fit on OXF labels and then frozen before ds004998 testing.",
        "ds004998 MAGE-G geometry is reconstructed from the frozen channel manifest as side-level usable STN contact sets, not as OXF bipolar-contact strings.",
        "The ds source/span terms therefore test geometry-aware transfer under a coarse side-montage approximation.",
        "No raw extraction, top_k, aperiodic_alpha, or existing frozen outputs are modified.",
    ]
    summary = {
        "key_results": {
            "seed": RANDOM_SEED,
            "top_k": TOP_K,
            "aperiodic_alpha": APERIODIC_ALPHA,
            "oxf_train_queries": int(oxf_base["query_pair_id"].nunique()),
            "ds_test_queries": int(ds_base["query_pair_id"].nunique()),
            "ds_native_v2d_top1": native_top1,
            "ds_native_v2d_mrr": native_mrr,
            "frozen_no_MAGE_G_ds_top1": no_mage_top1,
            "frozen_no_MAGE_G_ds_mrr": no_mage_mrr,
            "frozen_MAGE_G_ds_top1": mage_top1,
            "frozen_MAGE_G_ds_mrr": mage_mrr,
            "delta_MAGE_G_minus_no_MAGE_G_top1": mage_top1 - no_mage_top1,
            "delta_MAGE_G_minus_no_MAGE_G_mrr": mage_mrr - no_mage_mrr,
            "ds_contact_geometry_pairs": int(len(ds_pair_geometry)),
            "ds_contact_geometry_available_pairs": geometry_available_pairs,
            "ds_contact_geometry_missing_pairs": geometry_missing_pairs,
            "ds_geometry_features_computed_count": geometry_features_computed_count,
            "ds_geometry_rows_with_missing_distance_default": ds_geometry_rows_with_missing_distance,
        },
        "inputs": {
            "oxf_raw_scores": str(OXF_RAW_SCORES),
            "oxf_mage_g_scores": str(OXF_MAGE_G_SCORES),
            "ds_stn_scores": str(DS_STN_SCORES),
            "ds_frozen_pairs": str(DS_FROZEN_PAIRS),
            "ds_channel_manifest": str(DS_CHANNEL_MANIFEST),
        },
        "geometry_transport": geometry_transport.to_dict("records"),
        "warnings": warnings,
    }

    paths = {
        "summary_json": write_json(summary, OUTPUT_DIR / "oxf_frozen_reverse_transfer_summary.json"),
        "summary_md": write_report(summary, OUTPUT_DIR / "oxf_frozen_reverse_transfer_summary.md"),
        "frozen_model_coefficients": write_csv(model_table(rankers), OUTPUT_DIR / "oxf_frozen_ranker_coefficients.csv"),
        "ds_pair_contact_geometry": write_csv(ds_pair_geometry, OUTPUT_DIR / "ds004998_pair_contact_geometry.csv"),
        "geometry_transport": write_csv(geometry_transport, OUTPUT_DIR / "ds004998_mage_g_geometry_transport_policy.csv"),
        "oxf_train_fit_scores": write_csv(oxf_train_scores, OUTPUT_DIR / "oxf_train_fit_candidate_scores.csv"),
        "oxf_train_fit_metrics": write_csv(oxf_train_metrics, OUTPUT_DIR / "oxf_train_fit_observed_metrics.csv"),
        "oxf_train_fit_diagnostics": write_csv(oxf_train_diag, OUTPUT_DIR / "oxf_train_fit_query_diagnostics.csv"),
        "ds_test_scores": write_csv(ds_test_scores, OUTPUT_DIR / "ds004998_oxf_frozen_candidate_scores.csv"),
        "ds_test_metrics": write_csv(ds_metrics, OUTPUT_DIR / "ds004998_oxf_frozen_observed_metrics.csv"),
        "ds_test_diagnostics": write_csv(ds_diag, OUTPUT_DIR / "ds004998_oxf_frozen_query_diagnostics.csv"),
        "ds_test_bootstrap_ci": write_csv(bootstrap_ci, OUTPUT_DIR / "ds004998_oxf_frozen_subject_bootstrap_ci.csv"),
        "ds_test_bootstrap_samples": write_csv(bootstrap_samples, OUTPUT_DIR / "ds004998_oxf_frozen_subject_bootstrap_samples.csv"),
        "comparison": write_csv(comparison, OUTPUT_DIR / "ds004998_oxf_frozen_comparison.csv"),
    }
    print(f"output folder: {OUTPUT_DIR}")
    for key, value in summary["key_results"].items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    print("warnings:")
    for warning in warnings:
        print(f"- {warning}")
    return {"paths": {key: str(path) for key, path in paths.items()}, "summary": summary}


def main() -> None:
    """Entry point."""

    run()


if __name__ == "__main__":
    main()
