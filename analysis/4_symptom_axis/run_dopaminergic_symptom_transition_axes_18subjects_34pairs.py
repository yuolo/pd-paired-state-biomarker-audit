"""Symptom-specific dopaminergic transition-axis analysis for ds004998.

This script implements a fixed-plan exploratory analysis over cached
BYJoWR-inclusive ds004998 outputs. It does not change frozen v2A retrieval
logic, does not reread raw FIF files, and does not use the Oxford/MRC external
dataset.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from itertools import combinations
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
from scipy.stats import pearsonr, rankdata, spearmanr  # noqa: E402
from sklearn.linear_model import HuberRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from scripts.run_retrieval_distance_geometry_improvement import to_builtin  # noqa: E402


RANDOM_SEED = 42
N_BOOT = 5000
N_PERM = 5000
RIDGE_ALPHA = 1.0
EXPECTED_COMPLETE_PAIRS = 34
EXPECTED_SUBJECTS = 18
TOP_K = 5
APERIODIC_ALPHA = 0.5

RAW_DS004998_DIR = Path("data/raw/ds004998")
FROZEN_DIR = Path("outputs/v2A_frozen_18subjects_34pairs")
COHORT_DIR = Path("outputs/ds004998_full_cohort_completeness_audit_18subjects_34pairs")
OUTPUT_DIR = Path("outputs/dopaminergic_symptom_transition_axes_18subjects_34pairs")
MAPPING_PATH = Path("config/dopaminergic_symptom_transition_feature_component_mapping.csv")

PARTICIPANTS_PATH = RAW_DS004998_DIR / "participants.tsv"
UPDRS_OFF_PATH = RAW_DS004998_DIR / "participants_updrs_off.tsv"
UPDRS_ON_PATH = RAW_DS004998_DIR / "participants_updrs_on.tsv"

REQUIRED_INPUTS = {
    "mapping": MAPPING_PATH,
    "participants": PARTICIPANTS_PATH,
    "updrs_off": UPDRS_OFF_PATH,
    "updrs_on": UPDRS_ON_PATH,
    "pairs": FROZEN_DIR / "frozen_v2a_pairs.csv",
    "frozen_summary": FROZEN_DIR / "frozen_v2a_summary.json",
    "logical_manifest": COHORT_DIR / "local_logical_recording_manifest.csv",
    "usable_pair_inventory": COHORT_DIR / "usable_pair_inventory.csv",
    "excluded_subjects": COHORT_DIR / "excluded_subjects.csv",
    "cohort_summary": COHORT_DIR / "cohort_completeness_summary.json",
    "extraction_failure_log": FROZEN_DIR / "tables" / "extraction_failure_log.csv",
}

ALLOWED_COMPONENTS = [
    "aperiodic_normalization",
    "beta_power_residual_suppression",
    "cortico_stn_beta_desynchronization",
    "gamma_restoration",
    "coupling_asymmetry_transition",
    "compact_v2a_transition",
]

PRIMARY_AXIS_COMPONENTS = [
    "aperiodic_normalization",
    "beta_power_residual_suppression",
    "cortico_stn_beta_desynchronization",
    "gamma_restoration",
    "coupling_asymmetry_transition",
]

SYMPTOM_SPECS = [
    ("AR", "contralateral", "contralateral_ar_response", "primary"),
    ("AR", "ipsilateral", "ipsilateral_ar_response", "control"),
    ("tremor", "contralateral", "contralateral_trem_response", "primary"),
    ("tremor", "ipsilateral", "ipsilateral_trem_response", "control"),
    ("axial", "contralateral", "axial_response", "secondary_nonlateralized"),
    ("axial", "ipsilateral", "axial_response", "secondary_nonlateralized_duplicate_for_fdr_family"),
    ("total", "contralateral", "sum_response", "secondary_nonlateralized"),
    ("total", "ipsilateral", "sum_response", "secondary_nonlateralized_duplicate_for_fdr_family"),
]

AXIS_TARGETS = [
    ("contralateral_ar_response", "AR_contralateral", "primary"),
    ("contralateral_trem_response", "tremor_contralateral", "primary_guarded"),
    ("sum_response", "total_updrs", "secondary"),
    ("axial_response", "axial", "secondary"),
]


@dataclass(frozen=True)
class AxisResult:
    axis_name: str
    target_column: str
    status: str
    predictions: pd.DataFrame
    weights: pd.DataFrame
    summary: dict[str, object]
    perm_samples: pd.DataFrame
    boot_ci: pd.DataFrame


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_json(data: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(data), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_text(lines: Iterable[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_csv_required(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path, **kwargs)


def read_tsv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required TSV: {path}")
    return pd.read_csv(path, sep="\t", encoding="utf-8-sig")


def read_json_required(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def bool_from_any(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / den


def finite_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> tuple[int, float, float]:
    data = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(data) < 4 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return int(len(data)), float("nan"), float("nan")
    if method == "pearson":
        stat, pval = pearsonr(data["x"].to_numpy(dtype=float), data["y"].to_numpy(dtype=float))
    else:
        stat, pval = spearmanr(data["x"].to_numpy(dtype=float), data["y"].to_numpy(dtype=float))
    return int(len(data)), float(stat), float(pval)


def spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 4:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    if len(np.unique(xv)) < 2 or len(np.unique(yv)) < 2:
        return float("nan")
    xr = rankdata(xv, method="average")
    yr = rankdata(yv, method="average")
    if np.nanstd(xr) == 0 or np.nanstd(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def huber_slope(x: pd.Series, y: pd.Series) -> tuple[int, float, float, str]:
    data = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(data) < 6 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return int(len(data)), float("nan"), float("nan"), "insufficient_variation"
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data[["x"]].to_numpy(dtype=float))
    model = HuberRegressor()
    model.fit(x_scaled, data["y"].to_numpy(dtype=float))
    return int(len(data)), float(model.coef_[0]), float(model.intercept_), "ok"


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return pd.Series(q, index=p_values.index)
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for pos in range(m - 1, -1, -1):
        running = min(running, ranked[pos] * m / (pos + 1))
        adjusted[pos] = running
    q[order] = np.minimum(adjusted, 1.0)
    return pd.Series(q, index=p_values.index)


def percentile_ci(values: Iterable[float]) -> tuple[float, float]:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def validate_static_mapping(mapping: pd.DataFrame, available_columns: set[str]) -> pd.DataFrame:
    required = {
        "source_column",
        "transition_column",
        "component",
        "expected_direction",
        "include_in_primary_axis",
        "notes",
    }
    missing = required - set(mapping.columns)
    if missing:
        raise AssertionError(f"feature-component mapping missing columns: {sorted(missing)}")
    unknown_components = sorted(set(mapping["component"].astype(str)) - set(ALLOWED_COMPONENTS))
    if unknown_components:
        raise AssertionError(f"feature-component mapping has unknown components: {unknown_components}")
    allowed_directions = {"positive", "negative", "magnitude"}
    unknown_directions = sorted(set(mapping["expected_direction"].astype(str)) - allowed_directions)
    if unknown_directions:
        raise AssertionError(f"feature-component mapping has unknown expected_direction values: {unknown_directions}")
    missing_transition = sorted(set(mapping["transition_column"].astype(str)) - available_columns)
    if missing_transition:
        raise AssertionError(f"mapping references missing transition columns: {missing_transition}")
    mapping = mapping.copy()
    mapping["include_in_primary_axis"] = mapping["include_in_primary_axis"].map(bool_from_any)
    mapping["mapping_frozen_before_analysis"] = True
    return mapping


def validate_inputs() -> dict[str, object]:
    missing = [str(path) for path in REQUIRED_INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + "; ".join(missing))

    pairs = read_csv_required(REQUIRED_INPUTS["pairs"], dtype={"run_off": str, "run_on": str})
    logical = read_csv_required(REQUIRED_INPUTS["logical_manifest"])
    usable = read_csv_required(REQUIRED_INPUTS["usable_pair_inventory"])
    excluded = read_csv_required(REQUIRED_INPUTS["excluded_subjects"])
    cohort_summary = read_json_required(REQUIRED_INPUTS["cohort_summary"])
    frozen_summary = read_json_required(REQUIRED_INPUTS["frozen_summary"])
    failure_log = read_csv_required(REQUIRED_INPUTS["extraction_failure_log"])

    complete_pairs = int(len(pairs))
    subjects = int(pairs["subject"].nunique())
    if complete_pairs != EXPECTED_COMPLETE_PAIRS:
        raise AssertionError(f"complete_pairs={complete_pairs}, expected={EXPECTED_COMPLETE_PAIRS}")
    if subjects != EXPECTED_SUBJECTS:
        raise AssertionError(f"subjects={subjects}, expected={EXPECTED_SUBJECTS}")
    if "sub-BYJoWR" not in set(pairs["subject"].astype(str)):
        raise AssertionError("sub-BYJoWR is not included in the 18-subject frozen pairs.")
    byjo = pairs[pairs["subject"].astype(str).eq("sub-BYJoWR")]
    if len(byjo) != 1 or str(byjo.iloc[0]["task_original"]) != "HoldR":
        raise AssertionError("sub-BYJoWR expected exactly one complete HoldR pair.")
    if str(byjo.iloc[0].get("run_off", "")) != "2" or str(byjo.iloc[0].get("run_on", "")) != "2":
        raise AssertionError("sub-BYJoWR expected run-2 MedOff/MedOn pair.")
    if "sub-BYJoWR" in set(excluded.get("subject", pd.Series(dtype=str)).astype(str)):
        raise AssertionError("sub-BYJoWR is still marked as excluded.")
    if not failure_log.empty:
        raise AssertionError("Extraction failure log should remain empty.")
    if int(frozen_summary.get("top_k", TOP_K)) != TOP_K:
        raise AssertionError("Frozen top_k changed.")
    if not math.isclose(float(frozen_summary.get("aperiodic_alpha", APERIODIC_ALPHA)), APERIODIC_ALPHA):
        raise AssertionError("Frozen aperiodic_alpha changed.")
    if any(pairs["task_family"].astype(str).str.lower().eq("rest")):
        raise AssertionError("Rest task leaked into frozen pairs.")
    jy = usable[
        usable["subject"].astype(str).eq("sub-jyC0j3")
        & usable["task"].astype(str).eq("MoveR")
    ]
    split_collapsed = bool(len(jy) == 1 and int(jy.iloc[0].get("medon_split_count", 0)) == 2)
    if not split_collapsed:
        raise AssertionError("sub-jyC0j3 MoveR MedOn split files were not collapsed logically.")

    validation = {
        "complete_pairs": complete_pairs,
        "subjects_with_complete_pairs": subjects,
        "downloaded_meg_subjects": int(cohort_summary.get("downloaded_meg_subjects", -1)),
        "downloaded_logical_holdmove_recordings": int(cohort_summary.get("downloaded_logical_holdmove_recordings", -1)),
        "sub_BYJoWR_included": True,
        "sub_BYJoWR_holdr_run2_pair": True,
        "rest_excluded": True,
        "split_files_collapsed_logically": split_collapsed,
        "extraction_failure_log_rows": int(len(failure_log)),
        "top_k": TOP_K,
        "aperiodic_alpha": APERIODIC_ALPHA,
        "frozen_v2A_changed": False,
        "external_oxford_mrc_dataset_used": False,
    }
    return {
        "pairs": pairs,
        "logical": logical,
        "usable": usable,
        "excluded": excluded,
        "cohort_summary": cohort_summary,
        "frozen_summary": frozen_summary,
        "validation": validation,
    }


def prepare_clinical_targets(pair_subjects: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    participants = read_tsv_required(PARTICIPANTS_PATH)
    off = read_tsv_required(UPDRS_OFF_PATH)
    on = read_tsv_required(UPDRS_ON_PATH)

    participants = participants.rename(columns={"participant_id": "subject"})
    off = off.rename(columns={"participant_id": "subject"})
    on = on.rename(columns={"participant_id": "subject"})

    off_cols = {
        "MEG_UPDRS_SAMEDAY": "updrs_off_sameday",
        "SUM": "updrs_off_sum",
        "AR right": "updrs_off_ar_right",
        "AR left": "updrs_off_ar_left",
        "trem right": "updrs_off_trem_right",
        "trem left": "updrs_off_trem_left",
        "AR sum": "updrs_off_ar_sum",
        "trem sum": "updrs_off_trem_sum",
        "axial": "updrs_off_axial",
    }
    on_cols = {
        "MEG_UPDRS_SAMEDAY": "updrs_on_sameday",
        "SUM": "updrs_on_sum",
        "AR right": "updrs_on_ar_right",
        "AR left": "updrs_on_ar_left",
        "trem right": "updrs_on_trem_right",
        "trem left": "updrs_on_trem_left",
        "AR sum": "updrs_on_ar_sum",
        "trem sum": "updrs_on_trem_sum",
        "axial": "updrs_on_axial",
    }
    keep_off = ["subject", *off_cols.keys()]
    keep_on = ["subject", *on_cols.keys()]
    clinical = participants.merge(off[keep_off].rename(columns=off_cols), on="subject", how="left")
    clinical = clinical.merge(on[keep_on].rename(columns=on_cols), on="subject", how="left")
    clinical = clinical[clinical["subject"].astype(str).isin(pair_subjects)].copy()

    numeric_cols = [
        "age",
        "updrs_off_sum",
        "updrs_off_ar_right",
        "updrs_off_ar_left",
        "updrs_off_trem_right",
        "updrs_off_trem_left",
        "updrs_off_ar_sum",
        "updrs_off_trem_sum",
        "updrs_off_axial",
        "updrs_on_sum",
        "updrs_on_ar_right",
        "updrs_on_ar_left",
        "updrs_on_trem_right",
        "updrs_on_trem_left",
        "updrs_on_ar_sum",
        "updrs_on_trem_sum",
        "updrs_on_axial",
    ]
    for col in numeric_cols:
        if col in clinical.columns:
            clinical[col] = pd.to_numeric(clinical[col], errors="coerce")
    clinical["disease_duration_years"] = clinical.get("diseaseDuration", pd.Series(dtype=object)).astype(str).str.extract(
        r"(-?\d+(?:\.\d+)?)"
    )[0]
    clinical["disease_duration_years"] = pd.to_numeric(clinical["disease_duration_years"], errors="coerce")

    clinical["sum_response"] = clinical["updrs_off_sum"] - clinical["updrs_on_sum"]
    clinical["sum_percent_response"] = safe_div(clinical["sum_response"], clinical["updrs_off_sum"])
    clinical["ar_right_response"] = clinical["updrs_off_ar_right"] - clinical["updrs_on_ar_right"]
    clinical["ar_left_response"] = clinical["updrs_off_ar_left"] - clinical["updrs_on_ar_left"]
    clinical["ar_sum_response"] = clinical["updrs_off_ar_sum"] - clinical["updrs_on_ar_sum"]
    clinical["ar_sum_percent_response"] = safe_div(clinical["ar_sum_response"], clinical["updrs_off_ar_sum"])
    clinical["trem_right_response"] = clinical["updrs_off_trem_right"] - clinical["updrs_on_trem_right"]
    clinical["trem_left_response"] = clinical["updrs_off_trem_left"] - clinical["updrs_on_trem_left"]
    clinical["trem_sum_response"] = clinical["updrs_off_trem_sum"] - clinical["updrs_on_trem_sum"]
    clinical["trem_sum_percent_response"] = safe_div(clinical["trem_sum_response"], clinical["updrs_off_trem_sum"])
    clinical["axial_response"] = clinical["updrs_off_axial"] - clinical["updrs_on_axial"]
    clinical["axial_percent_response"] = safe_div(clinical["axial_response"], clinical["updrs_off_axial"])
    clinical["clinical_anchor_available"] = clinical[["updrs_off_sum", "updrs_on_sum"]].notna().all(axis=1)

    item_mapping = pd.DataFrame(
        [
            {
                "target": "AR_right",
                "off_column": "AR right",
                "on_column": "AR right",
                "body_side": "right",
                "contralateral_stn": "left",
                "role": "primary_lateralized_ar",
            },
            {
                "target": "AR_left",
                "off_column": "AR left",
                "on_column": "AR left",
                "body_side": "left",
                "contralateral_stn": "right",
                "role": "primary_lateralized_ar",
            },
            {
                "target": "trem_right",
                "off_column": "trem right",
                "on_column": "trem right",
                "body_side": "right",
                "contralateral_stn": "left",
                "role": "primary_guarded_lateralized_tremor",
            },
            {
                "target": "trem_left",
                "off_column": "trem left",
                "on_column": "trem left",
                "body_side": "left",
                "contralateral_stn": "right",
                "role": "primary_guarded_lateralized_tremor",
            },
            {
                "target": "axial",
                "off_column": "axial",
                "on_column": "axial",
                "body_side": "nonlateralized",
                "contralateral_stn": "not_applicable",
                "role": "secondary_nonlateralized",
            },
            {
                "target": "SUM",
                "off_column": "SUM",
                "on_column": "SUM",
                "body_side": "nonlateralized",
                "contralateral_stn": "not_applicable",
                "role": "secondary_total",
            },
        ]
    )

    more_rows = []
    lateral_rows = []
    for _, row in clinical.iterrows():
        subject = str(row["subject"])
        ar_right = row["updrs_off_ar_right"]
        ar_left = row["updrs_off_ar_left"]
        trem_right = row["updrs_off_trem_right"]
        trem_left = row["updrs_off_trem_left"]
        ar_side = more_affected_body_side(ar_left, ar_right)
        trem_side = more_affected_body_side(trem_left, trem_right)
        more_rows.append(
            {
                "subject": subject,
                "ar_more_affected_body_side": ar_side,
                "ar_more_affected_stn_hemisphere": contralateral_stn(ar_side),
                "ar_off_left": ar_left,
                "ar_off_right": ar_right,
                "ar_tie": ar_side == "tie",
                "trem_more_affected_body_side": trem_side,
                "trem_more_affected_stn_hemisphere": contralateral_stn(trem_side),
                "trem_off_left": trem_left,
                "trem_off_right": trem_right,
                "trem_tie": trem_side == "tie",
            }
        )
        for hemi in ["left", "right"]:
            contra_body = "right" if hemi == "left" else "left"
            ipsi_body = "left" if hemi == "left" else "right"
            lateral_rows.append(
                {
                    "subject": subject,
                    "hemisphere": hemi,
                    "contralateral_body_side": contra_body,
                    "ipsilateral_body_side": ipsi_body,
                    "contralateral_ar_off": row[f"updrs_off_ar_{contra_body}"],
                    "ipsilateral_ar_off": row[f"updrs_off_ar_{ipsi_body}"],
                    "contralateral_ar_response": row[f"ar_{contra_body}_response"],
                    "ipsilateral_ar_response": row[f"ar_{ipsi_body}_response"],
                    "contralateral_ar_percent_response": safe_scalar_div(
                        row[f"ar_{contra_body}_response"], row[f"updrs_off_ar_{contra_body}"]
                    ),
                    "ipsilateral_ar_percent_response": safe_scalar_div(
                        row[f"ar_{ipsi_body}_response"], row[f"updrs_off_ar_{ipsi_body}"]
                    ),
                    "contralateral_trem_off": row[f"updrs_off_trem_{contra_body}"],
                    "ipsilateral_trem_off": row[f"updrs_off_trem_{ipsi_body}"],
                    "contralateral_trem_response": row[f"trem_{contra_body}_response"],
                    "ipsilateral_trem_response": row[f"trem_{ipsi_body}_response"],
                    "contralateral_trem_percent_response": safe_scalar_div(
                        row[f"trem_{contra_body}_response"], row[f"updrs_off_trem_{contra_body}"]
                    ),
                    "ipsilateral_trem_percent_response": safe_scalar_div(
                        row[f"trem_{ipsi_body}_response"], row[f"updrs_off_trem_{ipsi_body}"]
                    ),
                    "sum_response": row["sum_response"],
                    "sum_percent_response": row["sum_percent_response"],
                    "axial_response": row["axial_response"],
                    "axial_percent_response": row["axial_percent_response"],
                    "clinical_anchor_available": row["clinical_anchor_available"],
                }
            )

    lateralized = pd.DataFrame(lateral_rows)
    more_affected = pd.DataFrame(more_rows)
    clinical = clinical.sort_values("subject").reset_index(drop=True)
    lateralized = lateralized.sort_values(["subject", "hemisphere"]).reset_index(drop=True)
    more_affected = more_affected.sort_values("subject").reset_index(drop=True)
    return clinical, item_mapping, lateralized, more_affected


def safe_scalar_div(num: object, den: object) -> float:
    num_f = pd.to_numeric(pd.Series([num]), errors="coerce").iloc[0]
    den_f = pd.to_numeric(pd.Series([den]), errors="coerce").iloc[0]
    if pd.isna(num_f) or pd.isna(den_f) or den_f == 0:
        return float("nan")
    return float(num_f / den_f)


def more_affected_body_side(left_value: object, right_value: object) -> str:
    left = pd.to_numeric(pd.Series([left_value]), errors="coerce").iloc[0]
    right = pd.to_numeric(pd.Series([right_value]), errors="coerce").iloc[0]
    if pd.isna(left) or pd.isna(right):
        return "unknown"
    if left > right:
        return "left"
    if right > left:
        return "right"
    return "tie"


def contralateral_stn(body_side: str) -> str:
    if body_side == "left":
        return "right"
    if body_side == "right":
        return "left"
    return body_side


def canonical_feature_name(transition_column: str) -> str:
    name = transition_column
    if name.startswith("y_"):
        name = name[2:]
    name = name.replace("stn_left_", "stn_hemi_")
    name = name.replace("stn_right_", "stn_hemi_")
    name = name.replace("motor_stn_left_", "motor_stn_hemi_")
    name = name.replace("motor_stn_right_", "motor_stn_hemi_")
    return name


def transition_applies_to_hemisphere(transition_column: str, hemisphere: str) -> bool:
    if "_left_" in transition_column:
        return hemisphere == "left"
    if "_right_" in transition_column:
        return hemisphere == "right"
    return True


def oriented_value(value: object, expected_direction: str) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float("nan")
    if expected_direction == "negative":
        return float(-numeric)
    if expected_direction == "magnitude":
        return float(abs(numeric))
    return float(numeric)


def raw_signed_value(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float("nan")
    return float(numeric)


def build_task_level_transition_features(pairs: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, pair in pairs.iterrows():
        for hemi in ["left", "right"]:
            out = {
                "pair_id": pair["pair_id"],
                "subject": pair["subject"],
                "hemisphere": hemi,
                "task_original": pair["task_original"],
                "task_family": pair["task_family"],
                "recording_side": pair["side"],
                "quality_flag": pair.get("quality_flag", "unknown"),
                "quality_match_source": pair.get("quality_match_source", ""),
            }
            component_values: dict[str, list[float]] = {component: [] for component in ALLOWED_COMPONENTS}
            component_signed: dict[str, list[float]] = {component: [] for component in ALLOWED_COMPONENTS}
            component_primary_values: dict[str, list[float]] = {component: [] for component in ALLOWED_COMPONENTS}
            for _, spec in mapping.iterrows():
                col = str(spec["transition_column"])
                if not transition_applies_to_hemisphere(col, hemi):
                    continue
                if col not in pair:
                    continue
                component = str(spec["component"])
                canonical = canonical_feature_name(col)
                raw = raw_signed_value(pair[col])
                oriented = oriented_value(pair[col], str(spec["expected_direction"]))
                out[f"feature__{canonical}"] = oriented
                out[f"signed_feature__{canonical}"] = raw
                component_values[component].append(oriented)
                component_signed[component].append(raw)
                if bool(spec["include_in_primary_axis"]):
                    component_primary_values[component].append(oriented)
            for component in ALLOWED_COMPONENTS:
                vals = pd.to_numeric(pd.Series(component_values[component]), errors="coerce").dropna()
                signed_vals = pd.to_numeric(pd.Series(component_signed[component]), errors="coerce").dropna()
                primary_vals = pd.to_numeric(pd.Series(component_primary_values[component]), errors="coerce").dropna()
                out[f"component__{component}__oriented_mean"] = float(vals.mean()) if len(vals) else float("nan")
                out[f"component__{component}__signed_mean"] = float(signed_vals.mean()) if len(signed_vals) else float("nan")
                out[f"component__{component}__norm"] = float(np.linalg.norm(vals.to_numpy(dtype=float))) if len(vals) else float("nan")
                out[f"component__{component}__n_features"] = int(len(vals))
                out[f"component__{component}__primary_oriented_mean"] = (
                    float(primary_vals.mean()) if len(primary_vals) else float("nan")
                )
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["subject", "hemisphere", "task_original"]).reset_index(drop=True)


def aggregate_hemisphere_features(task_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        col
        for col in task_features.columns
        if col.startswith("feature__")
        or col.startswith("signed_feature__")
        or col.startswith("component__")
    ]
    grouped = task_features.groupby(["subject", "hemisphere"], sort=True)
    mean_features = grouped[feature_cols].mean(numeric_only=True).reset_index()
    meta_rows = []
    for (subject, hemi), group in grouped:
        tasks = sorted(set(group["task_family"].astype(str)))
        qflags = sorted(set(group.get("quality_flag", pd.Series(["unknown"])).astype(str)))
        meta_rows.append(
            {
                "subject": subject,
                "hemisphere": hemi,
                "n_task_pairs_available": int(group["task_original"].nunique()),
                "has_hold_pair": bool("Hold" in tasks),
                "has_move_pair": bool("Move" in tasks),
                "task_availability_pattern": "+".join(tasks) if tasks else "none",
                "task_originals": ";".join(sorted(set(group["task_original"].astype(str)))),
                "quality_flags": ";".join(qflags),
                "has_unknown_quality": bool("unknown" in qflags),
                "has_bad_quality": bool("bad" in qflags),
            }
        )
    meta = pd.DataFrame(meta_rows)
    hemi = meta.merge(mean_features, on=["subject", "hemisphere"], how="left")
    task_audit = (
        hemi.groupby("task_availability_pattern")
        .agg(
            n_hemisphere_rows=("subject", "count"),
            n_subjects=("subject", "nunique"),
            has_unknown_quality_rate=("has_unknown_quality", "mean"),
            has_bad_quality_rate=("has_bad_quality", "mean"),
        )
        .reset_index()
    )
    return hemi.sort_values(["subject", "hemisphere"]).reset_index(drop=True), task_audit


def tremor_effective_sample_audit(clinical: pd.DataFrame, lateralized: pd.DataFrame) -> pd.DataFrame:
    trem_subject_off = clinical["updrs_off_trem_sum"]
    trem_subject_response = clinical["trem_sum_response"]
    hemi_off = lateralized["contralateral_trem_off"]
    hemi_response = lateralized["contralateral_trem_response"]

    rows = []
    for level, values in [
        ("subject_total_tremor_off", trem_subject_off),
        ("subject_total_tremor_response", trem_subject_response),
        ("hemisphere_contralateral_tremor_off", hemi_off),
        ("hemisphere_contralateral_tremor_response", hemi_response),
    ]:
        series = pd.to_numeric(values, errors="coerce").dropna()
        nonzero = series[series.ne(0)]
        rows.append(
            {
                "measure": level,
                "n_total": int(len(series)),
                "n_nonzero": int(len(nonzero)),
                "nonzero_rate": float(len(nonzero) / len(series)) if len(series) else float("nan"),
                "mean": float(series.mean()) if len(series) else float("nan"),
                "median": float(series.median()) if len(series) else float("nan"),
                "std": float(series.std(ddof=1)) if len(series) > 1 else float("nan"),
                "min": float(series.min()) if len(series) else float("nan"),
                "max": float(series.max()) if len(series) else float("nan"),
                "n_unique_values": int(series.nunique()) if len(series) else 0,
                "n_unique_nonzero_values": int(nonzero.nunique()) if len(nonzero) else 0,
            }
        )
    audit = pd.DataFrame(rows)
    subject_nonzero_off = int(audit.loc[audit["measure"].eq("subject_total_tremor_off"), "n_nonzero"].iloc[0])
    subject_nonzero_response = int(audit.loc[audit["measure"].eq("subject_total_tremor_response"), "n_nonzero"].iloc[0])
    unique_nonzero_response = int(
        audit.loc[audit["measure"].eq("subject_total_tremor_response"), "n_unique_nonzero_values"].iloc[0]
    )
    eligible = subject_nonzero_off >= 10 and subject_nonzero_response >= 8 and unique_nonzero_response >= 4
    status = "primary_axis_eligible" if eligible else "exploratory_or_underpowered"
    audit["tremor_axis_status"] = status
    audit["threshold_subject_nonzero_off_min"] = 10
    audit["threshold_subject_nonzero_response_min"] = 8
    audit["threshold_unique_nonzero_response_min"] = 4
    return audit


def merge_targets_and_features(hemi_features: pd.DataFrame, lateralized: pd.DataFrame) -> pd.DataFrame:
    merged = hemi_features.merge(lateralized, on=["subject", "hemisphere"], how="left")
    if merged[["contralateral_ar_response", "contralateral_trem_response"]].isna().all(axis=None):
        raise AssertionError("Clinical lateralized targets did not merge into hemisphere features.")
    return merged


def component_association_feature(component: str) -> str:
    return f"component__{component}__oriented_mean"


def subject_cluster_bootstrap_corr(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    subjects: np.ndarray,
    seed: int,
    n_boot: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = pd.to_numeric(data[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(data[y_col], errors="coerce").to_numpy(dtype=float)
    subject_array = data["subject"].astype(str).to_numpy()
    subject_to_idx = {subject: np.where(subject_array == subject)[0] for subject in subjects}
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([subject_to_idx[subject] for subject in sampled])
        values.append(spearman_np(x[idx], y[idx]))
    return percentile_ci(values)


def subject_label_permutation_p(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    observed: float,
    seed: int,
    n_perm: int,
) -> float:
    if not np.isfinite(observed):
        return float("nan")
    subjects = np.array(sorted(data["subject"].astype(str).unique()))
    x = pd.to_numeric(data[x_col], errors="coerce").to_numpy(dtype=float)
    row_subjects = data["subject"].astype(str).to_numpy()
    row_hemis = data["hemisphere"].astype(str).to_numpy()
    subject_target = {
        (str(row["subject"]), str(row["hemisphere"])): row[y_col]
        for _, row in data[["subject", "hemisphere", y_col]].iterrows()
    }
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        shuffled = rng.permutation(subjects)
        subject_map = dict(zip(subjects, shuffled))
        perm_y = np.array(
            [
                subject_target.get((subject_map[row_subjects[i]], row_hemis[i]), np.nan)
                for i in range(len(row_subjects))
            ],
            dtype=float,
        )
        null.append(spearman_np(x, perm_y))
    null_arr = pd.to_numeric(pd.Series(null), errors="coerce").dropna().to_numpy(dtype=float)
    if len(null_arr) == 0:
        return float("nan")
    return float((1 + np.sum(null_arr >= observed)) / (1 + len(null_arr)))


def component_symptom_associations(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subjects = np.array(sorted(data["subject"].astype(str).unique()))
    rows = []
    for symptom, laterality, target_col, family in SYMPTOM_SPECS:
        for component in ALLOWED_COMPONENTS:
            x_col = component_association_feature(component)
            n, rho, pval = finite_corr(data[x_col], data[target_col], method="spearman")
            _, slope, intercept, huber_status = huber_slope(data[x_col], data[target_col])
            ci_low, ci_high = subject_cluster_bootstrap_corr(
                data, x_col, target_col, subjects, RANDOM_SEED + len(rows), N_BOOT
            )
            perm_p = subject_label_permutation_p(
                data, x_col, target_col, rho, RANDOM_SEED + 1000 + len(rows), N_PERM
            )
            rows.append(
                {
                    "symptom": symptom,
                    "laterality": laterality,
                    "target_column": target_col,
                    "hypothesis_family": family,
                    "component": component,
                    "feature_column": x_col,
                    "n_rows": n,
                    "n_subjects": int(data.loc[data[target_col].notna() & data[x_col].notna(), "subject"].nunique()),
                    "spearman_r": rho,
                    "spearman_p": pval,
                    "association_bootstrap_ci_low": ci_low,
                    "association_bootstrap_ci_high": ci_high,
                    "subject_label_permutation_p_directional": perm_p,
                    "huber_slope": slope,
                    "huber_intercept": intercept,
                    "huber_status": huber_status,
                    "predicted_direction": "positive",
                    "direction_matches_prediction": bool(np.isfinite(rho) and rho > 0),
                    "primary_supported_directionally": bool(family == "primary" and np.isfinite(rho) and rho > 0),
                    "n_bootstrap": N_BOOT,
                    "n_permutations": N_PERM,
                    "seed": RANDOM_SEED,
                }
            )
    assoc = pd.DataFrame(rows)
    assoc["fdr_q_spearman_full_family"] = bh_fdr(assoc["spearman_p"])
    assoc["fdr_q_permutation_full_family"] = bh_fdr(assoc["subject_label_permutation_p_directional"])
    primary = assoc[assoc["hypothesis_family"].eq("primary")].copy()
    return assoc, assoc.copy(), primary


def contralateral_ipsilateral_control(data: pd.DataFrame, associations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    subjects = np.array(sorted(data["subject"].astype(str).unique()))
    subject_array = data["subject"].astype(str).to_numpy()
    subject_to_idx = {subject: np.where(subject_array == subject)[0] for subject in subjects}
    for symptom in ["AR", "tremor"]:
        for component in ALLOWED_COMPONENTS:
            contra = associations[
                associations["symptom"].eq(symptom)
                & associations["laterality"].eq("contralateral")
                & associations["component"].eq(component)
            ].iloc[0]
            ipsi = associations[
                associations["symptom"].eq(symptom)
                & associations["laterality"].eq("ipsilateral")
                & associations["component"].eq(component)
            ].iloc[0]
            observed = abs(float(contra["spearman_r"])) - abs(float(ipsi["spearman_r"]))
            boot_values = []
            c_target = f"contralateral_{'ar' if symptom == 'AR' else 'trem'}_response"
            i_target = f"ipsilateral_{'ar' if symptom == 'AR' else 'trem'}_response"
            x_col = component_association_feature(component)
            x = pd.to_numeric(data[x_col], errors="coerce").to_numpy(dtype=float)
            y_c = pd.to_numeric(data[c_target], errors="coerce").to_numpy(dtype=float)
            y_i = pd.to_numeric(data[i_target], errors="coerce").to_numpy(dtype=float)
            for _ in range(N_BOOT):
                sampled = rng.choice(subjects, size=len(subjects), replace=True)
                idx = np.concatenate([subject_to_idx[subject] for subject in sampled])
                r_c = spearman_np(x[idx], y_c[idx])
                r_i = spearman_np(x[idx], y_i[idx])
                boot_values.append(abs(r_c) - abs(r_i))
            ci_low, ci_high = percentile_ci(boot_values)
            null_values = []
            for _ in range(N_PERM):
                perm_c = y_c.copy()
                perm_i = y_i.copy()
                for subject in subjects:
                    if rng.random() < 0.5:
                        idx = subject_to_idx[subject]
                        old_c = perm_c[idx].copy()
                        perm_c[idx] = perm_i[idx]
                        perm_i[idx] = old_c
                r_c = spearman_np(x, perm_c)
                r_i = spearman_np(x, perm_i)
                null_values.append(abs(r_c) - abs(r_i))
            null_arr = pd.to_numeric(pd.Series(null_values), errors="coerce").dropna().to_numpy(dtype=float)
            pval = float((1 + np.sum(null_arr >= observed)) / (1 + len(null_arr))) if len(null_arr) else float("nan")
            rows.append(
                {
                    "symptom": symptom,
                    "component": component,
                    "contra_spearman_r": float(contra["spearman_r"]),
                    "ipsi_spearman_r": float(ipsi["spearman_r"]),
                    "abs_contra_minus_abs_ipsi": observed,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "permutation_p": pval,
                    "contra_direction_matches_prediction": bool(contra["direction_matches_prediction"]),
                    "contra_stronger_than_ipsi": bool(observed > 0),
                    "n_bootstrap": N_BOOT,
                    "n_permutations": N_PERM,
                    "seed": RANDOM_SEED,
                }
            )
    return pd.DataFrame(rows)


def component_collinearity(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [component_association_feature(component) for component in ALLOWED_COMPONENTS]
    renamed = {component_association_feature(component): component for component in ALLOWED_COMPONENTS}
    numeric = data[cols].rename(columns=renamed).apply(pd.to_numeric, errors="coerce")
    spearman = numeric.corr(method="spearman")
    pearson = numeric.corr(method="pearson")
    matrix_rows = []
    for component_a in ALLOWED_COMPONENTS:
        for component_b in ALLOWED_COMPONENTS:
            matrix_rows.append(
                {
                    "component_a": component_a,
                    "component_b": component_b,
                    "spearman_r": spearman.loc[component_a, component_b],
                    "pearson_r": pearson.loc[component_a, component_b],
                }
            )
    summary_rows = []
    cluster_rows = []
    for component in ALLOWED_COMPONENTS:
        others = spearman.loc[component].drop(index=component)
        max_abs = float(others.abs().max()) if len(others) else float("nan")
        strongest = str(others.abs().idxmax()) if len(others) else ""
        summary_rows.append(
            {
                "component": component,
                "max_abs_spearman_with_other_component": max_abs,
                "strongest_correlated_component": strongest,
                "collinearity_flag_abs_r_ge_0_70": bool(np.isfinite(max_abs) and max_abs >= 0.70),
            }
        )
    for a, b in combinations(ALLOWED_COMPONENTS, 2):
        rho = float(spearman.loc[a, b])
        if np.isfinite(rho) and abs(rho) >= 0.70:
            cluster_rows.append(
                {
                    "component_a": a,
                    "component_b": b,
                    "spearman_r": rho,
                    "interpretation": "coupled_component_cluster_do_not_claim_unique_component_dominance",
                }
            )
    return pd.DataFrame(matrix_rows), pd.DataFrame(summary_rows), pd.DataFrame(cluster_rows)


def prepare_axis_features(data: pd.DataFrame) -> list[str]:
    cols = [component_association_feature(component) for component in PRIMARY_AXIS_COMPONENTS]
    cols.extend([f"component__{component}__norm" for component in PRIMARY_AXIS_COMPONENTS])
    return [col for col in cols if col in data.columns]


def impute_with_train_medians(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    medians = train_x.median(numeric_only=True)
    train = train_x.fillna(medians).fillna(0.0)
    test = test_x.fillna(medians).fillna(0.0)
    return train, test


def fit_predict_ridge_closed_form(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit fixed-alpha ridge with train-only scaling using a closed-form solver."""

    train_x, test_x = impute_with_train_medians(train_x, test_x)
    return fit_predict_ridge_array(
        train_x.to_numpy(dtype=float),
        pd.to_numeric(train_y, errors="coerce").to_numpy(dtype=float),
        test_x.to_numpy(dtype=float),
    )


def fit_predict_ridge_array(
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast fixed-alpha ridge for numeric arrays."""

    medians = np.nanmedian(x_train_raw, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    train = np.where(np.isfinite(x_train_raw), x_train_raw, medians)
    test = np.where(np.isfinite(x_test_raw), x_test_raw, medians)
    x_mean = np.nanmean(x_train_raw, axis=0)
    x_mean[~np.isfinite(x_mean)] = 0.0
    x_std = np.nanstd(train, axis=0)
    x_std[~np.isfinite(x_std) | (x_std == 0)] = 1.0
    x_train = (train - x_mean) / x_std
    x_test = (test - x_mean) / x_std
    y = y_train_raw.astype(float)
    y_mean = float(np.nanmean(y))
    y_centered = y - y_mean
    penalty = RIDGE_ALPHA * np.eye(x_train.shape[1], dtype=float)
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_centered)
    pred = x_test @ beta + y_mean
    return pred, beta


def run_loso_ridge(data: pd.DataFrame, target_col: str, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_rows = []
    weight_rows = []
    subjects = sorted(data["subject"].astype(str).unique())
    for held_subject in subjects:
        train_mask = ~data["subject"].astype(str).eq(held_subject)
        test_mask = data["subject"].astype(str).eq(held_subject)
        train = data.loc[train_mask].copy()
        test = data.loc[test_mask].copy()
        train = train[train[target_col].notna()].copy()
        test = test[test[target_col].notna()].copy()
        if len(train) < 8 or train[target_col].nunique() < 2 or test.empty:
            for _, row in test.iterrows():
                pred_rows.append(
                    {
                        "held_out_subject": held_subject,
                        "subject": row["subject"],
                        "hemisphere": row["hemisphere"],
                        "target_column": target_col,
                        "y_true": row[target_col],
                        "y_pred": np.nan,
                        "fold_status": "insufficient_training_data",
                    }
                )
            continue
        y_pred, coef = fit_predict_ridge_closed_form(train[feature_cols], train[target_col], test[feature_cols])
        for feature, weight in zip(feature_cols, coef):
            weight_rows.append(
                {
                    "held_out_subject": held_subject,
                    "target_column": target_col,
                    "feature": feature,
                    "weight": float(weight),
                    "ridge_alpha": RIDGE_ALPHA,
                }
            )
        for pred, (_, row) in zip(y_pred, test.iterrows()):
            pred_rows.append(
                {
                    "held_out_subject": held_subject,
                    "subject": row["subject"],
                    "hemisphere": row["hemisphere"],
                    "target_column": target_col,
                    "y_true": row[target_col],
                    "y_pred": float(pred),
                    "fold_status": "ok",
                }
            )
    return pd.DataFrame(pred_rows), pd.DataFrame(weight_rows)


def prediction_metrics(predictions: pd.DataFrame) -> dict[str, float | int | str]:
    ok = predictions[predictions["fold_status"].eq("ok")].copy()
    ok = ok.dropna(subset=["y_true", "y_pred"])
    if len(ok) < 4 or ok["y_true"].nunique() < 2 or ok["y_pred"].nunique() < 2:
        return {
            "n_predictions": int(len(ok)),
            "spearman_r": float("nan"),
            "pearson_r": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "status": "insufficient_prediction_variation",
        }
    _, sr, sp = finite_corr(ok["y_pred"], ok["y_true"], "spearman")
    _, pr, pp = finite_corr(ok["y_pred"], ok["y_true"], "pearson")
    err = ok["y_pred"].to_numpy(dtype=float) - ok["y_true"].to_numpy(dtype=float)
    return {
        "n_predictions": int(len(ok)),
        "spearman_r": sr,
        "spearman_p": sp,
        "pearson_r": pr,
        "pearson_p": pp,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "status": "ok",
    }


def fast_loso_ridge_spearman(
    data: pd.DataFrame,
    y_values: np.ndarray,
    feature_cols: list[str],
) -> float:
    """Return LOSO ridge Spearman without constructing fold DataFrames."""

    x_all = data[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    subjects = data["subject"].astype(str).to_numpy()
    unique_subjects = np.array(sorted(np.unique(subjects)))
    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    for held_subject in unique_subjects:
        train_mask = (subjects != held_subject) & np.isfinite(y_values)
        test_mask = (subjects == held_subject) & np.isfinite(y_values)
        if int(train_mask.sum()) < 8 or int(test_mask.sum()) == 0:
            continue
        if len(np.unique(y_values[train_mask])) < 2:
            continue
        pred, _ = fit_predict_ridge_array(x_all[train_mask], y_values[train_mask], x_all[test_mask])
        y_true_all.extend(y_values[test_mask].tolist())
        y_pred_all.extend(pred.tolist())
    if len(y_true_all) < 4:
        return float("nan")
    return spearman_np(np.asarray(y_pred_all, dtype=float), np.asarray(y_true_all, dtype=float))


def weight_stability(weights: pd.DataFrame, feature_cols: list[str]) -> dict[str, object]:
    if weights.empty:
        return {
            "weight_stability_status": "no_weights",
            "fraction_features_sign_stable_80pct": 0.0,
            "median_pairwise_fold_weight_spearman": float("nan"),
            "axis_weight_stable": False,
        }
    pivot = weights.pivot_table(index="held_out_subject", columns="feature", values="weight", aggfunc="mean")
    pivot = pivot.reindex(columns=feature_cols)
    sign_rows = []
    for feature in feature_cols:
        vals = pd.to_numeric(pivot[feature], errors="coerce").dropna()
        if vals.empty:
            sign_rows.append((feature, 0.0, False))
            continue
        median = float(vals.median())
        median_sign = 1.0 if median >= 0 else -1.0
        sign_match = float((np.sign(vals.to_numpy(dtype=float)) == median_sign).mean())
        sign_rows.append((feature, sign_match, sign_match >= 0.80))
    pairwise = []
    for fold_a, fold_b in combinations(pivot.index, 2):
        a = pivot.loc[fold_a]
        b = pivot.loc[fold_b]
        valid = a.notna() & b.notna()
        if valid.sum() >= 3 and a[valid].nunique() > 1 and b[valid].nunique() > 1:
            pairwise.append(float(spearmanr(a[valid], b[valid]).statistic))
    median_pairwise = float(np.nanmedian(pairwise)) if pairwise else float("nan")
    stable_features = [row for row in sign_rows if row[2]]
    fraction_stable = float(len(stable_features) / len(sign_rows)) if sign_rows else 0.0
    axis_stable = bool(fraction_stable >= 0.80 and np.isfinite(median_pairwise) and median_pairwise >= 0.50)
    return {
        "weight_stability_status": "ok",
        "fraction_features_sign_stable_80pct": fraction_stable,
        "n_features_sign_stable": int(len(stable_features)),
        "n_features_total": int(len(sign_rows)),
        "median_pairwise_fold_weight_spearman": median_pairwise,
        "axis_weight_stable": axis_stable,
        "sign_consistency_threshold": 0.80,
        "fold_weight_spearman_threshold": 0.50,
    }


def permute_subject_targets(data: pd.DataFrame, target_col: str, rng: np.random.Generator) -> pd.DataFrame:
    subjects = np.array(sorted(data["subject"].astype(str).unique()))
    shuffled = rng.permutation(subjects)
    subject_map = dict(zip(subjects, shuffled))
    source_targets = {
        (str(row["subject"]), str(row["hemisphere"])): row[target_col]
        for _, row in data[["subject", "hemisphere", target_col]].iterrows()
    }
    out = data.copy()
    out[target_col] = [
        source_targets.get((subject_map[str(row["subject"])], str(row["hemisphere"])), np.nan)
        for _, row in data[["subject", "hemisphere"]].iterrows()
    ]
    return out


def bootstrap_prediction_ci(predictions: pd.DataFrame, axis_name: str) -> pd.DataFrame:
    subjects = np.array(sorted(predictions["subject"].astype(str).unique()))
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for _ in range(N_BOOT):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        boot = pd.concat([predictions[predictions["subject"].astype(str).eq(subject)] for subject in sampled])
        metrics = prediction_metrics(boot)
        rows.append(metrics)
    out = []
    for metric in ["spearman_r", "pearson_r", "mae", "rmse"]:
        low, high = percentile_ci([row.get(metric, np.nan) for row in rows])
        out.append(
            {
                "axis_name": axis_name,
                "metric": metric,
                "ci_lower_95": low,
                "ci_upper_95": high,
                "n_bootstrap": N_BOOT,
                "seed": RANDOM_SEED,
            }
        )
    return pd.DataFrame(out)


def run_axis_model(
    data: pd.DataFrame,
    target_col: str,
    axis_name: str,
    role: str,
    feature_cols: list[str],
    tremor_axis_eligible: bool,
) -> AxisResult:
    if axis_name.startswith("tremor") and not tremor_axis_eligible:
        return AxisResult(
            axis_name=axis_name,
            target_column=target_col,
            status="skipped_tremor_effective_sample_below_threshold",
            predictions=pd.DataFrame(),
            weights=pd.DataFrame(),
            summary={
                "axis_name": axis_name,
                "target_column": target_col,
                "axis_role": role,
                "status": "skipped_tremor_effective_sample_below_threshold",
                "ridge_alpha": RIDGE_ALPHA,
            },
            perm_samples=pd.DataFrame(),
            boot_ci=pd.DataFrame(),
        )
    predictions, weights = run_loso_ridge(data, target_col, feature_cols)
    metrics = prediction_metrics(predictions)
    stability = weight_stability(weights, feature_cols)
    observed = metrics.get("spearman_r", np.nan)
    rng = np.random.default_rng(RANDOM_SEED)
    perm_rows = []
    if np.isfinite(observed):
        subjects = np.array(sorted(data["subject"].astype(str).unique()))
        row_subjects = data["subject"].astype(str).to_numpy()
        row_hemis = data["hemisphere"].astype(str).to_numpy()
        subject_target = {
            (str(row["subject"]), str(row["hemisphere"])): row[target_col]
            for _, row in data[["subject", "hemisphere", target_col]].iterrows()
        }
        for perm_idx in range(N_PERM):
            shuffled = rng.permutation(subjects)
            subject_map = dict(zip(subjects, shuffled))
            perm_y = np.array(
                [
                    subject_target.get((subject_map[row_subjects[i]], row_hemis[i]), np.nan)
                    for i in range(len(row_subjects))
                ],
                dtype=float,
            )
            perm_spearman = fast_loso_ridge_spearman(data, perm_y, feature_cols)
            perm_rows.append(
                {
                    "axis_name": axis_name,
                    "permutation_index": perm_idx,
                    "spearman_r": perm_spearman,
                    "pearson_r": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                }
            )
    perm_samples = pd.DataFrame(perm_rows)
    if len(perm_samples) and np.isfinite(observed):
        null = pd.to_numeric(perm_samples["spearman_r"], errors="coerce").dropna().to_numpy(dtype=float)
        perm_p = float((1 + np.sum(null >= observed)) / (1 + len(null))) if len(null) else float("nan")
    else:
        perm_p = float("nan")
    boot_ci = bootstrap_prediction_ci(predictions, axis_name) if not predictions.empty else pd.DataFrame()
    summary = {
        "axis_name": axis_name,
        "target_column": target_col,
        "axis_role": role,
        "status": metrics.get("status", "unknown"),
        "ridge_alpha": RIDGE_ALPHA,
        "n_loso_folds": int(data["subject"].nunique()),
        "n_predictions": metrics.get("n_predictions", 0),
        "spearman_r": metrics.get("spearman_r", np.nan),
        "spearman_p": metrics.get("spearman_p", np.nan),
        "pearson_r": metrics.get("pearson_r", np.nan),
        "pearson_p": metrics.get("pearson_p", np.nan),
        "mae": metrics.get("mae", np.nan),
        "rmse": metrics.get("rmse", np.nan),
        "permutation_p_spearman_directional": perm_p,
        "n_permutations": N_PERM if np.isfinite(observed) else 0,
        **stability,
    }
    return AxisResult(axis_name, target_col, str(metrics.get("status", "unknown")), predictions, weights, summary, perm_samples, boot_ci)


def axis_model_eligibility(tremor_audit: pd.DataFrame, data: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    status = str(tremor_audit["tremor_axis_status"].iloc[0])
    trem_eligible = status == "primary_axis_eligible"
    rows = []
    for target_col, axis_name, role in AXIS_TARGETS:
        target = pd.to_numeric(data[target_col], errors="coerce").dropna()
        eligible = len(target) >= 8 and target.nunique() >= 2
        if axis_name.startswith("tremor") and not trem_eligible:
            eligible = False
            reason = "tremor_effective_sample_below_threshold"
        elif not eligible:
            reason = "insufficient_target_variation"
        else:
            reason = "eligible"
        rows.append(
            {
                "axis_name": axis_name,
                "target_column": target_col,
                "axis_role": role,
                "eligible": bool(eligible),
                "reason": reason,
                "n_nonmissing_rows": int(len(target)),
                "n_unique_target_values": int(target.nunique()),
                "tremor_axis_status": status if axis_name.startswith("tremor") else "not_applicable",
            }
        )
    return pd.DataFrame(rows), trem_eligible


def more_affected_side_analysis(
    data: pd.DataFrame,
    more_affected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    detail_rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    components = ["beta_power_residual_suppression", "cortico_stn_beta_desynchronization"]
    for symptom, response_base, label_col, tie_col in [
        ("AR", "ar", "ar_more_affected_stn_hemisphere", "ar_tie"),
        ("tremor", "trem", "trem_more_affected_stn_hemisphere", "trem_tie"),
    ]:
        for component in components:
            component_col = component_association_feature(component)
            subj_rows = []
            for _, mrow in more_affected.iterrows():
                if bool(mrow[tie_col]) or str(mrow[label_col]) not in {"left", "right"}:
                    continue
                subject = str(mrow["subject"])
                more_hemi = str(mrow[label_col])
                less_hemi = "right" if more_hemi == "left" else "left"
                more_row = data[data["subject"].astype(str).eq(subject) & data["hemisphere"].astype(str).eq(more_hemi)]
                less_row = data[data["subject"].astype(str).eq(subject) & data["hemisphere"].astype(str).eq(less_hemi)]
                if more_row.empty or less_row.empty:
                    continue
                target_col = f"contralateral_{response_base}_response"
                subj_rows.append(
                    {
                        "subject": subject,
                        "symptom": symptom,
                        "component": component,
                        "more_affected_stn_hemisphere": more_hemi,
                        "less_affected_stn_hemisphere": less_hemi,
                        "more_affected_component_score": more_row.iloc[0][component_col],
                        "less_affected_component_score": less_row.iloc[0][component_col],
                        "more_affected_response": more_row.iloc[0][target_col],
                        "less_affected_response": less_row.iloc[0][target_col],
                    }
                )
            table = pd.DataFrame(subj_rows)
            if table.empty:
                rows.append(
                    {
                        "symptom": symptom,
                        "component": component,
                        "n_subjects": 0,
                        "more_affected_spearman_r": np.nan,
                        "less_affected_spearman_r": np.nan,
                        "abs_more_minus_abs_less": np.nan,
                        "status": "no_non_tie_subjects",
                    }
                )
                detail_rows.append(
                    {
                        "symptom": symptom,
                        "component": component,
                        "n_subjects": 0,
                        "abs_more_minus_abs_less": np.nan,
                        "bootstrap_ci_low": np.nan,
                        "bootstrap_ci_high": np.nan,
                        "n_bootstrap": N_BOOT,
                        "seed": RANDOM_SEED,
                        "status": "no_non_tie_subjects",
                    }
                )
                continue
            _, more_r, more_p = finite_corr(table["more_affected_component_score"], table["more_affected_response"])
            _, less_r, less_p = finite_corr(table["less_affected_component_score"], table["less_affected_response"])
            observed = abs(more_r) - abs(less_r)
            boot_values = []
            subjects = table["subject"].astype(str).to_numpy()
            more_x = pd.to_numeric(table["more_affected_component_score"], errors="coerce").to_numpy(dtype=float)
            more_y = pd.to_numeric(table["more_affected_response"], errors="coerce").to_numpy(dtype=float)
            less_x = pd.to_numeric(table["less_affected_component_score"], errors="coerce").to_numpy(dtype=float)
            less_y = pd.to_numeric(table["less_affected_response"], errors="coerce").to_numpy(dtype=float)
            for _ in range(N_BOOT):
                sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
                idx = np.concatenate([np.where(subjects == sampled_subject)[0] for sampled_subject in sampled_subjects])
                boot_more = spearman_np(more_x[idx], more_y[idx])
                boot_less = spearman_np(less_x[idx], less_y[idx])
                boot_values.append(abs(boot_more) - abs(boot_less))
            ci_low, ci_high = percentile_ci(boot_values)
            rows.append(
                {
                    "symptom": symptom,
                    "component": component,
                    "n_subjects": int(table["subject"].nunique()),
                    "more_affected_spearman_r": more_r,
                    "more_affected_spearman_p": more_p,
                    "less_affected_spearman_r": less_r,
                    "less_affected_spearman_p": less_p,
                    "abs_more_minus_abs_less": observed,
                    "status": "ok",
                }
            )
            detail_rows.append(
                {
                    "symptom": symptom,
                    "component": component,
                    "n_subjects": int(table["subject"].nunique()),
                    "abs_more_minus_abs_less": observed,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "n_bootstrap": N_BOOT,
                    "seed": RANDOM_SEED,
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(detail_rows)


def sensitivity_outputs(
    data: pd.DataFrame,
    task_features: pd.DataFrame,
    lateralized: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sens_rows = []
    subsets = {
        "main_18subjects": data,
        "exclude_BYJoWR_17subjects": data[~data["subject"].astype(str).eq("sub-BYJoWR")],
        "known_quality_only": data[~data.get("has_unknown_quality", pd.Series(False, index=data.index)).astype(bool)],
        "exclude_bad_quality": data[~data.get("has_bad_quality", pd.Series(False, index=data.index)).astype(bool)],
        "left_hemisphere_only": data[data["hemisphere"].astype(str).eq("left")],
        "right_hemisphere_only": data[data["hemisphere"].astype(str).eq("right")],
        "both_task_available": data[data["task_availability_pattern"].astype(str).eq("Hold+Move")],
    }
    for name, subset in subsets.items():
        _, ar_r, _ = finite_corr(
            subset[component_association_feature("beta_power_residual_suppression")],
            subset["contralateral_ar_response"],
        )
        _, trem_r, _ = finite_corr(
            subset[component_association_feature("beta_power_residual_suppression")],
            subset["contralateral_trem_response"],
        )
        sens_rows.append(
            {
                "subset": name,
                "n_rows": int(len(subset)),
                "n_subjects": int(subset["subject"].nunique()) if len(subset) else 0,
                "beta_vs_contralateral_ar_spearman": ar_r,
                "beta_vs_contralateral_tremor_spearman": trem_r,
            }
        )

    task_rows = []
    for task in ["Hold", "Move"]:
        task_subset = task_features[task_features["task_family"].astype(str).eq(task)].copy()
        if task_subset.empty:
            continue
        task_hemi, _ = aggregate_hemisphere_features(task_subset)
        merged = merge_targets_and_features(task_hemi, lateralized)
        for component in ["aperiodic_normalization", "beta_power_residual_suppression", "cortico_stn_beta_desynchronization"]:
            _, ar_r, _ = finite_corr(merged[component_association_feature(component)], merged["contralateral_ar_response"])
            _, trem_r, _ = finite_corr(
                merged[component_association_feature(component)], merged["contralateral_trem_response"]
            )
            task_rows.append(
                {
                    "task_family": task,
                    "component": component,
                    "n_rows": int(len(merged)),
                    "n_subjects": int(merged["subject"].nunique()),
                    "contralateral_ar_spearman": ar_r,
                    "contralateral_tremor_spearman": trem_r,
                }
            )

    quality = pd.DataFrame(
        [
            {
                "quality_subset": "all",
                "n_rows": int(len(data)),
                "n_subjects": int(data["subject"].nunique()),
            },
            {
                "quality_subset": "exclude_unknown",
                "n_rows": int((~data["has_unknown_quality"].astype(bool)).sum()),
                "n_subjects": int(data.loc[~data["has_unknown_quality"].astype(bool), "subject"].nunique()),
            },
            {
                "quality_subset": "exclude_bad",
                "n_rows": int((~data["has_bad_quality"].astype(bool)).sum()),
                "n_subjects": int(data.loc[~data["has_bad_quality"].astype(bool), "subject"].nunique()),
            },
        ]
    )

    flip = data.copy()
    for base in ["ar", "trem"]:
        contra = f"contralateral_{base}_response"
        ipsi = f"ipsilateral_{base}_response"
        old = flip[contra].copy()
        flip[contra] = flip[ipsi].to_numpy()
        flip[ipsi] = old.to_numpy()
    flip_rows = []
    for component in ALLOWED_COMPONENTS:
        _, ar_r, _ = finite_corr(flip[component_association_feature(component)], flip["contralateral_ar_response"])
        _, trem_r, _ = finite_corr(flip[component_association_feature(component)], flip["contralateral_trem_response"])
        flip_rows.append(
            {
                "component": component,
                "flipped_contralateral_ar_spearman": ar_r,
                "flipped_contralateral_tremor_spearman": trem_r,
            }
        )

    cohort = pd.DataFrame(
        [
            {
                "cohort": "18subjects_34pairs",
                "n_rows": int(len(data)),
                "n_subjects": int(data["subject"].nunique()),
                "BYJoWR_included": True,
            },
            {
                "cohort": "17subjects_33pairs_exclude_BYJoWR",
                "n_rows": int(len(data[~data["subject"].astype(str).eq("sub-BYJoWR")])),
                "n_subjects": int(data.loc[~data["subject"].astype(str).eq("sub-BYJoWR"), "subject"].nunique()),
                "BYJoWR_included": False,
            },
        ]
    )

    return pd.DataFrame(sens_rows), quality, pd.DataFrame(task_rows), pd.DataFrame(flip_rows), cohort


def component_profile(
    associations: pd.DataFrame,
    collinearity_summary: pd.DataFrame,
    axis_summaries: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile = associations[
        associations["symptom"].isin(["AR", "tremor"])
        & associations["laterality"].eq("contralateral")
    ].copy()
    profile = profile.merge(
        collinearity_summary[["component", "collinearity_flag_abs_r_ge_0_70", "strongest_correlated_component"]],
        on="component",
        how="left",
    )
    profile["interpretation_caveat"] = np.where(
        profile["collinearity_flag_abs_r_ge_0_70"].astype(bool),
        "component_collinearity_detected_do_not_claim_unique_component_dominance",
        "no_high_component_collinearity_flag",
    )
    weights = []
    for _, row in axis_summaries.iterrows():
        weights.append(
            {
                "axis_name": row["axis_name"],
                "axis_weight_stable": row.get("axis_weight_stable", False),
                "median_pairwise_fold_weight_spearman": row.get("median_pairwise_fold_weight_spearman", np.nan),
                "fraction_features_sign_stable_80pct": row.get("fraction_features_sign_stable_80pct", np.nan),
                "mechanistic_interpretation_allowed": bool(row.get("axis_weight_stable", False)),
            }
        )
    return profile, pd.DataFrame(weights)


def plot_outputs(
    output_dir: Path,
    associations: pd.DataFrame,
    lateralized: pd.DataFrame,
    collinearity_matrix: pd.DataFrame,
    task_audit: pd.DataFrame,
    axis_predictions: pd.DataFrame,
    axis_weights: pd.DataFrame,
    more_affected_summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    heat = associations[
        associations["laterality"].eq("contralateral")
        & associations["symptom"].isin(["AR", "tremor", "axial", "total"])
    ].pivot_table(index="component", columns="symptom", values="spearman_r", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(heat.fillna(0).to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_title("Contralateral component-symptom Spearman r")
    fig.colorbar(im, ax=ax, label="Spearman r")
    fig.tight_layout()
    fig.savefig(output_dir / "symptom_component_heatmap.png", dpi=180)
    plt.close(fig)

    ctrl = associations[associations["symptom"].isin(["AR", "tremor"])].copy()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = []
    x = np.arange(len(ALLOWED_COMPONENTS))
    width = 0.2
    for idx, (symptom, laterality) in enumerate([("AR", "contralateral"), ("AR", "ipsilateral"), ("tremor", "contralateral"), ("tremor", "ipsilateral")]):
        vals = [
            ctrl[
                ctrl["component"].eq(component)
                & ctrl["symptom"].eq(symptom)
                & ctrl["laterality"].eq(laterality)
            ]["spearman_r"].iloc[0]
            for component in ALLOWED_COMPONENTS
        ]
        ax.bar(x + (idx - 1.5) * width, vals, width=width, label=f"{symptom} {laterality}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [c.replace("_", "\n") for c in ALLOWED_COMPONENTS], fontsize=7)
    ax.set_ylabel("Spearman r")
    ax.set_title("Contralateral vs ipsilateral associations")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "contralateral_vs_ipsilateral_associations.png", dpi=180)
    plt.close(fig)

    for axis_name, filename in [("AR_contralateral", "ar_axis_loso_prediction.png"), ("tremor_contralateral", "tremor_axis_loso_prediction.png")]:
        sub = axis_predictions[axis_predictions["axis_name"].eq(axis_name)].dropna(subset=["y_true", "y_pred"])
        fig, ax = plt.subplots(figsize=(5, 4.5))
        if not sub.empty:
            ax.scatter(sub["y_true"], sub["y_pred"], s=28)
            lo = float(np.nanmin([sub["y_true"].min(), sub["y_pred"].min()]))
            hi = float(np.nanmax([sub["y_true"].max(), sub["y_pred"].max()]))
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
        ax.set_xlabel("Observed")
        ax.set_ylabel("LOSO predicted")
        ax.set_title(axis_name)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    if not axis_weights.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        w = axis_weights.groupby(["axis_name", "feature"])["weight"].median().reset_index()
        ar = w[w["axis_name"].eq("AR_contralateral")]
        ax.bar(np.arange(len(ar)), ar["weight"])
        ax.set_xticks(np.arange(len(ar)), [f.replace("component__", "").replace("__oriented_mean", "") for f in ar["feature"]], rotation=45, ha="right")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Median LOSO weights: AR axis")
        fig.tight_layout()
        fig.savefig(output_dir / "axis_weight_stability.png", dpi=180)
        plt.close(fig)

    profile = associations[
        associations["laterality"].eq("contralateral")
        & associations["symptom"].isin(["AR", "tremor"])
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot = profile.pivot_table(index="component", columns="symptom", values="spearman_r", aggfunc="mean")
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Spearman r")
    ax.set_title("Mechanistic component profiles")
    fig.tight_layout()
    fig.savefig(output_dir / "mechanistic_component_profiles.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(lateralized["contralateral_ar_response"], lateralized["contralateral_trem_response"], c=np.arange(len(lateralized)), cmap="viridis")
    ax.set_xlabel("Contralateral AR response")
    ax.set_ylabel("Contralateral tremor response")
    ax.set_title("Subject-level symptom transition map")
    fig.tight_layout()
    fig.savefig(output_dir / "subject_level_symptom_transition_map.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    if not more_affected_summary.empty:
        labels = more_affected_summary["symptom"] + "\n" + more_affected_summary["component"].str.replace("_", "\n")
        ax.bar(np.arange(len(more_affected_summary)), more_affected_summary["abs_more_minus_abs_less"])
        ax.set_xticks(np.arange(len(more_affected_summary)), labels, rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("|r more affected| - |r less affected|")
    ax.set_title("More-affected-side beta consistency")
    fig.tight_layout()
    fig.savefig(output_dir / "more_affected_side_beta_consistency.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(lateralized["contralateral_trem_response"].dropna(), bins=10, edgecolor="black")
    ax.set_xlabel("Contralateral tremor response")
    ax.set_ylabel("Hemisphere rows")
    ax.set_title("Tremor response distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "tremor_response_distribution.png", dpi=180)
    plt.close(fig)

    matrix = collinearity_matrix.pivot_table(index="component_a", columns="component_b", values="spearman_r")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(matrix.fillna(0).to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=7)
    ax.set_title("Component collinearity")
    fig.colorbar(im, ax=ax, label="Spearman r")
    fig.tight_layout()
    fig.savefig(output_dir / "component_collinearity_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    if not task_audit.empty:
        ax.bar(task_audit["task_availability_pattern"], task_audit["n_subjects"])
    ax.set_ylabel("Subjects")
    ax.set_title("Task availability")
    fig.tight_layout()
    fig.savefig(output_dir / "task_availability_by_subject.png", dpi=180)
    plt.close(fig)


def build_summary(
    validation: dict[str, object],
    tremor_audit: pd.DataFrame,
    associations: pd.DataFrame,
    control: pd.DataFrame,
    axis_summary: pd.DataFrame,
    collinearity_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> dict[str, object]:
    primary = associations[associations["hypothesis_family"].eq("primary")].copy()
    supported = primary[
        primary["direction_matches_prediction"].astype(bool)
        & (primary["fdr_q_permutation_full_family"] <= 0.05)
    ]
    control_supported = control[
        control["contra_direction_matches_prediction"].astype(bool)
        & control["contra_stronger_than_ipsi"].astype(bool)
        & (control["permutation_p"] <= 0.05)
    ]
    honest_negative = bool(supported.empty and control_supported.empty)
    return {
        "analysis": "dopaminergic_symptom_transition_axes_18subjects_34pairs",
        "validation": validation,
        "random_seed": RANDOM_SEED,
        "n_bootstrap": N_BOOT,
        "n_permutations": N_PERM,
        "ridge_alpha": RIDGE_ALPHA,
        "huber_regression": "sklearn.linear_model.HuberRegressor default parameters",
        "static_feature_component_mapping": str(MAPPING_PATH),
        "tremor_axis_status": str(tremor_audit["tremor_axis_status"].iloc[0]),
        "n_primary_associations_directional_fdr_supported": int(len(supported)),
        "n_contra_greater_than_ipsi_supported": int(len(control_supported)),
        "honest_negative_primary_result": honest_negative,
        "axis_summary": axis_summary.to_dict(orient="records"),
        "high_collinearity_components": collinearity_summary[
            collinearity_summary["collinearity_flag_abs_r_ge_0_70"].astype(bool)
        ].to_dict(orient="records"),
        "sensitivity_summary": sensitivity.to_dict(orient="records"),
        "warnings": [
            "Exploratory symptom-specific transition analysis; frozen retrieval logic unchanged.",
            "No clinical prediction, treatment recommendation, DBS optimization, or causal medication-effect claim is made.",
            "No Oxford/MRC external dataset was used.",
            "Two hemispheres from one subject are clustered; LOSO removes both hemispheres.",
            "Tremor axis is guarded by effective-sample thresholds.",
        ],
    }


def write_markdown_summary(summary: dict[str, object], output_dir: Path) -> None:
    axis_rows = summary.get("axis_summary", [])
    lines = [
        "# Dopaminergic Symptom Transition Axes",
        "",
        "Scope: exploratory symptom-specific MedOff-to-MedOn transition analysis over cached ds004998 18-subject outputs.",
        "",
        "## Validation",
    ]
    for key, value in summary["validation"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Key Settings",
            f"- ridge alpha: {summary['ridge_alpha']}",
            f"- bootstrap/permutation seed: {summary['random_seed']}",
            f"- static mapping: `{summary['static_feature_component_mapping']}`",
            f"- tremor axis status: {summary['tremor_axis_status']}",
            "",
            "## Axis Summary",
        ]
    )
    if axis_rows:
        for row in axis_rows:
            lines.append(
                "- {axis_name}: status={status}, spearman={spearman_r:.3f}, permutation_p={permutation_p_spearman_directional}, weight_stable={axis_weight_stable}".format(
                    axis_name=row.get("axis_name", ""),
                    status=row.get("status", ""),
                    spearman_r=float(row.get("spearman_r", np.nan)) if pd.notna(row.get("spearman_r", np.nan)) else float("nan"),
                    permutation_p_spearman_directional=row.get("permutation_p_spearman_directional", np.nan),
                    axis_weight_stable=row.get("axis_weight_stable", False),
                )
            )
    else:
        lines.append("- No axis models completed.")
    lines.extend(
        [
            "",
            "## Primary Result Boundary",
            f"- primary directional FDR-supported associations: {summary['n_primary_associations_directional_fdr_supported']}",
            f"- contralateral stronger than ipsilateral supported cases: {summary['n_contra_greater_than_ipsi_supported']}",
            f"- honest negative primary result: {summary['honest_negative_primary_result']}",
            "",
            "## Warnings",
        ]
    )
    lines.extend(f"- {warning}" for warning in summary["warnings"])
    write_text(lines, output_dir / "symptom_transition_axes_summary.md")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = validate_inputs()
    pairs = inputs["pairs"]
    mapping_raw = read_csv_required(MAPPING_PATH)
    mapping = validate_static_mapping(mapping_raw, set(pairs.columns))
    pair_subjects = sorted(pairs["subject"].astype(str).unique())

    clinical, item_mapping, lateralized, more_affected = prepare_clinical_targets(pair_subjects)
    tremor_audit = tremor_effective_sample_audit(clinical, lateralized)

    task_features = build_task_level_transition_features(pairs, mapping)
    hemi_features, task_audit = aggregate_hemisphere_features(task_features)
    data = merge_targets_and_features(hemi_features, lateralized)

    associations, fdr_assoc, primary_assoc = component_symptom_associations(data)
    control = contralateral_ipsilateral_control(data, associations)
    col_matrix, col_summary, col_clusters = component_collinearity(data)

    axis_eligibility, tremor_axis_eligible = axis_model_eligibility(tremor_audit, data)
    axis_features = prepare_axis_features(data)
    axis_results = []
    for target_col, axis_name, role in AXIS_TARGETS:
        eligible = bool(axis_eligibility.loc[axis_eligibility["axis_name"].eq(axis_name), "eligible"].iloc[0])
        if not eligible and not axis_name.startswith("tremor"):
            axis_results.append(
                AxisResult(
                    axis_name,
                    target_col,
                    "skipped_axis_not_eligible",
                    pd.DataFrame(),
                    pd.DataFrame(),
                    {
                        "axis_name": axis_name,
                        "target_column": target_col,
                        "axis_role": role,
                        "status": "skipped_axis_not_eligible",
                        "ridge_alpha": RIDGE_ALPHA,
                    },
                    pd.DataFrame(),
                    pd.DataFrame(),
                )
            )
        else:
            axis_results.append(run_axis_model(data, target_col, axis_name, role, axis_features, tremor_axis_eligible))

    pred_tables = []
    weight_tables = []
    perm_tables = []
    boot_tables = []
    summary_rows = []
    for result in axis_results:
        summary_rows.append(result.summary)
        if not result.predictions.empty:
            pred = result.predictions.copy()
            pred["axis_name"] = result.axis_name
            pred_tables.append(pred)
        if not result.weights.empty:
            weights = result.weights.copy()
            weights["axis_name"] = result.axis_name
            weight_tables.append(weights)
        if not result.perm_samples.empty:
            perm_tables.append(result.perm_samples)
        if not result.boot_ci.empty:
            boot_tables.append(result.boot_ci)
    axis_summary = pd.DataFrame(summary_rows)
    axis_predictions = pd.concat(pred_tables, ignore_index=True) if pred_tables else pd.DataFrame()
    axis_weights = pd.concat(weight_tables, ignore_index=True) if weight_tables else pd.DataFrame()
    axis_perm = pd.concat(perm_tables, ignore_index=True) if perm_tables else pd.DataFrame()
    axis_boot = pd.concat(boot_tables, ignore_index=True) if boot_tables else pd.DataFrame()

    more_summary, more_detail = more_affected_side_analysis(data, more_affected)
    sens, quality, task_sens, flip, cohort_sens = sensitivity_outputs(data, task_features, lateralized)
    profile, mech_weights = component_profile(associations, col_summary, axis_summary)

    write_csv(mapping, OUTPUT_DIR / "feature_component_mapping.csv")
    write_csv(clinical, OUTPUT_DIR / "clinical_target_inventory.csv")
    write_csv(item_mapping, OUTPUT_DIR / "updrs_item_mapping.csv")
    write_csv(lateralized, OUTPUT_DIR / "lateralized_symptom_targets.csv")
    write_csv(more_affected, OUTPUT_DIR / "more_affected_side_labels.csv")
    write_csv(tremor_audit, OUTPUT_DIR / "tremor_effective_sample_audit.csv")
    write_csv(hemi_features, OUTPUT_DIR / "hemisphere_transition_features.csv")
    write_csv(task_features, OUTPUT_DIR / "task_level_transition_features.csv")
    write_csv(data, OUTPUT_DIR / "component_transition_scores.csv")
    write_csv(task_audit, OUTPUT_DIR / "task_availability_audit.csv")
    write_csv(associations, OUTPUT_DIR / "component_symptom_associations.csv")
    write_csv(fdr_assoc, OUTPUT_DIR / "component_symptom_associations_fdr_full_family.csv")
    write_csv(primary_assoc, OUTPUT_DIR / "primary_hypothesis_associations.csv")
    write_csv(control, OUTPUT_DIR / "contralateral_ipsilateral_control.csv")
    write_csv(associations[["symptom", "laterality", "component", "spearman_r", "association_bootstrap_ci_low", "association_bootstrap_ci_high"]], OUTPUT_DIR / "association_bootstrap_ci.csv")
    write_csv(associations[["symptom", "laterality", "component", "spearman_r", "subject_label_permutation_p_directional"]], OUTPUT_DIR / "association_permutation_null.csv")
    write_csv(col_matrix, OUTPUT_DIR / "component_collinearity_matrix.csv")
    write_csv(col_summary, OUTPUT_DIR / "component_collinearity_summary.csv")
    write_csv(col_clusters, OUTPUT_DIR / "component_cluster_interpretation.csv")
    write_csv(axis_eligibility, OUTPUT_DIR / "axis_model_eligibility.csv")
    write_csv(axis_predictions, OUTPUT_DIR / "symptom_axis_loso_predictions.csv")
    write_csv(axis_weights, OUTPUT_DIR / "symptom_axis_model_weights.csv")
    write_csv(axis_summary, OUTPUT_DIR / "symptom_axis_summary.csv")
    write_csv(axis_perm, OUTPUT_DIR / "symptom_axis_permutation_null.csv")
    write_csv(axis_boot, OUTPUT_DIR / "symptom_axis_bootstrap_ci.csv")
    write_csv(more_summary, OUTPUT_DIR / "more_affected_side_beta_consistency.csv")
    write_csv(more_detail, OUTPUT_DIR / "more_affected_side_bootstrap_ci.csv")
    write_csv(mech_weights, OUTPUT_DIR / "mechanistic_component_weights.csv")
    write_csv(profile, OUTPUT_DIR / "symptom_component_profile.csv")
    write_csv(sens, OUTPUT_DIR / "sensitivity_summary.csv")
    write_csv(quality, OUTPUT_DIR / "quality_sensitivity.csv")
    write_csv(task_sens, OUTPUT_DIR / "task_sensitivity.csv")
    write_csv(flip, OUTPUT_DIR / "hemisphere_flip_null.csv")
    write_csv(cohort_sens, OUTPUT_DIR / "cohort_17_vs_18_sensitivity.csv")

    plot_outputs(OUTPUT_DIR, associations, lateralized, col_matrix, task_audit, axis_predictions, axis_weights, more_summary)

    summary = build_summary(inputs["validation"], tremor_audit, associations, control, axis_summary, col_summary, sens)
    write_json(summary, OUTPUT_DIR / "symptom_transition_axes_summary.json")
    write_markdown_summary(summary, OUTPUT_DIR)

    print("dopaminergic symptom transition axes completed")
    print(f"complete_pairs: {inputs['validation']['complete_pairs']}")
    print(f"subjects_with_complete_pairs: {inputs['validation']['subjects_with_complete_pairs']}")
    print(f"sub-BYJoWR included: {inputs['validation']['sub_BYJoWR_included']}")
    print(f"tremor_axis_status: {summary['tremor_axis_status']}")
    print(f"primary directional FDR-supported associations: {summary['n_primary_associations_directional_fdr_supported']}")
    print(f"honest_negative_primary_result: {summary['honest_negative_primary_result']}")
    print(f"output path: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
