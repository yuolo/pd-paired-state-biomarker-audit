#!/usr/bin/env python3
"""Audit OXF STN montage harmonization and contact-geometry branches.

This script is a diagnostic layer for the external OXF Rest-STN cohort. It
uses cached channel-level features and cached v2E pair tables only. It does
not reread raw MATLAB/SMR/FIF data, does not tune retrieval parameters, and
does not modify the frozen ds004998 v2A pipeline.

The audit separates:

1. Conservative exact-contact, same-spacing bipolar matches.
2. Numeric-token repairs, reported as sensitivity only.
3. Constituent-contact availability without a valid bipolar match.
4. Exploratory pseudo-monopolar/contact-proxy reconstruction from cached
   channel features, reported as exploratory and not as ground truth.
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
from scripts.run_oxf_rest_stn_branch_diagnostic import (  # noqa: E402
    COMMON_COMPACT,
    evaluate_one_feature_set_fast,
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
)


OUTPUT_DIR = Path("outputs/oxf_montage_harmonization_audit")
V2E_DIR = Path("outputs/oxf_stn_physiology_locked_v2e")
CHANNEL_FEATURES_PATH = V2E_DIR / "v2e_physiology_channel_features.csv"
V2E_PAIRS_PATH = V2E_DIR / "v2e_pairs.csv"

FEATURES = list(v2e.PHYSIOLOGY_FEATURES)
PRIMARY_FEATURES = list(COMMON_COMPACT)
FOCUS_VARIANTS = [
    oxf_base.V2A_NAME,
    V2C_DECONFOUNDED_NAME,
    V2D_QUALITY_DECONFOUNDED_NAME,
]

BRANCHES = [
    {
        "branch_id": "exact_same_spacing_compact_residual_7",
        "branch_role": "conservative",
        "source": "cached_v2e_exact_contact_pairs",
        "spacing_filter": "same_spacing_bipolar",
        "interpretation": "Exact OFF/ON channel token, valid bipolar geometry, and same inter-contact distance.",
    },
    {
        "branch_id": "exact_narrow_only_compact_residual_7",
        "branch_role": "conservative_sensitivity",
        "source": "cached_v2e_exact_contact_pairs",
        "spacing_filter": "narrow",
        "interpretation": "Exact OFF/ON channel token restricted to adjacent-contact bipolar pairs.",
    },
    {
        "branch_id": "exact_broad_only_compact_residual_7",
        "branch_role": "conservative_sensitivity",
        "source": "cached_v2e_exact_contact_pairs",
        "spacing_filter": "broad",
        "interpretation": "Exact OFF/ON channel token restricted to wider-contact bipolar pairs.",
    },
    {
        "branch_id": "pseudo_contact_proxy_compact_residual_7",
        "branch_role": "exploratory",
        "source": "cached_channel_feature_contact_proxy",
        "spacing_filter": "not_applicable",
        "interpretation": "Feature-level constituent-contact proxy from cached channels; not raw pseudo-monopolar ground truth.",
    },
    {
        "branch_id": "all_pairs_exact_else_median_compact_residual_7",
        "branch_role": "coverage_complete",
        "source": "hierarchy_exact_else_median",
        "spacing_filter": "all_pairs",
        "interpretation": "All 30 pairs using exact-contact rows when available and median-channel fallback otherwise.",
    },
    {
        "branch_id": "all_pairs_exact_numeric_else_median_compact_residual_7",
        "branch_role": "coverage_complete_sensitivity",
        "source": "hierarchy_exact_numeric_else_median",
        "spacing_filter": "all_pairs",
        "interpretation": "All 30 pairs using exact-contact rows, numeric-token recovery when exact is missing, and median fallback otherwise.",
    },
]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return pd.read_csv(path)


def write_csv(data: pd.DataFrame, path: Path) -> str:
    data.to_csv(path, index=False)
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


def compact_digit_string(value: object) -> str:
    return "".join(re.findall(r"\d", str(value)))


def contact_digits(value: object) -> list[int]:
    digits = compact_digit_string(value)
    return [int(char) for char in digits]


def unique_digits(value: object) -> list[int]:
    return sorted(set(contact_digits(value)))


def has_segment_marker(value: object) -> bool:
    token = str(value).strip().lower()
    return bool(re.search(r"[sv]", token))


def geometry_from_channel(channel: object, side: object) -> dict[str, object]:
    """Infer channel geometry from cached OXF channel labels.

    The labels are treated conservatively. Two numeric contacts are interpreted
    as bipolar; adjacent pairs are "narrow" and wider pairs are "broad".
    Labels containing segment markers or three or more digits are "segment".
    Single-contact labels do not define a bipolar spacing and are marked
    unknown for spacing.
    """

    exact = v2e.exact_contact_token(channel, side)
    numeric = v2e.numeric_contact_token(channel, side)
    digits = unique_digits(exact)
    n_digits = len(digits)
    span = int(max(digits) - min(digits)) if n_digits >= 2 else np.nan
    segment_like = has_segment_marker(exact) or n_digits >= 3
    if n_digits == 0:
        spacing_class = "unknown"
        geometry_class = "unknown"
    elif segment_like:
        spacing_class = "segment"
        geometry_class = "segment_or_directional"
    elif n_digits == 1:
        spacing_class = "unknown"
        geometry_class = "single_or_monopolar_label"
    elif n_digits == 2 and span == 1:
        spacing_class = "narrow"
        geometry_class = "valid_bipolar"
    elif n_digits == 2 and span > 1:
        spacing_class = "broad"
        geometry_class = "valid_bipolar"
    else:
        spacing_class = "unknown"
        geometry_class = "unknown"
    pair_key = "-".join(str(digit) for digit in digits) if n_digits >= 2 else ""
    return {
        "exact_contact_token": exact,
        "numeric_contact_token": numeric,
        "contact_digits": "".join(str(digit) for digit in digits),
        "contact_pair_key": pair_key,
        "n_constituent_contacts": int(n_digits),
        "contact_span": span,
        "spacing_class": spacing_class,
        "channel_geometry_class": geometry_class,
        "valid_bipolar_geometry": bool(spacing_class in {"narrow", "broad"}),
    }


def annotate_channel_inventory(channel_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in channel_features.iterrows():
        geom = geometry_from_channel(row.get("channel", ""), row.get("side", ""))
        rows.append({**row.to_dict(), **geom})
    out = pd.DataFrame(rows)
    out["feature_status"] = out.get("feature_status", "").astype(str)
    return out


def select_off_channel(off: pd.DataFrame) -> pd.Series:
    data = off.copy()
    data["_selection_score"] = pd.to_numeric(data.get("beta_peak_amplitude"), errors="coerce")
    if data["_selection_score"].notna().sum() == 0:
        data["_selection_score"] = pd.to_numeric(data.get("stn_broad_beta_log_power"), errors="coerce")
    data["_selection_score"] = data["_selection_score"].fillna(-np.inf)
    return data.sort_values(["_selection_score", "channel"], ascending=[False, True]).iloc[0]


def _token_set(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame:
        return set()
    return {str(value) for value in frame[column].dropna().astype(str) if str(value)}


def _constituent_digit_set(frame: pd.DataFrame) -> set[str]:
    digits: set[str] = set()
    for value in frame.get("contact_digits", pd.Series(dtype=object)).dropna().astype(str):
        digits.update(char for char in value if char.isdigit())
    return digits


def contact_action(repair_class: str) -> str:
    if repair_class == "exact_same_spacing_available":
        return "Use in conservative exact-contact same-spacing branch."
    if repair_class == "exact_available_unknown_or_segment_spacing":
        return "Keep out of same-spacing primary; report exact label availability separately."
    if repair_class == "numeric_only_recovered":
        return "Sensitivity only until channel nomenclature equivalence is independently verified."
    if repair_class == "constituent_contacts_available_not_valid_bipolar":
        return "Do not reconstruct as a valid bipolar match without raw montage verification."
    if repair_class == "no_valid_on_contact":
        return "Exclude from exact-contact branch; report as montage coverage loss."
    return "Unavailable without additional metadata."


def build_montage_coverage_cases(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = inventory[inventory["feature_status"].astype(str).eq("ok")].copy()
    for (subject, side), group in ok.groupby(["subject", "side"], dropna=False):
        off = group[group["medication_state"].astype(str).eq("OFF")].copy()
        on = group[group["medication_state"].astype(str).eq("ON")].copy()
        if off.empty or on.empty:
            rows.append(
                {
                    "pair_id": f"{subject}_{side}",
                    "subject": subject,
                    "side": side,
                    "selection_status": "missing_off_or_on",
                    "repair_class": "missing_off_or_on",
                    "scientific_action": contact_action("missing_off_or_on"),
                }
            )
            continue
        selected = select_off_channel(off)
        exact = str(selected["exact_contact_token"])
        numeric = str(selected["numeric_contact_token"])
        off_digits = set(str(selected["contact_digits"]))
        exact_on = on[on["exact_contact_token"].astype(str).eq(exact)].copy()
        numeric_on = on[on["numeric_contact_token"].astype(str).eq(numeric)].copy()
        on_digits = _constituent_digit_set(on)
        exact_paired = not exact_on.empty
        numeric_paired = not numeric_on.empty
        same_spacing = False
        on_exact_spacing = ""
        on_exact_span = np.nan
        on_exact_channel = ""
        if exact_paired:
            exact_on = exact_on.sort_values("channel")
            on_row = exact_on.iloc[0]
            on_exact_spacing = str(on_row.get("spacing_class", ""))
            on_exact_span = safe_float(on_row.get("contact_span"))
            on_exact_channel = str(on_row.get("channel", ""))
            same_spacing = (
                bool(selected.get("valid_bipolar_geometry", False))
                and bool(on_row.get("valid_bipolar_geometry", False))
                and str(selected.get("spacing_class", "")) == on_exact_spacing
                and safe_float(selected.get("contact_span")) == on_exact_span
            )
        if same_spacing:
            repair_class = "exact_same_spacing_available"
        elif exact_paired:
            repair_class = "exact_available_unknown_or_segment_spacing"
        elif numeric_paired:
            repair_class = "numeric_only_recovered"
        elif off_digits and off_digits.issubset(on_digits):
            repair_class = "constituent_contacts_available_not_valid_bipolar"
        elif len(on) > 0:
            repair_class = "no_valid_on_contact"
        else:
            repair_class = "no_on_channel_inventory"

        rows.append(
            {
                "pair_id": f"{subject}_{side}",
                "subject": subject,
                "side": side,
                "selection_status": "paired" if exact_paired else "not_exact_paired",
                "off_selected_channel": selected.get("channel", ""),
                "off_exact_contact_token": exact,
                "off_numeric_contact_token": numeric,
                "off_contact_digits": selected.get("contact_digits", ""),
                "off_contact_pair_key": selected.get("contact_pair_key", ""),
                "off_spacing_class": selected.get("spacing_class", ""),
                "off_contact_span": selected.get("contact_span", np.nan),
                "off_valid_bipolar_geometry": bool(selected.get("valid_bipolar_geometry", False)),
                "exact_on_selected_channel": on_exact_channel,
                "exact_on_spacing_class": on_exact_spacing,
                "exact_on_contact_span": on_exact_span,
                "same_intercontact_distance": bool(same_spacing),
                "numeric_on_selected_channels": ";".join(numeric_on["channel"].astype(str).tolist()) if numeric_paired else "",
                "available_on_channels": ";".join(on["channel"].astype(str).tolist()),
                "available_on_exact_tokens": ";".join(sorted(_token_set(on, "exact_contact_token"))),
                "available_on_numeric_tokens": ";".join(sorted(_token_set(on, "numeric_contact_token"))),
                "available_on_constituent_digits": "".join(sorted(on_digits)),
                "exact_paired": bool(exact_paired),
                "numeric_paired": bool(numeric_paired),
                "repair_class": repair_class,
                "scientific_action": contact_action(repair_class),
            }
        )
    return pd.DataFrame(rows).sort_values(["repair_class", "subject", "side"]).reset_index(drop=True)


def summarize_coverage(cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        cases.groupby("repair_class", dropna=False)
        .agg(n_pairs=("pair_id", "size"), n_subjects=("subject", "nunique"))
        .reset_index()
    )
    summary["rate"] = summary["n_pairs"] / len(cases) if len(cases) else np.nan
    spacing = (
        cases.groupby(["off_spacing_class", "repair_class"], dropna=False)
        .agg(n_pairs=("pair_id", "size"), n_subjects=("subject", "nunique"))
        .reset_index()
    )
    spacing["rate_within_all_pairs"] = spacing["n_pairs"] / len(cases) if len(cases) else np.nan
    return summary.sort_values("n_pairs", ascending=False), spacing.sort_values(["off_spacing_class", "repair_class"])


def channel_geometry_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in inventory.groupby(["medication_state", "side", "spacing_class", "channel_geometry_class"], dropna=False):
        medication, side, spacing, geometry = keys
        rows.append(
            {
                "medication_state": medication,
                "side": side,
                "spacing_class": spacing,
                "channel_geometry_class": geometry,
                "n_channel_rows": int(len(group)),
                "n_subjects": int(group["subject"].nunique()),
                "n_unique_channel_labels": int(group["channel"].astype(str).nunique()),
                "example_channels": ";".join(sorted(group["channel"].astype(str).unique())[:12]),
            }
        )
    return pd.DataFrame(rows).sort_values(["medication_state", "side", "spacing_class"])


def build_pair_row_from_rows(
    subject: str,
    side: str,
    feature_set: str,
    off_row: pd.Series,
    on_row: pd.Series,
    selection: dict[str, object],
) -> dict[str, object]:
    quality = "unknown"
    if bool(off_row.get("flag_erna_label", False)) or bool(on_row.get("flag_erna_label", False)):
        quality = "erna_label_present"
    row: dict[str, object] = {
        "pair_id": f"{subject}_{side}",
        "subject": subject,
        "task_original": "Rest",
        "task_family": "Rest",
        "side": side,
        "quality_flag": quality,
        "feature_set": feature_set,
        "off_file_path": off_row.get("file_path", ""),
        "on_file_path": on_row.get("file_path", ""),
        "off_selected_channel": off_row.get("channel", ""),
        "on_selected_channel": on_row.get("channel", ""),
        "off_sampling_frequency": off_row.get("sampling_frequency", np.nan),
        "on_sampling_frequency": on_row.get("sampling_frequency", np.nan),
        **selection,
    }
    for feature in FEATURES:
        row[f"x_{feature}"] = safe_float(off_row.get(feature))
        row[f"medon_{feature}"] = safe_float(on_row.get(feature))
        row[f"y_{feature}"] = safe_float(on_row.get(feature))
    return row


def exact_pairs_with_geometry(v2e_pairs: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    exact = v2e_pairs[v2e_pairs["feature_set"].astype(str).eq("off_beta_peak_contact_exact")].copy()
    cols = [
        "pair_id",
        "off_spacing_class",
        "off_contact_span",
        "off_contact_pair_key",
        "off_valid_bipolar_geometry",
        "same_intercontact_distance",
        "repair_class",
    ]
    return exact.merge(cases[cols], on="pair_id", how="left")


def filter_exact_branch(exact_pairs: pd.DataFrame, spacing_filter: str) -> pd.DataFrame:
    data = exact_pairs[exact_pairs["same_intercontact_distance"].astype(bool)].copy()
    if spacing_filter in {"narrow", "broad"}:
        data = data[data["off_spacing_class"].astype(str).eq(spacing_filter)].copy()
    data["feature_set"] = spacing_filter
    return data.reset_index(drop=True)


def all_pairs_hierarchy_branch(
    v2e_pairs: pd.DataFrame,
    coverage_cases: pd.DataFrame,
    include_numeric_repair: bool,
) -> pd.DataFrame:
    """Build a coverage-complete all-pairs branch with fixed provenance.

    This deliberately keeps all 30 complete OXF hemisphere pairs. It is useful
    for sample-complete diagnostics, but each row carries the montage source so
    exact-contact and fallback rows are not silently treated as equivalent.
    """

    exact = v2e_pairs[v2e_pairs["feature_set"].astype(str).eq("off_beta_peak_contact_exact")].copy()
    numeric = v2e_pairs[v2e_pairs["feature_set"].astype(str).eq("off_beta_peak_contact_numeric")].copy()
    median = v2e_pairs[v2e_pairs["feature_set"].astype(str).eq("median_channel_reference_with_bursts")].copy()
    exact_by_id = {str(row["pair_id"]): row for _, row in exact.iterrows()}
    numeric_by_id = {str(row["pair_id"]): row for _, row in numeric.iterrows()}
    median_by_id = {str(row["pair_id"]): row for _, row in median.iterrows()}
    rows = []
    for _, case in coverage_cases.sort_values(["subject", "side"]).iterrows():
        pair_id = str(case["pair_id"])
        repair_class = str(case.get("repair_class", ""))
        source = "missing"
        row = None
        if pair_id in exact_by_id and repair_class.startswith("exact_"):
            row = exact_by_id[pair_id].copy()
            source = "exact_contact"
        elif include_numeric_repair and pair_id in numeric_by_id and repair_class == "numeric_only_recovered":
            row = numeric_by_id[pair_id].copy()
            source = "numeric_contact_repair"
        elif pair_id in median_by_id:
            row = median_by_id[pair_id].copy()
            source = "median_channel_fallback"
        if row is None:
            continue
        row = row.to_dict()
        row["montage_hierarchy_source"] = source
        row["montage_repair_class"] = repair_class
        row["off_spacing_class"] = case.get("off_spacing_class", "")
        row["same_intercontact_distance"] = bool(case.get("same_intercontact_distance", False))
        row["off_valid_bipolar_geometry"] = bool(case.get("off_valid_bipolar_geometry", False))
        row["montage_hierarchy_warning"] = (
            "heterogeneous_all_pairs_branch_report_source_strata"
            if source != "exact_contact"
            else "exact_contact_row"
        )
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def pseudo_contact_rows(inventory: pd.DataFrame) -> pd.DataFrame:
    """Build feature-level contact proxies from cached channel features.

    This is not raw pseudo-monopolar reconstruction. It pools cached features
    from channels that include a given constituent contact. Segment labels are
    excluded from the proxy to avoid mixing directional/segmented labels with
    ring-contact geometry.
    """

    ok = inventory[inventory["feature_status"].astype(str).eq("ok")].copy()
    ok = ok[ok["spacing_class"].astype(str).isin(["narrow", "broad", "unknown"])].copy()
    rows = []
    metadata = [
        "file_path",
        "file_name",
        "subject",
        "side",
        "medication_state",
        "flag_erna_label",
        "loader",
        "sampling_frequency",
    ]
    for keys, group in ok.groupby(["subject", "side", "medication_state"], dropna=False):
        subject, side, medication = keys
        contacts = sorted({digit for value in group["contact_digits"].astype(str) for digit in value if digit.isdigit()})
        for contact in contacts:
            source = group[group["contact_digits"].astype(str).map(lambda value: contact in set(value))].copy()
            if source.empty:
                continue
            first = source.iloc[0]
            values = source[FEATURES].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
            spacing_classes = sorted(set(source["spacing_class"].astype(str)))
            rows.append(
                {
                    **{col: first.get(col, np.nan) for col in metadata},
                    "channel": f"{str(side).lower()[0].upper()}pseudo{contact}",
                    "pseudo_contact": contact,
                    "feature_status": "ok",
                    "source_channels": ";".join(source["channel"].astype(str).tolist()),
                    "source_spacing_classes": ";".join(spacing_classes),
                    "n_source_channels": int(len(source)),
                    "n_valid_bipolar_source_channels": int(source["valid_bipolar_geometry"].astype(bool).sum()),
                    "n_single_or_unknown_source_channels": int((source["spacing_class"].astype(str) == "unknown").sum()),
                    "pseudo_proxy_warning": "feature_level_constituent_contact_proxy_not_raw_pseudomonopolar",
                    **values.to_dict(),
                }
            )
    return pd.DataFrame(rows).sort_values(["subject", "side", "medication_state", "pseudo_contact"]).reset_index(drop=True)


def build_pseudo_pairs(pseudo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    selection_rows = []
    for (subject, side), group in pseudo.groupby(["subject", "side"], dropna=False):
        off = group[group["medication_state"].astype(str).eq("OFF")].copy()
        on = group[group["medication_state"].astype(str).eq("ON")].copy()
        if off.empty or on.empty:
            selection_rows.append(
                {
                    "pair_id": f"{subject}_{side}",
                    "subject": subject,
                    "side": side,
                    "selection_status": "missing_off_or_on",
                }
            )
            continue
        selected = select_off_channel(off.rename(columns={"pseudo_contact": "pseudo_contact"}))
        contact = str(selected.get("pseudo_contact", ""))
        on_match = on[on["pseudo_contact"].astype(str).eq(contact)].copy()
        if on_match.empty:
            selection_rows.append(
                {
                    "pair_id": f"{subject}_{side}",
                    "subject": subject,
                    "side": side,
                    "selection_status": "no_on_pseudo_contact_match",
                    "off_selected_pseudo_contact": contact,
                    "available_on_pseudo_contacts": ";".join(on["pseudo_contact"].astype(str).tolist()),
                    "off_source_channels": selected.get("source_channels", ""),
                }
            )
            continue
        on_row = on_match.sort_values("pseudo_contact").iloc[0]
        selection = {
            "selection_rule": "off_beta_peak_pseudo_contact_proxy",
            "match_mode": "same_pseudo_contact",
            "selection_status": "paired",
            "off_contact_token": contact,
            "on_contact_token": contact,
            "off_selected_pseudo_contact": contact,
            "on_selected_pseudo_contact": contact,
            "off_source_channels": selected.get("source_channels", ""),
            "on_source_channels": on_row.get("source_channels", ""),
            "off_n_source_channels": selected.get("n_source_channels", np.nan),
            "on_n_source_channels": on_row.get("n_source_channels", np.nan),
            "off_source_spacing_classes": selected.get("source_spacing_classes", ""),
            "on_source_spacing_classes": on_row.get("source_spacing_classes", ""),
            "pseudo_proxy_warning": "exploratory_feature_level_proxy_not_primary",
        }
        pair_rows.append(
            build_pair_row_from_rows(
                str(subject),
                str(side),
                "pseudo_contact_proxy",
                selected,
                on_row,
                selection,
            )
        )
        selection_rows.append({"pair_id": f"{subject}_{side}", "subject": subject, "side": side, **selection})
    return pd.DataFrame(pair_rows), pd.DataFrame(selection_rows)


def available_features(pairs: pd.DataFrame, features: list[str]) -> list[str]:
    usable = []
    for feature in features:
        cols = [f"x_{feature}", f"medon_{feature}"]
        if any(col not in pairs for col in cols):
            continue
        values = pairs[cols].apply(pd.to_numeric, errors="coerce")
        if values.notna().all(axis=1).sum() >= 3:
            usable.append(feature)
    return usable


def evaluate_branch(branch: dict[str, object], pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    branch_id = str(branch["branch_id"])
    if pairs.empty:
        empty = pd.DataFrame()
        return {
            "candidate_scores": empty,
            "observed_metrics": pd.DataFrame(
                [
                    {
                        "branch_id": branch_id,
                        "branch_role": branch["branch_role"],
                        "skip_reason": "no_pairs_after_montage_filter",
                        "n_pairs": 0,
                        "n_subjects": 0,
                    }
                ]
            ),
            "query_diagnostics": empty,
            "failure_cases": empty,
            "hard_negative_summary": empty,
            "hard_negative_cases": empty,
        }
    usable = available_features(pairs, PRIMARY_FEATURES)
    if len(usable) == 0 or pairs["subject"].nunique() < 3:
        empty = pd.DataFrame()
        return {
            "candidate_scores": empty,
            "observed_metrics": pd.DataFrame(
                [
                    {
                        "branch_id": branch_id,
                        "branch_role": branch["branch_role"],
                        "skip_reason": "too_few_subjects_or_no_features",
                        "n_pairs": int(len(pairs)),
                        "n_subjects": int(pairs["subject"].nunique()),
                        "usable_features": ";".join(usable),
                    }
                ]
            ),
            "query_diagnostics": empty,
            "failure_cases": empty,
            "hard_negative_summary": empty,
            "hard_negative_cases": empty,
        }
    result = evaluate_one_feature_set_fast(
        pairs=pairs,
        feature_set=branch_id,
        usable_features=usable,
        rerank_features=usable,
    )
    out: dict[str, pd.DataFrame] = {}
    rename = {
        "scores": "candidate_scores",
        "metrics": "observed_metrics",
        "diagnostics": "query_diagnostics",
    }
    for old_key, table in result.items():
        new_key = rename.get(old_key, old_key)
        frame = table.copy()
        if not frame.empty:
            frame.insert(0, "branch_interpretation", branch["interpretation"])
            frame.insert(0, "branch_role", branch["branch_role"])
            frame.insert(0, "branch_id", branch_id)
            frame.insert(0, "usable_features", ";".join(usable))
        out[new_key] = frame
    return out


def concat_result(results: list[dict[str, pd.DataFrame]], key: str) -> pd.DataFrame:
    frames = [result[key] for result in results if key in result and not result[key].empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def focus_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "variant_name" not in metrics:
        return metrics
    out = metrics[
        metrics["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & metrics["variant_name"].astype(str).isin(FOCUS_VARIANTS)
    ].copy()
    out["is_primary_conservative_branch"] = (
        out["branch_id"].astype(str).eq("exact_same_spacing_compact_residual_7")
        & out["variant_name"].astype(str).eq(V2C_DECONFOUNDED_NAME)
    )
    return out.sort_values(["top1", "mrr"], ascending=False).reset_index(drop=True)


def bootstrap_by_branch(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["branch_id", "branch_role", "usable_features"]
    for keys, group in diagnostics.groupby(group_cols, dropna=False):
        clean = group.drop(columns=[col for col in group_cols if col in group], errors="ignore")
        summary, _samples = v2e.subject_bootstrap_ci(clean)
        if summary.empty:
            continue
        for col, value in zip(group_cols, keys, strict=False):
            summary.insert(0, col, value)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_branch_tables(
    inventory: pd.DataFrame,
    coverage_cases: pd.DataFrame,
    v2e_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    exact = exact_pairs_with_geometry(v2e_pairs, coverage_cases)
    pseudo_rows = pseudo_contact_rows(inventory)
    pseudo_pairs, pseudo_selection = build_pseudo_pairs(pseudo_rows)
    branch_pairs: dict[str, pd.DataFrame] = {}
    for branch in BRANCHES:
        branch_id = str(branch["branch_id"])
        if branch["source"] == "cached_channel_feature_contact_proxy":
            data = pseudo_pairs.copy()
        elif branch["source"] == "hierarchy_exact_else_median":
            data = all_pairs_hierarchy_branch(v2e_pairs, coverage_cases, include_numeric_repair=False)
        elif branch["source"] == "hierarchy_exact_numeric_else_median":
            data = all_pairs_hierarchy_branch(v2e_pairs, coverage_cases, include_numeric_repair=True)
        else:
            data = filter_exact_branch(exact, str(branch["spacing_filter"]))
        data["feature_set"] = branch_id
        branch_pairs[branch_id] = data.reset_index(drop=True)
    branch_pair_summary = pd.DataFrame(
        [
            {
                "branch_id": branch["branch_id"],
                "branch_role": branch["branch_role"],
                "source": branch["source"],
                "spacing_filter": branch["spacing_filter"],
                "n_pairs": int(len(branch_pairs[str(branch["branch_id"])])),
                "n_subjects": int(branch_pairs[str(branch["branch_id"])]["subject"].nunique())
                if not branch_pairs[str(branch["branch_id"])].empty
                else 0,
                "interpretation": branch["interpretation"],
            }
            for branch in BRANCHES
        ]
    )
    return branch_pair_summary, pseudo_rows, pseudo_selection, branch_pairs


def all_pairs_source_summary(branch_pairs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for branch_id in [
        "all_pairs_exact_else_median_compact_residual_7",
        "all_pairs_exact_numeric_else_median_compact_residual_7",
    ]:
        table = branch_pairs.get(branch_id, pd.DataFrame())
        if table.empty or "montage_hierarchy_source" not in table:
            continue
        for keys, group in table.groupby(["montage_hierarchy_source", "montage_repair_class"], dropna=False):
            source, repair = keys
            rows.append(
                {
                    "branch_id": branch_id,
                    "montage_hierarchy_source": source,
                    "montage_repair_class": repair,
                    "n_pairs": int(len(group)),
                    "n_subjects": int(group["subject"].nunique()),
                }
            )
    return pd.DataFrame(rows).sort_values(["branch_id", "montage_hierarchy_source", "montage_repair_class"])


def make_plots(coverage_summary: pd.DataFrame, focus: pd.DataFrame) -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not coverage_summary.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            data = coverage_summary.sort_values("n_pairs", ascending=True)
            ax.barh(data["repair_class"], data["n_pairs"], color="#4c78a8")
            ax.set_xlabel("Pairs")
            ax.set_ylabel("")
            ax.set_title("OXF montage coverage classes")
            fig.tight_layout()
            path = OUTPUT_DIR / "montage_coverage_classes.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths.append(str(path))
        if not focus.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            data = focus[focus["variant_name"].astype(str).isin([V2C_DECONFOUNDED_NAME, V2D_QUALITY_DECONFOUNDED_NAME])].copy()
            labels = data["branch_id"].astype(str) + "\n" + data["variant_name"].astype(str).str.replace("_", " ", regex=False)
            ax.bar(range(len(data)), pd.to_numeric(data["top1"], errors="coerce"), color="#59a14f")
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Top1")
            ax.set_title("Retrieval by montage branch")
            fig.tight_layout()
            path = OUTPUT_DIR / "montage_branch_top1.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths.append(str(path))
    except Exception as exc:  # noqa: BLE001
        (OUTPUT_DIR / "plot_warning.txt").write_text(f"Plot generation skipped: {exc}\n", encoding="utf-8")
    return paths


def write_report(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# OXF Montage Harmonization Audit",
        "",
        "Diagnostic only. Cached OXF features, fixed top_k=5, alpha=0.5, no ds004998 pipeline changes.",
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

    channel_features = read_csv(CHANNEL_FEATURES_PATH)
    v2e_pairs = read_csv(V2E_PAIRS_PATH)
    inventory = annotate_channel_inventory(channel_features)
    coverage_cases = build_montage_coverage_cases(inventory)
    coverage_summary, spacing_summary = summarize_coverage(coverage_cases)
    geometry_summary = channel_geometry_summary(inventory)
    branch_pair_summary, pseudo_rows, pseudo_selection, branch_pairs = build_branch_tables(
        inventory,
        coverage_cases,
        v2e_pairs,
    )
    hierarchy_source_summary = all_pairs_source_summary(branch_pairs)
    branch_results = [evaluate_branch(branch, branch_pairs[str(branch["branch_id"])]) for branch in BRANCHES]
    scores = concat_result(branch_results, "candidate_scores")
    metrics = concat_result(branch_results, "observed_metrics")
    diagnostics = concat_result(branch_results, "query_diagnostics")
    failures = concat_result(branch_results, "failure_cases")
    hard_summary = concat_result(branch_results, "hard_summary")
    hard_cases = concat_result(branch_results, "hard_cases")
    focus = focus_metrics(metrics)
    bootstrap = bootstrap_by_branch(diagnostics)
    plot_paths = make_plots(coverage_summary, focus)

    outputs = {
        "channel_montage_inventory": write_csv(inventory, OUTPUT_DIR / "channel_montage_inventory.csv"),
        "channel_geometry_summary": write_csv(geometry_summary, OUTPUT_DIR / "channel_geometry_summary.csv"),
        "montage_coverage_cases": write_csv(coverage_cases, OUTPUT_DIR / "montage_coverage_cases.csv"),
        "montage_coverage_summary": write_csv(coverage_summary, OUTPUT_DIR / "montage_coverage_summary.csv"),
        "spacing_class_summary": write_csv(spacing_summary, OUTPUT_DIR / "spacing_class_summary.csv"),
        "branch_pair_summary": write_csv(branch_pair_summary, OUTPUT_DIR / "branch_pair_summary.csv"),
        "all_pairs_hierarchy_source_summary": write_csv(
            hierarchy_source_summary,
            OUTPUT_DIR / "all_pairs_hierarchy_source_summary.csv",
        ),
        "pseudo_contact_proxy_features": write_csv(pseudo_rows, OUTPUT_DIR / "pseudo_contact_proxy_features.csv"),
        "pseudo_contact_proxy_selection": write_csv(pseudo_selection, OUTPUT_DIR / "pseudo_contact_proxy_selection.csv"),
        "candidate_scores": write_csv(scores, OUTPUT_DIR / "montage_branch_candidate_scores.csv"),
        "observed_metrics": write_csv(metrics, OUTPUT_DIR / "montage_branch_observed_metrics.csv"),
        "focus_metrics": write_csv(focus, OUTPUT_DIR / "montage_branch_focus_metrics.csv"),
        "query_diagnostics": write_csv(diagnostics, OUTPUT_DIR / "montage_branch_query_diagnostics.csv"),
        "bootstrap_ci": write_csv(bootstrap, OUTPUT_DIR / "montage_branch_bootstrap_ci.csv"),
        "failure_cases": write_csv(failures, OUTPUT_DIR / "montage_branch_failure_cases.csv"),
        "hard_negative_summary": write_csv(hard_summary, OUTPUT_DIR / "montage_branch_hard_negative_summary.csv"),
        "hard_negative_cases": write_csv(hard_cases, OUTPUT_DIR / "montage_branch_hard_negative_cases.csv"),
    }
    for branch_id, table in branch_pairs.items():
        outputs[f"branch_pairs_{branch_id}"] = write_csv(table, OUTPUT_DIR / f"{branch_id}_pairs.csv")

    def metric_value(branch_id: str, variant: str, metric: str) -> float:
        if focus.empty:
            return np.nan
        rows = focus[
            focus["branch_id"].astype(str).eq(branch_id)
            & focus["variant_name"].astype(str).eq(variant)
        ]
        return safe_float(rows.iloc[0].get(metric)) if not rows.empty else np.nan

    def coverage_n(label: str) -> int:
        rows = coverage_summary[coverage_summary["repair_class"].astype(str).eq(label)]
        return int(rows.iloc[0]["n_pairs"]) if not rows.empty else 0

    key_results = {
        "seed": RANDOM_SEED,
        "top_k": TOP_K,
        "aperiodic_alpha": APERIODIC_ALPHA,
        "total_complete_off_on_hemisphere_pairs": int(len(coverage_cases)),
        "exact_same_spacing_pairs": coverage_n("exact_same_spacing_available"),
        "numeric_only_recovered_pairs": coverage_n("numeric_only_recovered"),
        "constituent_contacts_available_not_valid_bipolar_pairs": coverage_n(
            "constituent_contacts_available_not_valid_bipolar"
        ),
        "no_valid_on_contact_pairs": coverage_n("no_valid_on_contact"),
        "exact_same_spacing_v2c_top1": metric_value(
            "exact_same_spacing_compact_residual_7",
            V2C_DECONFOUNDED_NAME,
            "top1",
        ),
        "exact_same_spacing_v2c_mrr": metric_value(
            "exact_same_spacing_compact_residual_7",
            V2C_DECONFOUNDED_NAME,
            "mrr",
        ),
        "exact_narrow_v2c_top1": metric_value("exact_narrow_only_compact_residual_7", V2C_DECONFOUNDED_NAME, "top1"),
        "exact_broad_v2c_top1": metric_value("exact_broad_only_compact_residual_7", V2C_DECONFOUNDED_NAME, "top1"),
        "pseudo_proxy_v2c_top1": metric_value("pseudo_contact_proxy_compact_residual_7", V2C_DECONFOUNDED_NAME, "top1"),
        "pseudo_proxy_v2c_mrr": metric_value("pseudo_contact_proxy_compact_residual_7", V2C_DECONFOUNDED_NAME, "mrr"),
        "all_pairs_exact_else_median_v2c_top1": metric_value(
            "all_pairs_exact_else_median_compact_residual_7",
            V2C_DECONFOUNDED_NAME,
            "top1",
        ),
        "all_pairs_exact_else_median_v2c_mrr": metric_value(
            "all_pairs_exact_else_median_compact_residual_7",
            V2C_DECONFOUNDED_NAME,
            "mrr",
        ),
        "all_pairs_exact_numeric_else_median_v2c_top1": metric_value(
            "all_pairs_exact_numeric_else_median_compact_residual_7",
            V2C_DECONFOUNDED_NAME,
            "top1",
        ),
        "all_pairs_exact_numeric_else_median_v2c_mrr": metric_value(
            "all_pairs_exact_numeric_else_median_compact_residual_7",
            V2C_DECONFOUNDED_NAME,
            "mrr",
        ),
    }

    warnings = [
        "Numeric-token contact repair is sensitivity only; it is not a primary montage-harmonized result.",
        "Pseudo-contact proxy is built from cached feature values, not raw pseudo-monopolar signal reconstruction.",
        "Segment/directional labels are excluded from the pseudo-contact proxy and are not merged into ring-contact bipolar classes.",
        "Exact narrow-only may have too few pairs for stable retrieval if the OXF selected contacts are mostly broad.",
        "Coverage-complete all-pairs branches are heterogeneous and must be reported with montage source strata.",
        "Frozen ds004998 v2A and previous OXF outputs are not modified.",
    ]
    summary = {
        "key_results": key_results,
        "outputs": outputs,
        "plots": plot_paths,
        "warnings": warnings,
    }
    write_json(summary, OUTPUT_DIR / "oxf_montage_harmonization_audit_summary.json")
    write_report(summary, OUTPUT_DIR / "oxf_montage_harmonization_audit_summary.md")

    print("OXF montage harmonization audit complete.")
    print(f"Output folder: {OUTPUT_DIR}")
    for key, value in key_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
