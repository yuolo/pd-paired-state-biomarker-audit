"""Run physiology-locked OXF STN feature variants for external retrieval.

This script is a follow-up to the first OXF STN-only validation. It tests
pre-specified, literature-motivated feature extraction changes that are
scientifically interpretable rather than tuned to retrieval performance:

1. OFF-only beta contact selection.
2. Exact or numeric ON contact matching, explicitly recorded.
3. Fixed broad-beta burst features.
4. The same top_k=5 and aperiodic_alpha=0.5 retrieval/reranking controls.

It does not modify the frozen ds004998 pipeline, does not tune parameters on
OXF, and does not make clinical prediction, treatment, DBS optimization, or
causal medication-effect claims.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts import run_oxf_external_stn_retrieval_validation as oxf_base  # noqa: E402
from scripts.run_v2a_scientific_audit_17subjects_33pairs import (  # noqa: E402
    APERIODIC_ALPHA,
    N_BOOT,
    N_PERM,
    RANDOM_SEED,
    TOP_K,
    query_metrics,
)
from scripts.run_v2b_gated_scientific_audit_17subjects_33pairs import (  # noqa: E402
    ALL_MEDON_CONDITION,
    HARD_NEGATIVE_CONDITION,
    PRIMARY_CONDITION,
    V2B_NAME,
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


OUTPUT_DIR = Path("outputs/oxf_stn_physiology_locked_v2e")
BASE_OXF_OUTPUT_DIR = Path("outputs/oxf_external_stn_retrieval_validation")
DEFAULT_DATA_ROOT = Path("data/oxf/DBS")
MAX_DURATION_SEC = 120.0
MIN_BURST_DURATION_SEC = 0.100
BETA_BAND = (13.0, 35.0)

BURST_FEATURES = [
    "beta_burst_rate",
    "beta_burst_mean_duration",
    "beta_burst_median_duration",
    "beta_burst_occupancy_fraction",
    "beta_burst_mean_amplitude",
    "beta_burst_amplitude_cv",
    "beta_burst_long_fraction_400ms",
    "beta_burst_long_fraction_600ms",
    "beta_burst_long_rate_400ms",
    "beta_burst_long_rate_600ms",
]
PHYSIOLOGY_FEATURES = [*oxf_base.ALL_FEATURES, *BURST_FEATURES]
PHYSIOLOGY_COMPACT_FEATURES = [*oxf_base.COMPACT_FEATURES, *BURST_FEATURES]

FEATURE_SET_SPECS = [
    {
        "feature_set": "median_channel_reference_with_bursts",
        "description": "Median across side-matched STN channels; current OXF reference plus burst features.",
        "source": "median",
        "selection_rule": "median",
        "match_mode": "not_applicable",
        "primary": False,
    },
    {
        "feature_set": "off_beta_peak_contact_exact",
        "description": "OFF-only max residual beta peak channel; ON channel matched by exact normalized contact token.",
        "source": "contact",
        "selection_rule": "off_beta_peak_amplitude",
        "match_mode": "exact",
        "primary": True,
    },
    {
        "feature_set": "off_beta_peak_contact_numeric",
        "description": "OFF-only max residual beta peak channel; ON channel matched by numeric contact token.",
        "source": "contact",
        "selection_rule": "off_beta_peak_amplitude",
        "match_mode": "numeric",
        "primary": False,
    },
    {
        "feature_set": "off_long_beta_burst_contact_numeric",
        "description": "OFF-only max long beta-burst fraction channel; ON channel matched by numeric contact token.",
        "source": "contact",
        "selection_rule": "off_long_beta_burst_fraction",
        "match_mode": "numeric",
        "primary": False,
    },
    {
        "feature_set": "off_combined_beta_contact_numeric",
        "description": "OFF-only combined beta peak plus long-burst rank channel; ON matched by numeric contact token.",
        "source": "contact",
        "selection_rule": "off_combined_beta_peak_burst_rank",
        "match_mode": "numeric",
        "primary": False,
    },
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--base-output-dir", default=str(BASE_OXF_OUTPUT_DIR))
    parser.add_argument("--max-duration-sec", type=float, default=MAX_DURATION_SEC)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def write_json(data: dict[str, object], path: Path) -> Path:
    """Write JSON output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(oxf_base.to_builtin(data), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(data: pd.DataFrame, path: Path) -> Path:
    """Write CSV output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return path


def write_text(lines: Iterable[str], path: Path) -> Path:
    """Write a text file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def safe_float(value: object, default: float = np.nan) -> float:
    """Convert to finite float if possible."""

    try:
        out = float(value)
    except Exception:  # noqa: BLE001
        return default
    return out if np.isfinite(out) else default


def stable_zscore(values: pd.Series) -> pd.Series:
    """Z-score with finite fallback, preserving index."""

    arr = pd.to_numeric(values, errors="coerce")
    mean = float(arr.mean()) if arr.notna().any() else 0.0
    std = float(arr.std(ddof=0)) if arr.notna().any() else 1.0
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return (arr - mean) / std


def load_or_build_channel_features(data_root: Path, base_output_dir: Path, max_duration_sec: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached OXF channel features or build them with the base script."""

    channel_path = base_output_dir / "oxf_stn_channel_features.csv"
    state_path = base_output_dir / "oxf_stn_state_features.csv"
    if channel_path.exists() and state_path.exists():
        return pd.read_csv(channel_path), pd.read_csv(state_path)
    inventory = oxf_base.discover_records(data_root)
    state_features, channel_features = oxf_base.extract_state_features(inventory, max_duration_sec)
    return channel_features, state_features


def beta_burst_events(mask: np.ndarray, envelope: np.ndarray, sfreq: float) -> list[dict[str, float]]:
    """Extract contiguous beta-burst events."""

    if len(mask) == 0:
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    starts = np.where(np.diff(padded.astype(int)) == 1)[0]
    ends = np.where(np.diff(padded.astype(int)) == -1)[0]
    events = []
    for start, end in zip(starts, ends, strict=False):
        duration = (end - start) / sfreq
        if duration < MIN_BURST_DURATION_SEC:
            continue
        amp = envelope[start:end]
        events.append(
            {
                "duration_sec": float(duration),
                "mean_amplitude": float(np.nanmean(amp)) if len(amp) else np.nan,
            }
        )
    return events


def beta_burst_feature_row(signal: np.ndarray, sfreq: float) -> dict[str, float]:
    """Compute fixed broad-beta burst features for one channel."""

    if oxf_base.scipy_signal is None or not np.isfinite(sfreq) or sfreq <= 2 * BETA_BAND[1] or len(signal) < int(sfreq):
        return {feature: np.nan for feature in BURST_FEATURES}
    try:
        sos = oxf_base.scipy_signal.butter(4, BETA_BAND, btype="bandpass", fs=sfreq, output="sos")
        filtered = oxf_base.scipy_signal.sosfiltfilt(sos, signal)
        envelope = np.abs(oxf_base.scipy_signal.hilbert(filtered))
    except Exception:  # noqa: BLE001
        return {feature: np.nan for feature in BURST_FEATURES}

    threshold = float(np.nanpercentile(envelope, 75))
    events = beta_burst_events(envelope > threshold, envelope, sfreq)
    total_duration = len(signal) / sfreq
    durations = np.asarray([event["duration_sec"] for event in events], dtype=float)
    amplitudes = np.asarray([event["mean_amplitude"] for event in events], dtype=float)
    long_400 = durations >= 0.400 if len(durations) else np.asarray([], dtype=bool)
    long_600 = durations >= 0.600 if len(durations) else np.asarray([], dtype=bool)
    amp_mean = float(np.nanmean(amplitudes)) if len(amplitudes) else np.nan
    amp_std = float(np.nanstd(amplitudes)) if len(amplitudes) else np.nan
    return {
        "beta_burst_rate": float(len(events) / total_duration) if total_duration > 0 else np.nan,
        "beta_burst_mean_duration": float(np.nanmean(durations)) if len(durations) else 0.0,
        "beta_burst_median_duration": float(np.nanmedian(durations)) if len(durations) else 0.0,
        "beta_burst_occupancy_fraction": float(np.nansum(durations) / total_duration) if total_duration > 0 else np.nan,
        "beta_burst_mean_amplitude": amp_mean,
        "beta_burst_amplitude_cv": float(amp_std / amp_mean) if np.isfinite(amp_mean) and abs(amp_mean) > 1e-12 else np.nan,
        "beta_burst_long_fraction_400ms": float(np.mean(long_400)) if len(long_400) else 0.0,
        "beta_burst_long_fraction_600ms": float(np.mean(long_600)) if len(long_600) else 0.0,
        "beta_burst_long_rate_400ms": float(np.sum(long_400) / total_duration) if total_duration > 0 else np.nan,
        "beta_burst_long_rate_600ms": float(np.sum(long_600) / total_duration) if total_duration > 0 else np.nan,
    }


def compute_channel_bursts(inventory: pd.DataFrame, max_duration_sec: float) -> pd.DataFrame:
    """Compute broad-beta burst features for every selected side-matched channel."""

    rows = []
    for _, record in inventory.iterrows():
        path = Path(record["file_path"])
        try:
            channel_titles, channels, sfreq, meta = oxf_base.read_record_channels(path, max_duration_sec)
            for title, signal in zip(channel_titles, channels, strict=False):
                features = beta_burst_feature_row(signal, sfreq)
                rows.append(
                    {
                        "file_path": str(path),
                        "subject": record["subject"],
                        "side": record["side"],
                        "medication_state": record["medication_state"],
                        "channel": title,
                        "sampling_frequency": sfreq,
                        "burst_loader": meta.get("loader", ""),
                        **features,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "file_path": str(path),
                    "subject": record["subject"],
                    "side": record["side"],
                    "medication_state": record["medication_state"],
                    "channel": "",
                    "sampling_frequency": np.nan,
                    "burst_loader": "load_failed",
                    "burst_error": str(exc),
                    **{feature: np.nan for feature in BURST_FEATURES},
                }
            )
    return pd.DataFrame(rows)


def load_or_build_bursts(data_root: Path, output_dir: Path, max_duration_sec: float) -> pd.DataFrame:
    """Load cached burst features or compute them."""

    path = output_dir / "v2e_channel_burst_features.csv"
    if path.exists():
        return pd.read_csv(path)
    inventory = oxf_base.discover_records(data_root)
    bursts = compute_channel_bursts(inventory, max_duration_sec)
    write_csv(bursts, path)
    return bursts


def merge_channel_features(channel_features: pd.DataFrame, burst_features: pd.DataFrame) -> pd.DataFrame:
    """Merge spectral/residual channel features with burst features."""

    keys = ["file_path", "subject", "side", "medication_state", "channel"]
    burst_cols = keys + BURST_FEATURES
    burst = burst_features[[col for col in burst_cols if col in burst_features.columns]].copy()
    merged = channel_features.merge(burst, on=keys, how="left")
    for feature in BURST_FEATURES:
        if feature not in merged:
            merged[feature] = np.nan
    return merged


def exact_contact_token(channel: object, side: object) -> str:
    """Normalize contact title while preserving exact numeric string."""

    text = str(channel).strip().lower()
    side_text = str(side).lower()
    if side_text == "left":
        text = re.sub(r"^le", "l", text)
        text = re.sub(r"^l", "", text)
    elif side_text == "right":
        text = re.sub(r"^re", "r", text)
        text = re.sub(r"^r", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def numeric_contact_token(channel: object, side: object) -> str:
    """Normalize contact title to numeric token only."""

    token = exact_contact_token(channel, side)
    digits = re.sub(r"[^0-9]", "", token)
    return digits.lstrip("0") or digits


def contact_token(channel: object, side: object, mode: str) -> str:
    """Return contact token for exact or numeric matching."""

    if mode == "exact":
        return exact_contact_token(channel, side)
    if mode == "numeric":
        return numeric_contact_token(channel, side)
    raise ValueError(f"Unsupported match mode: {mode}")


def select_off_channel(off: pd.DataFrame, rule: str) -> pd.Series:
    """Select one OFF channel using a fixed physiological rule."""

    data = off.copy()
    if rule == "off_beta_peak_amplitude":
        data["_score"] = pd.to_numeric(data["beta_peak_amplitude"], errors="coerce")
        if data["_score"].notna().sum() == 0:
            data["_score"] = pd.to_numeric(data["stn_broad_beta_log_power"], errors="coerce")
    elif rule == "off_long_beta_burst_fraction":
        data["_score"] = pd.to_numeric(data["beta_burst_long_fraction_400ms"], errors="coerce")
        if data["_score"].notna().sum() == 0 or float(data["_score"].max()) <= 0.0:
            data["_score"] = pd.to_numeric(data["beta_peak_amplitude"], errors="coerce")
    elif rule == "off_combined_beta_peak_burst_rank":
        peak = stable_zscore(data["beta_peak_amplitude"])
        burst = stable_zscore(data["beta_burst_long_fraction_400ms"])
        data["_score"] = peak.fillna(0.0) + burst.fillna(0.0)
    else:
        raise ValueError(f"Unsupported selection rule: {rule}")
    data["_score"] = pd.to_numeric(data["_score"], errors="coerce").fillna(-np.inf)
    return data.sort_values(["_score", "channel"], ascending=[False, True]).iloc[0]


def pair_feature_availability(pairs: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return finite-value availability for a pair table."""

    rows = []
    for feature in features:
        cols = [f"x_{feature}", f"medon_{feature}"]
        if any(col not in pairs for col in cols):
            finite_fraction = 0.0
            usable = False
        else:
            values = pairs[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            finite_fraction = float(finite.mean()) if finite.size else 0.0
            usable = bool(finite.all()) if finite.size else False
        rows.append(
            {
                "feature": feature,
                "feature_group": feature_group(feature),
                "finite_value_fraction": finite_fraction,
                "usable_for_retrieval": usable,
            }
        )
    return pd.DataFrame(rows)


def feature_group(feature: str) -> str:
    """Map a v2E feature to a diagnostic family."""

    if feature in BURST_FEATURES:
        return "beta_burst_features"
    if feature in oxf_base.COMPACT_FEATURES:
        return "compact_aperiodic_residual"
    return "stn_spectral"


def build_pair_row(
    subject: str,
    side: str,
    feature_set: str,
    off_row: pd.Series,
    on_row: pd.Series,
    selection_row: dict[str, object],
) -> dict[str, object]:
    """Build one pair row from selected OFF and ON channel rows."""

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
        **selection_row,
    }
    for feature in PHYSIOLOGY_FEATURES:
        row[f"x_{feature}"] = safe_float(off_row.get(feature))
        row[f"medon_{feature}"] = safe_float(on_row.get(feature))
        row[f"y_{feature}"] = safe_float(on_row.get(feature))
    return row


def build_median_pairs(channel_features: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build median-channel reference pairs with burst features included."""

    state_rows = []
    for keys, group in channel_features.groupby(["subject", "side", "medication_state"], dropna=False):
        subject, side, medication = keys
        feature_values = group[PHYSIOLOGY_FEATURES].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
        first = group.iloc[0]
        state_rows.append(
            {
                "subject": subject,
                "side": side,
                "medication_state": medication,
                "file_path": first.get("file_path", ""),
                "flag_erna_label": bool(first.get("flag_erna_label", False)),
                "sampling_frequency": first.get("sampling_frequency", np.nan),
                "channel": "median_side_matched_channels",
                **feature_values.to_dict(),
            }
        )
    states = pd.DataFrame(state_rows)
    pair_rows = []
    selection_rows = []
    for (subject, side), group in states.groupby(["subject", "side"], dropna=False):
        off = group[group["medication_state"].astype(str).eq("OFF")]
        on = group[group["medication_state"].astype(str).eq("ON")]
        if off.empty or on.empty:
            selection_rows.append(
                {
                    "feature_set": feature_set,
                    "subject": subject,
                    "side": side,
                    "selection_status": "missing_off_or_on",
                }
            )
            continue
        selection = {
            "selection_rule": "median",
            "match_mode": "not_applicable",
            "selection_status": "paired",
            "off_contact_token": "",
            "on_contact_token": "",
        }
        pair_rows.append(build_pair_row(subject, side, feature_set, off.iloc[0], on.iloc[0], selection))
        selection_rows.append({"feature_set": feature_set, "subject": subject, "side": side, **selection})
    return pd.DataFrame(pair_rows), pd.DataFrame(selection_rows)


def build_contact_pairs(
    channel_features: pd.DataFrame,
    feature_set: str,
    selection_rule: str,
    match_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build OFF-contact-selected pair table."""

    pair_rows = []
    selection_rows = []
    usable = channel_features[channel_features["feature_status"].astype(str).eq("ok")].copy()
    for (subject, side), group in usable.groupby(["subject", "side"], dropna=False):
        off = group[group["medication_state"].astype(str).eq("OFF")].copy()
        on = group[group["medication_state"].astype(str).eq("ON")].copy()
        if off.empty or on.empty:
            selection_rows.append(
                {
                    "feature_set": feature_set,
                    "subject": subject,
                    "side": side,
                    "selection_rule": selection_rule,
                    "match_mode": match_mode,
                    "selection_status": "missing_off_or_on",
                }
            )
            continue
        selected = select_off_channel(off, selection_rule)
        off_token = contact_token(selected["channel"], side, match_mode)
        on = on.copy()
        on["_token"] = on["channel"].map(lambda value: contact_token(value, side, match_mode))
        exact_on = on[on["_token"].astype(str).eq(str(off_token))].sort_values("channel")
        if exact_on.empty:
            selection_rows.append(
                {
                    "feature_set": feature_set,
                    "subject": subject,
                    "side": side,
                    "selection_rule": selection_rule,
                    "match_mode": match_mode,
                    "selection_status": "no_on_contact_match",
                    "off_selected_channel": selected.get("channel", ""),
                    "off_contact_token": off_token,
                    "available_on_channels": ";".join(on["channel"].astype(str).tolist()),
                }
            )
            continue
        on_selected = exact_on.iloc[0]
        selection = {
            "selection_rule": selection_rule,
            "match_mode": match_mode,
            "selection_status": "paired",
            "off_contact_token": off_token,
            "on_contact_token": on_selected["_token"],
            "off_selection_score_beta_peak": safe_float(selected.get("beta_peak_amplitude")),
            "off_selection_score_long_burst_fraction": safe_float(selected.get("beta_burst_long_fraction_400ms")),
        }
        pair_rows.append(build_pair_row(subject, side, feature_set, selected, on_selected, selection))
        selection_rows.append(
            {
                "feature_set": feature_set,
                "subject": subject,
                "side": side,
                "off_selected_channel": selected.get("channel", ""),
                "on_selected_channel": on_selected.get("channel", ""),
                **selection,
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(selection_rows)


def prune_pair_features(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Drop feature columns with any missing OFF/ON value."""

    availability = pair_feature_availability(pairs, PHYSIOLOGY_FEATURES)
    usable_features = availability.loc[availability["usable_for_retrieval"].astype(bool), "feature"].astype(str).tolist()
    drop_features = [feature for feature in PHYSIOLOGY_FEATURES if feature not in usable_features]
    drop_cols = [f"{prefix}{feature}" for feature in drop_features for prefix in ["x_", "medon_", "y_"]]
    kept = pairs.drop(columns=[col for col in drop_cols if col in pairs.columns], errors="ignore").copy()
    compact = [feature for feature in PHYSIOLOGY_COMPACT_FEATURES if feature in usable_features]
    return kept, availability, usable_features, compact


def build_all_pair_tables(channel_features: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build all pre-specified v2E feature-set pair tables."""

    pair_tables: dict[str, pd.DataFrame] = {}
    availability_tables = []
    selection_tables = []
    spec_rows = []
    for spec in FEATURE_SET_SPECS:
        name = str(spec["feature_set"])
        if spec["source"] == "median":
            pairs, selection = build_median_pairs(channel_features, name)
        else:
            pairs, selection = build_contact_pairs(
                channel_features,
                name,
                str(spec["selection_rule"]),
                str(spec["match_mode"]),
            )
        if pairs.empty:
            availability = pair_feature_availability(pairs, PHYSIOLOGY_FEATURES)
            usable_features: list[str] = []
            compact: list[str] = []
            pruned = pairs
        else:
            pruned, availability, usable_features, compact = prune_pair_features(pairs)
        pruned["feature_set"] = name
        availability["feature_set"] = name
        selection["feature_set"] = name
        pair_tables[name] = pruned
        availability_tables.append(availability)
        selection_tables.append(selection)
        spec_rows.append(
            {
                **spec,
                "n_pairs": int(len(pruned)),
                "n_subjects": int(pruned["subject"].nunique()) if not pruned.empty else 0,
                "n_usable_features": int(len(usable_features)),
                "n_compact_features": int(len(compact)),
                "usable_features": ";".join(usable_features),
                "compact_features": ";".join(compact),
            }
        )
    return (
        pair_tables,
        pd.concat(selection_tables, ignore_index=True) if selection_tables else pd.DataFrame(),
        pd.concat(availability_tables, ignore_index=True) if availability_tables else pd.DataFrame(),
        pd.DataFrame(spec_rows),
    )


def evaluate_one_feature_set(
    pairs: pd.DataFrame,
    feature_set: str,
    usable_features: list[str],
    compact_features: list[str],
) -> dict[str, pd.DataFrame]:
    """Evaluate v2A/v2B/v2C/v2D for one feature-set pair table."""

    if pairs.empty or not usable_features or not compact_features or pairs["subject"].nunique() < 3:
        empty = pd.DataFrame()
        return {
            "scores": empty,
            "forward_scores": empty,
            "reverse_scores": empty,
            "metrics": empty,
            "diagnostics": empty,
            "components": empty,
            "comparison_to_v2a": empty,
            "comparison_to_v2b": empty,
            "bootstrap_ci": empty,
            "bootstrap_samples": empty,
            "failure_cases": empty,
            "hard_summary": empty,
            "hard_cases": empty,
        }
    clean_features = compact_features
    forward_scores, _candidate_pools = oxf_base.evaluate_forward_scores(
        pairs,
        usable_features,
        compact_features,
        clean_features,
    )
    reverse_scores = oxf_base.evaluate_reverse_v2b(pairs, usable_features, compact_features, clean_features)
    v2c_cycle = cycle_rerank_scores(forward_scores, reverse_scores, V2C_CYCLE_NAME, deconfounded=False)
    v2c_deconfounded = cycle_rerank_scores(forward_scores, reverse_scores, V2C_DECONFOUNDED_NAME, deconfounded=True)
    v2d_tables = []
    component_tables = []
    for variant in V2D_VARIANTS:
        scores, components = rerank_v2d_variant(forward_scores, reverse_scores, variant)
        v2d_tables.append(scores)
        component_tables.append(components)
    scores = pd.concat([forward_scores, v2c_cycle, v2c_deconfounded, *v2d_tables], ignore_index=True)
    scores = rank_scores(normalize_scores(scores))
    components = pd.concat(component_tables, ignore_index=True) if component_tables else pd.DataFrame()
    metrics, diagnostics = build_metrics_tables(scores)
    comparison_v2a = comparison_to_baseline(metrics, oxf_base.V2A_NAME)
    comparison_v2b = comparison_to_baseline(metrics, V2B_NAME)
    bootstrap_ci, bootstrap_samples = subject_bootstrap_ci(diagnostics)
    hard_summary, hard_cases = oxf_base.hard_negative_cases(scores)
    failures = failure_cases(diagnostics)
    tables = {
        "scores": scores,
        "forward_scores": forward_scores,
        "reverse_scores": reverse_scores,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "components": components,
        "comparison_to_v2a": comparison_v2a,
        "comparison_to_v2b": comparison_v2b,
        "bootstrap_ci": bootstrap_ci,
        "bootstrap_samples": bootstrap_samples,
        "failure_cases": failures,
        "hard_summary": hard_summary,
        "hard_cases": hard_cases,
    }
    for table in tables.values():
        if not table.empty:
            table.insert(0, "feature_set", feature_set)
    return tables


def random_label_null_by_feature_set(
    diagnostics: pd.DataFrame,
    scores: pd.DataFrame,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run random-label null separately for each feature set."""

    summary_tables = []
    sample_tables = []
    for feature_set in sorted(scores["feature_set"].astype(str).unique()):
        s = scores[scores["feature_set"].astype(str).eq(feature_set)].drop(columns=["feature_set"], errors="ignore")
        d = diagnostics[diagnostics["feature_set"].astype(str).eq(feature_set)].drop(columns=["feature_set"], errors="ignore")
        summary, samples = oxf_base.random_label_null(d, s, n_perm, seed)
        if not summary.empty:
            summary.insert(0, "feature_set", feature_set)
            summary_tables.append(summary)
        if not samples.empty:
            samples.insert(0, "feature_set", feature_set)
            sample_tables.append(samples)
    return (
        pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame(),
        pd.concat(sample_tables, ignore_index=True) if sample_tables else pd.DataFrame(),
    )


def matched_side_null_by_feature_set(
    diagnostics: pd.DataFrame,
    scores: pd.DataFrame,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run matched-side null separately for each feature set."""

    summary_tables = []
    sample_tables = []
    for feature_set in sorted(scores["feature_set"].astype(str).unique()):
        s = scores[scores["feature_set"].astype(str).eq(feature_set)].drop(columns=["feature_set"], errors="ignore")
        d = diagnostics[diagnostics["feature_set"].astype(str).eq(feature_set)].drop(columns=["feature_set"], errors="ignore")
        summary, samples = oxf_base.matched_side_null(d, s, n_perm, seed)
        if not summary.empty:
            summary.insert(0, "feature_set", feature_set)
            summary_tables.append(summary)
        if not samples.empty:
            samples.insert(0, "feature_set", feature_set)
            sample_tables.append(samples)
    return (
        pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame(),
        pd.concat(sample_tables, ignore_index=True) if sample_tables else pd.DataFrame(),
    )


def loso_by_feature_set(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Run LOSO sensitivity separately for each feature set."""

    tables = []
    for feature_set in sorted(diagnostics["feature_set"].astype(str).unique()):
        d = diagnostics[diagnostics["feature_set"].astype(str).eq(feature_set)].drop(columns=["feature_set"], errors="ignore")
        loso = oxf_base.loso_sensitivity(d)
        if not loso.empty:
            loso.insert(0, "feature_set", feature_set)
            tables.append(loso)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def summarize_best(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize all-MedOn performance by feature set and variant."""

    focus_variants = [
        oxf_base.V2A_NAME,
        V2B_NAME,
        V2C_DECONFOUNDED_NAME,
        V2D_QUALITY_DECONFOUNDED_NAME,
    ]
    return (
        metrics[
            metrics["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
            & metrics["variant_name"].astype(str).isin(focus_variants)
        ]
        .sort_values(["top1", "mrr"], ascending=False)
        .reset_index(drop=True)
    )


def plot_top1(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Plot all-MedOn top1 by feature set and variant."""

    focus = summarize_best(metrics)
    if focus.empty:
        return
    pivot = focus.pivot_table(index="feature_set", columns="variant_name", values="top1", aggfunc="first")
    fig, ax = plt.subplots(figsize=(13, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.45, color="gray", linestyle="--", linewidth=1, label="0.45 reference")
    ax.axhline(0.55, color="black", linestyle=":", linewidth=1, label="0.55 target")
    ax.set_ylabel("All-MedOn top1")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "figures" / "v2e_all_medon_top1_by_feature_set.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(summary: dict[str, object], path: Path) -> Path:
    """Write Markdown report."""

    lines = [
        "# OXF STN Physiology-Locked v2E",
        "",
        "Scope: pre-specified physiological feature extraction variants on OXF STN-only ON/OFF data.",
        "",
        "## Validation",
    ]
    for key, value in summary["validation"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Best All-MedOn Rows"])
    for row in summary["best_all_medon_rows"][:12]:
        lines.append(
            f"- {row['feature_set']} / {row['variant_name']}: "
            f"top1={float(row['top1']):.3f}, MRR={float(row['mrr']):.3f}, "
            f"pairs={int(row['n_pairs'])}, failures={int(row['failures'])}"
        )
    lines.extend(["", "## Warnings"])
    for warning in summary["warnings"]:
        lines.append(f"- {warning}")
    return write_text(lines, path)


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run the v2E OXF physiology-locked analysis."""

    if TOP_K != 5:
        raise AssertionError(f"top_k changed: {TOP_K}")
    if not math.isclose(float(APERIODIC_ALPHA), 0.5):
        raise AssertionError(f"aperiodic_alpha changed: {APERIODIC_ALPHA}")

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    base_output_dir = Path(args.base_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_root.exists():
        raise FileNotFoundError(f"OXF data root not found: {data_root}")

    channel_features, state_features = load_or_build_channel_features(data_root, base_output_dir, float(args.max_duration_sec))
    inventory = oxf_base.discover_records(data_root)
    bursts = load_or_build_bursts(data_root, output_dir, float(args.max_duration_sec))
    physiology_channels = merge_channel_features(channel_features, bursts)
    pair_tables, selection_table, availability_table, feature_set_summary = build_all_pair_tables(physiology_channels)

    evaluation_tables: dict[str, list[pd.DataFrame]] = {
        "scores": [],
        "forward_scores": [],
        "reverse_scores": [],
        "metrics": [],
        "diagnostics": [],
        "components": [],
        "comparison_to_v2a": [],
        "comparison_to_v2b": [],
        "bootstrap_ci": [],
        "bootstrap_samples": [],
        "failure_cases": [],
        "hard_summary": [],
        "hard_cases": [],
    }
    pruned_pair_tables = []
    for _, row in feature_set_summary.iterrows():
        feature_set = str(row["feature_set"])
        pairs = pair_tables[feature_set]
        usable_features = [value for value in str(row["usable_features"]).split(";") if value]
        compact_features = [value for value in str(row["compact_features"]).split(";") if value]
        if not pairs.empty:
            pruned_pair_tables.append(pairs)
        tables = evaluate_one_feature_set(pairs, feature_set, usable_features, compact_features)
        for key, table in tables.items():
            if not table.empty:
                evaluation_tables[key].append(table)

    concat = {
        key: pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        for key, tables in evaluation_tables.items()
    }
    all_pairs = pd.concat(pruned_pair_tables, ignore_index=True) if pruned_pair_tables else pd.DataFrame()
    metrics = concat["metrics"]
    diagnostics = concat["diagnostics"]
    scores = concat["scores"]
    random_summary, random_samples = random_label_null_by_feature_set(diagnostics, scores, int(args.n_perm), int(args.seed))
    matched_summary, matched_samples = matched_side_null_by_feature_set(diagnostics, scores, int(args.n_perm), int(args.seed))
    loso = loso_by_feature_set(diagnostics)
    best = summarize_best(metrics)

    validation = {
        "data_root": str(data_root),
        "discovered_mat_files": int(len(inventory)),
        "state_feature_rows": int(len(state_features)),
        "channel_feature_rows": int(len(channel_features)),
        "burst_feature_rows": int(len(bursts)),
        "feature_sets_evaluated": int(len(feature_set_summary)),
        "top_k": TOP_K,
        "aperiodic_alpha": APERIODIC_ALPHA,
        "n_perm": int(args.n_perm),
        "n_boot": int(N_BOOT),
        "seed": int(args.seed),
    }
    warnings = [
        "v2E is a separate physiology-locked OXF STN-only analysis; frozen ds004998 v2A/v2B/v2C/v2D outputs are unchanged.",
        "OFF-contact selection uses only OFF physiology, not retrieval outcome.",
        "Exact contact matching is primary; numeric contact matching is reported as secondary/exploratory because montage labels can be ambiguous.",
        "Beta-burst features use a fixed 13-35 Hz band, 75th percentile envelope threshold, and 100 ms minimum duration.",
        "OXF remains Rest/STN-only and cannot validate the full ds004998 MEG+STN Hold/Move pipeline.",
        "No clinical prediction, treatment, DBS optimization, stimulation planning, or causal medication-effect claim is made.",
    ]
    primary_exact = feature_set_summary[
        feature_set_summary["feature_set"].astype(str).eq("off_beta_peak_contact_exact")
    ]
    if not primary_exact.empty and int(primary_exact.iloc[0]["n_pairs"]) < 30:
        warnings.append("Primary exact-contact v2E drops pairs without exact ON contact match; numeric matching is secondary.")

    summary = {
        "validation": validation,
        "feature_set_summary": feature_set_summary.to_dict("records"),
        "best_all_medon_rows": best.head(20).to_dict("records") if not best.empty else [],
        "warnings": warnings,
        "claim_boundary": {
            "physiology_locked_feature_extraction": True,
            "parameter_tuning": False,
            "full_meg_stn_validation": False,
            "clinical_prediction_or_treatment_claim": False,
        },
    }

    paths = {
        "summary_json": write_json(summary, output_dir / "v2e_summary.json"),
        "summary_md": write_report(summary, output_dir / "v2e_summary.md"),
        "feature_set_summary": write_csv(feature_set_summary, output_dir / "v2e_feature_set_summary.csv"),
        "channel_burst_features": write_csv(bursts, output_dir / "v2e_channel_burst_features.csv"),
        "physiology_channel_features": write_csv(physiology_channels, output_dir / "v2e_physiology_channel_features.csv"),
        "contact_selection": write_csv(selection_table, output_dir / "v2e_contact_selection.csv"),
        "feature_availability": write_csv(availability_table, output_dir / "v2e_feature_availability.csv"),
        "pairs": write_csv(all_pairs, output_dir / "v2e_pairs.csv"),
        "observed_metrics": write_csv(metrics, output_dir / "v2e_observed_metrics.csv"),
        "candidate_scores": write_csv(scores, output_dir / "v2e_candidate_scores.csv"),
        "query_diagnostics": write_csv(diagnostics, output_dir / "v2e_query_diagnostics.csv"),
        "comparison_to_v2a": write_csv(concat["comparison_to_v2a"], output_dir / "v2e_comparison_to_v2a.csv"),
        "comparison_to_v2b": write_csv(concat["comparison_to_v2b"], output_dir / "v2e_comparison_to_v2b.csv"),
        "bootstrap_ci": write_csv(concat["bootstrap_ci"], output_dir / "v2e_subject_bootstrap_ci.csv"),
        "bootstrap_samples": write_csv(concat["bootstrap_samples"], output_dir / "v2e_subject_bootstrap_samples.csv"),
        "random_label_null_summary": write_csv(random_summary, output_dir / "v2e_random_label_null_summary.csv"),
        "random_label_null_samples": write_csv(random_samples, output_dir / "v2e_random_label_null_samples.csv"),
        "matched_side_null_summary": write_csv(matched_summary, output_dir / "v2e_matched_side_null_summary.csv"),
        "matched_side_null_samples": write_csv(matched_samples, output_dir / "v2e_matched_side_null_samples.csv"),
        "loso_sensitivity": write_csv(loso, output_dir / "v2e_loso_sensitivity.csv"),
        "hard_negative_summary": write_csv(concat["hard_summary"], output_dir / "v2e_same_subject_hard_negative_summary.csv"),
        "hard_negative_cases": write_csv(concat["hard_cases"], output_dir / "v2e_same_subject_hard_negative_cases.csv"),
        "failure_cases": write_csv(concat["failure_cases"], output_dir / "v2e_failure_cases.csv"),
        "v2d_components": write_csv(concat["components"], output_dir / "v2e_v2d_components.csv"),
    }
    plot_top1(metrics, output_dir)

    print(f"data_root: {data_root}")
    print(f"feature_sets_evaluated: {len(feature_set_summary)}")
    print(f"top_k: {TOP_K}")
    print(f"aperiodic_alpha: {APERIODIC_ALPHA}")
    print("feature set summary:")
    for _, row in feature_set_summary.iterrows():
        print(
            f"- {row['feature_set']}: pairs={int(row['n_pairs'])}, "
            f"subjects={int(row['n_subjects'])}, usable_features={int(row['n_usable_features'])}, "
            f"compact_features={int(row['n_compact_features'])}"
        )
    print("best all-MedOn rows:")
    for _, row in best.head(12).iterrows():
        print(
            f"- {row['feature_set']} / {row['variant_name']}: "
            f"top1={float(row['top1']):.3f}, MRR={float(row['mrr']):.3f}, "
            f"pairs={int(row['n_pairs'])}, failures={int(row['failures'])}"
        )
    print(f"output folder: {output_dir}")
    print("warnings:")
    for warning in warnings:
        print(f"- {warning}")
    return {"paths": {key: str(path) for key, path in paths.items()}, "summary": summary}


def main() -> None:
    """Entry point."""

    run(parse_args())


if __name__ == "__main__":
    main()
