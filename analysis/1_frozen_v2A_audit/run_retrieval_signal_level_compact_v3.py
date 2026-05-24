"""Signal-level compact V3 retrieval feature-improvement phase.

This script tests whether compact, predefined signal-level STN burst and
PSD-derived aperiodic/residual features improve the current v2 paired-state
retrieval scorer. It does not download data, does not introduce deep learning,
does not perform exhaustive feature search, and does not make clinical, DBS,
treatment, stimulation-planning, or MedOn-as-healthy claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt, welch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.load_ds004998_bids import (  # noqa: E402
    load_lfp_montage_mapping,
    parse_bids_entities,
    read_raw_header,
)
from src.preprocessing.channel_selection import (  # noqa: E402
    meg_picks,
    motor_cortical_proxy_picks,
    stn_lfp_picks,
    stn_lfp_picks_by_side,
)
from src.preprocessing.spectral_features import BANDS  # noqa: E402
from src.evaluation.predictive_rescue_analysis import (  # noqa: E402
    build_base_candidate_sets,
    read_quality_table,
    read_state_vectors,
    read_subspace_definitions,
    transform,
)
from scripts.run_retrieval_distance_geometry_improvement import (  # noqa: E402
    VariantSpec,
    cosine_distance,
    evaluate_variant as evaluate_v2_variant,
    ordered_unique,
    pair_value,
    query_change_log_for_variant,
    query_change_summary,
    query_diagnostics_from_scores,
    random_label_negative_control,
    same_subject_hard_negative as same_subject_hard_negative_v2,
    summarize_diagnostics,
    to_builtin,
    write_csv,
    write_text,
)


STATE_VECTORS_PATH = Path("outputs/tables/ds004998_state_vectors_enhanced_coupling_expanded.csv")
WINDOW_FEATURES_PATH = Path("outputs/tables/ds004998_window_features_enhanced_coupling_expanded.csv")
QUALITY_TABLE_PATH = Path("outputs/tables/real_recording_quality.csv")
SUBSPACE_DEFINITIONS_PATH = Path("outputs/tables/subspace_definitions.csv")
PROXY_V3_RESULTS_PATH = Path("outputs/retrieval_v3_feature_improvement/v3_scorer_results.csv")
OUTPUT_DIR = Path("outputs/retrieval_signal_level_compact_v3")
FIGURES_DIR = OUTPUT_DIR / "figures"

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
N_RANDOM_LABEL_PERMUTATIONS = 1000
V2_SCORER_NAME = "group_balanced_cosine_full28_plus_coupling"
MAX_COMPACT_BURST_FEATURES = 12
MAX_COMPACT_APERIODIC_FEATURES = 12
DO_NOT_DOWNLOAD_DATA = True
MIN_DISTRACTORS = 2
MAX_SIGNAL_DURATION_SEC = 120.0
MAX_MEG_CHANNELS_FOR_PSD = 64
MRR_TOLERANCE = 0.01

BURST_BANDS = {
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 35.0),
    "broad_beta": (13.0, 35.0),
    "gamma": BANDS.get("gamma", (35.0, 70.0)),
}
BETA_MIN_DURATION_SEC = 0.100
GAMMA_MIN_DURATION_SEC = 0.050
BURST_THRESHOLD_PERCENTILE = 75.0
PSD_FMIN = 3.0
PSD_FMAX = 80.0
LINE_NOISE_EXCLUDE = (48.0, 52.0)
SPECTRAL_BANDS = {
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 35.0),
    "broad_beta": (13.0, 35.0),
    "gamma": BANDS.get("gamma", (35.0, 70.0)),
}

COMPACT_BURST_FEATURES = [
    "stn_low_beta_burst_occupancy_fraction",
    "stn_high_beta_burst_occupancy_fraction",
    "stn_broad_beta_burst_occupancy_fraction",
    "stn_low_beta_mean_burst_duration",
    "stn_high_beta_mean_burst_duration",
    "stn_broad_beta_long_burst_fraction",
    "stn_low_beta_burst_rate",
    "stn_high_beta_burst_rate",
    "stn_low_beta_high_beta_burst_ratio",
    "stn_broad_beta_left_right_asymmetry",
    "task_side_stn_broad_beta_burst_occupancy_fraction",
    "opposite_side_stn_broad_beta_burst_occupancy_fraction",
][:MAX_COMPACT_BURST_FEATURES]

COMPACT_APERIODIC_FEATURES = [
    "stn_aperiodic_slope",
    "stn_aperiodic_offset",
    "stn_low_beta_residual_power",
    "stn_high_beta_residual_power",
    "stn_broad_beta_residual_power",
    "stn_beta_peak_amplitude",
    "stn_beta_peak_frequency",
    "motor_aperiodic_slope",
    "motor_low_beta_residual_power",
    "motor_high_beta_residual_power",
    "meg_aperiodic_slope",
    "meg_gamma_residual_power",
][:MAX_COMPACT_APERIODIC_FEATURES]


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class CompactScorerSpec:
    """Predefined compact V3 scorer."""

    name: str
    feature_key: str
    diagnostic_only: bool = False
    note: str = ""


def write_json(data: dict[str, object], path: Path) -> Path:
    """Write JSON with parent directory creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(data), indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_csv_or_empty(path: str | Path, warnings: list[str]) -> pd.DataFrame:
    """Read a CSV table or return an empty frame."""

    path = Path(path)
    if not path.exists():
        warnings.append(f"Missing input: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"run": str})
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Could not read {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def task_side_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure task_original/task_family/side columns."""

    out = data.copy()
    if "task_original" not in out:
        source = out["task"] if "task" in out else out["condition"] if "condition" in out else pd.Series("unknown", index=out.index)
        out["task_original"] = source.fillna("unknown").astype(str)
    if "task_family" not in out:
        task = out["task_original"].fillna("unknown").astype(str)
        out["task_family"] = np.where(task.str.lower().str.startswith("hold"), "Hold", np.where(task.str.lower().str.startswith("move"), "Move", task))
    if "side" not in out:
        task = out["task_original"].fillna("").astype(str)
        out["side"] = np.where(task.str.endswith("L"), "L", np.where(task.str.endswith("R"), "R", "none"))
    return out


def numeric_feature_columns(state_vectors: pd.DataFrame) -> list[str]:
    """Return numeric feature columns for compact state-vector pairing."""

    metadata = {
        "subject",
        "session",
        "condition",
        "medication",
        "task",
        "run",
        "task_original",
        "task_family",
        "side",
        "n_windows",
        "source",
        "methodological_label",
        "quality_flag",
    }
    return [
        column
        for column in state_vectors.columns
        if column not in metadata and pd.api.types.is_numeric_dtype(state_vectors[column])
    ]


QUALITY_SEVERITY = {"good": 0, "acceptable": 1, "caution": 2, "low_quality": 3, "unknown": 4}


def worst_quality(values: Iterable[object]) -> str:
    """Return worst quality flag."""

    flags = [str(value) for value in values if str(value) and str(value).lower() != "nan"]
    if not flags:
        return "unknown"
    return sorted(flags, key=lambda flag: QUALITY_SEVERITY.get(flag, 4), reverse=True)[0]


def attach_pair_quality(pairs: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Attach subject/task quality flags."""

    out = pairs.copy()
    if out.empty:
        return out
    if quality.empty:
        out["quality_flag"] = "unknown"
        out["quality_match_source"] = "missing_quality_table"
        return out
    q = task_side_columns(quality)
    for column in ["subject", "task_original", "quality_flag"]:
        if column not in q:
            q[column] = "unknown"
        q[column] = q[column].fillna("unknown").astype(str)
    lookup = q.groupby(["subject", "task_original"], dropna=False)["quality_flag"].agg(worst_quality).to_dict()
    out["quality_flag"] = [lookup.get((str(row.subject), str(row.task_original)), "unknown") for row in out.itertuples()]
    out["quality_match_source"] = "subject_task_worst_recording_quality"
    return out


def build_paired_examples_all_features(state_vectors: pd.DataFrame, quality: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build MedOff/MedOn pairs using all numeric compact state-vector features."""

    features = numeric_feature_columns(state_vectors)
    rows: list[dict[str, object]] = []
    logs: list[dict[str, object]] = []
    for (subject, task_original), subset in state_vectors.groupby(["subject", "task_original"], dropna=False):
        meds = subset["medication"].fillna("").astype(str).str.lower()
        off = subset[meds.eq("off")]
        on = subset[meds.eq("on")]
        status = "paired" if not off.empty and not on.empty else "missing_medoff_or_medon"
        logs.append({"subject": subject, "task_original": task_original, "n_medoff_rows": len(off), "n_medon_rows": len(on), "status": status})
        if off.empty or on.empty:
            continue
        off_values = off[features].mean(numeric_only=True)
        on_values = on[features].mean(numeric_only=True)
        first = subset.iloc[0]
        record: dict[str, object] = {
            "pair_id": f"{subject}|{task_original}",
            "subject": str(subject),
            "session": str(first.get("session", "unknown")),
            "task_original": str(task_original),
            "task_family": str(first.get("task_family", "unknown")),
            "side": str(first.get("side", "unknown")),
            "run_off": ";".join(off.get("run", pd.Series("unknown", index=off.index)).dropna().astype(str).unique()),
            "run_on": ";".join(on.get("run", pd.Series("unknown", index=on.index)).dropna().astype(str).unique()),
        }
        for feature in features:
            record[f"x_{feature}"] = float(off_values[feature])
            record[f"medon_{feature}"] = float(on_values[feature])
            record[f"y_{feature}"] = float(on_values[feature] - off_values[feature])
        rows.append(record)
    return attach_pair_quality(pd.DataFrame(rows), quality), pd.DataFrame(logs)


def unique_recording_rows(window_features: pd.DataFrame) -> pd.DataFrame:
    """Return one row per locally processed recording."""

    if window_features.empty or "file_path" not in window_features:
        return pd.DataFrame()
    columns = [column for column in ["file_path", "subject", "session", "task", "run", "medication", "condition"] if column in window_features.columns]
    return window_features[columns].drop_duplicates().reset_index(drop=True)


def raw_signal_availability(recordings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Inspect local FIF availability and required channel groups."""

    rows: list[dict[str, object]] = []
    for _, rec in recordings.iterrows():
        path = Path(str(rec["file_path"]))
        base = rec.to_dict()
        base["file_path"] = str(path)
        base["local_file_available"] = bool(path.exists())
        if not path.exists():
            rows.append({**base, "usable_for_signal_level": False, "qc_status": "missing_local_file"})
            continue
        try:
            raw = read_raw_header(path)
            mapping = load_lfp_montage_mapping(path)
            stn = stn_lfp_picks(raw, lfp_mapping=mapping)
            motor = motor_cortical_proxy_picks(raw)
            meg = meg_picks(raw)
            rows.append(
                {
                    **base,
                    "sampling_frequency_hz": float(raw.info["sfreq"]),
                    "recording_duration_sec": float(raw.n_times / raw.info["sfreq"]),
                    "n_channels": int(len(raw.ch_names)),
                    "n_stn_lfp_channels": int(len(stn)),
                    "n_motor_proxy_channels": int(len(motor)),
                    "n_meg_channels": int(len(meg)),
                    "stn_lfp_available": bool(stn),
                    "motor_available": bool(motor),
                    "meg_available": bool(meg),
                    "lfp_mapping_channels": int(len(mapping)),
                    "usable_for_signal_level": bool(stn and raw.info["sfreq"] > 0),
                    "qc_status": "ok" if stn and raw.info["sfreq"] > 0 else "missing_stn_or_sampling_rate",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({**base, "usable_for_signal_level": False, "qc_status": f"read_error:{type(exc).__name__}", "error_message": str(exc)})
    table = pd.DataFrame(rows)
    summary = {
        "n_recordings": int(len(table)),
        "n_local_files": int(table.get("local_file_available", pd.Series(dtype=bool)).sum()) if not table.empty else 0,
        "n_usable_for_signal_level": int(table.get("usable_for_signal_level", pd.Series(dtype=bool)).sum()) if not table.empty else 0,
        "do_not_download_data": DO_NOT_DOWNLOAD_DATA,
        "raw_signal_available": bool(not table.empty and table.get("usable_for_signal_level", pd.Series(dtype=bool)).any()),
    }
    return table, summary


def signal_segment(raw, picks: list[int], max_duration_sec: float = MAX_SIGNAL_DURATION_SEC) -> np.ndarray:
    """Load a compact segment for selected picks."""

    if not picks:
        return np.empty((0, 0))
    sfreq = float(raw.info["sfreq"])
    stop = min(raw.n_times, int(max_duration_sec * sfreq))
    if stop <= 8:
        return np.empty((0, 0))
    return raw.get_data(picks=picks, start=0, stop=stop, reject_by_annotation="omit")


def find_burst_runs(mask: np.ndarray, sfreq: float, min_duration_sec: float) -> list[tuple[int, int]]:
    """Find contiguous burst runs satisfying a minimum duration."""

    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    min_samples = max(1, int(round(min_duration_sec * sfreq)))
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=False) if end - start >= min_samples]


def burst_stats_for_group(
    data: np.ndarray,
    sfreq: float,
    source: str,
    band_name: str,
    band: tuple[float, float],
    recording_id: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Compute signal-level burst events and summary stats for one source/band."""

    if data.size == 0 or sfreq <= 0:
        return [], {}
    high = min(float(band[1]), sfreq / 2.0 - 1.0)
    low = float(band[0])
    if high <= low:
        return [], {}
    try:
        sos = butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
        filtered = sosfiltfilt(sos, data, axis=-1)
    except Exception:  # noqa: BLE001
        return [], {}
    envelope = np.abs(hilbert(filtered, axis=-1))
    min_duration = GAMMA_MIN_DURATION_SEC if band_name == "gamma" else BETA_MIN_DURATION_SEC
    total_sec = data.shape[-1] / sfreq
    event_rows: list[dict[str, object]] = []
    channel_stats = []
    for channel_idx in range(envelope.shape[0]):
        env = envelope[channel_idx]
        threshold = float(np.nanpercentile(env, BURST_THRESHOLD_PERCENTILE))
        median = float(np.nanmedian(env))
        mad = float(np.nanmedian(np.abs(env - median))) or 1.0
        norm_env = (env - median) / (1.4826 * mad + 1e-8)
        runs = find_burst_runs(env >= threshold, sfreq, min_duration)
        durations = [(end - start) / sfreq for start, end in runs]
        amplitudes = [float(np.nanmean(norm_env[start:end])) for start, end in runs]
        occupancy = float(np.sum(durations) / max(total_sec, 1e-8))
        long_threshold = 0.300 if band_name != "gamma" else 0.150
        channel_stats.append(
            {
                "rate": len(runs) / max(total_sec / 60.0, 1e-8),
                "occupancy": occupancy,
                "mean_duration": float(np.nanmean(durations)) if durations else 0.0,
                "median_duration": float(np.nanmedian(durations)) if durations else 0.0,
                "max_duration": float(np.nanmax(durations)) if durations else 0.0,
                "mean_amplitude": float(np.nanmean(amplitudes)) if amplitudes else 0.0,
                "long_fraction": float(np.mean(np.asarray(durations) >= long_threshold)) if durations else 0.0,
            }
        )
        for start, end in runs:
            event_rows.append(
                {
                    **recording_id,
                    "source": source,
                    "band": band_name,
                    "channel_index_within_source": channel_idx,
                    "start_sec": float(start / sfreq),
                    "end_sec": float(end / sfreq),
                    "duration_sec": float((end - start) / sfreq),
                    "mean_normalized_amplitude": float(np.nanmean(norm_env[start:end])),
                    "threshold_percentile": BURST_THRESHOLD_PERCENTILE,
                    "min_duration_sec": min_duration,
                }
            )
    stats = pd.DataFrame(channel_stats)
    if stats.empty:
        return event_rows, {}
    prefix = f"{source}_{band_name}"
    summary = {
        f"{prefix}_burst_rate": float(stats["rate"].mean()),
        f"{prefix}_burst_occupancy_fraction": float(stats["occupancy"].mean()),
        f"{prefix}_mean_burst_duration": float(stats["mean_duration"].mean()),
        f"{prefix}_median_burst_duration": float(stats["median_duration"].mean()),
        f"{prefix}_max_burst_duration": float(stats["max_duration"].mean()),
        f"{prefix}_mean_burst_amplitude": float(stats["mean_amplitude"].mean()),
        f"{prefix}_long_burst_fraction": float(stats["long_fraction"].mean()),
    }
    return event_rows, summary


def spectral_params_from_data(data: np.ndarray, sfreq: float, source: str) -> dict[str, float]:
    """Compute PSD-level aperiodic fallback and residual-band features."""

    if data.size == 0 or sfreq <= 0:
        return {}
    n_times = data.shape[-1]
    nperseg = min(n_times, int(max(sfreq * 2.0, 64)))
    if nperseg < 16:
        return {}
    freqs, psd = welch(np.asarray(data, dtype=np.float32), fs=sfreq, axis=-1, nperseg=nperseg)
    mean_psd = np.nanmean(psd, axis=0)
    mask = (freqs >= PSD_FMIN) & (freqs <= min(PSD_FMAX, sfreq / 2.0 - 1.0))
    mask &= ~((freqs >= LINE_NOISE_EXCLUDE[0]) & (freqs <= LINE_NOISE_EXCLUDE[1]))
    if mask.sum() < 5:
        return {}
    x = np.log10(freqs[mask])
    y = np.log10(np.maximum(mean_psd[mask], 1e-300))
    slope, offset = np.polyfit(x, y, deg=1)
    predicted = offset + slope * np.log10(np.maximum(freqs, 1e-12))
    residual = np.log10(np.maximum(mean_psd, 1e-300)) - predicted
    out = {
        f"{source}_aperiodic_slope": float(slope),
        f"{source}_aperiodic_offset": float(offset),
        f"{source}_aperiodic_exponent_or_slope": float(slope),
    }
    for band_name, (low, high) in SPECTRAL_BANDS.items():
        high = min(high, sfreq / 2.0 - 1.0)
        band_mask = (freqs >= low) & (freqs <= high)
        band_mask &= ~((freqs >= LINE_NOISE_EXCLUDE[0]) & (freqs <= LINE_NOISE_EXCLUDE[1]))
        out[f"{source}_{band_name}_residual_power"] = float(np.nanmean(residual[band_mask])) if band_mask.any() else np.nan
    beta_mask = (freqs >= 13.0) & (freqs <= min(35.0, sfreq / 2.0 - 1.0))
    if beta_mask.any():
        idx = np.where(beta_mask)[0][int(np.nanargmax(residual[beta_mask]))]
        out[f"{source}_beta_peak_frequency"] = float(freqs[idx])
        out[f"{source}_beta_peak_amplitude"] = float(residual[idx])
        out[f"{source}_beta_peak_bandwidth"] = float(freqs[1] - freqs[0]) if len(freqs) > 1 else np.nan
    gamma_mask = (freqs >= SPECTRAL_BANDS["gamma"][0]) & (freqs <= min(SPECTRAL_BANDS["gamma"][1], sfreq / 2.0 - 1.0))
    gamma_mask &= ~((freqs >= LINE_NOISE_EXCLUDE[0]) & (freqs <= LINE_NOISE_EXCLUDE[1]))
    if gamma_mask.any():
        out[f"{source}_gamma_peak_amplitude"] = float(np.nanmax(residual[gamma_mask]))
    return out


def extract_signal_level_features(availability: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame, dict[str, object]]:
    """Extract compact signal-level burst and PSD features from local FIF files."""

    usable = availability[availability.get("usable_for_signal_level", False).astype(bool)].copy() if not availability.empty else pd.DataFrame()
    if usable.empty:
        empty_inventory = {"available": False, "warning": "No usable local raw signals; signal-level extraction skipped."}
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), empty_inventory, pd.DataFrame(), empty_inventory

    burst_events: list[dict[str, object]] = []
    burst_state_rows: list[dict[str, object]] = []
    spectral_rows: list[dict[str, object]] = []
    for _, rec in usable.iterrows():
        path = Path(str(rec["file_path"]))
        try:
            raw = read_raw_header(path)
            mapping = load_lfp_montage_mapping(path)
            sfreq = float(raw.info["sfreq"])
            stn_picks_all = stn_lfp_picks(raw, lfp_mapping=mapping)
            side_picks = stn_lfp_picks_by_side(raw, lfp_mapping=mapping)
            motor_picks = motor_cortical_proxy_picks(raw)
            meg_pick_list = meg_picks(raw)[:MAX_MEG_CHANNELS_FOR_PSD]
            stn_data = signal_segment(raw, stn_picks_all)
            left_data = signal_segment(raw, side_picks.get("left", []))
            right_data = signal_segment(raw, side_picks.get("right", []))
            motor_data = signal_segment(raw, motor_picks)
            meg_data = signal_segment(raw, meg_pick_list)
            base = {key: rec.get(key, "") for key in ["subject", "session", "task", "medication", "run", "condition", "file_path"]}
            base["task_original"] = rec.get("task", rec.get("condition", "unknown"))
            task_text = str(base["task_original"])
            base["task_family"] = "Hold" if task_text.startswith("Hold") else "Move" if task_text.startswith("Move") else task_text
            base["side"] = "L" if task_text.endswith("L") else "R" if task_text.endswith("R") else "none"
            burst_record = dict(base)
            for source, data in [("stn", stn_data), ("stn_left", left_data), ("stn_right", right_data)]:
                for band_name, band in BURST_BANDS.items():
                    events, stats = burst_stats_for_group(data, sfreq, source, band_name, band, base)
                    burst_events.extend(events)
                    burst_record.update(stats)
            for band in BURST_BANDS:
                low_name = f"stn_left_{band}_burst_occupancy_fraction"
                right_name = f"stn_right_{band}_burst_occupancy_fraction"
                left = burst_record.get(low_name, np.nan)
                right = burst_record.get(right_name, np.nan)
                if np.isfinite(left) and np.isfinite(right):
                    burst_record[f"stn_{band}_left_right_asymmetry"] = float((left - right) / (abs(left) + abs(right) + 1e-8))
            low = burst_record.get("stn_low_beta_burst_occupancy_fraction", np.nan)
            high = burst_record.get("stn_high_beta_burst_occupancy_fraction", np.nan)
            if np.isfinite(low) and np.isfinite(high):
                burst_record["stn_low_beta_high_beta_burst_ratio"] = float(low / (high + 1e-8))
            for band in BURST_BANDS:
                left = burst_record.get(f"stn_left_{band}_burst_occupancy_fraction", np.nan)
                right = burst_record.get(f"stn_right_{band}_burst_occupancy_fraction", np.nan)
                task_side = left if base["side"] == "L" else right if base["side"] == "R" else np.nan
                opposite_side = right if base["side"] == "L" else left if base["side"] == "R" else np.nan
                burst_record[f"task_side_stn_{band}_burst_occupancy_fraction"] = float(task_side) if np.isfinite(task_side) else np.nan
                burst_record[f"opposite_side_stn_{band}_burst_occupancy_fraction"] = float(opposite_side) if np.isfinite(opposite_side) else np.nan
            # Compact aliases requested by the V3 spec.
            for band in ["low_beta", "high_beta", "broad_beta"]:
                if f"stn_{band}_mean_burst_duration" in burst_record:
                    burst_record[f"stn_{band}_mean_burst_duration"] = burst_record[f"stn_{band}_mean_burst_duration"]
            burst_state_rows.append(burst_record)

            spectral_record = dict(base)
            for source, data in [("stn", stn_data), ("stn_left", left_data), ("stn_right", right_data), ("motor", motor_data), ("meg", meg_data)]:
                spectral_record.update(spectral_params_from_data(data, sfreq, source))
            spectral_rows.append(spectral_record)
        except Exception as exc:  # noqa: BLE001
            burst_state_rows.append({**rec.to_dict(), "extraction_error": f"{type(exc).__name__}: {exc}"})

    burst_state = pd.DataFrame(burst_state_rows)
    spectral_state = pd.DataFrame(spectral_rows)
    burst_feature_cols = [column for column in burst_state.columns if "burst" in column]
    spectral_feature_cols = [
        column
        for column in spectral_state.columns
        if "aperiodic" in column or "residual_power" in column or "peak_" in column
    ]
    burst_inventory = {
        "available": bool(burst_feature_cols),
        "method": "signal_level_bandpass_hilbert_envelope",
        "threshold": f"{BURST_THRESHOLD_PERCENTILE}th percentile per recording/channel/band",
        "beta_min_duration_sec": BETA_MIN_DURATION_SEC,
        "gamma_min_duration_sec": GAMMA_MIN_DURATION_SEC,
        "n_state_rows": int(len(burst_state)),
        "n_event_rows": int(len(burst_events)),
        "n_features_full": int(len(burst_feature_cols)),
        "features_full": burst_feature_cols,
    }
    spectral_inventory = {
        "available": bool(spectral_feature_cols),
        "method": "welch_psd_log_log_aperiodic_fit_fallback",
        "uses_specparam_or_fooof": False,
        "n_state_rows": int(len(spectral_state)),
        "n_features_full": int(len(spectral_feature_cols)),
        "features_full": spectral_feature_cols,
        "meg_sensor_summary_max_channels": MAX_MEG_CHANNELS_FOR_PSD,
    }
    return pd.DataFrame(burst_events), burst_state, burst_state, burst_inventory, spectral_state, spectral_inventory


def compact_feature_table(full: pd.DataFrame, compact_features: list[str], id_cols: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Extract compact predefined feature table and inventory."""

    if full.empty:
        return pd.DataFrame(), {"available": False, "selected_features": [], "missing_features": compact_features}
    selected = [feature for feature in compact_features if feature in full.columns]
    missing = [feature for feature in compact_features if feature not in full.columns]
    columns = [column for column in id_cols if column in full.columns]
    table = full[[*columns, *selected]].copy() if selected else full[columns].copy()
    return table, {
        "available": bool(selected),
        "selected_features": selected,
        "missing_features": missing,
        "n_selected_features": int(len(selected)),
        "max_allowed_features": int(len(compact_features)),
    }


def merge_compact_features(state_vectors: pd.DataFrame, compact_tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge compact feature tables into the v2 state-vector table."""

    merged = task_side_columns(state_vectors)
    for table in compact_tables:
        if table.empty:
            continue
        keys = [key for key in ["subject", "session", "task", "medication", "run"] if key in merged.columns and key in table.columns]
        if not keys:
            keys = [key for key in ["subject", "task_original", "medication"] if key in merged.columns and key in table.columns]
        feature_cols = [column for column in table.columns if column not in keys and column not in {"task_original", "task_family", "side", "condition", "file_path"}]
        before = set(merged.columns)
        merged = merged.merge(table[[*keys, *feature_cols]], on=keys, how="left")
        for column in [column for column in merged.columns if column not in before]:
            if pd.api.types.is_numeric_dtype(merged[column]):
                merged[column] = merged[column].fillna(0.0)
    return merged


def build_compact_subspaces(existing: dict[str, list[str]], state_vectors: pd.DataFrame, burst_features: list[str], aperiodic_features: list[str]) -> dict[str, list[str]]:
    """Build compact V3 subspace map."""

    available = set(numeric_feature_columns(state_vectors))
    full28 = [feature for feature in existing.get("full_28", []) if feature in available]
    coupling = [
        feature
        for feature in available
        if "coupling" in feature.lower() or "cortico_stn" in feature.lower()
    ]
    burst = [feature for feature in burst_features if feature in available]
    aperiodic_bg = [feature for feature in aperiodic_features if feature in available and "aperiodic" in feature]
    residual = [feature for feature in aperiodic_features if feature in available and "aperiodic" not in feature]
    raw_beta_gamma = [
        feature
        for feature in full28
        if (feature.startswith("meg_") or feature.startswith("motor_") or feature.startswith("stn_"))
        and ("beta" in feature.lower() or "gamma" in feature.lower())
        and "power" in feature.lower()
    ]
    return {
        "full28": full28,
        "coupling": sorted(coupling),
        "compact_signal_level_stn_bursts": burst,
        "compact_aperiodic_background": aperiodic_bg,
        "compact_periodic_residual_power": residual,
        "v2_reference": ordered_unique([*full28, *sorted(coupling)]),
        "compact_v3_bursts_only_addition": ordered_unique([*full28, *sorted(coupling), *burst]),
        "compact_v3_aperiodic_only_addition": ordered_unique([*full28, *sorted(coupling), *aperiodic_bg, *residual]),
        "compact_v3_bursts_plus_aperiodic": ordered_unique([*full28, *sorted(coupling), *burst, *aperiodic_bg, *residual]),
        "compact_v3_residual_power_replaces_raw_beta_gamma_diagnostic": ordered_unique(
            [*[feature for feature in full28 if feature not in raw_beta_gamma], *sorted(coupling), *residual]
        ),
    }


def compact_group(feature: str) -> str:
    """Map feature to compact V3 group for group-balanced scoring."""

    lower = feature.lower()
    if "burst" in lower:
        return "compact_signal_level_stn_bursts"
    if "aperiodic" in lower:
        return "compact_aperiodic_background"
    if "residual_power" in lower or "peak_" in lower:
        return "compact_periodic_residual_power"
    if "coupling" in lower or "cortico_stn" in lower:
        return "coupling"
    return "full28"


def group_indices(features: list[str]) -> dict[str, list[int]]:
    """Return compact group indices."""

    groups: dict[str, list[int]] = {}
    for idx, feature in enumerate(features):
        groups.setdefault(compact_group(feature), []).append(idx)
    return {group: idxs for group, idxs in groups.items() if idxs}


def fit_scaler(train: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fit train-only scaler."""

    values = np.vstack([train[[f"x_{f}" for f in features]].to_numpy(dtype=float), train[[f"medon_{f}" for f in features]].to_numpy(dtype=float)])
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0, ddof=1)
    std = np.where(~np.isfinite(std) | (std < 1e-8), 1.0, std)
    return mean, std


def evaluate_compact_group_balanced(pairs: pd.DataFrame, base_candidates: pd.DataFrame, features: list[str], scorer: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate compact group-balanced cosine scorer."""

    if not features:
        return pd.DataFrame(), pd.DataFrame()
    groups = group_indices(features)
    pair_lookup = pairs.set_index("pair_id", drop=False)
    rows: list[dict[str, object]] = []
    for heldout_subject in sorted(pairs["subject"].astype(str).unique()):
        train = pairs[~pairs["subject"].astype(str).eq(heldout_subject)].copy()
        test_ids = pairs.loc[pairs["subject"].astype(str).eq(heldout_subject), "pair_id"].astype(str).tolist()
        mean, std = fit_scaler(train, features)
        for query_id in test_ids:
            query = pair_lookup.loc[query_id]
            qx = transform(pair_value(query, "x_", features), mean, std)
            candidates = base_candidates[base_candidates["query_pair_id"].astype(str).eq(str(query_id))]
            for _, candidate_meta in candidates.iterrows():
                candidate = pair_lookup.loc[str(candidate_meta["candidate_pair_id"])]
                cm = transform(pair_value(candidate, "medon_", features), mean, std)
                score = float(np.nanmean([cosine_distance(qx[idxs], cm[idxs]) for idxs in groups.values()]))
                rows.append(
                    {
                        **candidate_meta.to_dict(),
                        "heldout_subject": heldout_subject,
                        "variant_name": scorer,
                        "score": score,
                        "n_features": int(len(features)),
                        "n_groups": int(len(groups)),
                        "feature_groups": ";".join(sorted(groups)),
                    }
                )
    scores = pd.DataFrame(rows).sort_values(["variant_name", "query_pair_id", "score"], ascending=True).copy()
    scores["rank"] = scores.groupby(["variant_name", "query_pair_id"], dropna=False).cumcount() + 1
    return scores, query_diagnostics_from_scores(scores)


def same_subject_compact(pairs: pd.DataFrame, features: list[str], scorer: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Same-subject hard-negative check for compact scorers."""

    rows = []
    groups = group_indices(features)
    for _, query in pairs.iterrows():
        subject = str(query["subject"])
        negatives = pairs[pairs["subject"].astype(str).eq(subject) & ~pairs["pair_id"].astype(str).eq(str(query["pair_id"]))]
        train = pairs[~pairs["subject"].astype(str).eq(subject)].copy()
        if negatives.empty or train.empty:
            continue
        mean, std = fit_scaler(train, features)
        qx = transform(pair_value(query, "x_", features), mean, std)
        candidate_rows = [query, *[row for _, row in negatives.iterrows()]]
        distances = []
        for candidate in candidate_rows:
            cm = transform(pair_value(candidate, "medon_", features), mean, std)
            distances.append(float(np.nanmean([cosine_distance(qx[idxs], cm[idxs]) for idxs in groups.values()])))
        true_distance = distances[0]
        neg = np.asarray(distances[1:], dtype=float)
        nearest_idx = int(np.nanargmin(neg)) if len(neg) else -1
        nearest = negatives.iloc[nearest_idx] if nearest_idx >= 0 else pd.Series(dtype=object)
        rows.append(
            {
                "variant_name": scorer,
                "query_subject": subject,
                "query_task": query["task_original"],
                "query_side": query["side"],
                "true_rank_among_same_subject_candidates": int(1 + np.sum(neg < true_distance)),
                "n_same_subject_candidates": int(len(candidate_rows)),
                "true_beats_all_same_subject_negatives": bool(np.all(true_distance <= neg)) if len(neg) else True,
                "nearest_same_subject_negative_task": nearest.get("task_original", ""),
                "nearest_same_subject_negative_side": nearest.get("side", ""),
                "distance_true": float(true_distance),
                "distance_nearest_same_subject_negative": float(neg[nearest_idx]) if nearest_idx >= 0 else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "variant_name": scorer,
        "queries_with_same_subject_hard_negatives": int(len(table)),
        "true_beats_all_rate": float(table["true_beats_all_same_subject_negatives"].mean()) if not table.empty else np.nan,
        "mean_true_rank": float(table["true_rank_among_same_subject_candidates"].mean()) if not table.empty else np.nan,
        "by_task": table.groupby("query_task")["true_beats_all_same_subject_negatives"].mean().to_dict() if not table.empty else {},
        "by_side": table.groupby("query_side")["true_beats_all_same_subject_negatives"].mean().to_dict() if not table.empty else {},
    }
    return table, summary


def summary_row(scorer: str, diag: pd.DataFrame, hard_summary: dict[str, object], n_features: int, n_groups: int, diagnostic_only: bool = False) -> dict[str, object]:
    """Build scorer result row."""

    metrics = summarize_diagnostics(diag)
    return {
        "scorer": scorer,
        "diagnostic_only": bool(diagnostic_only),
        "n_features": int(n_features),
        "n_groups": int(n_groups),
        **metrics,
        "same_subject_hard_negative_success": hard_summary.get("true_beats_all_rate", np.nan),
    }


def bootstrap_ci(diag: pd.DataFrame, hard: pd.DataFrame, n_bootstrap: int, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """Bootstrap query-level retrieval metrics."""

    rng = np.random.default_rng(seed)
    metric_columns = {
        "top1": "top_ranked_is_true_pair",
        "mrr": "reciprocal_rank",
        "percentile_rank": "percentile_rank",
        "retrieval_margin": "retrieval_margin",
    }
    observed = {metric: float(diag[column].astype(float).mean()) for metric, column in metric_columns.items()}
    if not hard.empty:
        observed["same_subject_hard_negative_success"] = float(hard["true_beats_all_same_subject_negatives"].astype(float).mean())
    boot = {metric: [] for metric in observed}
    values = {metric: diag[column].astype(float).to_numpy() for metric, column in metric_columns.items()}
    hard_values = hard["true_beats_all_same_subject_negatives"].astype(float).to_numpy() if not hard.empty else np.asarray([])
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(diag), size=len(diag))
        for metric in metric_columns:
            boot[metric].append(float(np.nanmean(values[metric][idx])))
        if len(hard_values):
            hidx = rng.integers(0, len(hard_values), size=len(hard_values))
            boot["same_subject_hard_negative_success"].append(float(np.nanmean(hard_values[hidx])))
    rows = []
    for metric, samples in boot.items():
        arr = np.asarray(samples, dtype=float)
        rows.append(
            {
                "metric": metric,
                "observed": observed[metric],
                "bootstrap_mean": float(np.nanmean(arr)),
                "ci_lower_95": float(np.nanpercentile(arr, 2.5)),
                "ci_upper_95": float(np.nanpercentile(arr, 97.5)),
                "n_bootstrap": int(n_bootstrap),
            }
        )
    table = pd.DataFrame(rows)
    margin = table[table["metric"].eq("retrieval_margin")]
    return table, {
        "rows": table.to_dict("records"),
        "margin_ci_crosses_zero": bool(not margin.empty and margin["ci_lower_95"].iloc[0] <= 0 <= margin["ci_upper_95"].iloc[0]),
    }


def select_compact_v3(results: pd.DataFrame, v2: dict[str, object], change_log: pd.DataFrame) -> dict[str, object]:
    """Select compact V3 only if all required gates pass."""

    reasons: dict[str, str] = {}
    eligible = []
    candidates = results[~results["scorer"].astype(str).eq("v2_reference")].copy()
    for _, row in candidates.iterrows():
        scorer = str(row["scorer"])
        changes = change_log[change_log["variant_name"].astype(str).eq(scorer)]
        fail_to_success = int(changes["change_type"].eq("fail_to_success").sum())
        success_to_failure = int(changes["change_type"].eq("success_to_failure").sum())
        improved_subjects = changes.loc[changes["change_type"].eq("fail_to_success"), "query_subject"].astype(str)
        concentrated = False
        if fail_to_success > 0 and not improved_subjects.empty:
            concentrated = bool(improved_subjects.value_counts().iloc[0] > max(1, int(np.ceil(0.6 * fail_to_success))))
        checks = {
            "top1_not_worse": float(row["top1"]) >= float(v2["top1"]),
            "mrr_not_meaningfully_worse": float(row["mrr"]) >= float(v2["mrr"]) - MRR_TOLERANCE,
            "same_subject_not_worse": float(row["same_subject_hard_negative_success"]) >= float(v2["same_subject_hard_negative_success"]),
            "hard_failures_not_higher": int(row["other_subject_same_task_side_failure_count"]) <= int(v2["other_subject_same_task_side_failure_count"]),
            "quality_failures_not_higher": int(row["quality_related_failure_count"]) <= int(v2["quality_related_failure_count"]),
            "net_positive_query_change": fail_to_success > success_to_failure,
            "improvements_not_one_subject": not concentrated,
            "not_diagnostic_only": not bool(row.get("diagnostic_only", False)),
        }
        if all(checks.values()):
            eligible.append(row)
            reasons[scorer] = "eligible"
        else:
            reasons[scorer] = "not selected: failed " + ", ".join([name for name, ok in checks.items() if not ok])
    if not eligible:
        return {
            "selected_compact_v3_variant_name": "",
            "reason_selected": "No compact V3 variant passed all required gates.",
            "reason_not_selected": reasons,
            "warning": "No compact V3 clearly improves over v2; v2 remains default.",
            "v2_remains_default": True,
            "v2_reference_metrics": v2,
        }
    frame = pd.DataFrame(eligible).sort_values(
        ["other_subject_same_task_side_failure_count", "quality_related_failure_count", "same_subject_hard_negative_success", "mrr", "top1"],
        ascending=[True, True, False, False, False],
    )
    selected = frame.iloc[0].to_dict()
    for key, value in list(reasons.items()):
        if key == selected["scorer"]:
            reasons.pop(key)
        elif value == "eligible":
            reasons[key] = "not selected: eligible but ranked lower by conservative ordering"
    return {
        "selected_compact_v3_variant_name": str(selected["scorer"]),
        "reason_selected": "Selected because all required compact V3 gates passed.",
        "reason_not_selected": reasons,
        "selected_compact_v3_metrics": selected,
        "v2_reference_metrics": v2,
        "warning": "",
        "v2_remains_default": False,
    }


def loso_selected(pairs: pd.DataFrame, subspaces: dict[str, list[str]], scorer: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Leave-one-subject-out stability for selected compact V3."""

    rows = []
    features = subspaces.get(scorer, [])
    for subject in sorted(pairs["subject"].astype(str).unique()):
        subset = pairs[~pairs["subject"].astype(str).eq(subject)].copy()
        base = build_base_candidate_sets(subset, MIN_DISTRACTORS)
        _, diag = evaluate_compact_group_balanced(subset, base, features, scorer)
        metrics = summarize_diagnostics(diag)
        rows.append({"removed_subject": subject, "n_pairs_remaining": len(subset), **metrics})
    table = pd.DataFrame(rows)
    return table, {
        "selected_compact_v3_variant": scorer,
        "min_top1": float(table["top1"].min()) if not table.empty else np.nan,
        "max_top1": float(table["top1"].max()) if not table.empty else np.nan,
        "mean_top1": float(table["top1"].mean()) if not table.empty else np.nan,
    }


def proxy_comparison(v2_row: dict[str, object], compact_results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare v2, previous proxy V3, and compact V3 candidates."""

    rows = [{**v2_row, "source": "v2_current"}]
    if PROXY_V3_RESULTS_PATH.exists():
        proxy = pd.read_csv(PROXY_V3_RESULTS_PATH)
        if not proxy.empty and "scorer" in proxy:
            proxy_non_v2 = proxy[~proxy["scorer"].astype(str).eq("v2_reference")].copy()
            if not proxy_non_v2.empty:
                best = proxy_non_v2.sort_values(["top1", "mrr"], ascending=False).iloc[0].to_dict()
                best["source"] = "previous_proxy_v3_best_raw"
                rows.append(best)
    for _, row in compact_results.iterrows():
        record = row.to_dict()
        record["source"] = "compact_signal_level_v3"
        rows.append(record)
    table = pd.DataFrame(rows)
    return table, {
        "proxy_v3_file_found": PROXY_V3_RESULTS_PATH.exists(),
        "n_rows": int(len(table)),
        "note": "Previous proxy V3 used window/band-power proxies; compact V3 uses local signal-level extraction where available.",
    }


def failure_comparison(v2_diag: pd.DataFrame, compact_diag: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare v2 failures to selected compact V3 failures."""

    rows = []
    v2 = v2_diag.set_index("query_pair_id", drop=False)
    compact = compact_diag.set_index("query_pair_id", drop=False) if compact_diag is not None and not compact_diag.empty else pd.DataFrame()
    for query_id in sorted(v2.index):
        v2row = v2.loc[query_id]
        crow = compact.loc[query_id] if not compact.empty and query_id in compact.index else None
        v2_success = bool(v2row["top_ranked_is_true_pair"])
        c_success = bool(crow["top_ranked_is_true_pair"]) if crow is not None else np.nan
        if crow is None:
            change_type = "no_compact_v3_selected"
        elif v2_success and c_success:
            change_type = "unchanged_success"
        elif (not v2_success) and (not c_success):
            change_type = "unchanged_failure"
        elif (not v2_success) and c_success:
            change_type = "fail_to_success"
        else:
            change_type = "success_to_failure"
        rows.append(
            {
                "query_id": query_id,
                "subject": v2row["query_subject"],
                "task": v2row["query_task"],
                "side": v2row["query_side"],
                "quality": v2row["query_quality"],
                "v2_true_rank": int(v2row["true_medon_rank"]),
                "compact_v3_true_rank": int(crow["true_medon_rank"]) if crow is not None else np.nan,
                "v2_top_candidate": v2row["top_ranked_candidate_subject"],
                "compact_v3_top_candidate": crow["top_ranked_candidate_subject"] if crow is not None else "",
                "change_type": change_type,
                "failure_type": crow["failure_type"] if crow is not None else v2row["failure_type"],
                "feature_groups_pulling_wrong_candidate_closer": "",
            }
        )
    table = pd.DataFrame(rows)
    return table, {
        "change_type_counts": table["change_type"].value_counts().to_dict(),
        "n_queries": int(len(table)),
    }


def make_figures(
    results: pd.DataFrame,
    change_log: pd.DataFrame,
    selected_bootstrap: pd.DataFrame,
    random_null: pd.DataFrame,
    selected_metrics: dict[str, object] | None,
    proxy_table: pd.DataFrame,
    burst_inventory: dict[str, object],
    aperiodic_inventory: dict[str, object],
) -> None:
    """Generate compact V3 figures."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not results.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(results))
        width = 0.38
        ax.bar(x - width / 2, results["top1"], width, label="top1")
        ax.bar(x + width / 2, results["mrr"], width, label="MRR")
        ax.set_xticks(x)
        ax.set_xticklabels(results["scorer"], rotation=55, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("V2 vs Compact V3 Top1/MRR")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "v2_vs_compact_v3_top1_mrr.png", dpi=200)
        plt.close(fig)

        for column, title, fname in [
            ("same_subject_hard_negative_success", "Same-Subject Hard-Negative Success", "same_subject_hard_negative_success.png"),
            ("other_subject_same_task_side_failure_count", "Other-Subject Same-Task/Side Failures", "other_subject_same_task_side_failures.png"),
            ("quality_related_failure_count", "Quality-Related Failures", "quality_related_failures.png"),
        ]:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(results["scorer"], results[column])
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=55)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / fname, dpi=200)
            plt.close(fig)

    if not change_log.empty:
        counts = (
            change_log[change_log["change_type"].isin(["fail_to_success", "success_to_failure"])]
            .groupby(["variant_name", "change_type"])
            .size()
            .unstack(fill_value=0)
        )
        if not counts.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            counts.plot(kind="bar", ax=ax)
            ax.set_title("Query Change Counts vs V2")
            ax.set_ylabel("Query count")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "query_change_counts_vs_v2.png", dpi=200)
            plt.close(fig)

    if not selected_bootstrap.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(
            selected_bootstrap["metric"],
            selected_bootstrap["observed"],
            yerr=[
                selected_bootstrap["observed"] - selected_bootstrap["ci_lower_95"],
                selected_bootstrap["ci_upper_95"] - selected_bootstrap["observed"],
            ],
            fmt="o",
        )
        ax.tick_params(axis="x", rotation=45)
        ax.set_title("Selected Compact V3 Bootstrap CI")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "selected_compact_v3_bootstrap_ci.png", dpi=200)
        plt.close(fig)

    if not random_null.empty and selected_metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(random_null["top1"], bins=25)
        ax.axvline(float(selected_metrics["top1"]), color="black", linestyle="--", label="observed")
        ax.legend()
        ax.set_title("Selected Compact V3 Random-Label Top1 Null")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "selected_compact_v3_random_label_top1_null.png", dpi=200)
        plt.close(fig)

    if not proxy_table.empty and "n_features" in proxy_table:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = proxy_table.get("scorer", proxy_table.get("source")).astype(str)
        ax.bar(labels, proxy_table["n_features"])
        ax.set_title("Feature Count Comparison")
        ax.tick_params(axis="x", rotation=55)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "feature_count_comparison.png", dpi=200)
        plt.close(fig)

    for inv, name in [(burst_inventory, "compact_burst_feature_availability"), (aperiodic_inventory, "compact_aperiodic_feature_availability")]:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["available", "features"], [int(bool(inv.get("available"))), int(inv.get("n_selected_features", inv.get("n_features_full", 0)))])
        ax.set_title(name.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200)
        plt.close(fig)


def write_report(
    v2_row: dict[str, object],
    raw_summary: dict[str, object],
    burst_inventory: dict[str, object],
    compact_burst_inventory: dict[str, object],
    spectral_inventory: dict[str, object],
    compact_aperiodic_inventory: dict[str, object],
    results: pd.DataFrame,
    selection: dict[str, object],
    path: Path,
) -> Path:
    """Write final compact signal-level V3 report."""

    rows = {str(row["scorer"]): row for row in results.to_dict("records")}
    bursts = rows.get("compact_v3_bursts_only_addition", {})
    aper = rows.get("compact_v3_aperiodic_only_addition", {})
    combined = rows.get("compact_v3_bursts_plus_aperiodic", {})
    selected = selection.get("selected_compact_v3_variant_name") or "none"
    lines = [
        "# Retrieval Signal-Level Compact V3",
        "",
        "## Purpose",
        "",
        "This technical report evaluates compact signal-level STN burst and PSD-derived aperiodic/residual features for paired-state retrieval. It does not discuss manuscripts, clinical use, DBS optimization, stimulation planning, or MedOn as a healthy state.",
        "",
        "## Direct Answers",
        "",
        f"1. Was v2 reproduced? Yes: top1={v2_row.get('top1')}, MRR={v2_row.get('mrr')}.",
        f"2. Were raw/time-series signals locally available? {raw_summary.get('raw_signal_available')} ({raw_summary.get('n_usable_for_signal_level')} usable recordings).",
        f"3. Were true signal-level STN burst features extracted? {burst_inventory.get('available')}; method={burst_inventory.get('method')}.",
        f"4. Were full PSD-level aperiodic/residual features extracted? {spectral_inventory.get('available')}; method={spectral_inventory.get('method')}.",
        f"5. Compact burst features used: {compact_burst_inventory.get('n_selected_features')}.",
        f"6. Compact aperiodic/residual features used: {compact_aperiodic_inventory.get('n_selected_features')}.",
        f"7. Did compact burst features improve over v2? top1={bursts.get('top1')}, MRR={bursts.get('mrr')}; compare v2 top1={v2_row.get('top1')}.",
        f"8. Did compact aperiodic features improve over v2? top1={aper.get('top1')}, MRR={aper.get('mrr')}.",
        f"9. Did compact burst + aperiodic improve over v2? top1={combined.get('top1')}, MRR={combined.get('mrr')}.",
        f"10. Did any compact V3 preserve or improve same-subject hard-negative success? selected={selected}; selection warning={selection.get('warning')}.",
        f"11. Did any compact V3 reduce other_subject_same_task_side failures? See compact_v3_scorer_results.csv; selected={selected}.",
        f"12. Did any compact V3 survive safety controls? {'yes' if selected != 'none' else 'no selected compact V3 reached controls'}.",
        f"13. Should compact V3 replace v2? {'no, v2 remains default' if selection.get('v2_remains_default') else 'yes, selected compact V3 passed gates'}.",
        f"14. If compact V3 failed, why? {selection.get('warning')}",
        "15. Next technical fix: inspect compact feature scaling and same-subject hard negatives before any additional predefined feature transforms.",
        "",
        "## Guardrail Notes",
        "",
        "- The compact V3 features are technical retrieval features only.",
        "- The script does not download data and does not alter the v2 default unless conservative gates pass.",
        "- Rank improvements are not interpreted as treatment, stimulation planning, or MedOn-as-healthy evidence.",
    ]
    return write_text(lines, path)


def run(args: argparse.Namespace) -> dict[str, Path]:
    """Run compact signal-level V3."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    state_vectors = task_side_columns(read_state_vectors(args.state_vectors, warnings))
    quality = read_quality_table(args.quality_table, warnings)
    existing_subspaces = read_subspace_definitions(args.subspace_definitions, state_vectors, warnings)
    window_features = task_side_columns(read_csv_or_empty(args.window_features, warnings))
    recordings = unique_recording_rows(window_features)

    availability, raw_summary = raw_signal_availability(recordings)
    write_csv(availability, OUTPUT_DIR / "raw_signal_availability_table.csv")
    write_json(raw_summary, OUTPUT_DIR / "raw_signal_availability_summary.json")

    if raw_summary.get("raw_signal_available"):
        burst_events, burst_full, _, burst_inventory, spectral_full, spectral_inventory = extract_signal_level_features(availability)
    else:
        burst_events = pd.DataFrame()
        burst_full = pd.DataFrame()
        spectral_full = pd.DataFrame()
        burst_inventory = {"available": False, "warning": "Raw signal unavailable; burst extraction skipped."}
        spectral_inventory = {"available": False, "warning": "Raw signal unavailable; PSD parameterization skipped."}

    id_cols = ["subject", "session", "task", "medication", "run", "condition", "task_original", "task_family", "side", "file_path"]
    compact_burst, compact_burst_inventory = compact_feature_table(burst_full, COMPACT_BURST_FEATURES, id_cols)
    compact_aper, compact_aper_inventory = compact_feature_table(spectral_full, COMPACT_APERIODIC_FEATURES, id_cols)

    write_csv(burst_events, OUTPUT_DIR / "signal_level_stn_burst_events.csv")
    write_csv(burst_full, OUTPUT_DIR / "signal_level_stn_burst_features_state_level_full.csv")
    write_json(burst_inventory, OUTPUT_DIR / "signal_level_stn_burst_feature_inventory.json")
    write_csv(compact_burst, OUTPUT_DIR / "compact_signal_level_burst_features.csv")
    write_json(compact_burst_inventory, OUTPUT_DIR / "compact_signal_level_burst_feature_inventory.json")
    write_csv(spectral_full, OUTPUT_DIR / "signal_level_aperiodic_features_state_level_full.csv")
    write_json(spectral_inventory, OUTPUT_DIR / "signal_level_aperiodic_feature_inventory.json")
    write_csv(compact_aper, OUTPUT_DIR / "compact_signal_level_aperiodic_features.csv")
    write_json(compact_aper_inventory, OUTPUT_DIR / "compact_signal_level_aperiodic_feature_inventory.json")

    compact_state = merge_compact_features(state_vectors, [compact_burst, compact_aper])
    write_csv(compact_state, OUTPUT_DIR / "compact_v3_state_vectors.csv")
    subspaces = build_compact_subspaces(
        existing_subspaces,
        compact_state,
        compact_burst_inventory.get("selected_features", []),
        compact_aper_inventory.get("selected_features", []),
    )
    write_json(
        {
            "compact_signal_level_stn_bursts": subspaces.get("compact_signal_level_stn_bursts", []),
            "compact_aperiodic_background": subspaces.get("compact_aperiodic_background", []),
            "compact_periodic_residual_power": subspaces.get("compact_periodic_residual_power", []),
            "existing_full28": subspaces.get("full28", []),
            "existing_coupling": subspaces.get("coupling", []),
        },
        OUTPUT_DIR / "compact_v3_feature_inventory.json",
    )

    pairs, _ = build_paired_examples_all_features(compact_state, quality)
    base_candidates = build_base_candidate_sets(pairs, MIN_DISTRACTORS)
    clean_features = [feature for feature in existing_subspaces.get("clean_stable_features", []) if f"x_{feature}" in pairs.columns]
    v2_features = subspaces["v2_reference"]
    v2_variant = VariantSpec(
        name="v2_reference",
        category="v2_reproduction",
        score_kind="group_balanced",
        feature_space="full_28_plus_coupling",
        distance="cosine",
    )
    v2_scores, v2_diag = evaluate_v2_variant(pairs, base_candidates, v2_features, v2_variant, clean_features)
    v2_hard, v2_hard_summary = same_subject_hard_negative_v2(pairs, v2_features, v2_variant, clean_features)
    v2_row = {
        "scorer": "v2_reference",
        "diagnostic_only": False,
        "n_features": len(v2_features),
        "n_groups": 2,
        **summarize_diagnostics(v2_diag),
        "same_subject_hard_negative_success": v2_hard_summary.get("true_beats_all_rate", np.nan),
    }
    write_csv(pd.DataFrame([v2_row]), OUTPUT_DIR / "v2_reproduction.csv")
    write_json(v2_row, OUTPUT_DIR / "v2_reproduction_summary.json")

    specs = [
        CompactScorerSpec("v2_reference", "v2_reference"),
        CompactScorerSpec("compact_v3_bursts_only_addition", "compact_v3_bursts_only_addition"),
        CompactScorerSpec("compact_v3_aperiodic_only_addition", "compact_v3_aperiodic_only_addition"),
        CompactScorerSpec("compact_v3_bursts_plus_aperiodic", "compact_v3_bursts_plus_aperiodic"),
        CompactScorerSpec("compact_v3_residual_power_replaces_raw_beta_gamma_diagnostic", "compact_v3_residual_power_replaces_raw_beta_gamma_diagnostic", diagnostic_only=True),
    ]
    rows = []
    diag_by_name = {"v2_reference": v2_diag}
    scores_by_name = {"v2_reference": v2_scores}
    hard_by_name = {"v2_reference": v2_hard}
    for spec in specs:
        features = subspaces.get(spec.feature_key, [])
        if not features:
            rows.append({"scorer": spec.name, "status": "skipped_missing_features", "diagnostic_only": spec.diagnostic_only, "n_features": 0, "n_groups": 0})
            continue
        if spec.name == "v2_reference":
            diag = v2_diag
            scores = v2_scores
            hard = v2_hard
            hard_summary = v2_hard_summary
        else:
            scores, diag = evaluate_compact_group_balanced(pairs, base_candidates, features, spec.name)
            hard, hard_summary = same_subject_compact(pairs, features, spec.name)
        diag_by_name[spec.name] = diag
        scores_by_name[spec.name] = scores
        hard_by_name[spec.name] = hard
        rows.append(summary_row(spec.name, diag, hard_summary, len(features), len(group_indices(features)), spec.diagnostic_only) | {"status": "ok", "note": spec.note})
    results = pd.DataFrame(rows)

    change_frames = []
    for name, diag in diag_by_name.items():
        if name != "v2_reference":
            change_frames.append(query_change_log_for_variant(v2_diag, diag, name))
    change_log = pd.concat(change_frames, ignore_index=True) if change_frames else pd.DataFrame()
    selection = select_compact_v3(results[results["status"].astype(str).eq("ok")], v2_row, change_log)
    selected_name = str(selection.get("selected_compact_v3_variant_name") or "")
    selected_diag = diag_by_name.get(selected_name)
    selected_scores = scores_by_name.get(selected_name)
    selected_hard = hard_by_name.get(selected_name, pd.DataFrame())
    selected_boot = pd.DataFrame()
    selected_boot_summary: dict[str, object] = {"warning": "No selected compact V3; bootstrap skipped because v2 remains default."}
    selected_random = pd.DataFrame()
    selected_random_summary: dict[str, object] = {"warning": "No selected compact V3; random-label control skipped because v2 remains default."}
    selected_loso = pd.DataFrame()
    selected_loso_summary: dict[str, object] = {"warning": "No selected compact V3; LOSO skipped because v2 remains default."}
    selected_hard_summary: dict[str, object] = {"warning": "No selected compact V3; hard-negative control skipped because v2 remains default."}
    if selected_name and selected_diag is not None and selected_scores is not None:
        selected_boot, selected_boot_summary = bootstrap_ci(selected_diag, selected_hard, args.n_bootstrap, args.random_seed)
        selected_random, selected_random_summary = random_label_negative_control(selected_scores, selected_diag, args.n_random_label_permutations, args.random_seed)
        selected_loso, selected_loso_summary = loso_selected(pairs, subspaces, selected_name)
        _, selected_hard_summary = same_subject_compact(pairs, subspaces[selected_name], selected_name)

    proxy_table, proxy_summary = proxy_comparison(v2_row, results)
    failure_table, failure_summary = failure_comparison(v2_diag, selected_diag)

    paths = {
        "results": write_csv(results, OUTPUT_DIR / "compact_v3_scorer_results.csv"),
        "summary": write_json({"rows": results.to_dict("records")}, OUTPUT_DIR / "compact_v3_scorer_summary.json"),
        "selection": write_json(selection, OUTPUT_DIR / "compact_v3_selection_summary.json"),
        "query_change": write_csv(change_log, OUTPUT_DIR / "compact_v3_query_change_log.csv"),
        "query_change_summary": write_json(query_change_summary(change_log), OUTPUT_DIR / "compact_v3_query_change_summary.json"),
        "selected_boot": write_csv(selected_boot, OUTPUT_DIR / "selected_compact_v3_bootstrap_ci.csv"),
        "selected_boot_summary": write_json(selected_boot_summary, OUTPUT_DIR / "selected_compact_v3_bootstrap_ci_summary.json"),
        "selected_random": write_csv(selected_random, OUTPUT_DIR / "selected_compact_v3_random_label_control.csv"),
        "selected_random_summary": write_json(selected_random_summary, OUTPUT_DIR / "selected_compact_v3_random_label_control_summary.json"),
        "selected_loso": write_csv(selected_loso, OUTPUT_DIR / "selected_compact_v3_loso.csv"),
        "selected_loso_summary": write_json(selected_loso_summary, OUTPUT_DIR / "selected_compact_v3_loso_summary.json"),
        "selected_hard": write_csv(selected_hard if selected_name else pd.DataFrame(), OUTPUT_DIR / "selected_compact_v3_same_subject_hard_negative.csv"),
        "selected_hard_summary": write_json(selected_hard_summary, OUTPUT_DIR / "selected_compact_v3_same_subject_hard_negative_summary.json"),
        "proxy_comparison": write_csv(proxy_table, OUTPUT_DIR / "v2_proxyv3_compactv3_comparison.csv"),
        "proxy_comparison_summary": write_json(proxy_summary, OUTPUT_DIR / "v2_proxyv3_compactv3_comparison_summary.json"),
        "failure_comparison": write_csv(failure_table, OUTPUT_DIR / "v2_vs_compactv3_failure_comparison.csv"),
        "failure_comparison_summary": write_json(failure_summary, OUTPUT_DIR / "v2_vs_compactv3_failure_comparison_summary.json"),
    }
    if warnings:
        paths["warnings"] = write_json({"warnings": warnings}, OUTPUT_DIR / "warnings.json")

    selected_metrics = selection.get("selected_compact_v3_metrics") if isinstance(selection.get("selected_compact_v3_metrics"), dict) else None
    make_figures(results, change_log, selected_boot, selected_random, selected_metrics, proxy_table, compact_burst_inventory, compact_aper_inventory)
    paths["report"] = write_report(
        v2_row,
        raw_summary,
        burst_inventory,
        compact_burst_inventory,
        spectral_inventory,
        compact_aper_inventory,
        results,
        selection,
        OUTPUT_DIR / "retrieval_signal_level_compact_v3_report.md",
    )

    best_compact = results[results["scorer"].astype(str).ne("v2_reference") & results["status"].astype(str).eq("ok")].sort_values(["top1", "mrr"], ascending=False)
    best_text = "unavailable" if best_compact.empty else f"{float(best_compact.iloc[0]['top1']):.3f} / {float(best_compact.iloc[0]['mrr']):.3f}"
    print("Retrieval signal-level compact V3 complete.")
    print(f"v2 top1/MRR: {float(v2_row['top1']):.3f} / {float(v2_row['mrr']):.3f}")
    print(f"raw signal availability: {raw_summary.get('raw_signal_available')} ({raw_summary.get('n_usable_for_signal_level')} usable recordings)")
    print(f"compact burst feature availability: {compact_burst_inventory.get('available')} ({compact_burst_inventory.get('n_selected_features')} features)")
    print(f"compact aperiodic feature availability: {compact_aper_inventory.get('available')} ({compact_aper_inventory.get('n_selected_features')} features)")
    print(f"best compact V3 top1/MRR: {best_text}")
    if selected_metrics:
        hard_delta = int(selected_metrics["other_subject_same_task_side_failure_count"]) - int(v2_row["other_subject_same_task_side_failure_count"])
        same_delta = float(selected_metrics["same_subject_hard_negative_success"]) - float(v2_row["same_subject_hard_negative_success"])
        print(f"same-subject hard-negative success change: {same_delta:.3f}")
        print(f"other_subject_same_task_side failure change: {hard_delta}")
        print("compact V3 passed selection: True")
        print("v2 remains default: False")
    else:
        print("same-subject hard-negative success change: no selected compact V3")
        print("other_subject_same_task_side failure change: no selected compact V3")
        print("compact V3 passed selection: False")
        print("v2 remains default: True")
    print(f"output path: {OUTPUT_DIR}")
    return paths


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run signal-level compact V3 retrieval analysis.")
    parser.add_argument("--state-vectors", default=str(STATE_VECTORS_PATH))
    parser.add_argument("--window-features", default=str(WINDOW_FEATURES_PATH))
    parser.add_argument("--quality-table", default=str(QUALITY_TABLE_PATH))
    parser.add_argument("--subspace-definitions", default=str(SUBSPACE_DEFINITIONS_PATH))
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--n-random-label-permutations", type=int, default=N_RANDOM_LABEL_PERMUTATIONS)
    return parser.parse_args()


def main() -> None:
    """Entry point."""

    run(parse_args())


if __name__ == "__main__":
    main()
