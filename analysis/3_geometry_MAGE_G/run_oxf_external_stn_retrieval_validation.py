"""Run locked STN-only external ON/OFF retrieval validation on local OXF data.

This is a separate external-validation layer. It does not modify the frozen
ds004998 v2A pipeline or reuse Oxford/MRC data inside ds004998 analyses. The
external data are treated as STN-only MATLAB exports, so this script evaluates
the transportable STN spectral plus compact aperiodic/residual feature branch
with fixed top_k=5 and aperiodic_alpha=0.5.

The main claim boundary is deliberate: these are paired-state identifiability
audits on an external STN-LFP ON/OFF cohort, not clinical prediction, treatment
recommendation, DBS optimization, or causal medication-effect estimation.
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

import matplotlib.pyplot as plt  # noqa: E402

try:
    import h5py  # noqa: E402
except Exception as exc:  # noqa: BLE001
    h5py = None
    H5PY_IMPORT_ERROR = str(exc)
else:
    H5PY_IMPORT_ERROR = ""

try:
    from scipy import io as scipy_io  # noqa: E402
    from scipy import signal as scipy_signal  # noqa: E402
except Exception as exc:  # noqa: BLE001
    scipy_io = None
    scipy_signal = None
    SCIPY_IMPORT_ERROR = str(exc)
else:
    SCIPY_IMPORT_ERROR = ""

from scripts.run_retrieval_aperiodic_assisted_v2 import (  # noqa: E402
    AssistedVariantSpec,
    aper_distance_scores,
    evaluate_assisted_variant,
)
from scripts.run_retrieval_distance_geometry_improvement import (  # noqa: E402
    VariantSpec,
    evaluate_variant,
    query_diagnostics_from_scores,
    to_builtin,
)
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


OUTPUT_DIR = Path("outputs/oxf_external_stn_retrieval_validation")
FIGURES_DIR = OUTPUT_DIR / "figures"
DEFAULT_DATA_ROOT = Path("data/oxf/DBS")
MAX_DURATION_SEC = 120.0
MIN_CHANNEL_DURATION_SEC = 20.0
EPS = 1e-30

SPECTRAL_FEATURES = [
    "stn_alpha_log_power",
    "stn_low_beta_log_power",
    "stn_high_beta_log_power",
    "stn_broad_beta_log_power",
    "stn_gamma_log_power",
    "stn_low_beta_high_beta_log_ratio",
    "stn_beta_gamma_log_ratio",
]
COMPACT_FEATURES = [
    "aperiodic_offset",
    "aperiodic_slope",
    "low_beta_residual_power",
    "high_beta_residual_power",
    "broad_beta_residual_power",
    "gamma_residual_power",
    "beta_peak_frequency",
    "beta_peak_amplitude",
    "gamma_peak_amplitude",
]
ALL_FEATURES = [*SPECTRAL_FEATURES, *COMPACT_FEATURES]
EVALUATED_CONDITIONS = [
    PRIMARY_CONDITION,
    HARD_NEGATIVE_CONDITION,
    ALL_MEDON_CONDITION,
    TRUE_PLUS_OTHER_CONDITION,
]
PRIMARY_CONDITION_INTERPRETATION = (
    "OXF matched-hemisphere analogue: true ON plus other-subject ON candidates "
    "from the same STN side."
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--max-duration-sec", type=float, default=MAX_DURATION_SEC)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


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
    """Write Markdown/text output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def safe_float(value: object, default: float = np.nan) -> float:
    """Convert a value to finite float if possible."""

    try:
        out = float(value)
    except Exception:  # noqa: BLE001
        return default
    return out if np.isfinite(out) else default


def infer_subject(path: Path) -> str:
    """Infer OXF subject id from file name."""

    return path.name.split("_")[0]


def infer_side(path: Path) -> str:
    """Infer STN side from file name."""

    text = path.name.lower()
    if "_lestn_" in text:
        return "left"
    if "_ristn_" in text:
        return "right"
    return "unknown"


def infer_medication(path: Path) -> str:
    """Infer medication label from file name."""

    text = path.name.lower()
    if "_off" in text:
        return "OFF"
    if "_on" in text:
        return "ON"
    return "unknown"


def decode_uint16_string(values: np.ndarray) -> str:
    """Decode MATLAB uint16 char arrays."""

    flat = np.asarray(values).reshape(-1)
    chars = []
    for value in flat:
        code = int(value)
        if code:
            chars.append(chr(code))
    return "".join(chars)


def scipy_titles_and_data(path: Path) -> tuple[list[str], float, np.ndarray, str]:
    """Load old-style MATLAB file with scipy."""

    if scipy_io is None:
        raise RuntimeError(f"scipy.io is unavailable: {SCIPY_IMPORT_ERROR}")
    mat = scipy_io.loadmat(path, squeeze_me=True, struct_as_record=False)
    smr = mat["SmrData"]
    titles = [str(value) for value in np.ravel(getattr(smr, "WvTits"))]
    fs = safe_float(getattr(smr, "Fs"))
    data = np.asarray(getattr(smr, "WvData"), dtype=float)
    return titles, fs, data, "scipy.io.loadmat"


def h5_titles(path: Path) -> tuple[list[str], float, tuple[int, ...]]:
    """Read HDF5 MATLAB titles and shape without loading all samples."""

    if h5py is None:
        raise RuntimeError(f"h5py is unavailable: {H5PY_IMPORT_ERROR}")
    with h5py.File(path, "r") as handle:
        titles_ds = handle["SmrData/WvTits"]
        titles = []
        for ref in np.ravel(titles_ds[()]):
            titles.append(decode_uint16_string(np.asarray(handle[ref])))
        fs = safe_float(np.asarray(handle["SmrData/Fs"])[0, 0])
        shape = tuple(int(dim) for dim in handle["SmrData/WvData"].shape)
    return titles, fs, shape


def h5_read_selected_channels(path: Path, selected_indices: list[int], max_samples: int) -> tuple[list[np.ndarray], str]:
    """Read selected HDF5 channels using partial slices."""

    if h5py is None:
        raise RuntimeError(f"h5py is unavailable: {H5PY_IMPORT_ERROR}")
    with h5py.File(path, "r") as handle:
        data = handle["SmrData/WvData"]
        titles, _fs, _shape = h5_titles(path)
        n_titles = len(titles)
        channels = []
        if data.ndim == 1:
            channels.append(np.asarray(data[:max_samples], dtype=float))
        elif data.shape[1] == n_titles:
            n = min(max_samples, int(data.shape[0]))
            for idx in selected_indices:
                channels.append(np.asarray(data[:n, idx], dtype=float))
        elif data.shape[0] == n_titles:
            n = min(max_samples, int(data.shape[1]))
            for idx in selected_indices:
                channels.append(np.asarray(data[idx, :n], dtype=float))
        else:
            raise ValueError(f"Cannot align WvData shape {data.shape} to {n_titles} titles.")
    return channels, "h5py_partial"


def is_strict_stn_channel(title: str, side: str) -> bool:
    """Return whether a title looks like a raw bipolar STN channel."""

    text = str(title).strip()
    low = text.lower()
    if not text or low.startswith(("f_", "s_", "ch", "amp")) or "dcrem" in low:
        return False
    if side == "left":
        return bool(re.match(r"^l(?:e)?\d+$", low))
    if side == "right":
        return bool(re.match(r"^r(?:e)?\d+$", low))
    return False


def is_fallback_stn_channel(title: str, side: str) -> bool:
    """Conservative fallback if strict channel names are absent."""

    text = str(title).strip()
    low = text.lower()
    if not text or low.startswith(("f_", "s_", "ch", "amp")) or "dcrem" in low:
        return False
    if side == "left":
        return low.startswith("l") and any(ch.isdigit() for ch in low)
    if side == "right":
        return low.startswith("r") and any(ch.isdigit() for ch in low)
    return False


def selected_channel_indices(titles: list[str], side: str) -> tuple[list[int], str]:
    """Select STN channels matching the file side."""

    strict = [idx for idx, title in enumerate(titles) if is_strict_stn_channel(title, side)]
    if strict:
        return strict, "strict_side_bipolar"
    fallback = [idx for idx, title in enumerate(titles) if is_fallback_stn_channel(title, side)]
    return fallback, "fallback_side_numeric" if fallback else "no_matching_stn_channel"


def clean_signal(values: np.ndarray) -> np.ndarray:
    """Clean one channel for conservative feature extraction."""

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return arr
    arr = arr - float(np.nanmean(arr))
    if len(arr) > 100:
        low, high = np.nanpercentile(arr, [0.1, 99.9])
        if np.isfinite(low) and np.isfinite(high) and high > low:
            arr = np.clip(arr, low, high)
    return arr


def read_record_channels(path: Path, max_duration_sec: float) -> tuple[list[str], list[np.ndarray], float, dict[str, object]]:
    """Read selected side-matched STN channels from one OXF MATLAB file."""

    side = infer_side(path)
    try:
        titles, fs, data, loader = scipy_titles_and_data(path)
        selected, selection_mode = selected_channel_indices(titles, side)
        if data.ndim == 1:
            channels = [data[: int(max_duration_sec * fs)]]
            channel_titles = [titles[selected[0]] if selected else f"{side}_1d_signal"]
        elif data.shape[0] == len(titles):
            max_samples = min(int(max_duration_sec * fs), int(data.shape[1]))
            channels = [np.asarray(data[idx, :max_samples], dtype=float) for idx in selected]
            channel_titles = [titles[idx] for idx in selected]
        elif data.shape[1] == len(titles):
            max_samples = min(int(max_duration_sec * fs), int(data.shape[0]))
            channels = [np.asarray(data[:max_samples, idx], dtype=float) for idx in selected]
            channel_titles = [titles[idx] for idx in selected]
        else:
            raise ValueError(f"Cannot align WvData shape {data.shape} to {len(titles)} titles.")
        shape = tuple(int(dim) for dim in data.shape)
    except NotImplementedError:
        titles, fs, shape = h5_titles(path)
        selected, selection_mode = selected_channel_indices(titles, side)
        max_samples = int(max_duration_sec * fs) if np.isfinite(fs) else 0
        channels, loader = h5_read_selected_channels(path, selected, max_samples)
        channel_titles = [titles[idx] for idx in selected]

    cleaned = [clean_signal(channel) for channel in channels]
    meta = {
        "loader": loader,
        "wvdata_shape": "x".join(str(dim) for dim in shape),
        "n_wvtitles": int(len(titles)),
        "selected_channel_count": int(len(cleaned)),
        "selected_channels": ";".join(channel_titles),
        "channel_selection_mode": selection_mode,
    }
    return channel_titles, cleaned, fs, meta


def gamma_band_for_sfreq(sfreq: float) -> tuple[float, float] | None:
    """Return a gamma band available below Nyquist."""

    if not np.isfinite(sfreq) or sfreq <= 0:
        return None
    nyquist = sfreq / 2.0
    if nyquist > 95:
        return (60.0, 90.0)
    if nyquist > 45:
        return (35.0, min(55.0, nyquist * 0.9))
    return None


def compute_psd(signal: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD."""

    if scipy_signal is None:
        raise RuntimeError(f"scipy.signal is unavailable: {SCIPY_IMPORT_ERROR}")
    if not np.isfinite(sfreq) or sfreq <= 0 or len(signal) < 128:
        return np.asarray([]), np.asarray([])
    nperseg = min(4096, max(256, len(signal) // 8), len(signal))
    freqs, psd = scipy_signal.welch(signal, fs=sfreq, nperseg=nperseg, detrend="constant")
    return freqs, psd


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    """Integrate PSD in a band."""

    mask = (freqs >= band[0]) & (freqs <= band[1])
    if mask.sum() < 2:
        return np.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))


def channel_feature_row(signal: np.ndarray, sfreq: float) -> dict[str, float]:
    """Compute STN spectral and compact aperiodic/residual features."""

    freqs, psd = compute_psd(signal, sfreq)
    if len(freqs) == 0:
        return {feature: np.nan for feature in ALL_FEATURES}

    bands = {
        "alpha": (8.0, 12.0),
        "low_beta": (13.0, 20.0),
        "high_beta": (20.0, 35.0),
        "broad_beta": (13.0, 35.0),
    }
    gamma = gamma_band_for_sfreq(sfreq)
    if gamma is not None:
        bands["gamma"] = gamma

    powers = {name: band_power(freqs, psd, band) for name, band in bands.items()}
    row: dict[str, float] = {
        "stn_alpha_log_power": float(np.log10(max(powers.get("alpha", np.nan), EPS))),
        "stn_low_beta_log_power": float(np.log10(max(powers.get("low_beta", np.nan), EPS))),
        "stn_high_beta_log_power": float(np.log10(max(powers.get("high_beta", np.nan), EPS))),
        "stn_broad_beta_log_power": float(np.log10(max(powers.get("broad_beta", np.nan), EPS))),
        "stn_gamma_log_power": float(np.log10(max(powers.get("gamma", np.nan), EPS))),
    }
    row["stn_low_beta_high_beta_log_ratio"] = row["stn_low_beta_log_power"] - row["stn_high_beta_log_power"]
    row["stn_beta_gamma_log_ratio"] = row["stn_broad_beta_log_power"] - row["stn_gamma_log_power"]

    max_fit_freq = min(95.0, float(np.nanmax(freqs)))
    fit_mask = (freqs >= 5.0) & (freqs <= max_fit_freq) & (psd > 0)
    fit_mask &= ~((freqs >= 45.0) & (freqs <= 55.0))
    if fit_mask.sum() < 8:
        for feature in COMPACT_FEATURES:
            row[feature] = np.nan
        return row

    x = np.log10(freqs[fit_mask])
    y = np.log10(psd[fit_mask])
    slope, offset = np.polyfit(x, y, 1)
    predicted = offset + slope * np.log10(np.maximum(freqs, 1e-6))
    residual = np.log10(np.maximum(psd, EPS)) - predicted
    row["aperiodic_offset"] = float(offset)
    row["aperiodic_slope"] = float(slope)
    for name, band in bands.items():
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if name in {"low_beta", "high_beta", "broad_beta", "gamma"}:
            row[f"{name}_residual_power"] = float(np.nanmean(residual[mask])) if mask.sum() else np.nan

    beta_mask = (freqs >= 13.0) & (freqs <= 35.0)
    if beta_mask.sum():
        beta_freqs = freqs[beta_mask]
        beta_resid = residual[beta_mask]
        idx = int(np.nanargmax(beta_resid))
        row["beta_peak_frequency"] = float(beta_freqs[idx])
        row["beta_peak_amplitude"] = float(beta_resid[idx])
    else:
        row["beta_peak_frequency"] = np.nan
        row["beta_peak_amplitude"] = np.nan
    if gamma is not None:
        gamma_mask = (freqs >= gamma[0]) & (freqs <= gamma[1])
        row["gamma_peak_amplitude"] = float(np.nanmax(residual[gamma_mask])) if gamma_mask.sum() else np.nan
    else:
        row["gamma_peak_amplitude"] = np.nan
    return row


def discover_records(data_root: Path) -> pd.DataFrame:
    """Discover individual ON/OFF STN MATLAB files."""

    files = sorted(path for path in data_root.rglob("*.mat") if path.name != "MATRIX_DBS.mat")
    rows = []
    for path in files:
        rows.append(
            {
                "file_path": str(path),
                "file_name": path.name,
                "subject": infer_subject(path),
                "side": infer_side(path),
                "medication_state": infer_medication(path),
                "file_size_bytes": int(path.stat().st_size),
                "flag_erna_label": bool("erna" in path.name.lower()),
            }
        )
    return pd.DataFrame(rows)


def extract_state_features(inventory: pd.DataFrame, max_duration_sec: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract one aggregated feature row per OXF file."""

    state_rows: list[dict[str, object]] = []
    channel_rows: list[dict[str, object]] = []
    for _, record in inventory.iterrows():
        path = Path(record["file_path"])
        base = record.to_dict()
        try:
            channel_titles, channels, sfreq, meta = read_record_channels(path, max_duration_sec)
            duration_rows = []
            feature_rows = []
            for title, signal in zip(channel_titles, channels, strict=False):
                duration_sec = len(signal) / sfreq if np.isfinite(sfreq) and sfreq > 0 else np.nan
                if len(signal) < 128 or (np.isfinite(duration_sec) and duration_sec < MIN_CHANNEL_DURATION_SEC):
                    feature = {feature_name: np.nan for feature_name in ALL_FEATURES}
                    status = "too_short"
                else:
                    feature = channel_feature_row(signal, sfreq)
                    status = "ok"
                channel_row = {
                    **base,
                    **meta,
                    "channel": title,
                    "sampling_frequency": sfreq,
                    "n_samples_used": int(len(signal)),
                    "duration_sec_used": duration_sec,
                    "feature_status": status,
                    **feature,
                }
                channel_rows.append(channel_row)
                if status == "ok":
                    duration_rows.append(duration_sec)
                    feature_rows.append(feature)
            if feature_rows:
                frame = pd.DataFrame(feature_rows)
                state_feature = frame.median(numeric_only=True).to_dict()
                state_status = "ok"
            else:
                state_feature = {feature_name: np.nan for feature_name in ALL_FEATURES}
                state_status = "no_usable_selected_channel"
            state_rows.append(
                {
                    **base,
                    **meta,
                    "sampling_frequency": sfreq,
                    "n_channels_used": int(len(feature_rows)),
                    "duration_sec_used_median": float(np.nanmedian(duration_rows)) if duration_rows else np.nan,
                    "state_feature_status": state_status,
                    **state_feature,
                }
            )
        except Exception as exc:  # noqa: BLE001
            empty_features = {feature_name: np.nan for feature_name in ALL_FEATURES}
            state_rows.append(
                {
                    **base,
                    "loader": "load_failed",
                    "wvdata_shape": "",
                    "n_wvtitles": 0,
                    "selected_channel_count": 0,
                    "selected_channels": "",
                    "channel_selection_mode": "load_failed",
                    "sampling_frequency": np.nan,
                    "n_channels_used": 0,
                    "duration_sec_used_median": np.nan,
                    "state_feature_status": "load_failed",
                    "load_error": str(exc),
                    **empty_features,
                }
            )
    return pd.DataFrame(state_rows), pd.DataFrame(channel_rows)


def feature_availability(pairs: pd.DataFrame, candidate_features: list[str]) -> pd.DataFrame:
    """Return feature availability across OFF and ON pair columns."""

    rows = []
    for feature in candidate_features:
        cols = [f"x_{feature}", f"medon_{feature}"]
        values = pairs[cols].apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=float))
        rows.append(
            {
                "feature": feature,
                "feature_group": "compact_aperiodic_residual" if feature in COMPACT_FEATURES else "stn_spectral",
                "finite_value_fraction": float(finite.mean()) if finite.size else 0.0,
                "usable_for_retrieval": bool(finite.all()) if finite.size else False,
            }
        )
    return pd.DataFrame(rows)


def build_pairs(state_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build complete OFF/ON hemisphere-level pairs."""

    complete_rows: list[dict[str, object]] = []
    incomplete_rows: list[dict[str, object]] = []
    usable = state_features[state_features["state_feature_status"].astype(str).eq("ok")].copy()
    for (subject, side), group in usable.groupby(["subject", "side"], dropna=False):
        off = group[group["medication_state"].astype(str).eq("OFF")]
        on = group[group["medication_state"].astype(str).eq("ON")]
        if off.empty or on.empty:
            incomplete_rows.append(
                {
                    "subject": subject,
                    "side": side,
                    "n_off": int(len(off)),
                    "n_on": int(len(on)),
                    "reason": "missing_OFF_or_ON",
                }
            )
            continue
        off_row = off.sort_values("file_name").iloc[0]
        on_row = on.sort_values("file_name").iloc[0]
        pair_id = f"{subject}_{side}"
        quality = "unknown"
        if bool(off_row.get("flag_erna_label", False)) or bool(on_row.get("flag_erna_label", False)):
            quality = "erna_label_present"
        row: dict[str, object] = {
            "pair_id": pair_id,
            "subject": subject,
            "task_original": "Rest",
            "task_family": "Rest",
            "side": side,
            "quality_flag": quality,
            "off_file_path": off_row["file_path"],
            "on_file_path": on_row["file_path"],
            "off_selected_channels": off_row.get("selected_channels", ""),
            "on_selected_channels": on_row.get("selected_channels", ""),
            "off_sampling_frequency": off_row.get("sampling_frequency", np.nan),
            "on_sampling_frequency": on_row.get("sampling_frequency", np.nan),
            "off_duration_sec_used": off_row.get("duration_sec_used_median", np.nan),
            "on_duration_sec_used": on_row.get("duration_sec_used_median", np.nan),
            "off_flag_erna_label": bool(off_row.get("flag_erna_label", False)),
            "on_flag_erna_label": bool(on_row.get("flag_erna_label", False)),
        }
        for feature in ALL_FEATURES:
            row[f"x_{feature}"] = safe_float(off_row.get(feature))
            row[f"medon_{feature}"] = safe_float(on_row.get(feature))
            row[f"y_{feature}"] = safe_float(on_row.get(feature))
        complete_rows.append(row)
    pairs = pd.DataFrame(complete_rows).sort_values(["subject", "side"]).reset_index(drop=True)
    availability = feature_availability(pairs, ALL_FEATURES) if not pairs.empty else pd.DataFrame()
    usable_features = availability.loc[availability["usable_for_retrieval"].astype(bool), "feature"].astype(str).tolist()
    keep_cols = [col for col in pairs.columns if not any(col.endswith(feature) for feature in ALL_FEATURES if feature not in usable_features)]
    pairs = pairs[keep_cols].copy() if not pairs.empty else pairs
    return pairs, availability, pd.DataFrame(incomplete_rows)


def candidate_row(query: pd.Series, candidate: pd.Series, is_true: bool, match_level: str) -> dict[str, object]:
    """Build one candidate row in the ds004998 retrieval schema."""

    return {
        "query_pair_id": query["pair_id"],
        "query_subject": query["subject"],
        "query_task_original": query["task_original"],
        "query_task_family": query["task_family"],
        "query_side": query["side"],
        "query_quality_flag": query.get("quality_flag", "unknown"),
        "candidate_pair_id": candidate["pair_id"],
        "candidate_subject": candidate["subject"],
        "candidate_task_original": candidate["task_original"],
        "candidate_task_family": candidate["task_family"],
        "candidate_side": candidate["side"],
        "candidate_quality_flag": candidate.get("quality_flag", "unknown"),
        "is_true_pair": bool(is_true),
        "match_level": match_level,
    }


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep first occurrence for each candidate id."""

    seen: set[str] = set()
    out = []
    for row in rows:
        candidate_id = str(row["candidate_pair_id"])
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        out.append(row)
    return out


def finalize_candidate_pool(rows: list[dict[str, object]], condition: str) -> pd.DataFrame:
    """Add pool counts and condition name."""

    data = pd.DataFrame(rows)
    if data.empty:
        return data
    data["candidate_pool_condition"] = condition
    data["number_of_candidates"] = data.groupby("query_pair_id", dropna=False)["candidate_pair_id"].transform("count").astype(int)
    return data


def build_candidate_pool(pairs: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Build OXF candidate-pool ladder using standard condition names."""

    rows: list[dict[str, object]] = []
    for _, query in pairs.iterrows():
        query_id = str(query["pair_id"])
        subject = str(query["subject"])
        side = str(query["side"])
        local: list[dict[str, object]] = [candidate_row(query, query, True, "true_pair")]
        if condition == PRIMARY_CONDITION:
            pool = pairs[~pairs["subject"].astype(str).eq(subject) & pairs["side"].astype(str).eq(side)]
            for _, candidate in pool.iterrows():
                local.append(candidate_row(query, candidate, False, "other_subject_same_hemisphere"))
        elif condition == HARD_NEGATIVE_CONDITION:
            pool = pairs[~pairs["subject"].astype(str).eq(subject) & pairs["side"].astype(str).eq(side)]
            for _, candidate in pool.iterrows():
                local.append(candidate_row(query, candidate, False, "other_subject_same_hemisphere"))
            same_subject = pairs[pairs["subject"].astype(str).eq(subject) & ~pairs["pair_id"].astype(str).eq(query_id)]
            for _, candidate in same_subject.iterrows():
                local.append(candidate_row(query, candidate, False, "same_subject_other_hemisphere"))
        elif condition == ALL_MEDON_CONDITION:
            for _, candidate in pairs.iterrows():
                local.append(candidate_row(query, candidate, str(candidate["pair_id"]) == query_id, "all_on_candidates"))
        elif condition == TRUE_PLUS_OTHER_CONDITION:
            pool = pairs[~pairs["subject"].astype(str).eq(subject)]
            for _, candidate in pool.iterrows():
                local.append(candidate_row(query, candidate, False, "all_other_subject_on"))
        else:
            raise ValueError(f"Unknown candidate-pool condition: {condition}")
        rows.extend(deduplicate(local))
    return finalize_candidate_pool(rows, condition)


def selected_v2b_variant():
    """Return the fixed selected v2B variant from ds004998 experiments."""

    for variant in V2B_VARIANTS:
        if variant.name == V2B_NAME:
            return variant
    raise AssertionError(f"Selected v2B variant missing: {V2B_NAME}")


def add_condition(data: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Add/replace candidate-pool condition."""

    out = data.copy()
    out["candidate_pool_condition"] = condition
    return out


def evaluate_forward_scores(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate v2, v2A, and selected v2B over all OXF pools."""

    v2_variant = VariantSpec(
        name=V2_NAME,
        category="external_stn_only_v2_candidate_generator",
        score_kind="group_balanced",
        feature_space="external_stn_spectral_plus_compact_aperiodic",
        distance="cosine",
        note="STN-only external transfer; no ds004998 scorer tuning.",
    )
    v2a_spec = AssistedVariantSpec(
        name=V2A_NAME,
        mode="two_stage",
        top_k=TOP_K,
        alpha=APERIODIC_ALPHA,
        note="Frozen v2A top-5 compact reranking transferred to OXF STN-only features.",
    )
    v2b_variant = selected_v2b_variant()

    score_tables = []
    pool_tables = []
    for condition in EVALUATED_CONDITIONS:
        pool = build_candidate_pool(pairs, condition)
        pool_tables.append(pool)
        v2_scores, _v2_diag = evaluate_variant(pairs, pool, v2_features, v2_variant, clean_features)
        aper_scores = aper_distance_scores(pairs, pool, compact_features, "aperiodic_distance")
        v2a_scores, _v2a_diag = evaluate_assisted_variant(v2_scores, aper_scores, v2a_spec)
        v2b_scores, _v2b_diag, _weights = evaluate_v2b_variant(pairs, pool, v2_features, compact_features, clean_features, v2b_variant)
        score_tables.extend(
            [
                add_condition(v2_scores, condition),
                add_condition(v2a_scores, condition),
                add_condition(v2b_scores, condition),
            ]
        )
    return normalize_scores(pd.concat(score_tables, ignore_index=True)), pd.concat(pool_tables, ignore_index=True)


def reverse_pairs(pairs: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Swap OFF and ON feature vectors for ON-to-OFF reverse retrieval."""

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


def evaluate_reverse_v2b(
    pairs: pd.DataFrame,
    v2_features: list[str],
    compact_features: list[str],
    clean_features: list[str],
) -> pd.DataFrame:
    """Evaluate selected v2B in reverse ON-to-OFF direction over all candidates."""

    reversed_pairs = reverse_pairs(pairs, v2_features)
    pool = build_candidate_pool(reversed_pairs, ALL_MEDON_CONDITION)
    scores, _diag, _weights = evaluate_v2b_variant(
        reversed_pairs, pool, v2_features, compact_features, clean_features, selected_v2b_variant()
    )
    return add_condition(normalize_scores(scores), ALL_MEDON_CONDITION)


def random_label_null(
    diagnostics: pd.DataFrame,
    scores: pd.DataFrame,
    n_perm: int,
    seed: int,
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
        for idx in range(n_perm):
            ranks = np.asarray([rng.integers(1, size + 1) for size in sizes], dtype=float)
            sample_rows.append(
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
        samples = pd.DataFrame([row for row in sample_rows if row["candidate_pool_condition"] == condition and row["variant_name"] == variant])
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


def matched_side_null(
    diagnostics: pd.DataFrame,
    scores: pd.DataFrame,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Null positive drawn from same-side false candidates when available."""

    rng = np.random.default_rng(seed)
    summary_rows = []
    sample_rows = []
    for (condition, variant), subset in scores.groupby(["candidate_pool_condition", "variant_name"], dropna=False):
        per_query = []
        for query_id, group in subset.groupby("query_pair_id", dropna=False):
            false_same_side = group[
                ~group["is_true_pair"].astype(bool)
                & group["candidate_side"].astype(str).eq(group["query_side"].astype(str))
            ].copy()
            if false_same_side.empty:
                continue
            true = group[group["is_true_pair"].astype(bool)]
            if true.empty:
                continue
            per_query.append(
                {
                    "query_pair_id": query_id,
                    "candidate_set_size": int(group["candidate_pair_id"].nunique()),
                    "true_rank": int(true.iloc[0]["rank"]),
                    "false_ranks": false_same_side["rank"].to_numpy(dtype=int),
                }
            )
        if not per_query:
            continue
        true_ranks = np.asarray([item["true_rank"] for item in per_query], dtype=float)
        sizes = np.asarray([item["candidate_set_size"] for item in per_query], dtype=float)
        observed = {
            "top1": float(np.mean(true_ranks == 1)),
            "mrr": float(np.mean(1.0 / true_ranks)),
            "percentile_rank": float(np.mean(1.0 - ((true_ranks - 1.0) / np.maximum(1.0, sizes - 1.0)))),
        }
        local_samples = []
        for idx in range(n_perm):
            ranks = np.asarray([rng.choice(item["false_ranks"]) for item in per_query], dtype=float)
            local_samples.append(
                {
                    "permutation_index": int(idx),
                    "candidate_pool_condition": condition,
                    "variant_name": variant,
                    "top1": float(np.mean(ranks == 1)),
                    "mrr": float(np.mean(1.0 / ranks)),
                    "percentile_rank": float(np.mean(1.0 - ((ranks - 1.0) / np.maximum(1.0, sizes - 1.0)))),
                    "seed": seed,
                    "null_type": "matched_side_false_positive",
                    "n_queries_with_matched_false_positive": int(len(per_query)),
                }
            )
        sample_rows.extend(local_samples)
        local_frame = pd.DataFrame(local_samples)
        for metric in ["top1", "mrr", "percentile_rank"]:
            null_values = local_frame[metric].to_numpy(dtype=float)
            obs = observed[metric]
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
                    "n_queries": int(len(per_query)),
                    "seed": seed,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def loso_sensitivity(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Drop one subject at a time and summarize metrics."""

    rows = []
    for (condition, variant), subset in diagnostics.groupby(["candidate_pool_condition", "variant_name"], dropna=False):
        full = query_metrics(subset)
        for subject in sorted(subset["query_subject"].astype(str).unique()):
            reduced = subset[~subset["query_subject"].astype(str).eq(subject)].copy()
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
    """Evaluate same-subject other-hemisphere hard negatives."""

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
                "query_side": true["query_side"],
                "true_rank_full_pool": int(true["rank"]),
                "true_rank_among_same_subject_candidates": int(rank_among),
                "true_beats_same_subject_hard_negative": bool(rank_among == 1),
                "best_same_subject_candidate_pair_id": best["candidate_pair_id"],
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


def feature_mapping(usable_features: list[str]) -> pd.DataFrame:
    """Map external features to fixed diagnostic families."""

    rows = []
    for feature in usable_features:
        group = "compact_aperiodic_residual" if feature in COMPACT_FEATURES else "stn_spectral"
        role = "v2A_compact_rerank_features" if feature in COMPACT_FEATURES else "v2_candidate_generator_stn_branch"
        rows.append({"feature": feature, "feature_group": group, "locked_role": role})
    return pd.DataFrame(rows)


def plot_metric(metrics: pd.DataFrame, metric: str, path: Path) -> None:
    """Plot metric by condition and variant."""

    if metrics.empty:
        return
    focus = metrics[metrics["variant_name"].isin([V2_NAME, V2A_NAME, V2B_NAME, V2C_DECONFOUNDED_NAME, V2D_QUALITY_DECONFOUNDED_NAME])].copy()
    if focus.empty:
        return
    pivot = focus.pivot_table(index="candidate_pool_condition", columns="variant_name", values=metric, aggfunc="first")
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05 if metric in {"top1", "mrr", "top3", "top5"} else None)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(summary: dict[str, object], path: Path) -> Path:
    """Write concise Markdown report."""

    validation = summary["validation"]
    lines = [
        "# OXF External STN Retrieval Validation",
        "",
        "Scope: STN-only external ON/OFF paired-state retrieval validation. No ds004998 frozen outputs were modified.",
        "",
        "## Validation",
        f"- data_root: {validation['data_root']}",
        f"- discovered_mat_files: {validation['discovered_mat_files']}",
        f"- complete_hemisphere_pairs: {validation['complete_hemisphere_pairs']}",
        f"- subjects_with_complete_pairs: {validation['subjects_with_complete_pairs']}",
        f"- left_pairs: {validation['left_pairs']}",
        f"- right_pairs: {validation['right_pairs']}",
        f"- top_k: {validation['top_k']}",
        f"- aperiodic_alpha: {validation['aperiodic_alpha']}",
        f"- primary_condition_interpretation: {PRIMARY_CONDITION_INTERPRETATION}",
        "",
        "## Key Metrics",
    ]
    for row in summary["key_metrics"]:
        lines.append(
            f"- {row['candidate_pool_condition']} / {row['variant_name']}: "
            f"top1={float(row['top1']):.3f}, MRR={float(row['mrr']):.3f}, failures={int(row['failures'])}"
        )
    lines.extend(["", "## Warnings"])
    for warning in summary["warnings"]:
        lines.append(f"- {warning}")
    return write_text(lines, path)


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run the full OXF external validation."""

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if TOP_K != 5:
        raise AssertionError(f"top_k changed: {TOP_K}")
    if not math.isclose(float(APERIODIC_ALPHA), 0.5):
        raise AssertionError(f"aperiodic_alpha changed: {APERIODIC_ALPHA}")
    if not data_root.exists():
        raise FileNotFoundError(f"OXF data root not found: {data_root}")

    inventory = discover_records(data_root)
    state_features, channel_features = extract_state_features(inventory, float(args.max_duration_sec))
    pairs, availability, incomplete = build_pairs(state_features)
    usable_features = availability.loc[availability["usable_for_retrieval"].astype(bool), "feature"].astype(str).tolist()
    v2_features = [feature for feature in ALL_FEATURES if feature in usable_features]
    compact_features = [feature for feature in COMPACT_FEATURES if feature in usable_features]
    clean_features = compact_features
    if len(pairs) == 0:
        raise AssertionError("No complete OXF OFF/ON hemisphere pairs were built.")
    if not v2_features or not compact_features:
        raise AssertionError("No usable external STN v2/compact features available.")

    forward_scores, candidate_pools = evaluate_forward_scores(pairs, v2_features, compact_features, clean_features)
    reverse_scores = evaluate_reverse_v2b(pairs, v2_features, compact_features, clean_features)
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
    bootstrap_ci, bootstrap_samples = subject_bootstrap_ci(diagnostics)
    random_summary, random_samples = random_label_null(diagnostics, all_scores, int(args.n_perm), int(args.seed))
    matched_summary, matched_samples = matched_side_null(diagnostics, all_scores, int(args.n_perm), int(args.seed))
    loso = loso_sensitivity(diagnostics)
    hard_summary, hard_cases = hard_negative_cases(all_scores)
    failures = failure_cases(diagnostics)
    mapping = feature_mapping(usable_features)

    validation = {
        "data_root": str(data_root),
        "discovered_mat_files": int(len(inventory)),
        "state_feature_rows": int(len(state_features)),
        "channel_feature_rows": int(len(channel_features)),
        "complete_hemisphere_pairs": int(len(pairs)),
        "subjects_with_complete_pairs": int(pairs["subject"].nunique()),
        "left_pairs": int(pairs["side"].astype(str).eq("left").sum()),
        "right_pairs": int(pairs["side"].astype(str).eq("right").sum()),
        "usable_feature_count": int(len(usable_features)),
        "usable_compact_feature_count": int(len(compact_features)),
        "top_k": TOP_K,
        "aperiodic_alpha": APERIODIC_ALPHA,
        "max_duration_sec": float(args.max_duration_sec),
        "primary_condition_interpretation": PRIMARY_CONDITION_INTERPRETATION,
    }
    warnings = [
        "External OXF validation is STN-only; it cannot validate the full ds004998 MEG+STN feature system.",
        "Primary OXF pool is a matched-hemisphere analogue of the ds004998 matched task/side pool.",
        "MATLAB v7.3 files are loaded with h5py; older MATLAB files are loaded with scipy.io.loadmat.",
        "Signals are truncated to a fixed maximum duration before feature extraction for harmonization.",
        "Quality flags are unavailable except filename-level ERNA labels; v2D quality-aware terms are therefore limited.",
        "No parameter tuning, feature selection search, clinical prediction, treatment, DBS, or causal medication claim is made.",
    ]
    if int(pairs["subject"].nunique()) < int(len(pairs)):
        warnings.append("Some subjects contribute two hemispheres; bootstrap and LOSO use subject-level grouping.")
    if state_features["state_feature_status"].astype(str).ne("ok").any():
        warnings.append("Some OXF state files failed feature extraction or had no usable side-matched STN channel.")
    if inventory["flag_erna_label"].astype(bool).any():
        warnings.append("Some ON/OFF filenames contain ERNA labels; these are flagged but not excluded.")

    key_variants = [V2_NAME, V2A_NAME, V2B_NAME, V2C_DECONFOUNDED_NAME, V2D_QUALITY_DECONFOUNDED_NAME]
    key_conditions = [PRIMARY_CONDITION, HARD_NEGATIVE_CONDITION, ALL_MEDON_CONDITION]
    key_metrics = metrics[
        metrics["variant_name"].astype(str).isin(key_variants)
        & metrics["candidate_pool_condition"].astype(str).isin(key_conditions)
    ].sort_values(["candidate_pool_condition", "variant_name"])
    summary = {
        "validation": validation,
        "key_metrics": key_metrics.to_dict("records"),
        "warnings": warnings,
        "claim_boundary": {
            "external_stn_only_validation": True,
            "full_meg_stn_validation": False,
            "frozen_ds004998_outputs_modified": False,
            "clinical_prediction_or_treatment_claim": False,
            "parameter_tuning": False,
        },
    }

    paths = {
        "summary_json": write_json(summary, output_dir / "oxf_external_validation_summary.json"),
        "summary_md": write_report(summary, output_dir / "oxf_external_validation_summary.md"),
        "inventory": write_csv(inventory, output_dir / "oxf_data_inventory.csv"),
        "state_features": write_csv(state_features, output_dir / "oxf_stn_state_features.csv"),
        "channel_features": write_csv(channel_features, output_dir / "oxf_stn_channel_features.csv"),
        "pairs": write_csv(pairs, output_dir / "oxf_onoff_pairs.csv"),
        "incomplete_pairs": write_csv(incomplete, output_dir / "oxf_incomplete_pairs.csv"),
        "feature_availability": write_csv(availability, output_dir / "oxf_feature_availability.csv"),
        "feature_mapping": write_csv(mapping, output_dir / "oxf_feature_component_mapping.csv"),
        "candidate_pools": write_csv(candidate_pools, output_dir / "oxf_candidate_pools.csv"),
        "candidate_scores": write_csv(all_scores, output_dir / "oxf_candidate_scores.csv"),
        "forward_scores": write_csv(forward_scores, output_dir / "oxf_forward_scores.csv"),
        "reverse_scores": write_csv(reverse_scores, output_dir / "oxf_reverse_v2b_scores.csv"),
        "metrics": write_csv(metrics, output_dir / "oxf_observed_metrics.csv"),
        "diagnostics": write_csv(diagnostics, output_dir / "oxf_query_diagnostics.csv"),
        "comparison_to_v2a": write_csv(comparison_v2a, output_dir / "oxf_comparison_to_v2a.csv"),
        "comparison_to_v2b": write_csv(comparison_v2b, output_dir / "oxf_comparison_to_v2b.csv"),
        "bootstrap_ci": write_csv(bootstrap_ci, output_dir / "oxf_subject_bootstrap_ci.csv"),
        "bootstrap_samples": write_csv(bootstrap_samples, output_dir / "oxf_subject_bootstrap_samples.csv"),
        "random_label_null_summary": write_csv(random_summary, output_dir / "oxf_random_label_null_summary.csv"),
        "random_label_null_samples": write_csv(random_samples, output_dir / "oxf_random_label_null_samples.csv"),
        "matched_side_null_summary": write_csv(matched_summary, output_dir / "oxf_matched_side_null_summary.csv"),
        "matched_side_null_samples": write_csv(matched_samples, output_dir / "oxf_matched_side_null_samples.csv"),
        "loso_sensitivity": write_csv(loso, output_dir / "oxf_loso_sensitivity.csv"),
        "hard_negative_summary": write_csv(hard_summary, output_dir / "oxf_same_subject_hard_negative_summary.csv"),
        "hard_negative_cases": write_csv(hard_cases, output_dir / "oxf_same_subject_hard_negative_cases.csv"),
        "failure_cases": write_csv(failures, output_dir / "oxf_failure_cases.csv"),
        "v2d_components": write_csv(components, output_dir / "oxf_v2d_components.csv"),
    }
    plot_metric(metrics, "top1", figures_dir / "oxf_top1_by_variant_condition.png")
    plot_metric(metrics, "mrr", figures_dir / "oxf_mrr_by_variant_condition.png")

    print(f"data_root: {data_root}")
    print(f"complete_hemisphere_pairs: {validation['complete_hemisphere_pairs']}")
    print(f"subjects_with_complete_pairs: {validation['subjects_with_complete_pairs']}")
    print(f"left_pairs/right_pairs: {validation['left_pairs']}/{validation['right_pairs']}")
    print(f"usable_features: {validation['usable_feature_count']}")
    print(f"top_k: {TOP_K}")
    print(f"aperiodic_alpha: {APERIODIC_ALPHA}")
    for condition in key_conditions:
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
