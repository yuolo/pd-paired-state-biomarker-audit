"""Spectral feature extraction for MEG/STN LFP recordings."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch


BANDS = {
    "alpha": (8.0, 12.0),
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 30.0),
    "broad_beta": (13.0, 30.0),
    "gamma": (35.0, 70.0),
}


def _safe_log10(value: float, eps: float = 1e-300) -> float:
    """Return log-scaled power with numerical protection."""

    return float(np.log10(max(float(value), eps)))


def band_power_from_data(
    data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
) -> float:
    """Compute mean log band power across channels."""

    if data.size == 0 or sfreq <= 0:
        return float("nan")
    n_times = data.shape[-1]
    nperseg = min(n_times, int(max(sfreq * 2, 32)))
    if nperseg < 8:
        return float("nan")
    freqs, psd = welch(data, fs=sfreq, axis=-1, nperseg=nperseg)
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return float("nan")
    power = np.trapezoid(psd[..., mask], freqs[mask], axis=-1)
    return _safe_log10(float(np.nanmean(power)))


def band_powers_from_data(
    data: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Compute all configured band powers from one Welch PSD."""

    bands = bands or BANDS
    if data.size == 0 or sfreq <= 0:
        return {band_name: float("nan") for band_name in bands}
    data = np.asarray(data, dtype=np.float32)
    n_times = data.shape[-1]
    nperseg = min(n_times, int(max(sfreq * 2, 32)))
    if nperseg < 8:
        return {band_name: float("nan") for band_name in bands}
    freqs, psd = welch(data, fs=sfreq, axis=-1, nperseg=nperseg)
    results: dict[str, float] = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            results[band_name] = float("nan")
            continue
        power = np.trapezoid(psd[..., mask], freqs[mask], axis=-1)
        results[band_name] = _safe_log10(float(np.nanmean(power)))
    return results


def summarize_band_powers(
    raw,
    picks: list[int],
    windows: pd.DataFrame,
    prefix: str,
    max_duration_sec: float | None = None,
) -> list[dict[str, float | int | str]]:
    """Compute band powers for each window and channel group."""

    sfreq = float(raw.info["sfreq"])
    rows = []
    for window_idx, window in windows.iterrows():
        duration = float(window["duration"])
        if max_duration_sec is not None:
            duration = min(duration, float(max_duration_sec))
        start = int(float(window["onset"]) * sfreq)
        stop = min(raw.n_times, start + int(duration * sfreq))
        if stop <= start or not picks:
            data = np.empty((0, 0))
        else:
            data = raw.get_data(picks=picks, start=start, stop=stop, reject_by_annotation="omit")
        record: dict[str, float | int | str] = {
            "window_index": int(window_idx),
            "condition": str(window["condition"]),
            "window_onset": float(window["onset"]),
            "window_duration": float(duration),
            f"{prefix}_n_channels": int(len(picks)),
        }
        for band_name, power in band_powers_from_data(data, sfreq, BANDS).items():
            record[f"{prefix}_{band_name}_power"] = power
        rows.append(record)
    return rows
