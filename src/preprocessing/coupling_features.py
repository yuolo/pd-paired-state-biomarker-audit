"""Cortico-STN coupling proxy features.

These features summarize magnitude-squared coherence between a sensor-level
motor MEG proxy and montage-mapped STN/LFP channels in ds004998. They are
exploratory in silico network features, not device settings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import coherence

from src.preprocessing.spectral_features import BANDS


COUPLING_BANDS = ["alpha", "low_beta", "high_beta", "broad_beta", "gamma"]
GLOBAL_COUPLING_COLUMNS = [
    "motor_stn_alpha_coupling",
    "motor_stn_low_beta_coupling",
    "motor_stn_high_beta_coupling",
    "motor_stn_broad_beta_coupling",
    "motor_stn_gamma_coupling",
]
SIDE_COUPLING_COLUMNS = [
    f"motor_stn_{side}_{band}_coupling"
    for side in ["left", "right"]
    for band in COUPLING_BANDS
]
TASK_SIDE_COUPLING_COLUMNS = [
    f"motor_stn_task_side_{band}_coupling"
    for band in COUPLING_BANDS
] + [
    f"motor_stn_opposite_task_side_{band}_coupling"
    for band in COUPLING_BANDS
]
COUPLING_ASYMMETRY_COLUMNS = [
    f"motor_stn_{band}_coupling_asymmetry"
    for band in COUPLING_BANDS
]
COUPLING_INDEX_COLUMNS = [
    "alpha_coupling_index",
    "beta_coupling_index",
    "gamma_coupling_index",
    "cortico_stn_coupling_index",
    "beta_coupling_asymmetry",
    "gamma_coupling_asymmetry",
]
ENHANCED_COUPLING_COLUMNS = [
    *GLOBAL_COUPLING_COLUMNS,
    *SIDE_COUPLING_COLUMNS,
    *TASK_SIDE_COUPLING_COLUMNS,
    *COUPLING_ASYMMETRY_COLUMNS,
    *COUPLING_INDEX_COLUMNS,
]


def band_coherence_from_data(
    left_data: np.ndarray,
    right_data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
) -> float:
    """Compute average coherence between mean left/right signals."""

    if left_data.size == 0 or right_data.size == 0 or sfreq <= 0:
        return float("nan")
    left = np.nanmean(left_data, axis=0)
    right = np.nanmean(right_data, axis=0)
    n_times = min(left.size, right.size)
    if n_times < 8:
        return float("nan")
    left = left[:n_times]
    right = right[:n_times]
    nperseg = min(n_times, int(max(sfreq * 2, 32)))
    freqs, coh = coherence(left, right, fs=sfreq, nperseg=nperseg)
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(coh[mask]))


def coherence_spectrum_from_data(
    left_data: np.ndarray,
    right_data: np.ndarray,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one coherence spectrum between mean channel-group signals."""

    if left_data.size == 0 or right_data.size == 0 or sfreq <= 0:
        return np.asarray([]), np.asarray([])
    left = np.nanmean(np.asarray(left_data, dtype=np.float32), axis=0)
    right = np.nanmean(np.asarray(right_data, dtype=np.float32), axis=0)
    n_times = min(left.size, right.size)
    if n_times < 8:
        return np.asarray([]), np.asarray([])
    left = left[:n_times]
    right = right[:n_times]
    nperseg = min(n_times, int(max(sfreq * 2, 32)))
    return coherence(left, right, fs=sfreq, nperseg=nperseg)


def band_coherence_from_spectrum(
    freqs: np.ndarray,
    coh: np.ndarray,
    band: tuple[float, float],
) -> float:
    """Average coherence over one band from a precomputed spectrum."""

    if freqs.size == 0 or coh.size == 0:
        return float("nan")
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(coh[mask]))


def band_coherences_from_data(
    left_data: np.ndarray,
    right_data: np.ndarray,
    sfreq: float,
    bands: list[str] | None = None,
) -> dict[str, float]:
    """Compute all band coherence values from one coherence spectrum."""

    band_names = bands or COUPLING_BANDS
    freqs, coh = coherence_spectrum_from_data(left_data, right_data, sfreq)
    return {
        band_name: band_coherence_from_spectrum(freqs, coh, BANDS[band_name])
        for band_name in band_names
    }


def _side_from_task(task: str | None) -> str:
    """Return task side label from HoldL/MoveR style task names."""

    text = str(task or "").strip().lower()
    if text.endswith("l"):
        return "left"
    if text.endswith("r"):
        return "right"
    return ""


def _opposite_side(side: str) -> str:
    """Return left/right opposite side, or empty when unavailable."""

    if side == "left":
        return "right"
    if side == "right":
        return "left"
    return ""


def _nanmean(values: list[float]) -> float:
    """Mean that returns NaN if every value is NaN."""

    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.all(~np.isfinite(array)):
        return float("nan")
    return float(np.nanmean(array))


def cortico_stn_coupling_features(
    motor_data: np.ndarray,
    stn_data: np.ndarray,
    sfreq: float,
    stn_left_data: np.ndarray | None = None,
    stn_right_data: np.ndarray | None = None,
    task: str | None = None,
) -> dict[str, float]:
    """Return multi-band motor-STN coherence and side-aware summaries."""

    stn_left_data = stn_left_data if stn_left_data is not None else np.empty((0, 0))
    stn_right_data = stn_right_data if stn_right_data is not None else np.empty((0, 0))
    features: dict[str, float] = {}
    pooled = band_coherences_from_data(motor_data, stn_data, sfreq, COUPLING_BANDS)
    left = band_coherences_from_data(motor_data, stn_left_data, sfreq, COUPLING_BANDS)
    right = band_coherences_from_data(motor_data, stn_right_data, sfreq, COUPLING_BANDS)

    for band_name in COUPLING_BANDS:
        global_value = pooled[band_name]
        left_value = left[band_name]
        right_value = right[band_name]
        features[f"motor_stn_{band_name}_coupling"] = global_value
        features[f"motor_stn_left_{band_name}_coupling"] = left_value
        features[f"motor_stn_right_{band_name}_coupling"] = right_value
        features[f"motor_stn_{band_name}_coupling_asymmetry"] = (
            right_value - left_value
            if np.isfinite(right_value) and np.isfinite(left_value)
            else float("nan")
        )

    task_side = _side_from_task(task)
    opposite = _opposite_side(task_side)
    for band_name in COUPLING_BANDS:
        features[f"motor_stn_task_side_{band_name}_coupling"] = features.get(
            f"motor_stn_{task_side}_{band_name}_coupling",
            float("nan"),
        )
        features[f"motor_stn_opposite_task_side_{band_name}_coupling"] = features.get(
            f"motor_stn_{opposite}_{band_name}_coupling",
            float("nan"),
        )

    beta_values = [
        features["motor_stn_low_beta_coupling"],
        features["motor_stn_high_beta_coupling"],
        features["motor_stn_broad_beta_coupling"],
    ]
    features["alpha_coupling_index"] = features["motor_stn_alpha_coupling"]
    features["beta_coupling_index"] = _nanmean(beta_values)
    features["gamma_coupling_index"] = features["motor_stn_gamma_coupling"]
    features["cortico_stn_coupling_index"] = _nanmean(
        [
            features["motor_stn_alpha_coupling"],
            *beta_values,
            features["motor_stn_gamma_coupling"],
        ]
    )
    beta_left = _nanmean(
        [
            features["motor_stn_left_low_beta_coupling"],
            features["motor_stn_left_high_beta_coupling"],
            features["motor_stn_left_broad_beta_coupling"],
        ]
    )
    beta_right = _nanmean(
        [
            features["motor_stn_right_low_beta_coupling"],
            features["motor_stn_right_high_beta_coupling"],
            features["motor_stn_right_broad_beta_coupling"],
        ]
    )
    features["beta_coupling_asymmetry"] = (
        beta_right - beta_left
        if np.isfinite(beta_right) and np.isfinite(beta_left)
        else float("nan")
    )
    features["gamma_coupling_asymmetry"] = features["motor_stn_gamma_coupling_asymmetry"]

    for column in ENHANCED_COUPLING_COLUMNS:
        features.setdefault(column, float("nan"))
    return features


def beta_coupling_features(
    motor_data: np.ndarray,
    stn_data: np.ndarray,
    sfreq: float,
) -> dict[str, float]:
    """Return beta-band cortico-STN coherence proxies."""

    features = cortico_stn_coupling_features(motor_data, stn_data, sfreq)
    return {
        "motor_stn_low_beta_coupling": features["motor_stn_low_beta_coupling"],
        "motor_stn_high_beta_coupling": features["motor_stn_high_beta_coupling"],
        "motor_stn_broad_beta_coupling": features["motor_stn_broad_beta_coupling"],
    }


def coupling_feature_inventory(
    window_features: pd.DataFrame,
    state_vectors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build an inventory for cortico-STN coupling features."""

    state_vectors = state_vectors if state_vectors is not None else pd.DataFrame()
    rows = []
    for feature in ENHANCED_COUPLING_COLUMNS:
        lower = feature.lower()
        if "gamma" in lower:
            band = "gamma"
        elif "alpha" in lower:
            band = "alpha"
        elif "low_beta" in lower:
            band = "low_beta"
        elif "high_beta" in lower:
            band = "high_beta"
        elif "broad_beta" in lower:
            band = "broad_beta"
        elif "beta" in lower:
            band = "beta"
        else:
            band = "multi_band"

        if "opposite_task_side" in lower:
            stn_side = "opposite_task_labeled"
        elif "task_side" in lower:
            stn_side = "task_labeled"
        elif "left" in lower:
            stn_side = "left"
        elif "right" in lower:
            stn_side = "right"
        else:
            stn_side = "pooled"

        if "asymmetry" in lower:
            feature_type = "side_asymmetry"
        elif "index" in lower:
            feature_type = "network_index"
        else:
            feature_type = "coherence"

        rows.append(
            {
                "feature": feature,
                "available_in_window_features": bool(feature in window_features.columns),
                "available_in_state_vectors": bool(feature in state_vectors.columns),
                "band": band,
                "motor_signal": "sensor_level_motor_proxy",
                "stn_side": stn_side,
                "feature_type": feature_type,
                "methodological_label": (
                    "magnitude-squared coherence proxy from simultaneous MEG and montage-mapped STN/LFP; "
                    "not a clinical control or device-setting feature"
                ),
            }
        )
    return pd.DataFrame(rows)
