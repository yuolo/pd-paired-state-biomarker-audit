"""Real ds004998 MEG/STN LFP feature extraction.

Features are sensor-level or channel-level methodological proxies unless
source localization is explicitly added later.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_loading.load_ds004998_bids import (
    inspect_ds004998_bids,
    load_lfp_montage_mapping,
    parse_bids_entities,
    read_raw_header,
)
from src.preprocessing.channel_selection import (
    meg_picks,
    motor_cortical_proxy_picks,
    stn_lfp_picks,
    stn_lfp_picks_by_side,
)
from src.preprocessing.coupling_features import (
    ENHANCED_COUPLING_COLUMNS,
    cortico_stn_coupling_features,
)
from src.preprocessing.epoching import load_event_windows
from src.preprocessing.spectral_features import BANDS, summarize_band_powers


FEATURE_PREFIXES = ["meg", "motor", "stn", "stn_left", "stn_right"]
COUPLING_COLUMNS = ENHANCED_COUPLING_COLUMNS
FEATURE_WINDOW_COLUMNS = [
    "file_path",
    "subject",
    "session",
    "task",
    "run",
    "medication",
    "condition",
    "window_index",
    "window_onset",
    "window_duration",
    *[
        f"{prefix}_{band_name}_power"
        for prefix in FEATURE_PREFIXES
        for band_name in BANDS
    ],
    *[f"{prefix}_n_channels" for prefix in FEATURE_PREFIXES],
    *COUPLING_COLUMNS,
    "source",
    "methodological_label",
]

FAILURE_LOG_COLUMNS = ["file_path", "subject", "session", "task", "run", "reason", "details"]
EXTRACTION_TIMING_COLUMNS = [
    "file_path",
    "subject",
    "session",
    "task",
    "run",
    "status",
    "runtime_sec",
    "n_feature_rows",
    "error_type",
    "error_message",
]


def empty_feature_table() -> pd.DataFrame:
    """Return an empty feature table with stable headers."""

    return pd.DataFrame(columns=FEATURE_WINDOW_COLUMNS)


def empty_failure_log() -> pd.DataFrame:
    """Return an empty extraction failure log with stable headers."""

    return pd.DataFrame(columns=FAILURE_LOG_COLUMNS)


def empty_timing_table() -> pd.DataFrame:
    """Return an empty extraction timing table with stable headers."""

    return pd.DataFrame(columns=EXTRACTION_TIMING_COLUMNS)


def _combine_feature_records(
    motor_records: list[dict[str, Any]],
    stn_records: list[dict[str, Any]],
    coupling_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Merge per-window feature dictionaries."""

    merged: dict[int, dict[str, Any]] = {}
    for collection in (motor_records, stn_records, coupling_records):
        for record in collection:
            idx = int(record["window_index"])
            merged.setdefault(idx, {}).update(record)
    return pd.DataFrame([merged[idx] for idx in sorted(merged)])


def extract_recording_features(
    file_path: str | Path,
    max_duration_sec: float | None = 120.0,
    max_windows: int | None = 12,
) -> pd.DataFrame:
    """Extract proxy electrophysiological features from one recording."""

    file_path = Path(file_path)
    entities = parse_bids_entities(file_path)
    raw = read_raw_header(file_path)
    lfp_mapping = load_lfp_montage_mapping(file_path)
    duration = float(raw.n_times / raw.info["sfreq"]) if raw.info["sfreq"] else 0.0
    windows = load_event_windows(
        file_path,
        recording_duration=duration,
        default_condition=entities["condition"],
        max_windows=max_windows,
        max_duration_sec=max_duration_sec,
    )
    if windows.empty:
        return empty_feature_table()

    all_meg_picks = meg_picks(raw)
    motor_picks = motor_cortical_proxy_picks(raw)
    stn_picks = stn_lfp_picks(raw, lfp_mapping=lfp_mapping)
    stn_side_picks = stn_lfp_picks_by_side(raw, lfp_mapping=lfp_mapping)
    meg_records = summarize_band_powers(
        raw,
        all_meg_picks,
        windows,
        prefix="meg",
        max_duration_sec=max_duration_sec,
    )
    motor_records = summarize_band_powers(
        raw,
        motor_picks,
        windows,
        prefix="motor",
        max_duration_sec=max_duration_sec,
    )
    stn_records = summarize_band_powers(
        raw,
        stn_picks,
        windows,
        prefix="stn",
        max_duration_sec=max_duration_sec,
    )
    stn_left_records = summarize_band_powers(
        raw,
        stn_side_picks.get("left", []),
        windows,
        prefix="stn_left",
        max_duration_sec=max_duration_sec,
    )
    stn_right_records = summarize_band_powers(
        raw,
        stn_side_picks.get("right", []),
        windows,
        prefix="stn_right",
        max_duration_sec=max_duration_sec,
    )

    sfreq = float(raw.info["sfreq"])
    coupling_records = []
    for window_idx, window in windows.iterrows():
        duration_window = float(window["duration"])
        if max_duration_sec is not None:
            duration_window = min(duration_window, float(max_duration_sec))
        start = int(float(window["onset"]) * sfreq)
        stop = min(raw.n_times, start + int(duration_window * sfreq))
        if stop <= start or not motor_picks or not stn_picks:
            motor_data = np.empty((0, 0))
            stn_data = np.empty((0, 0))
            stn_left_data = np.empty((0, 0))
            stn_right_data = np.empty((0, 0))
        else:
            motor_data = raw.get_data(
                picks=motor_picks,
                start=start,
                stop=stop,
                reject_by_annotation="omit",
            )
            stn_data = raw.get_data(
                picks=stn_picks,
                start=start,
                stop=stop,
                reject_by_annotation="omit",
            )
            stn_left_data = (
                raw.get_data(
                    picks=stn_side_picks.get("left", []),
                    start=start,
                    stop=stop,
                    reject_by_annotation="omit",
                )
                if stn_side_picks.get("left")
                else np.empty((0, 0))
            )
            stn_right_data = (
                raw.get_data(
                    picks=stn_side_picks.get("right", []),
                    start=start,
                    stop=stop,
                    reject_by_annotation="omit",
                )
                if stn_side_picks.get("right")
                else np.empty((0, 0))
            )
        coupling_records.append(
            {
                "window_index": int(window_idx),
                **cortico_stn_coupling_features(
                    motor_data,
                    stn_data,
                    sfreq,
                    stn_left_data=stn_left_data,
                    stn_right_data=stn_right_data,
                    task=entities["task"],
                ),
            }
        )

    features = _combine_feature_records(
        meg_records + motor_records,
        stn_records + stn_left_records + stn_right_records,
        coupling_records,
    )
    for key in ["subject", "session", "task", "run", "medication"]:
        features.insert(0, key, entities[key])
    features.insert(0, "file_path", str(file_path))
    features["source"] = "ds004998_real_sensor_proxy"
    features["methodological_label"] = (
        "sensor-level MEG, montage-mapped EEG/LFP, and cortico-STN coherence proxy; "
        "no anatomical source localization in v2"
    )
    for column in FEATURE_WINDOW_COLUMNS:
        if column not in features:
            features[column] = np.nan if column not in {"source", "methodological_label"} else ""
    return features[FEATURE_WINDOW_COLUMNS]


def extract_recording_features_timed(
    file_path: str | Path,
    max_duration_sec: float | None = 120.0,
    max_windows: int | None = 12,
) -> tuple[pd.DataFrame, dict[str, object] | None, dict[str, object]]:
    """Extract one recording and return features, error record, and timing."""

    start = time.perf_counter()
    entities = parse_bids_entities(file_path)
    timing: dict[str, object] = {
        "file_path": str(file_path),
        "subject": entities.get("subject", ""),
        "session": entities.get("session", ""),
        "task": entities.get("task", ""),
        "run": entities.get("run", ""),
        "status": "ok",
        "runtime_sec": 0.0,
        "n_feature_rows": 0,
        "error_type": "",
        "error_message": "",
    }
    try:
        features = extract_recording_features(
            file_path,
            max_duration_sec=max_duration_sec,
            max_windows=max_windows,
        )
        timing["n_feature_rows"] = int(len(features))
        return features, None, timing | {"runtime_sec": time.perf_counter() - start}
    except Exception as exc:  # noqa: BLE001
        timing.update(
            {
                "status": "error",
                "runtime_sec": time.perf_counter() - start,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        return empty_feature_table(), {
            "subject": entities.get("subject", ""),
            "session": entities.get("session", ""),
            "task": entities.get("task", ""),
            "run": entities.get("run", ""),
            "file_path": str(file_path),
            "usable": False,
            "qc_status": f"feature_error:{type(exc).__name__}",
            "duration_sec": 0.0,
            "sampling_rate_hz": 0.0,
            "n_channels": 0,
            "n_meg_channels": 0,
            "n_lfp_stn_channels": 0,
            "notes": str(exc),
        }, timing


def build_extraction_failure_log(
    recordings: pd.DataFrame,
    qc: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Build a failure log for empty or partially failed extraction runs."""

    rows = []
    if recordings.empty:
        rows.append(
            {
                "file_path": "",
                "subject": "",
                "session": "",
                "task": "",
                "run": "",
                "reason": "no_recordings_found",
                "details": "No FIF recordings were discovered under the supplied data root.",
            }
        )
    for _, row in qc.iterrows():
        status = str(row.get("qc_status", ""))
        if status != "ok" or status.startswith("feature_error"):
            rows.append(
                {
                    "file_path": row.get("file_path", ""),
                    "subject": row.get("subject", ""),
                    "session": row.get("session", ""),
                    "task": row.get("task", ""),
                    "run": row.get("run", ""),
                    "reason": status or "not_usable",
                    "details": row.get("notes", ""),
                }
            )
    if features.empty and not rows:
        rows.append(
            {
                "file_path": "",
                "subject": "",
                "session": "",
                "task": "",
                "run": "",
                "reason": "no_feature_rows",
                "details": "Recordings were inspected but no feature rows were extracted.",
            }
        )
    return pd.DataFrame(rows, columns=FAILURE_LOG_COLUMNS)


def extract_ds004998_features(
    data_root: str | Path,
    max_recordings: int | None = None,
    max_duration_sec: float | None = 120.0,
    max_windows: int | None = 12,
    n_jobs: int = 1,
    parallel_backend: str = "sequential",
    recording_batch_size: int | None = None,
    return_timing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inspect ds004998 and extract real feature windows from usable recordings."""

    recordings, channels, qc = inspect_ds004998_bids(data_root, max_recordings=max_recordings)
    if recordings.empty:
        if return_timing:
            return recordings, channels, qc, pd.DataFrame(), empty_timing_table()
        return recordings, channels, qc, pd.DataFrame()

    feature_tables = []
    timing_rows = []
    usable = recordings[recordings["usable"].astype(bool)]
    records = usable.to_dict(orient="records")
    n_jobs = max(1, int(n_jobs or 1))
    backend = str(parallel_backend or "sequential").lower()
    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor

    def handle_result(result: tuple[pd.DataFrame, dict[str, object] | None, dict[str, object]]) -> None:
        features_one, error_record, timing = result
        timing_rows.append(timing)
        if not features_one.empty:
            feature_tables.append(features_one)
        if error_record is not None:
            qc.loc[len(qc)] = error_record

    if n_jobs == 1 or backend == "sequential":
        for row in records:
            handle_result(
                extract_recording_features_timed(
                    row["file_path"],
                    max_duration_sec=max_duration_sec,
                    max_windows=max_windows,
                )
            )
    elif backend in {"process", "thread"}:
        batch_size = max(1, int(recording_batch_size or len(records) or 1))
        for batch_start in range(0, len(records), batch_size):
            batch = records[batch_start : batch_start + batch_size]
            with executor_cls(max_workers=n_jobs) as executor:
                futures = [
                    executor.submit(
                        extract_recording_features_timed,
                        row["file_path"],
                        max_duration_sec,
                        max_windows,
                    )
                    for row in batch
                ]
                for future in as_completed(futures):
                    handle_result(future.result())
    else:
        raise ValueError(f"Unknown parallel_backend: {parallel_backend}")

    features = pd.concat(feature_tables, ignore_index=True) if feature_tables else empty_feature_table()
    timing = pd.DataFrame(timing_rows, columns=EXTRACTION_TIMING_COLUMNS)
    if return_timing:
        return recordings, channels, qc, features, timing
    return recordings, channels, qc, features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract real ds004998 proxy features.")
    parser.add_argument("--data-root", default="data/raw/ds004998")
    parser.add_argument("--out", default="outputs/tables/ds004998_window_features.csv")
    parser.add_argument("--max-recordings", type=int, default=1)
    parser.add_argument("--max-duration-sec", type=float, default=120.0)
    parser.add_argument("--max-windows", type=int, default=12)
    args = parser.parse_args()

    _, _, _, features = extract_ds004998_features(
        args.data_root,
        max_recordings=args.max_recordings,
        max_duration_sec=args.max_duration_sec,
        max_windows=args.max_windows,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    print(f"Wrote {len(features)} feature rows to {out}.")


if __name__ == "__main__":
    main()
