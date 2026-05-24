"""Epoch/window construction for ds004998 feature extraction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


WINDOW_COLUMNS = ["onset", "duration", "condition", "trial_type", "source"]


def infer_events_path(recording_path: str | Path) -> Path:
    """Infer a BIDS events.tsv path from a MEG FIF path."""

    path = Path(recording_path)
    name = path.name
    if name.endswith("_meg.fif"):
        return path.with_name(name.replace("_meg.fif", "_events.tsv"))
    return path.with_name(path.stem + "_events.tsv")


def load_event_windows(
    recording_path: str | Path,
    recording_duration: float,
    default_condition: str = "unknown",
    max_windows: int | None = None,
    max_duration_sec: float | None = None,
    fixed_window_sec: float = 10.0,
) -> pd.DataFrame:
    """Load BIDS events as feature windows, falling back to whole recording."""

    events_path = infer_events_path(recording_path)
    if events_path.exists():
        events = pd.read_csv(events_path, sep="\t")
        onset_col = "onset" if "onset" in events else None
        duration_col = "duration" if "duration" in events else None
        trial_col = "trial_type" if "trial_type" in events else None
        if onset_col and duration_col:
            windows = pd.DataFrame(
                {
                    "onset": pd.to_numeric(events[onset_col], errors="coerce").fillna(0.0),
                    "duration": pd.to_numeric(events[duration_col], errors="coerce").fillna(0.0),
                    "condition": events[trial_col].astype(str) if trial_col else default_condition,
                    "trial_type": events[trial_col].astype(str) if trial_col else default_condition,
                    "source": str(events_path),
                }
            )
            windows = windows[(windows["duration"] > 0) & (windows["onset"] >= 0)]
            windows = windows[windows["onset"] < recording_duration]
            non_bad = ~windows["condition"].astype(str).str.lower().str.startswith("bad")
            preferred = windows[non_bad]
            condition_hint = str(default_condition).lower()
            if condition_hint and condition_hint != "unknown":
                matching = preferred[
                    preferred["condition"].astype(str).str.lower().str.contains(condition_hint)
                ]
                if not matching.empty:
                    preferred = matching
            if not preferred.empty:
                windows = preferred
            if max_duration_sec is not None:
                max_duration = float(max_duration_sec)
                windows["duration"] = windows["duration"].clip(upper=max_duration)
                windows = windows[windows["duration"] > 0]
            if max_windows is not None:
                windows = windows.head(max_windows)
            if not windows.empty:
                return windows.reset_index(drop=True)[WINDOW_COLUMNS]

    duration = max(0.0, float(recording_duration))
    if max_duration_sec is not None:
        duration = min(duration, float(max_duration_sec))
    if duration <= 0:
        return pd.DataFrame(columns=WINDOW_COLUMNS)

    fixed_window_sec = max(1.0, float(fixed_window_sec))
    n_possible = max(1, int(duration // fixed_window_sec))
    if max_windows is not None:
        n_possible = min(n_possible, max_windows)
    rows = []
    for idx in range(n_possible):
        onset = idx * fixed_window_sec
        if onset >= duration:
            break
        window_duration = min(fixed_window_sec, duration - onset)
        rows.append(
            {
                "onset": onset,
                "duration": window_duration,
                "condition": default_condition,
                "trial_type": default_condition,
                "source": "fixed_window_fallback",
            }
        )
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)
