"""Build ds004998 real-data state vectors for inverse design."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.coupling_features import (
    ENHANCED_COUPLING_COLUMNS,
    coupling_feature_inventory,
)
from src.utils.io_utils import write_table

BASE_REAL_STATE_FEATURE_COLUMNS = [
    "meg_alpha_power",
    "meg_low_beta_power",
    "meg_high_beta_power",
    "meg_broad_beta_power",
    "meg_gamma_power",
    "motor_alpha_power",
    "motor_low_beta_power",
    "motor_high_beta_power",
    "motor_broad_beta_power",
    "motor_gamma_power",
    "stn_alpha_power",
    "stn_low_beta_power",
    "stn_high_beta_power",
    "stn_broad_beta_power",
    "stn_gamma_power",
    "stn_left_alpha_power",
    "stn_left_low_beta_power",
    "stn_left_high_beta_power",
    "stn_left_broad_beta_power",
    "stn_left_gamma_power",
    "stn_right_alpha_power",
    "stn_right_low_beta_power",
    "stn_right_high_beta_power",
    "stn_right_broad_beta_power",
    "stn_right_gamma_power",
    "motor_stn_low_beta_coupling",
    "motor_stn_high_beta_coupling",
    "motor_stn_broad_beta_coupling",
]
REAL_STATE_FEATURE_COLUMNS = [
    *BASE_REAL_STATE_FEATURE_COLUMNS,
    *[
        column
        for column in ENHANCED_COUPLING_COLUMNS
        if column not in BASE_REAL_STATE_FEATURE_COLUMNS
    ],
]


STATE_METADATA_COLUMNS = [
    "subject",
    "session",
    "condition",
    "medication",
    "task",
    "run",
]
STATE_SIDE_COLUMNS = ["task_original", "task_family", "side"]


def _task_side_labels(task: str) -> tuple[str, str, str]:
    """Return original task, task family, and side label."""

    original = str(task or "unknown")
    lower = original.lower()
    if lower.startswith("hold"):
        family = "Hold"
    elif lower.startswith("move"):
        family = "Move"
    elif lower.startswith("rest"):
        family = "Rest"
    else:
        family = original or "unknown"
    if lower.endswith("l"):
        side = "L"
    elif lower.endswith("r"):
        side = "R"
    elif family == "Rest":
        side = "none"
    else:
        side = "unknown"
    return original, family, side


def available_real_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return real feature columns available in a table."""

    return [
        column
        for column in REAL_STATE_FEATURE_COLUMNS
        if column in features.columns and pd.api.types.is_numeric_dtype(features[column])
    ]


def build_real_state_vectors(features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate feature windows into subject/session/condition state vectors."""

    columns = STATE_METADATA_COLUMNS + REAL_STATE_FEATURE_COLUMNS + [
        "n_windows",
        "source",
        "methodological_label",
        *STATE_SIDE_COLUMNS,
    ]
    if features.empty:
        return pd.DataFrame(columns=columns)

    feature_columns = available_real_feature_columns(features)
    if not feature_columns:
        return pd.DataFrame(columns=columns)

    data = features.copy()
    for column in STATE_METADATA_COLUMNS:
        if column not in data:
            data[column] = "unknown"
        data[column] = data[column].fillna("unknown").astype(str)

    data["_window_count"] = 1
    grouped = (
        data.groupby(STATE_METADATA_COLUMNS, dropna=False, as_index=False)
        .agg({**{column: "mean" for column in feature_columns}, "_window_count": "sum"})
        .rename(columns={"_window_count": "n_windows"})
    )

    for column in REAL_STATE_FEATURE_COLUMNS:
        if column not in grouped:
            grouped[column] = np.nan
    grouped["source"] = "ds004998_real_sensor_proxy"
    grouped["methodological_label"] = (
        "dataset-internal MEG/STN LFP state proxy; no HCP electrophysiology reference"
    )
    labels = grouped["task"].apply(_task_side_labels)
    grouped["task_original"] = [item[0] for item in labels]
    grouped["task_family"] = [item[1] for item in labels]
    grouped["side"] = [item[2] for item in labels]
    return grouped[columns]


def build_subject_level_features(state_vectors: pd.DataFrame) -> pd.DataFrame:
    """Aggregate state vectors to subject-level summaries."""

    if state_vectors.empty:
        return pd.DataFrame(columns=["subject", "condition", "medication"] + REAL_STATE_FEATURE_COLUMNS)
    feature_columns = available_real_feature_columns(state_vectors)
    return (
        state_vectors.groupby(["subject", "condition", "medication"], as_index=False)[feature_columns]
        .mean()
        .sort_values(["subject", "condition", "medication"])
    )


def build_condition_level_features(state_vectors: pd.DataFrame) -> pd.DataFrame:
    """Aggregate state vectors to condition-level summaries."""

    if state_vectors.empty:
        return pd.DataFrame(columns=["condition", "medication"] + REAL_STATE_FEATURE_COLUMNS)
    feature_columns = available_real_feature_columns(state_vectors)
    return (
        state_vectors.groupby(["condition", "medication"], as_index=False)[feature_columns]
        .mean()
        .sort_values(["condition", "medication"])
    )


def write_real_state_outputs(
    feature_windows: pd.DataFrame,
    output_dir: str | Path = "outputs/tables",
) -> dict[str, Path]:
    """Write ds004998 feature and state-vector outputs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_vectors = build_real_state_vectors(feature_windows)
    subject_level = build_subject_level_features(state_vectors)
    condition_level = build_condition_level_features(state_vectors)
    paths = {
        "window_features": output_dir / "ds004998_window_features.csv",
        "window_features_enhanced_coupling": output_dir / "ds004998_window_features_enhanced_coupling.csv",
        "subject_level": output_dir / "ds004998_subject_level_features.csv",
        "condition_level": output_dir / "ds004998_condition_level_features.csv",
        "state_vectors": output_dir / "ds004998_state_vectors.csv",
        "state_vectors_enhanced_coupling": output_dir / "ds004998_state_vectors_enhanced_coupling.csv",
        "coupling_feature_inventory": output_dir / "cortico_stn_coupling_feature_inventory.csv",
    }
    if feature_windows.empty and len(feature_windows.columns) == 0:
        from src.preprocessing.meg_lfp_preprocessing import empty_feature_table

        feature_windows = empty_feature_table()
    write_table(feature_windows, paths["window_features"], formats=("csv", "parquet"))
    write_table(feature_windows, paths["window_features_enhanced_coupling"], formats=("csv", "parquet"))
    write_table(subject_level, paths["subject_level"], formats=("csv", "parquet"))
    write_table(condition_level, paths["condition_level"], formats=("csv", "parquet"))
    write_table(state_vectors, paths["state_vectors"], formats=("csv", "parquet"))
    write_table(state_vectors, paths["state_vectors_enhanced_coupling"], formats=("csv", "parquet"))
    write_table(
        coupling_feature_inventory(feature_windows, state_vectors),
        paths["coupling_feature_inventory"],
        formats=("csv", "parquet"),
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real ds004998 state vectors.")
    parser.add_argument("--features", default="outputs/tables/ds004998_window_features.csv")
    parser.add_argument("--output-dir", default="outputs/tables")
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    paths = write_real_state_outputs(features, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
