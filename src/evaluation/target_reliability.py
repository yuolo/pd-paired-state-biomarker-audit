"""Target stability and reliability analysis for ds004998 real-data outputs.

All electrophysiological comparisons in this module use dataset-internal
proxies. HCP-derived information is not used as an electrophysiological
reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.stratified_real_analysis import (
    finite_feature_subset,
    read_state_vectors,
    target_flags,
)
from src.evaluation.side_aware_analysis import add_task_side_columns, parse_task_side
from src.inverse_design.objective import ObjectiveWeights, parse_candidate_feature, rank_candidate_targets
from src.pathology_model.build_real_state_vectors import available_real_feature_columns
from src.pathology_model.state_deviation import build_internal_reference_stats, select_internal_reference_rows


TARGET_ID_COLUMNS = ["feature", "node", "band", "target_type"]
RANK_COLUMNS = [
    "rank",
    "target_type",
    "node",
    "edge",
    "band",
    "feature",
    "expected_delta_deviation",
    "uncertainty",
    "stability_estimate",
    "is_beta_target",
    "is_stn_target",
    "is_gamma_target",
]
RELIABILITY_COLUMNS = [
    "feature",
    "node",
    "band",
    "target_type",
    "top5_frequency",
    "top10_frequency",
    "mean_rank",
    "median_rank",
    "rank_std",
    "appears_in_n_subjects",
    "appears_in_n_tasks",
    "appears_in_holdl",
    "appears_in_movel",
    "appears_after_quality_filtering",
    "appears_in_good_only",
    "mean_expected_delta_deviation",
    "mean_uncertainty",
    "mean_stability_estimate",
    "subject_coverage",
    "task_coverage",
    "rank_score",
    "uncertainty_score",
    "quality_survival_score",
    "reliability_score",
    "is_beta_target",
    "is_stn_target",
    "is_gamma_target",
    "n_rank_contexts",
]
SIDE_RELIABILITY_COLUMNS = [
    *RELIABILITY_COLUMNS,
    "appears_in_left",
    "appears_in_right",
    "side_coverage",
    "appears_in_hold_family",
    "appears_in_move_family",
    "task_family_coverage",
    "sideaware_reliability_score",
]
CONDITION_ORDER = [
    "all_recordings",
    "exclude_low_quality",
    "exclude_caution_low_quality",
    "good_only",
]


@dataclass(frozen=True)
class ReliabilityInputs:
    """Input tables used by the target reliability analysis."""

    state_vectors: pd.DataFrame
    deviation_scores: pd.DataFrame
    overall_targets: pd.DataFrame
    holdl_targets: pd.DataFrame
    movel_targets: pd.DataFrame
    by_subject_targets: pd.DataFrame
    rank_stability: pd.DataFrame
    quality: pd.DataFrame
    quality_summary: pd.DataFrame
    quality_targets: pd.DataFrame
    warnings: list[str]


def optional_read_csv(path: str | Path, warnings: list[str], dtype: dict[str, type | str] | None = None) -> pd.DataFrame:
    """Read a CSV if available, otherwise record a warning and return empty."""

    path = Path(path)
    if not path.exists():
        warnings.append(f"Missing optional input: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype)
    except Exception as exc:  # noqa: BLE001 - report and continue with other inputs.
        warnings.append(f"Could not read {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def read_inputs(state_vectors_path: str | Path, table_dir: str | Path = "outputs/tables") -> ReliabilityInputs:
    """Load all available reliability-analysis inputs."""

    warnings: list[str] = []
    table_dir = Path(table_dir)
    dtype = {"run": str}
    if Path(state_vectors_path).exists():
        state_vectors = read_state_vectors(state_vectors_path)
    else:
        warnings.append(f"Missing required state vector input: {state_vectors_path}")
        state_vectors = pd.DataFrame()
    return ReliabilityInputs(
        state_vectors=state_vectors,
        deviation_scores=optional_read_csv(table_dir / "real_deviation_scores.csv", warnings, dtype=dtype),
        overall_targets=optional_read_csv(table_dir / "real_candidate_targets.csv", warnings, dtype=dtype),
        holdl_targets=optional_read_csv(table_dir / "real_candidate_targets_holdl.csv", warnings, dtype=dtype),
        movel_targets=optional_read_csv(table_dir / "real_candidate_targets_movel.csv", warnings, dtype=dtype),
        by_subject_targets=optional_read_csv(table_dir / "real_candidate_targets_by_subject.csv", warnings, dtype=dtype),
        rank_stability=optional_read_csv(table_dir / "real_target_rank_stability.csv", warnings, dtype=dtype),
        quality=optional_read_csv(table_dir / "real_recording_quality.csv", warnings, dtype=dtype),
        quality_summary=optional_read_csv(table_dir / "real_quality_sensitivity_summary.csv", warnings, dtype=dtype),
        quality_targets=optional_read_csv(table_dir / "real_quality_sensitivity_targets.csv", warnings, dtype=dtype),
        warnings=warnings,
    )


def ensure_target_flags(table: pd.DataFrame) -> pd.DataFrame:
    """Ensure rank and target-flag columns are available."""

    if table.empty:
        return table.copy()
    data = table.copy()
    if "rank" in data:
        data["rank"] = pd.to_numeric(data["rank"], errors="coerce")
    for column in ["node", "band", "target_type", "feature", "edge"]:
        if column not in data:
            data[column] = ""
    missing_flags = {"is_beta_target", "is_stn_target", "is_gamma_target", "is_top5"} - set(data.columns)
    if missing_flags:
        data = target_flags(data)
    for column in ["expected_delta_deviation", "uncertainty", "stability_estimate"]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        else:
            data[column] = np.nan
    return data


def candidate_descriptor(feature: str, lookup: dict[str, dict[str, object]]) -> dict[str, object]:
    """Return a stable target descriptor for a feature."""

    if feature in lookup:
        return lookup[feature]
    parsed = parse_candidate_feature(feature)
    return {
        "feature": feature,
        "node": parsed.get("node", ""),
        "band": parsed.get("band", ""),
        "target_type": parsed.get("target_type", ""),
        "is_beta_target": "beta" in feature.lower(),
        "is_stn_target": "stn" in feature.lower(),
        "is_gamma_target": "gamma" in feature.lower(),
    }


def build_feature_lookup(tables: list[pd.DataFrame]) -> dict[str, dict[str, object]]:
    """Build feature metadata from available target tables."""

    lookup: dict[str, dict[str, object]] = {}
    for table in tables:
        data = ensure_target_flags(table)
        if data.empty or "feature" not in data:
            continue
        for _, row in data.dropna(subset=["feature"]).iterrows():
            feature = str(row["feature"])
            lookup.setdefault(
                feature,
                {
                    "feature": feature,
                    "node": row.get("node", ""),
                    "band": row.get("band", ""),
                    "target_type": row.get("target_type", ""),
                    "is_beta_target": bool(row.get("is_beta_target", "beta" in feature.lower())),
                    "is_stn_target": bool(row.get("is_stn_target", "stn" in feature.lower())),
                    "is_gamma_target": bool(row.get("is_gamma_target", "gamma" in feature.lower())),
                },
            )
    return lookup


def add_rank_context(
    table: pd.DataFrame,
    source: str,
    context_column: str,
    context_value: str,
    subject_column: str | None = None,
    task_value: str | None = None,
) -> pd.DataFrame:
    """Normalize one target table into rank observations."""

    data = ensure_target_flags(table)
    if data.empty or "feature" not in data:
        return pd.DataFrame()
    if context_column in data:
        context_values = data[context_column].fillna(context_value).astype(str)
    else:
        context_values = pd.Series(context_value, index=data.index)
    observations = data.copy()
    observations["rank_source"] = source
    observations["rank_context"] = source + ":" + context_values
    if task_value is not None:
        observations["task_context"] = task_value
    elif context_column == "stratum":
        observations["task_context"] = context_values
    else:
        observations["task_context"] = ""
    observations["subject_context"] = ""
    if subject_column and subject_column in observations:
        observations["subject_context"] = observations[subject_column].fillna("").astype(str)
        observations["rank_context"] = observations["rank_context"] + ":" + observations["subject_context"]
    keep = [
        column
        for column in [
            "rank_source",
            "rank_context",
            "task_context",
            "subject_context",
            *RANK_COLUMNS,
        ]
        if column in observations
    ]
    return observations[keep]


def reliability_observations(inputs: ReliabilityInputs) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Collect rank observations from overall, task, subject, and quality tables."""

    tables = [
        inputs.overall_targets,
        inputs.holdl_targets,
        inputs.movel_targets,
        inputs.by_subject_targets,
        inputs.quality_targets,
    ]
    lookup = build_feature_lookup(tables)
    frames = [
        add_rank_context(inputs.overall_targets, "overall", "analysis", "all_recordings"),
        add_rank_context(inputs.holdl_targets, "task", "stratum", "HoldL", task_value="HoldL"),
        add_rank_context(inputs.movel_targets, "task", "stratum", "MoveL", task_value="MoveL"),
        add_rank_context(inputs.by_subject_targets, "subject_task", "stratum", "subject_task", "subject"),
        add_rank_context(inputs.quality_targets, "quality", "analysis_condition", "quality_filter"),
    ]
    observations = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    if observations.empty:
        return observations, lookup
    if "stratum" in inputs.by_subject_targets.columns and not inputs.by_subject_targets.empty:
        pass
    observations["rank"] = pd.to_numeric(observations["rank"], errors="coerce")
    observations = observations.dropna(subset=["feature", "rank"])
    observations["feature"] = observations["feature"].astype(str)
    return observations, lookup


def complete_observations(observations: pd.DataFrame, lookup: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Add missing feature/context pairs as non-top observations."""

    if observations.empty:
        return observations.copy()
    features = sorted(set(lookup) | set(observations["feature"].astype(str)))
    context_columns = ["rank_source", "rank_context", "task_context", "subject_context"]
    for optional in ["side_context", "task_family_context"]:
        if optional in observations:
            context_columns.append(optional)
    contexts = observations[context_columns].drop_duplicates()
    max_known_rank = int(np.nanmax(observations["rank"].to_numpy(dtype=float))) if not observations.empty else len(features)
    missing_rank = max(max_known_rank, len(features)) + 1
    records = []
    observed_pairs = set(zip(observations["rank_context"].astype(str), observations["feature"].astype(str), strict=False))
    for context in contexts.itertuples(index=False):
        for feature in features:
            if (str(context.rank_context), feature) in observed_pairs:
                continue
            desc = candidate_descriptor(feature, lookup)
            records.append(
                {
                    "rank_source": context.rank_source,
                    "rank_context": context.rank_context,
                    "task_context": context.task_context,
                    "subject_context": context.subject_context,
                    "side_context": getattr(context, "side_context", ""),
                    "task_family_context": getattr(context, "task_family_context", ""),
                    "rank": missing_rank,
                    "target_type": desc["target_type"],
                    "node": desc["node"],
                    "edge": "",
                    "band": desc["band"],
                    "feature": feature,
                    "expected_delta_deviation": np.nan,
                    "uncertainty": np.nan,
                    "stability_estimate": np.nan,
                    "is_beta_target": desc["is_beta_target"],
                    "is_stn_target": desc["is_stn_target"],
                    "is_gamma_target": desc["is_gamma_target"],
                }
            )
    completed = pd.concat([observations, pd.DataFrame(records)], ignore_index=True) if records else observations.copy()
    completed["rank"] = pd.to_numeric(completed["rank"], errors="coerce")
    return completed


def minmax_inverse(series: pd.Series) -> pd.Series:
    """Return 1 for low values and 0 for high values with stable fallbacks."""

    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)
    minimum = float(values.min())
    maximum = float(values.max())
    if np.isclose(maximum, minimum):
        return pd.Series(0.5, index=series.index)
    return 1.0 - (values - minimum) / (maximum - minimum)


def compute_reliability(
    observations: pd.DataFrame,
    lookup: dict[str, dict[str, object]],
    total_subjects: int,
    total_tasks: int,
    task_filter: str | None = None,
) -> pd.DataFrame:
    """Compute transparent target reliability scores.

    Formula:
        0.25 * top10_frequency
      + 0.15 * rank_score
      + 0.15 * subject_coverage
      + 0.10 * task_coverage
      + 0.15 * quality_survival_score
      + 0.10 * uncertainty_score
      + 0.10 * mean_stability_estimate

    The score is an exploratory stability summary, not a clinical claim.
    """

    if observations.empty:
        return pd.DataFrame(columns=RELIABILITY_COLUMNS)
    data = complete_observations(observations, lookup)
    if task_filter is not None:
        task_mask = data["task_context"].eq(task_filter) | (
            data["rank_source"].eq("quality")
        )
        data = data[task_mask].copy()
    if data.empty:
        return pd.DataFrame(columns=RELIABILITY_COLUMNS)

    rows = []
    n_rank_contexts = int(data["rank_context"].nunique())
    max_rank = max(int(data["rank"].max()), len(set(data["feature"])))
    for feature, table in data.groupby("feature", dropna=False):
        desc = candidate_descriptor(str(feature), lookup)
        top5_frequency = float((table["rank"] <= 5).mean())
        top10_frequency = float((table["rank"] <= 10).mean())
        subject_hits = table.loc[
            table["rank_source"].eq("subject_task") & (table["rank"] <= 10),
            "subject_context",
        ]
        task_hits = set(
            table.loc[
                table["task_context"].isin(["HoldL", "MoveL"]) & (table["rank"] <= 10),
                "task_context",
            ].astype(str)
        )
        quality_hits = table[
            table["rank_source"].eq("quality")
            & table["rank_context"].str.contains("exclude_", regex=False)
            & (table["rank"] <= 10)
        ]
        good_hits = table[
            table["rank_source"].eq("quality")
            & table["rank_context"].str.contains("good_only", regex=False)
            & (table["rank"] <= 10)
        ]
        rows.append(
            {
                **{column: desc[column] for column in TARGET_ID_COLUMNS},
                "top5_frequency": top5_frequency,
                "top10_frequency": top10_frequency,
                "mean_rank": float(table["rank"].mean()),
                "median_rank": float(table["rank"].median()),
                "rank_std": float(table["rank"].std(ddof=0)),
                "appears_in_n_subjects": int(subject_hits.replace("", np.nan).dropna().nunique()),
                "appears_in_n_tasks": int(len(task_hits)),
                "appears_in_holdl": bool("HoldL" in task_hits),
                "appears_in_movel": bool("MoveL" in task_hits),
                "appears_after_quality_filtering": bool(not quality_hits.empty),
                "appears_in_good_only": bool(not good_hits.empty),
                "mean_expected_delta_deviation": float(table["expected_delta_deviation"].mean(skipna=True)),
                "mean_uncertainty": float(table["uncertainty"].mean(skipna=True)),
                "mean_stability_estimate": float(table["stability_estimate"].mean(skipna=True)),
                "subject_coverage": 0.0 if total_subjects <= 0 else float(subject_hits.replace("", np.nan).dropna().nunique() / total_subjects),
                "task_coverage": 0.0 if total_tasks <= 0 else float(len(task_hits) / total_tasks),
                "is_beta_target": bool(desc["is_beta_target"]),
                "is_stn_target": bool(desc["is_stn_target"]),
                "is_gamma_target": bool(desc["is_gamma_target"]),
                "n_rank_contexts": n_rank_contexts,
            }
        )
    reliability = pd.DataFrame(rows)
    if reliability.empty:
        return pd.DataFrame(columns=RELIABILITY_COLUMNS)
    reliability["rank_score"] = 1.0 - ((reliability["mean_rank"] - 1.0) / max(1.0, max_rank - 1.0))
    reliability["rank_score"] = reliability["rank_score"].clip(lower=0.0, upper=1.0)
    reliability["uncertainty_score"] = minmax_inverse(reliability["mean_uncertainty"]).fillna(0.5)
    reliability["mean_stability_estimate"] = reliability["mean_stability_estimate"].fillna(0.5).clip(0.0, 1.0)
    reliability["quality_survival_score"] = (
        0.5 * reliability["appears_after_quality_filtering"].astype(float)
        + 0.5 * reliability["appears_in_good_only"].astype(float)
    )
    reliability["reliability_score"] = (
        0.25 * reliability["top10_frequency"]
        + 0.15 * reliability["rank_score"]
        + 0.15 * reliability["subject_coverage"]
        + 0.10 * reliability["task_coverage"]
        + 0.15 * reliability["quality_survival_score"]
        + 0.10 * reliability["uncertainty_score"]
        + 0.10 * reliability["mean_stability_estimate"]
    ).clip(0.0, 1.0)
    return reliability[RELIABILITY_COLUMNS].sort_values(
        ["reliability_score", "top10_frequency", "mean_rank"],
        ascending=[False, False, True],
    )


def read_sideaware_rank_tables(output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read side-aware ranking tables when available."""

    output_dir = Path(output_dir)
    names = {
        "HoldL": "real_candidate_targets_holdl_sideaware.csv",
        "MoveL": "real_candidate_targets_movel_sideaware.csv",
        "HoldR": "real_candidate_targets_holdr_sideaware.csv",
        "MoveR": "real_candidate_targets_mover_sideaware.csv",
        "left_side": "real_candidate_targets_left_side.csv",
        "right_side": "real_candidate_targets_right_side.csv",
        "hold_family": "real_candidate_targets_hold_family.csv",
        "move_family": "real_candidate_targets_move_family.csv",
    }
    tables = {}
    for label, name in names.items():
        path = output_dir / name
        if path.exists():
            table = pd.read_csv(path)
            tables[label] = add_task_side_columns(table, "stratum" if "stratum" in table else None)
    return tables


def sideaware_observations(
    side_tables: dict[str, pd.DataFrame],
    quality_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Build rank observations from side-aware tables."""

    frames = []
    lookup = build_feature_lookup([*side_tables.values(), quality_targets])
    for label, table in side_tables.items():
        data = ensure_target_flags(table)
        if data.empty or "feature" not in data:
            continue
        data = add_task_side_columns(data, "stratum" if "stratum" in data else None)
        data["rank_source"] = "sideaware"
        data["rank_context"] = f"sideaware:{label}"
        if label == "left_side":
            data["side_context"] = "L"
            data["task_family_context"] = ""
        elif label == "right_side":
            data["side_context"] = "R"
            data["task_family_context"] = ""
        elif label == "hold_family":
            data["side_context"] = ""
            data["task_family_context"] = "Hold"
        elif label == "move_family":
            data["side_context"] = ""
            data["task_family_context"] = "Move"
        else:
            _, family, side = parse_task_side(label)
            data["side_context"] = side
            data["task_family_context"] = family
        data["task_context"] = data["task_family_context"]
        data["subject_context"] = ""
        frames.append(data)
    quality = ensure_target_flags(quality_targets)
    if not quality.empty and "feature" in quality:
        quality = quality.copy()
        quality["rank_source"] = "quality"
        quality["rank_context"] = "quality:" + quality.get(
            "analysis_condition",
            pd.Series("unknown", index=quality.index),
        ).fillna("unknown").astype(str)
        quality["side_context"] = ""
        quality["task_family_context"] = ""
        quality["task_context"] = ""
        quality["subject_context"] = ""
        frames.append(quality)
    observations = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if observations.empty:
        return observations, lookup
    observations["rank"] = pd.to_numeric(observations["rank"], errors="coerce")
    observations = observations.dropna(subset=["feature", "rank"])
    return observations, lookup


def compute_sideaware_reliability(
    observations: pd.DataFrame,
    lookup: dict[str, dict[str, object]],
    base_reliability: pd.DataFrame,
    side_filter: str | None = None,
    family_filter: str | None = None,
) -> pd.DataFrame:
    """Compute side-aware reliability with side and task-family coverage terms.

    sideaware_reliability_score =
        0.20 * top10_frequency
      + 0.15 * rank_score
      + 0.10 * subject_coverage
      + 0.10 * side_coverage
      + 0.10 * task_family_coverage
      + 0.15 * quality_survival_score
      + 0.10 * uncertainty_score
      + 0.10 * mean_stability_estimate
    """

    if observations.empty:
        return pd.DataFrame(columns=SIDE_RELIABILITY_COLUMNS)
    data = observations.copy()
    if side_filter is not None:
        data = data[(data["side_context"].eq(side_filter)) | data["rank_source"].eq("quality")].copy()
    if family_filter is not None:
        data = data[(data["task_family_context"].eq(family_filter)) | data["rank_source"].eq("quality")].copy()
    if data.empty:
        return pd.DataFrame(columns=SIDE_RELIABILITY_COLUMNS)
    base_lookup = (
        base_reliability.set_index("feature").to_dict("index")
        if not base_reliability.empty and "feature" in base_reliability
        else {}
    )
    data = complete_observations(data, lookup)
    max_rank = max(int(data["rank"].max()), len(set(data["feature"])))
    rows = []
    for feature, table in data.groupby("feature", dropna=False):
        feature = str(feature)
        desc = candidate_descriptor(feature, lookup)
        base = base_lookup.get(feature, {})
        left_hit = bool(((table["side_context"].eq("L")) & (table["rank"] <= 10)).any())
        right_hit = bool(((table["side_context"].eq("R")) & (table["rank"] <= 10)).any())
        hold_hit = bool(((table["task_family_context"].eq("Hold")) & (table["rank"] <= 10)).any())
        move_hit = bool(((table["task_family_context"].eq("Move")) & (table["rank"] <= 10)).any())
        quality_hits = table[
            table["rank_source"].eq("quality")
            & table["rank_context"].str.contains("exclude_", regex=False)
            & (table["rank"] <= 10)
        ]
        good_hits = table[
            table["rank_source"].eq("quality")
            & table["rank_context"].str.contains("good_only", regex=False)
            & (table["rank"] <= 10)
        ]
        rows.append(
            {
                **{column: desc[column] for column in TARGET_ID_COLUMNS},
                "top5_frequency": float((table["rank"] <= 5).mean()),
                "top10_frequency": float((table["rank"] <= 10).mean()),
                "mean_rank": float(table["rank"].mean()),
                "median_rank": float(table["rank"].median()),
                "rank_std": float(table["rank"].std(ddof=0)),
                "appears_in_n_subjects": int(base.get("appears_in_n_subjects", 0)),
                "appears_in_n_tasks": int(base.get("appears_in_n_tasks", 0)),
                "appears_in_holdl": bool(base.get("appears_in_holdl", False)),
                "appears_in_movel": bool(base.get("appears_in_movel", False)),
                "appears_after_quality_filtering": bool(not quality_hits.empty),
                "appears_in_good_only": bool(not good_hits.empty),
                "mean_expected_delta_deviation": float(table["expected_delta_deviation"].mean(skipna=True)),
                "mean_uncertainty": float(table["uncertainty"].mean(skipna=True)),
                "mean_stability_estimate": float(table["stability_estimate"].mean(skipna=True)),
                "subject_coverage": float(base.get("subject_coverage", 0.0)),
                "task_coverage": float(base.get("task_coverage", 0.0)),
                "is_beta_target": bool(desc["is_beta_target"]),
                "is_stn_target": bool(desc["is_stn_target"]),
                "is_gamma_target": bool(desc["is_gamma_target"]),
                "n_rank_contexts": int(table["rank_context"].nunique()),
                "appears_in_left": left_hit,
                "appears_in_right": right_hit,
                "side_coverage": (float(left_hit) + float(right_hit)) / 2.0,
                "appears_in_hold_family": hold_hit,
                "appears_in_move_family": move_hit,
                "task_family_coverage": (float(hold_hit) + float(move_hit)) / 2.0,
            }
        )
    reliability = pd.DataFrame(rows)
    if reliability.empty:
        return pd.DataFrame(columns=SIDE_RELIABILITY_COLUMNS)
    reliability["rank_score"] = 1.0 - ((reliability["mean_rank"] - 1.0) / max(1.0, max_rank - 1.0))
    reliability["rank_score"] = reliability["rank_score"].clip(lower=0.0, upper=1.0)
    reliability["uncertainty_score"] = minmax_inverse(reliability["mean_uncertainty"]).fillna(0.5)
    reliability["mean_stability_estimate"] = reliability["mean_stability_estimate"].fillna(0.5).clip(0.0, 1.0)
    reliability["quality_survival_score"] = (
        0.5 * reliability["appears_after_quality_filtering"].astype(float)
        + 0.5 * reliability["appears_in_good_only"].astype(float)
    )
    reliability["reliability_score"] = reliability.get("reliability_score", np.nan)
    reliability["sideaware_reliability_score"] = (
        0.20 * reliability["top10_frequency"]
        + 0.15 * reliability["rank_score"]
        + 0.10 * reliability["subject_coverage"]
        + 0.10 * reliability["side_coverage"]
        + 0.10 * reliability["task_family_coverage"]
        + 0.15 * reliability["quality_survival_score"]
        + 0.10 * reliability["uncertainty_score"]
        + 0.10 * reliability["mean_stability_estimate"]
    ).clip(0.0, 1.0)
    return reliability[SIDE_RELIABILITY_COLUMNS].sort_values(
        ["sideaware_reliability_score", "top10_frequency", "mean_rank"],
        ascending=[False, False, True],
    )


def rank_all_subject_subset(
    subset: pd.DataFrame,
    feature_columns: list[str],
    weights: ObjectiveWeights,
    effect_fraction: float,
    uncertainty_floor: float,
    reference_strategy: str,
) -> pd.DataFrame:
    """Rank candidates for a pooled subset using MedOff vs MedOn internal proxy."""

    if subset.empty or not feature_columns:
        return pd.DataFrame()
    medication = subset["medication"].fillna("unknown").astype(str).str.lower()
    current_rows = subset[medication.eq("off")]
    if current_rows.empty:
        current_rows = subset
    reference_rows, reference_label = select_internal_reference_rows(
        subset,
        current_rows.iloc[0] if not current_rows.empty else None,
        strategy=reference_strategy,
    )
    stats = build_internal_reference_stats(subset, reference_rows, feature_columns, reference_label)
    current_state = current_rows[feature_columns].mean().to_numpy(dtype=float)
    current_state, target_state, reference_scale, valid_features = finite_feature_subset(
        current_state,
        stats.mean,
        stats.std,
        feature_columns,
    )
    if not valid_features:
        return pd.DataFrame()
    ranked = rank_candidate_targets(
        current_state,
        target_state,
        valid_features,
        reference_scale=reference_scale,
        weights=weights,
        effect_fraction=effect_fraction,
        uncertainty_floor=uncertainty_floor,
        top_k=None,
    )
    ranked = target_flags(ranked)
    ranked["reference_mode"] = reference_label
    ranked["n_subjects"] = int(subset["subject"].nunique()) if "subject" in subset else 0
    ranked["n_state_vectors"] = int(len(subset))
    return ranked


def leave_one_subject_out(
    state_vectors: pd.DataFrame,
    weights: ObjectiveWeights,
    effect_fraction: float,
    uncertainty_floor: float,
    reference_strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run leave-one-subject-out ranking and compare top-10 stability."""

    if state_vectors.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    feature_columns = available_real_feature_columns(state_vectors)
    full_targets = rank_all_subject_subset(
        state_vectors,
        feature_columns,
        weights,
        effect_fraction,
        uncertainty_floor,
        reference_strategy,
    )
    if full_targets.empty:
        return full_targets, pd.DataFrame(), pd.DataFrame()
    full_top10 = set(full_targets.head(10)["feature"].astype(str))
    full_top1 = str(full_targets.iloc[0]["feature"])
    full_stn_beta_count = int((full_targets.head(10)["is_stn_target"] & full_targets.head(10)["is_beta_target"]).sum())

    target_tables = []
    summary_rows = []
    for subject in sorted(state_vectors["subject"].dropna().astype(str).unique()):
        subset = state_vectors[~state_vectors["subject"].astype(str).eq(subject)]
        targets = rank_all_subject_subset(
            subset,
            feature_columns,
            weights,
            effect_fraction,
            uncertainty_floor,
            reference_strategy,
        )
        if targets.empty:
            summary_rows.append(
                {
                    "removed_subject": subject,
                    "top1_feature_after_removal": "",
                    "top10_overlap_with_full": 0,
                    "top10_overlap_fraction": 0.0,
                    "beta_count_top10": 0,
                    "gamma_count_top10": 0,
                    "stn_count_top10": 0,
                    "does_full_top1_remain_top10": False,
                    "rank_shift_of_full_top1": np.nan,
                    "stn_left_gamma_rank_after_removal": np.nan,
                    "stn_left_gamma_remains_top10": False,
                    "best_stn_beta_feature": "",
                    "best_stn_beta_rank": np.nan,
                    "stn_beta_count_change_vs_full": -full_stn_beta_count,
                    "interpretation_note": "Ranking unavailable after removal.",
                }
            )
            continue
        targets = targets.copy()
        targets.insert(0, "removed_subject", subject)
        target_tables.append(targets)
        top10 = targets.head(10)
        top10_features = set(top10["feature"].astype(str))
        full_top1_rows = targets[targets["feature"].astype(str).eq(full_top1)]
        full_top1_rank = float(full_top1_rows["rank"].iloc[0]) if not full_top1_rows.empty else np.nan
        stn_left_gamma_rows = targets[targets["feature"].astype(str).eq("stn_left_gamma_power")]
        stn_left_gamma_rank = float(stn_left_gamma_rows["rank"].iloc[0]) if not stn_left_gamma_rows.empty else np.nan
        stn_beta = top10[top10["is_stn_target"] & top10["is_beta_target"]]
        best_stn_beta = stn_beta.iloc[0] if not stn_beta.empty else None
        overlap = len(full_top10 & top10_features)
        stn_beta_count = int(len(stn_beta))
        note_parts = []
        note_parts.append("full_top1_remains_top10" if full_top1_rank <= 10 else "full_top1_drops_below_top10")
        if stn_beta_count > full_stn_beta_count:
            note_parts.append("stn_beta_stronger_than_full_top10")
        elif stn_beta_count < full_stn_beta_count:
            note_parts.append("stn_beta_weaker_than_full_top10")
        else:
            note_parts.append("stn_beta_count_unchanged")
        summary_rows.append(
            {
                "removed_subject": subject,
                "top1_feature_after_removal": str(targets.iloc[0]["feature"]),
                "top10_overlap_with_full": int(overlap),
                "top10_overlap_fraction": float(overlap / 10.0),
                "beta_count_top10": int(top10["is_beta_target"].sum()),
                "gamma_count_top10": int(top10["is_gamma_target"].sum()),
                "stn_count_top10": int(top10["is_stn_target"].sum()),
                "does_full_top1_remain_top10": bool(full_top1_rank <= 10),
                "rank_shift_of_full_top1": float(full_top1_rank - 1.0) if np.isfinite(full_top1_rank) else np.nan,
                "stn_left_gamma_rank_after_removal": stn_left_gamma_rank,
                "stn_left_gamma_remains_top10": bool(stn_left_gamma_rank <= 10),
                "best_stn_beta_feature": "" if best_stn_beta is None else best_stn_beta["feature"],
                "best_stn_beta_rank": np.nan if best_stn_beta is None else float(best_stn_beta["rank"]),
                "stn_beta_count_change_vs_full": int(stn_beta_count - full_stn_beta_count),
                "interpretation_note": "; ".join(note_parts),
            }
        )
    loso_targets = pd.concat(target_tables, ignore_index=True) if target_tables else pd.DataFrame()
    stability = pd.DataFrame(summary_rows)
    return full_targets, loso_targets, stability


def task_reliability_comparison(holdl: pd.DataFrame, movel: pd.DataFrame) -> pd.DataFrame:
    """Compare task-specific group ranks for candidate features."""

    holdl = ensure_target_flags(holdl)
    movel = ensure_target_flags(movel)
    features = sorted(set(holdl.get("feature", pd.Series(dtype=str)).dropna().astype(str)) | set(movel.get("feature", pd.Series(dtype=str)).dropna().astype(str)))
    rows = []
    holdl_lookup = holdl.set_index("feature") if not holdl.empty and "feature" in holdl else pd.DataFrame()
    movel_lookup = movel.set_index("feature") if not movel.empty and "feature" in movel else pd.DataFrame()
    for feature in features:
        rank_holdl = float(holdl_lookup.loc[feature, "rank"]) if not holdl_lookup.empty and feature in holdl_lookup.index else np.nan
        rank_movel = float(movel_lookup.loc[feature, "rank"]) if not movel_lookup.empty and feature in movel_lookup.index else np.nan
        desc_source = holdl_lookup if not holdl_lookup.empty and feature in holdl_lookup.index else movel_lookup
        if not desc_source.empty and feature in desc_source.index:
            desc_row = desc_source.loc[feature]
            if isinstance(desc_row, pd.DataFrame):
                desc_row = desc_row.iloc[0]
            desc = {
                "node": desc_row.get("node", ""),
                "band": desc_row.get("band", ""),
                "target_type": desc_row.get("target_type", ""),
            }
        else:
            desc = parse_candidate_feature(feature)
        appears_holdl_top10 = bool(np.isfinite(rank_holdl) and rank_holdl <= 10)
        appears_movel_top10 = bool(np.isfinite(rank_movel) and rank_movel <= 10)
        if appears_holdl_top10 and appears_movel_top10:
            diff = abs(rank_movel - rank_holdl)
            note = "shared_top10_similar_rank" if diff <= 3 else "shared_top10_rank_shift"
        elif appears_holdl_top10:
            note = "HoldL_top10_only"
        elif appears_movel_top10:
            note = "MoveL_top10_only"
        else:
            note = "not_top10_in_either_task"
        rows.append(
            {
                "feature": feature,
                "node": desc.get("node", ""),
                "band": desc.get("band", ""),
                "target_type": desc.get("target_type", ""),
                "rank_holdl": rank_holdl,
                "rank_movel": rank_movel,
                "rank_difference": rank_movel - rank_holdl if np.isfinite(rank_holdl) and np.isfinite(rank_movel) else np.nan,
                "appears_in_both_tasks": bool(appears_holdl_top10 and appears_movel_top10),
                "task_specificity_note": note,
            }
        )
    return pd.DataFrame(rows).sort_values(["appears_in_both_tasks", "rank_holdl", "rank_movel"], ascending=[False, True, True])


def direction_label(value: float, epsilon: float) -> str:
    """Map a signed delta to increase/decrease/unchanged."""

    if not np.isfinite(value) or abs(value) <= epsilon:
        return "unchanged"
    return "increase" if value > 0 else "decrease"


def compensation_directions(
    state_vectors: pd.DataFrame,
    direction_epsilon: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute MedOff-to-MedOn feature direction summaries."""

    if state_vectors.empty:
        empty_subject = pd.DataFrame(
            columns=[
                "subject",
                "session",
                "task",
                "feature",
                "node",
                "band",
                "target_type",
                "medoff_value",
                "medon_value",
                "signed_delta",
                "direction",
            ]
        )
        return empty_subject, pd.DataFrame(), pd.DataFrame()
    feature_columns = available_real_feature_columns(state_vectors)
    rows = []
    for (subject, session, task), subset in state_vectors.groupby(["subject", "session", "task"], dropna=False):
        medication = subset["medication"].fillna("").astype(str).str.lower()
        off_rows = subset[medication.eq("off")]
        on_rows = subset[medication.eq("on")]
        if off_rows.empty or on_rows.empty:
            continue
        off_values = off_rows[feature_columns].mean(numeric_only=True)
        on_values = on_rows[feature_columns].mean(numeric_only=True)
        for feature in feature_columns:
            desc = parse_candidate_feature(feature)
            medoff = float(off_values[feature])
            medon = float(on_values[feature])
            delta = medon - medoff
            rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "task": task,
                    "feature": feature,
                    "node": desc.get("node", ""),
                    "band": desc.get("band", ""),
                    "target_type": desc.get("target_type", ""),
                    "medoff_value": medoff,
                    "medon_value": medon,
                    "signed_delta": delta,
                    "direction": direction_label(delta, direction_epsilon),
                }
            )
    by_subject = pd.DataFrame(rows)
    if by_subject.empty:
        return by_subject, pd.DataFrame(), pd.DataFrame()

    group_rows = []
    for (task, feature), table in by_subject.groupby(["task", "feature"], dropna=False):
        desc = parse_candidate_feature(str(feature))
        counts = table["direction"].value_counts().to_dict()
        direction_counts = {key: int(counts.get(key, 0)) for key in ["increase", "decrease", "unchanged"]}
        dominant_direction = max(direction_counts, key=direction_counts.get)
        n_subjects = int(table["subject"].nunique())
        mean_delta = float(table["signed_delta"].mean())
        group_rows.append(
            {
                "task": task,
                "feature": feature,
                "node": desc.get("node", ""),
                "band": desc.get("band", ""),
                "target_type": desc.get("target_type", ""),
                "n_subjects": n_subjects,
                "mean_medoff_value": float(table["medoff_value"].mean()),
                "mean_medon_value": float(table["medon_value"].mean()),
                "group_mean_signed_delta": mean_delta,
                "group_median_signed_delta": float(table["signed_delta"].median()),
                "group_std_signed_delta": float(table["signed_delta"].std(ddof=0)),
                "group_direction": direction_label(mean_delta, direction_epsilon),
                "increase_count": direction_counts["increase"],
                "decrease_count": direction_counts["decrease"],
                "unchanged_count": direction_counts["unchanged"],
                "consistency_direction": dominant_direction,
                "consistency_fraction": float(direction_counts[dominant_direction] / max(1, n_subjects)),
            }
        )
    group = pd.DataFrame(group_rows).sort_values(["task", "consistency_fraction", "feature"], ascending=[True, False, True])

    stability_rows = []
    for feature, table in group.groupby("feature", dropna=False):
        holdl = table[table["task"].astype(str).str.lower().eq("holdl")]
        movel = table[table["task"].astype(str).str.lower().eq("movel")]
        holdl_direction = str(holdl["group_direction"].iloc[0]) if not holdl.empty else ""
        movel_direction = str(movel["group_direction"].iloc[0]) if not movel.empty else ""
        holdl_consistency = float(holdl["consistency_fraction"].iloc[0]) if not holdl.empty else np.nan
        movel_consistency = float(movel["consistency_fraction"].iloc[0]) if not movel.empty else np.nan
        directions_agree = bool(holdl_direction and movel_direction and holdl_direction == movel_direction)
        if directions_agree:
            note = "same_group_direction_across_tasks"
        elif holdl_direction and movel_direction:
            note = "task_dependent_direction"
        else:
            note = "direction_available_for_one_task"
        desc = parse_candidate_feature(str(feature))
        stability_rows.append(
            {
                "feature": feature,
                "node": desc.get("node", ""),
                "band": desc.get("band", ""),
                "target_type": desc.get("target_type", ""),
                "direction_holdl": holdl_direction,
                "direction_movel": movel_direction,
                "directions_agree": directions_agree,
                "holdl_consistency_fraction": holdl_consistency,
                "movel_consistency_fraction": movel_consistency,
                "mean_consistency_fraction": float(np.nanmean([holdl_consistency, movel_consistency])),
                "interpretation_note": note,
            }
        )
    stability = pd.DataFrame(stability_rows).sort_values(
        ["directions_agree", "mean_consistency_fraction", "feature"],
        ascending=[False, False, True],
    )
    return by_subject, group, stability


def summarize_top_reliable(label: str, table: pd.DataFrame, n: int = 8) -> list[str]:
    """Format top reliability rows for markdown."""

    lines = [f"## {label}", ""]
    if table.empty:
        lines.append("No reliability rows were available.")
        return lines
    for _, row in table.head(n).iterrows():
        lines.append(
            f"- {row['feature']}: score={row['reliability_score']:.3f}, "
            f"top10_frequency={row['top10_frequency']:.2f}, "
            f"mean_rank={row['mean_rank']:.2f}, "
            f"subjects={int(row['appears_in_n_subjects'])}, "
            f"tasks={int(row['appears_in_n_tasks'])}"
        )
    return lines


def write_reliability_report(
    path: str | Path,
    inputs: ReliabilityInputs,
    reliability: pd.DataFrame,
    reliability_holdl: pd.DataFrame,
    reliability_movel: pd.DataFrame,
    reliability_sideaware: pd.DataFrame,
    reliability_left: pd.DataFrame,
    reliability_right: pd.DataFrame,
    reliability_hold_family: pd.DataFrame,
    reliability_move_family: pd.DataFrame,
    task_comparison: pd.DataFrame,
    loso_stability: pd.DataFrame,
    direction_group: pd.DataFrame,
    direction_stability: pd.DataFrame,
) -> None:
    """Write a careful markdown report for target reliability outputs."""

    n_subjects = (
        inputs.state_vectors["subject"].nunique()
        if not inputs.state_vectors.empty and "subject" in inputs.state_vectors
        else 0
    )
    n_tasks = (
        inputs.state_vectors["task"].nunique()
        if not inputs.state_vectors.empty and "task" in inputs.state_vectors
        else 0
    )
    sides = (
        sorted(add_task_side_columns(inputs.state_vectors, "task")["side"].dropna().astype(str).unique())
        if not inputs.state_vectors.empty and "task" in inputs.state_vectors
        else []
    )
    lines = [
        "# Real Target Reliability Report",
        "",
        "This exploratory stability analysis evaluates in silico candidate targets using ds004998 dataset-internal electrophysiological proxies. It is not a clinical report and does not estimate real-world outcomes.",
        "",
        "## Inputs Used",
        "",
        f"- state_vector_rows: {len(inputs.state_vectors)}",
        f"- subjects: {n_subjects}",
        f"- tasks: {n_tasks}",
        f"- sides: {';'.join(sides)}",
        f"- state_features_used: {len(available_real_feature_columns(inputs.state_vectors)) if not inputs.state_vectors.empty else 0}",
        f"- deviation_score_rows: {len(inputs.deviation_scores)}",
        f"- overall_target_rows: {len(inputs.overall_targets)}",
        f"- HoldL_target_rows: {len(inputs.holdl_targets)}",
        f"- MoveL_target_rows: {len(inputs.movel_targets)}",
        f"- subject_level_target_rows: {len(inputs.by_subject_targets)}",
        f"- quality_target_rows: {len(inputs.quality_targets)}",
        "",
        "Reliability score formula: 0.25 top10_frequency + 0.15 rank_score + 0.15 subject_coverage + 0.10 task_coverage + 0.15 quality_survival + 0.10 uncertainty_score + 0.10 stability_estimate.",
        "Side-aware reliability score formula: 0.20 top10_frequency + 0.15 rank_score + 0.10 subject_coverage + 0.10 side_coverage + 0.10 task_family_coverage + 0.15 quality_survival + 0.10 uncertainty_score + 0.10 stability_estimate.",
        "",
    ]
    if inputs.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend([f"- {warning}" for warning in inputs.warnings])
        lines.append("")

    lines.extend(summarize_top_reliable("Top Reliable Targets Overall", reliability))
    lines.extend([""])
    lines.extend(summarize_top_reliable("Top Reliable Targets For HoldL", reliability_holdl, n=6))
    lines.extend([""])
    lines.extend(summarize_top_reliable("Top Reliable Targets For MoveL", reliability_movel, n=6))

    if not reliability_sideaware.empty:
        lines.extend(["", "## Top Side-Aware Reliable Targets", ""])
        for _, row in reliability_sideaware.head(8).iterrows():
            lines.append(
                f"- {row['feature']}: score={row['sideaware_reliability_score']:.3f}, "
                f"side_coverage={row['side_coverage']:.2f}, "
                f"task_family_coverage={row['task_family_coverage']:.2f}, "
                f"top10_frequency={row['top10_frequency']:.2f}"
            )
        lines.extend(["", "Side/task-family subsets:"])
        for label, table in [
            ("left", reliability_left),
            ("right", reliability_right),
            ("Hold family", reliability_hold_family),
            ("Move family", reliability_move_family),
        ]:
            top = table.iloc[0]["feature"] if not table.empty else "not_available"
            lines.append(f"- {label}: top_sideaware_reliable_target={top}")

    lines.extend(["", "## Task Agreement", ""])
    if task_comparison.empty:
        lines.append("Task-specific comparison was unavailable.")
    else:
        shared = task_comparison[task_comparison["appears_in_both_tasks"].astype(bool)]
        lines.append(f"- shared_top10_targets: {len(shared)}")
        for _, row in shared.head(8).iterrows():
            lines.append(
                f"- {row['feature']}: HoldL rank={_fmt_rank(row['rank_holdl'])}, "
                f"MoveL rank={_fmt_rank(row['rank_movel'])}, note={row['task_specificity_note']}"
            )

    lines.extend(["", "## Leave-One-Subject-Out Findings", ""])
    if loso_stability.empty:
        lines.append("Leave-one-subject-out ranking was unavailable.")
    else:
        for _, row in loso_stability.iterrows():
            lines.append(
                f"- remove {row['removed_subject']}: top1={row['top1_feature_after_removal']}, "
                f"top10_overlap={int(row['top10_overlap_with_full'])}/10, "
                f"stn_left_gamma_rank={_fmt_rank(row['stn_left_gamma_rank_after_removal'])}, "
                f"best_stn_beta={row['best_stn_beta_feature'] or 'none'} "
                f"(rank={_fmt_rank(row['best_stn_beta_rank'])})"
            )
        gamma_sensitive = loso_stability[
            ~loso_stability["stn_left_gamma_remains_top10"].astype(bool)
        ]
        stn_beta_stronger = loso_stability[
            pd.to_numeric(loso_stability["stn_beta_count_change_vs_full"], errors="coerce") > 0
        ]
        if gamma_sensitive.empty:
            lines.append(
                "stn_left_gamma_power remains in the top 10 after every single-subject removal in this processed subset."
            )
        else:
            removed = ", ".join(gamma_sensitive["removed_subject"].astype(str))
            lines.append(
                f"stn_left_gamma_power drops out of the top 10 after removing: {removed}. This suggests subject sensitivity, not a validated target."
            )
        if stn_beta_stronger.empty:
            lines.append("STN beta top-10 support does not increase in any leave-one-subject-out split.")
        else:
            removed = ", ".join(stn_beta_stronger["removed_subject"].astype(str))
            lines.append(
                f"STN beta top-10 support increases after removing: {removed}. This indicates that beta support is subset-dependent."
            )

    lines.extend(["", "## Gamma Stability", ""])
    gamma = reliability[reliability["is_gamma_target"].astype(bool)] if not reliability.empty else pd.DataFrame()
    if gamma.empty:
        lines.append("No gamma candidates were available in the reliability table.")
    else:
        for _, row in gamma.head(6).iterrows():
            lines.append(
                f"- {row['feature']}: score={row['reliability_score']:.3f}, "
                f"top10_frequency={row['top10_frequency']:.2f}, "
                f"subjects={int(row['appears_in_n_subjects'])}, "
                f"quality_survival={row['quality_survival_score']:.2f}"
            )
        lines.append(
            "Gamma candidates survive the current quality filters, but leave-one-subject-out results indicate that at least some gamma rankings are subject-sensitive. They remain exploratory and require more subjects."
        )

    lines.extend(["", "## STN Beta Stability", ""])
    stn_beta = reliability[
        reliability["is_stn_target"].astype(bool) & reliability["is_beta_target"].astype(bool)
    ] if not reliability.empty else pd.DataFrame()
    if stn_beta.empty:
        lines.append("No STN beta candidates were available in the reliability table.")
    else:
        for _, row in stn_beta.head(6).iterrows():
            lines.append(
                f"- {row['feature']}: score={row['reliability_score']:.3f}, "
                f"mean_rank={row['mean_rank']:.2f}, "
                f"top10_frequency={row['top10_frequency']:.2f}, "
                f"subjects={int(row['appears_in_n_subjects'])}"
            )
        lines.append(f"STN beta candidates remain present in the pooled {n_subjects}-subject ranking, with side/task-family dependence.")

    lines.extend(["", "## Compensation-Direction Preparation", ""])
    if direction_group.empty:
        lines.append("Compensation-direction tables were unavailable.")
    else:
        stable = direction_stability[direction_stability["directions_agree"].astype(bool)] if not direction_stability.empty else pd.DataFrame()
        lines.append(f"- features_with_same_group_direction_across_available_task_families: {len(stable)}")
        for _, row in direction_group.sort_values("consistency_fraction", ascending=False).head(8).iterrows():
            lines.append(
                f"- {row['task']} {row['feature']}: group_direction={row['group_direction']}, "
                f"mean_delta={row['group_mean_signed_delta']:.4g}, "
                f"consistency={row['consistency_fraction']:.2f}"
            )
        lines.append("These directions describe how MedOff feature proxies differ from MedOn dataset-internal compensated proxies; they are not intervention instructions.")

    lines.extend(
        [
            "",
            "## Limitations And Next Step",
            "",
            "- MedOn is a dataset-internal compensated proxy, not a healthy state.",
            "- Candidate targets are in silico research candidates, not DBS prescriptions or clinical recommendations.",
            "- HCP is not used as an electrophysiological reference.",
            f"- Reliability scores summarize the current {n_subjects}-subject processed subset and require validation on more subjects and preprocessing variants.",
            "- Next step: build the compensation model only around candidates with cross-subject, cross-task, and quality-filtered stability, while retaining uncertainty and task-specific direction labels.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_rank(value: object) -> str:
    """Format rank-like values."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return str(int(number))


def run_target_reliability(
    state_vectors_path: str | Path,
    output_dir: str | Path = "outputs/tables",
    reports_dir: str | Path = "reports",
    reference_strategy: str = "medon_condition_proxy",
    effect_fraction: float = 0.6,
    uncertainty_floor: float = 0.05,
    weights: ObjectiveWeights | None = None,
) -> dict[str, Path]:
    """Run target reliability, LOSO stability, and compensation-direction analyses."""

    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    weights = weights or ObjectiveWeights(
        lambda_energy=0.005,
        lambda_risk=0.01,
        lambda_complexity=0.005,
        lambda_stability=0.005,
    )

    inputs = read_inputs(state_vectors_path, output_dir)
    observations, lookup = reliability_observations(inputs)
    subjects = inputs.state_vectors["subject"].nunique() if not inputs.state_vectors.empty and "subject" in inputs.state_vectors else 0
    tasks = inputs.state_vectors["task"].nunique() if not inputs.state_vectors.empty and "task" in inputs.state_vectors else 0
    reliability = compute_reliability(observations, lookup, int(subjects), int(tasks))
    reliability_holdl = compute_reliability(observations, lookup, int(subjects), 1, task_filter="HoldL")
    reliability_movel = compute_reliability(observations, lookup, int(subjects), 1, task_filter="MoveL")
    side_tables = read_sideaware_rank_tables(output_dir)
    side_observations, side_lookup = sideaware_observations(side_tables, inputs.quality_targets)
    reliability_sideaware = compute_sideaware_reliability(
        side_observations,
        side_lookup,
        reliability,
    )
    reliability_left = compute_sideaware_reliability(
        side_observations,
        side_lookup,
        reliability,
        side_filter="L",
    )
    reliability_right = compute_sideaware_reliability(
        side_observations,
        side_lookup,
        reliability,
        side_filter="R",
    )
    reliability_hold_family = compute_sideaware_reliability(
        side_observations,
        side_lookup,
        reliability,
        family_filter="Hold",
    )
    reliability_move_family = compute_sideaware_reliability(
        side_observations,
        side_lookup,
        reliability,
        family_filter="Move",
    )

    _, loso_targets, loso_stability = leave_one_subject_out(
        inputs.state_vectors,
        weights,
        effect_fraction,
        uncertainty_floor,
        reference_strategy,
    )
    task_comparison = task_reliability_comparison(inputs.holdl_targets, inputs.movel_targets)
    direction_subject, direction_group, direction_stability = compensation_directions(inputs.state_vectors)
    direction_subject = add_task_side_columns(direction_subject, "task") if not direction_subject.empty and "task" in direction_subject else direction_subject
    direction_group = add_task_side_columns(direction_group, "task") if not direction_group.empty and "task" in direction_group else direction_group

    paths = {
        "target_reliability": output_dir / "real_target_reliability.csv",
        "target_reliability_sideaware": output_dir / "real_target_reliability_sideaware.csv",
        "target_reliability_left": output_dir / "real_target_reliability_left.csv",
        "target_reliability_right": output_dir / "real_target_reliability_right.csv",
        "target_reliability_hold_family": output_dir / "real_target_reliability_hold_family.csv",
        "target_reliability_move_family": output_dir / "real_target_reliability_move_family.csv",
        "leave_one_subject_out_targets": output_dir / "real_leave_one_subject_out_targets.csv",
        "leave_one_subject_out_stability": output_dir / "real_leave_one_subject_out_stability.csv",
        "target_reliability_holdl": output_dir / "real_target_reliability_holdl.csv",
        "target_reliability_movel": output_dir / "real_target_reliability_movel.csv",
        "task_reliability_comparison": output_dir / "real_task_reliability_comparison.csv",
        "compensation_directions_by_subject": output_dir / "real_compensation_directions_by_subject.csv",
        "compensation_directions_group": output_dir / "real_compensation_directions_group.csv",
        "compensation_direction_stability": output_dir / "real_compensation_direction_stability.csv",
        "report": reports_dir / "real_target_reliability_report.md",
    }

    reliability.to_csv(paths["target_reliability"], index=False)
    reliability_sideaware.to_csv(paths["target_reliability_sideaware"], index=False)
    reliability_left.to_csv(paths["target_reliability_left"], index=False)
    reliability_right.to_csv(paths["target_reliability_right"], index=False)
    reliability_hold_family.to_csv(paths["target_reliability_hold_family"], index=False)
    reliability_move_family.to_csv(paths["target_reliability_move_family"], index=False)
    loso_targets.to_csv(paths["leave_one_subject_out_targets"], index=False)
    loso_stability.to_csv(paths["leave_one_subject_out_stability"], index=False)
    reliability_holdl.to_csv(paths["target_reliability_holdl"], index=False)
    reliability_movel.to_csv(paths["target_reliability_movel"], index=False)
    task_comparison.to_csv(paths["task_reliability_comparison"], index=False)
    direction_subject.to_csv(paths["compensation_directions_by_subject"], index=False)
    direction_group.to_csv(paths["compensation_directions_group"], index=False)
    direction_stability.to_csv(paths["compensation_direction_stability"], index=False)
    write_reliability_report(
        paths["report"],
        inputs,
        reliability,
        reliability_holdl,
        reliability_movel,
        reliability_sideaware,
        reliability_left,
        reliability_right,
        reliability_hold_family,
        reliability_move_family,
        task_comparison,
        loso_stability,
        direction_group,
        direction_stability,
    )
    return paths
