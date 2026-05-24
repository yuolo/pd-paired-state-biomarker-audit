"""In silico compensation simulation for ds004998 feature states.

This module tests abstract feature-level shifts from MedOff toward MedOn
dataset-internal compensated proxies. It does not model real stimulation,
device settings, or care recommendations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.inverse_design.objective import parse_candidate_feature
from src.pathology_model.build_real_state_vectors import available_real_feature_columns
from src.pathology_model.state_deviation import state_deviation_score

_default_cache = Path.cwd() / ".cache" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(_default_cache))
os.environ.setdefault("MPLBACKEND", "Agg")
_default_cache.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.side_aware_analysis import (  # noqa: E402
    add_task_side_columns,
    parse_task_side,
    write_integrated_summary,
)


ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
STRATEGIES = [
    "no_action",
    "random_candidate",
    "strongest_candidate",
    "most_reliable_candidate",
    "best_stn_beta_candidate",
    "best_stn_gamma_candidate",
    "top_k_reliable_candidates",
    "top_k_stn_candidates",
]
METADATA_COLUMNS = ["subject", "session", "condition", "medication", "task", "run"]
SIMULATION_COLUMNS = [
    "subject",
    "session",
    "task",
    "task_original",
    "task_family",
    "side",
    "run",
    "reference_scope",
    "reference_n",
    "strategy",
    "selected_features",
    "alpha",
    "baseline_deviation",
    "post_compensation_deviation",
    "absolute_deviation_reduction",
    "percent_deviation_reduction",
    "intervention_magnitude",
    "intervention_magnitude_penalty",
    "multi_feature_complexity_penalty",
    "unstable_direction_penalty",
    "net_compensation_score",
    "n_shifted_features",
    "direction_consistency_score",
    "quality_flag",
    "methodological_note",
]


@dataclass(frozen=True)
class CompensationInputs:
    """Loaded input tables for the compensation simulation."""

    state_vectors: pd.DataFrame
    deviation_scores: pd.DataFrame
    reliability: pd.DataFrame
    directions_by_subject: pd.DataFrame
    directions_group: pd.DataFrame
    direction_stability: pd.DataFrame
    candidate_targets: pd.DataFrame
    reliability_holdl: pd.DataFrame
    reliability_movel: pd.DataFrame
    quality: pd.DataFrame
    warnings: list[str]


@dataclass(frozen=True)
class StrategySelection:
    """Selected feature set for one simulation strategy."""

    strategy: str
    features: list[str]
    source: str


def read_optional_table(path: str | Path, warnings: list[str], dtype: dict[str, type | str] | None = None) -> pd.DataFrame:
    """Read a CSV table if present; otherwise record a warning."""

    path = Path(path)
    if not path.exists():
        warnings.append(f"Missing optional input: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Could not read {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def read_state_vectors(path: str | Path, warnings: list[str]) -> pd.DataFrame:
    """Read state vectors while preserving identifiers."""

    path = Path(path)
    if not path.exists():
        warnings.append(f"Missing required state vector input: {path}")
        return pd.DataFrame()
    dtype = {column: str for column in METADATA_COLUMNS}
    data = pd.read_csv(path, dtype=dtype)
    for column in METADATA_COLUMNS:
        if column in data:
            data[column] = data[column].fillna("unknown").astype(str)
    return add_task_side_columns(data, "task")


def load_compensation_inputs(
    state_vectors_path: str | Path,
    reliability_path: str | Path,
    directions_path: str | Path,
    tables_dir: str | Path = "outputs/tables",
) -> CompensationInputs:
    """Load requested inputs and warnings."""

    warnings: list[str] = []
    tables_dir = Path(tables_dir)
    dtype = {"run": str}
    return CompensationInputs(
        state_vectors=read_state_vectors(state_vectors_path, warnings),
        deviation_scores=read_optional_table(tables_dir / "real_deviation_scores.csv", warnings, dtype=dtype),
        reliability=read_optional_table(reliability_path, warnings, dtype=dtype),
        directions_by_subject=read_optional_table(
            tables_dir / "real_compensation_directions_by_subject.csv",
            warnings,
            dtype=dtype,
        ),
        directions_group=read_optional_table(directions_path, warnings, dtype=dtype),
        direction_stability=read_optional_table(
            tables_dir / "real_compensation_direction_stability.csv",
            warnings,
            dtype=dtype,
        ),
        candidate_targets=read_optional_table(tables_dir / "real_candidate_targets.csv", warnings, dtype=dtype),
        reliability_holdl=read_optional_table(tables_dir / "real_target_reliability_holdl.csv", warnings, dtype=dtype),
        reliability_movel=read_optional_table(tables_dir / "real_target_reliability_movel.csv", warnings, dtype=dtype),
        quality=read_optional_table(tables_dir / "real_recording_quality.csv", warnings, dtype=dtype),
        warnings=warnings,
    )


def normalize_bool(series: pd.Series) -> pd.Series:
    """Convert booleans or boolean-like strings to bool."""

    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def ranked_feature_list(
    table: pd.DataFrame,
    feature_columns: list[str],
    sort_columns: list[str],
    ascending: list[bool],
    stn: bool | None = None,
    beta: bool | None = None,
    gamma: bool | None = None,
) -> list[str]:
    """Return candidate features from a ranked table with optional filters."""

    if table.empty or "feature" not in table:
        return []
    data = table.copy()
    for column in ["is_stn_target", "is_beta_target", "is_gamma_target"]:
        if column not in data:
            text = (
                data["feature"].fillna("").astype(str)
                + " "
                + data.get("node", pd.Series("", index=data.index)).fillna("").astype(str)
                + " "
                + data.get("band", pd.Series("", index=data.index)).fillna("").astype(str)
            ).str.lower()
            if column == "is_stn_target":
                data[column] = text.str.contains("stn")
            elif column == "is_beta_target":
                data[column] = text.str.contains("beta")
            else:
                data[column] = text.str.contains("gamma")
        else:
            data[column] = normalize_bool(data[column])
    if stn is not None:
        data = data[data["is_stn_target"].eq(stn)]
    if beta is not None:
        data = data[data["is_beta_target"].eq(beta)]
    if gamma is not None:
        data = data[data["is_gamma_target"].eq(gamma)]
    data = data[data["feature"].isin(feature_columns)]
    if data.empty:
        return []
    for column in sort_columns:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    valid_sort = [column for column in sort_columns if column in data]
    valid_ascending = ascending[: len(valid_sort)]
    if valid_sort:
        data = data.sort_values(valid_sort, ascending=valid_ascending)
    return list(dict.fromkeys(data["feature"].dropna().astype(str)))


def select_strategies(
    inputs: CompensationInputs,
    feature_columns: list[str],
    top_k: int = 3,
    seed: int = 42,
) -> tuple[list[StrategySelection], list[str]]:
    """Select feature groups for all requested simulation strategies."""

    warnings: list[str] = []
    rng = np.random.default_rng(seed)
    available = list(feature_columns)
    random_feature = str(rng.choice(available)) if available else ""
    strongest = ranked_feature_list(inputs.candidate_targets, available, ["rank"], [True])
    reliable = ranked_feature_list(
        inputs.reliability,
        available,
        ["reliability_score", "top10_frequency", "mean_rank"],
        [False, False, True],
    )
    stn_beta = ranked_feature_list(
        inputs.reliability,
        available,
        ["reliability_score", "top10_frequency", "mean_rank"],
        [False, False, True],
        stn=True,
        beta=True,
    )
    if not stn_beta:
        stn_beta = ranked_feature_list(inputs.candidate_targets, available, ["rank"], [True], stn=True, beta=True)
    stn_gamma = ranked_feature_list(
        inputs.reliability,
        available,
        ["reliability_score", "top10_frequency", "mean_rank"],
        [False, False, True],
        stn=True,
        gamma=True,
    )
    if not stn_gamma:
        stn_gamma = ranked_feature_list(inputs.candidate_targets, available, ["rank"], [True], stn=True, gamma=True)
    stn_reliable = ranked_feature_list(
        inputs.reliability,
        available,
        ["reliability_score", "top10_frequency", "mean_rank"],
        [False, False, True],
        stn=True,
    )

    selections = [
        StrategySelection("no_action", [], "none"),
        StrategySelection("random_candidate", [random_feature] if random_feature else [], "seeded_random_feature"),
        StrategySelection("strongest_candidate", strongest[:1], "real_candidate_targets_rank"),
        StrategySelection("most_reliable_candidate", reliable[:1], "real_target_reliability_score"),
        StrategySelection("best_stn_beta_candidate", stn_beta[:1], "reliable_stn_beta_candidate"),
        StrategySelection("best_stn_gamma_candidate", stn_gamma[:1], "reliable_stn_gamma_candidate"),
        StrategySelection("top_k_reliable_candidates", reliable[:top_k], f"top_{top_k}_real_target_reliability"),
        StrategySelection("top_k_stn_candidates", stn_reliable[:top_k], f"top_{top_k}_stn_reliability"),
    ]
    for selection in selections:
        if selection.strategy != "no_action" and not selection.features:
            warnings.append(f"No available feature was selected for strategy: {selection.strategy}")
    return selections, warnings


def build_direction_lookup(
    state_vectors: pd.DataFrame,
    directions_group: pd.DataFrame,
    feature_columns: list[str],
    direction_epsilon: float = 1e-6,
) -> dict[tuple[str, str], dict[str, float | str]]:
    """Build task/feature compensation directions from group direction table."""

    lookup: dict[tuple[str, str], dict[str, float | str]] = {}
    if not directions_group.empty:
        for _, row in directions_group.iterrows():
            feature = str(row.get("feature", ""))
            task = str(row.get("task", ""))
            if feature not in feature_columns or not task:
                continue
            signed_delta = float(pd.to_numeric(pd.Series([row.get("group_mean_signed_delta", np.nan)]), errors="coerce").iloc[0])
            consistency = float(pd.to_numeric(pd.Series([row.get("consistency_fraction", np.nan)]), errors="coerce").iloc[0])
            direction = "unchanged"
            if np.isfinite(signed_delta) and abs(signed_delta) > direction_epsilon:
                direction = "increase" if signed_delta > 0 else "decrease"
            lookup[(task, feature)] = {
                "signed_delta": signed_delta if np.isfinite(signed_delta) else 0.0,
                "direction": direction,
                "consistency_fraction": consistency if np.isfinite(consistency) else 0.0,
                "source": "real_compensation_directions_group",
            }
    if state_vectors.empty:
        return lookup
    medication = state_vectors["medication"].fillna("").astype(str).str.lower()
    for task, subset in state_vectors.groupby("task", dropna=False):
        off = subset[medication.loc[subset.index].eq("off")]
        on = subset[medication.loc[subset.index].eq("on")]
        if off.empty or on.empty:
            continue
        for feature in feature_columns:
            key = (str(task), feature)
            if key in lookup:
                continue
            delta = float(on[feature].mean() - off[feature].mean())
            direction = "unchanged"
            if abs(delta) > direction_epsilon:
                direction = "increase" if delta > 0 else "decrease"
            lookup[key] = {
                "signed_delta": delta,
                "direction": direction,
                "consistency_fraction": 0.0,
                "source": "state_vector_group_delta_fallback",
            }
    return lookup


def feature_scale_for_task(state_vectors: pd.DataFrame, task: str, feature_columns: list[str], min_scale: float = 0.1) -> np.ndarray:
    """Compute dataset-internal feature scale for a task."""

    task_rows = state_vectors[state_vectors["task"].astype(str).eq(str(task))]
    scale_source = task_rows if len(task_rows) >= 2 else state_vectors
    values = scale_source[feature_columns].to_numpy(dtype=float)
    ddof = 1 if values.shape[0] > 1 else 0
    scale = np.nanstd(values, axis=0, ddof=ddof)
    scale = np.nan_to_num(scale, nan=min_scale, posinf=min_scale, neginf=min_scale)
    return np.where(scale < min_scale, min_scale, scale)


def medoff_states(state_vectors: pd.DataFrame) -> pd.DataFrame:
    """Return MedOff rows for simulation."""

    if state_vectors.empty or "medication" not in state_vectors:
        return pd.DataFrame()
    medication = state_vectors["medication"].fillna("").astype(str).str.lower()
    return state_vectors[medication.eq("off")].copy()


def reference_states_for_row(
    state_vectors: pd.DataFrame,
    row: pd.Series,
    feature_columns: list[str],
) -> list[dict[str, object]]:
    """Return same-subject and group task MedOn proxy references for one MedOff row."""

    medication = state_vectors["medication"].fillna("").astype(str).str.lower()
    same_task = state_vectors["task"].astype(str).eq(str(row["task"]))
    same_subject = state_vectors["subject"].astype(str).eq(str(row["subject"]))
    medon = medication.eq("on")
    refs: list[dict[str, object]] = []
    subject_ref = state_vectors[same_task & same_subject & medon]
    if not subject_ref.empty:
        refs.append(
            {
                "reference_scope": "subject_medon_proxy",
                "reference_n": int(len(subject_ref)),
                "state": subject_ref[feature_columns].mean().to_numpy(dtype=float),
            }
        )
    group_ref = state_vectors[same_task & medon]
    if not group_ref.empty:
        refs.append(
            {
                "reference_scope": "group_task_medon_proxy",
                "reference_n": int(len(group_ref)),
                "state": group_ref[feature_columns].mean().to_numpy(dtype=float),
            }
        )
    return refs


def quality_lookup(quality: pd.DataFrame) -> dict[tuple[str, str, str, str], str]:
    """Build quality flag lookup by subject/task/medication/run."""

    if quality.empty:
        return {}
    q = quality.copy()
    for column in ["subject", "task", "medication", "run"]:
        if column in q:
            q[column] = q[column].fillna("unknown").astype(str)
        else:
            q[column] = "unknown"
    return {
        (row.subject, row.task, row.medication, row.run): str(row.quality_flag)
        for row in q.itertuples()
        if hasattr(row, "quality_flag")
    }


def apply_feature_shift(
    state: np.ndarray,
    task: str,
    feature_columns: list[str],
    selected_features: list[str],
    alpha: float,
    direction_lookup: dict[tuple[str, str], dict[str, float | str]],
    scale: np.ndarray,
) -> tuple[np.ndarray, float, int, float]:
    """Apply alpha-scaled compensation-direction shifts to selected features."""

    shifted = state.copy()
    applied = np.zeros_like(state, dtype=float)
    consistency_values = []
    n_shifted = 0
    feature_index = {feature: idx for idx, feature in enumerate(feature_columns)}
    for feature in selected_features:
        if feature not in feature_index:
            continue
        info = direction_lookup.get((str(task), feature), {})
        signed_delta = float(info.get("signed_delta", 0.0))
        consistency = float(info.get("consistency_fraction", 0.0))
        delta = float(alpha) * signed_delta
        idx = feature_index[feature]
        shifted[idx] += delta
        applied[idx] = delta
        if abs(delta) > 1e-12:
            n_shifted += 1
            consistency_values.append(consistency)
    safe_scale = np.where(scale <= 0, 1.0, scale)
    intervention_magnitude = float(np.sqrt(np.mean((applied / safe_scale) ** 2))) if n_shifted else 0.0
    direction_consistency = float(np.mean(consistency_values)) if consistency_values else 1.0
    return shifted, intervention_magnitude, n_shifted, direction_consistency


def simulate_rows(
    inputs: CompensationInputs,
    selections: list[StrategySelection],
    direction_lookup: dict[tuple[str, str], dict[str, float | str]],
    feature_columns: list[str],
    alpha_values: list[float],
    lambda_magnitude: float,
    lambda_complexity: float,
    lambda_instability: float,
) -> pd.DataFrame:
    """Run feature-level compensation simulation for every MedOff state."""

    off = medoff_states(inputs.state_vectors)
    if off.empty or not feature_columns:
        return pd.DataFrame(columns=SIMULATION_COLUMNS)
    quality = quality_lookup(inputs.quality)
    rows = []
    for _, medoff in off.iterrows():
        task = str(medoff["task"])
        task_original, task_family, side = parse_task_side(task)
        scale = feature_scale_for_task(inputs.state_vectors, task, feature_columns)
        current = medoff[feature_columns].to_numpy(dtype=float)
        refs = reference_states_for_row(inputs.state_vectors, medoff, feature_columns)
        quality_flag = quality.get(
            (
                str(medoff.get("subject", "unknown")),
                task,
                str(medoff.get("medication", "unknown")),
                str(medoff.get("run", "unknown")),
            ),
            "unknown",
        )
        for reference in refs:
            reference_state = np.asarray(reference["state"], dtype=float)
            baseline = state_deviation_score(current, reference_state, scale)
            for selection in selections:
                selected_features = [feature for feature in selection.features if feature in feature_columns]
                for alpha in alpha_values:
                    shifted, magnitude, n_shifted, consistency = apply_feature_shift(
                        current,
                        task,
                        feature_columns,
                        selected_features,
                        alpha,
                        direction_lookup,
                        scale,
                    )
                    post = state_deviation_score(shifted, reference_state, scale)
                    reduction = baseline - post
                    percent = 100.0 * reduction / baseline if np.isfinite(baseline) and baseline > 0 else np.nan
                    instability = 0.0 if n_shifted == 0 else max(0.0, 1.0 - consistency)
                    net_score = (
                        reduction
                        - lambda_magnitude * magnitude
                        - lambda_complexity * n_shifted
                        - lambda_instability * instability
                    )
                    rows.append(
                        {
                            "subject": medoff.get("subject", "unknown"),
                            "session": medoff.get("session", "unknown"),
                            "task": task,
                            "task_original": task_original,
                            "task_family": task_family,
                            "side": side,
                            "run": medoff.get("run", "unknown"),
                            "reference_scope": reference["reference_scope"],
                            "reference_n": int(reference["reference_n"]),
                            "strategy": selection.strategy,
                            "selected_features": ";".join(selected_features),
                            "alpha": float(alpha),
                            "baseline_deviation": baseline,
                            "post_compensation_deviation": post,
                            "absolute_deviation_reduction": reduction,
                            "percent_deviation_reduction": percent,
                            "intervention_magnitude": magnitude,
                            "intervention_magnitude_penalty": lambda_magnitude * magnitude,
                            "multi_feature_complexity_penalty": lambda_complexity * n_shifted,
                            "unstable_direction_penalty": lambda_instability * instability,
                            "net_compensation_score": net_score,
                            "n_shifted_features": int(n_shifted),
                            "direction_consistency_score": consistency,
                            "quality_flag": quality_flag,
                            "methodological_note": (
                                "in silico feature-level shift toward dataset-internal MedOn compensated proxy; "
                                "not a real intervention model"
                            ),
                        }
                    )
    return pd.DataFrame(rows, columns=SIMULATION_COLUMNS)


def summarize_strategies(simulation: pd.DataFrame) -> pd.DataFrame:
    """Summarize compensation outcomes by strategy, alpha, and reference scope."""

    if simulation.empty:
        return pd.DataFrame()
    grouped = simulation.groupby(["reference_scope", "strategy", "alpha"], dropna=False)
    summary = grouped.agg(
        n_rows=("subject", "size"),
        n_subjects=("subject", "nunique"),
        n_tasks=("task", "nunique"),
        mean_baseline_deviation=("baseline_deviation", "mean"),
        mean_post_compensation_deviation=("post_compensation_deviation", "mean"),
        mean_absolute_deviation_reduction=("absolute_deviation_reduction", "mean"),
        median_absolute_deviation_reduction=("absolute_deviation_reduction", "median"),
        mean_percent_deviation_reduction=("percent_deviation_reduction", "mean"),
        improvement_rate=("absolute_deviation_reduction", lambda x: float((x > 0).mean())),
        mean_intervention_magnitude=("intervention_magnitude", "mean"),
        mean_n_shifted_features=("n_shifted_features", "mean"),
        mean_direction_consistency=("direction_consistency_score", "mean"),
        mean_net_compensation_score=("net_compensation_score", "mean"),
    ).reset_index()
    summary["methodological_note"] = "summary over in silico feature-level compensation simulation rows"
    return summary.sort_values(["reference_scope", "mean_net_compensation_score"], ascending=[True, False])


def best_by_subject(simulation: pd.DataFrame) -> pd.DataFrame:
    """Select the best active strategy/alpha row per subject and task."""

    if simulation.empty:
        return pd.DataFrame()
    primary = simulation[simulation["reference_scope"].eq("subject_medon_proxy")].copy()
    if primary.empty:
        primary = simulation.copy()
    active = primary[(primary["alpha"] > 0) & ~primary["strategy"].eq("no_action")].copy()
    if active.empty:
        active = primary.copy()
    grouped = (
        active.groupby(["subject", "task", "strategy", "alpha"], as_index=False)
        .agg(
            mean_absolute_deviation_reduction=("absolute_deviation_reduction", "mean"),
            mean_net_compensation_score=("net_compensation_score", "mean"),
            mean_intervention_magnitude=("intervention_magnitude", "mean"),
            selected_features=("selected_features", "first"),
            quality_flag=("quality_flag", "first"),
        )
        .sort_values(["subject", "task", "mean_net_compensation_score"], ascending=[True, True, False])
    )
    return grouped.groupby(["subject", "task"], as_index=False).head(1).reset_index(drop=True)


def best_by_task(summary: pd.DataFrame) -> pd.DataFrame:
    """Select best strategy/alpha per task under subject-level proxy."""

    if summary.empty:
        return pd.DataFrame()
    return summary.copy()


def summarize_by_task(simulation: pd.DataFrame) -> pd.DataFrame:
    """Summarize best active strategies by task."""

    if simulation.empty:
        return pd.DataFrame()
    primary = simulation[simulation["reference_scope"].eq("subject_medon_proxy")].copy()
    if primary.empty:
        primary = simulation.copy()
    active = primary[(primary["alpha"] > 0) & ~primary["strategy"].eq("no_action")].copy()
    if active.empty:
        active = primary.copy()
    grouped = active.groupby(["task", "strategy", "alpha"], as_index=False).agg(
        n_subjects=("subject", "nunique"),
        mean_absolute_deviation_reduction=("absolute_deviation_reduction", "mean"),
        mean_percent_deviation_reduction=("percent_deviation_reduction", "mean"),
        improvement_rate=("absolute_deviation_reduction", lambda x: float((x > 0).mean())),
        mean_intervention_magnitude=("intervention_magnitude", "mean"),
        mean_net_compensation_score=("net_compensation_score", "mean"),
        selected_features=("selected_features", "first"),
    )
    grouped["beats_no_action_by_net_score"] = grouped["mean_net_compensation_score"] > 0
    grouped = grouped.sort_values(["task", "mean_net_compensation_score"], ascending=[True, False])
    return grouped.groupby("task", as_index=False).head(1).reset_index(drop=True)


def summarize_sideaware_strategies(simulation: pd.DataFrame) -> pd.DataFrame:
    """Summarize strategy outcomes by original task, task family, and side."""

    if simulation.empty:
        return pd.DataFrame()
    group_cols = ["reference_scope", "task_original", "task_family", "side", "strategy", "alpha"]
    grouped = simulation.groupby(group_cols, dropna=False).agg(
        n_rows=("subject", "size"),
        n_subjects=("subject", "nunique"),
        mean_baseline_deviation=("baseline_deviation", "mean"),
        mean_post_compensation_deviation=("post_compensation_deviation", "mean"),
        mean_absolute_deviation_reduction=("absolute_deviation_reduction", "mean"),
        mean_percent_deviation_reduction=("percent_deviation_reduction", "mean"),
        improvement_rate=("absolute_deviation_reduction", lambda x: float((x > 0).mean())),
        mean_intervention_magnitude=("intervention_magnitude", "mean"),
        mean_n_shifted_features=("n_shifted_features", "mean"),
        mean_direction_consistency=("direction_consistency_score", "mean"),
        mean_net_compensation_score=("net_compensation_score", "mean"),
        selected_features=("selected_features", "first"),
    ).reset_index()
    grouped["beats_no_action_by_net_score"] = grouped["mean_net_compensation_score"] > 0
    grouped["methodological_note"] = "side-aware in silico feature-level compensation summary"
    return grouped


def best_active_by_group(simulation: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Best active strategy by side or task family."""

    if simulation.empty or group_column not in simulation:
        return pd.DataFrame()
    primary = simulation[simulation["reference_scope"].eq("subject_medon_proxy")].copy()
    if primary.empty:
        primary = simulation.copy()
    active = primary[(primary["alpha"] > 0) & ~primary["strategy"].eq("no_action")].copy()
    if active.empty:
        active = primary.copy()
    grouped = active.groupby([group_column, "strategy", "alpha"], as_index=False).agg(
        n_subjects=("subject", "nunique"),
        n_state_rows=("subject", "size"),
        mean_absolute_deviation_reduction=("absolute_deviation_reduction", "mean"),
        mean_percent_deviation_reduction=("percent_deviation_reduction", "mean"),
        improvement_rate=("absolute_deviation_reduction", lambda x: float((x > 0).mean())),
        mean_intervention_magnitude=("intervention_magnitude", "mean"),
        mean_net_compensation_score=("net_compensation_score", "mean"),
        selected_features=("selected_features", "first"),
    )
    grouped["beats_no_action_by_net_score"] = grouped["mean_net_compensation_score"] > 0
    grouped = grouped.sort_values([group_column, "mean_net_compensation_score"], ascending=[True, False])
    return grouped.groupby(group_column, as_index=False).head(1).reset_index(drop=True)


def simulate_trajectories(
    inputs: CompensationInputs,
    selections: list[StrategySelection],
    direction_lookup: dict[tuple[str, str], dict[str, float | str]],
    feature_columns: list[str],
    simulation: pd.DataFrame,
    transition_steps: int = 8,
    maintenance_steps: int = 8,
) -> pd.DataFrame:
    """Simulate abstract transition and maintenance alpha schedules."""

    off = medoff_states(inputs.state_vectors)
    if off.empty or simulation.empty:
        return pd.DataFrame(
            columns=[
                "subject",
                "task",
                "task_original",
                "task_family",
                "side",
                "strategy",
                "step",
                "regime",
                "alpha",
                "deviation",
                "intervention_magnitude",
            ]
        )
    primary = simulation[simulation["reference_scope"].eq("subject_medon_proxy")]
    if primary.empty:
        primary = simulation
    best_alpha = (
        primary.groupby(["subject", "task", "strategy", "alpha"], as_index=False)["net_compensation_score"]
        .mean()
        .sort_values(["subject", "task", "strategy", "net_compensation_score"], ascending=[True, True, True, False])
        .groupby(["subject", "task", "strategy"], as_index=False)
        .head(1)
        .set_index(["subject", "task", "strategy"])["alpha"]
        .to_dict()
    )
    selection_lookup = {selection.strategy: selection.features for selection in selections}
    rows = []
    for _, medoff in off.iterrows():
        task = str(medoff["task"])
        task_original, task_family, side = parse_task_side(task)
        current = medoff[feature_columns].to_numpy(dtype=float)
        scale = feature_scale_for_task(inputs.state_vectors, task, feature_columns)
        refs = reference_states_for_row(inputs.state_vectors, medoff, feature_columns)
        subject_refs = [ref for ref in refs if ref["reference_scope"] == "subject_medon_proxy"]
        if not subject_refs:
            continue
        reference_state = np.asarray(subject_refs[0]["state"], dtype=float)
        for strategy in STRATEGIES:
            target_alpha = float(best_alpha.get((str(medoff["subject"]), task, strategy), 0.0))
            maintenance_alpha = min(target_alpha, 0.25)
            schedule = []
            for step in range(transition_steps + 1):
                alpha = target_alpha * step / max(1, transition_steps)
                schedule.append((step, "transition", alpha))
            for step in range(1, maintenance_steps + 1):
                schedule.append((transition_steps + step, "maintenance", maintenance_alpha))
            selected = selection_lookup.get(strategy, [])
            for step, regime, alpha in schedule:
                shifted, magnitude, _, _ = apply_feature_shift(
                    current,
                    task,
                    feature_columns,
                    selected,
                    alpha,
                    direction_lookup,
                    scale,
                )
                deviation = state_deviation_score(shifted, reference_state, scale)
                rows.append(
                    {
                        "subject": medoff.get("subject", "unknown"),
                        "task": task,
                        "task_original": task_original,
                        "task_family": task_family,
                        "side": side,
                        "strategy": strategy,
                        "step": int(step),
                        "regime": regime,
                        "alpha": float(alpha),
                        "deviation": deviation,
                        "intervention_magnitude": magnitude,
                    }
                )
    return pd.DataFrame(rows)


def best_summary_row(
    summary: pd.DataFrame,
    strategy: str | None = None,
    task: str | None = None,
    active_only: bool = True,
    exclude_no_action: bool = True,
) -> pd.Series | None:
    """Return best primary summary row optionally filtered by strategy/task."""

    if summary.empty:
        return None
    data = summary.copy()
    if "reference_scope" in data:
        primary = data[data["reference_scope"].eq("subject_medon_proxy")]
        if not primary.empty:
            data = primary
    if strategy is not None:
        data = data[data["strategy"].eq(strategy)]
    if task is not None and "task" in data:
        data = data[data["task"].astype(str).eq(str(task))]
    if active_only and "alpha" in data:
        data = data[data["alpha"] > 0]
    if exclude_no_action and "strategy" in data:
        data = data[~data["strategy"].eq("no_action")]
    if data.empty:
        return None
    sort_columns = ["mean_net_compensation_score"]
    ascending = [False]
    if not exclude_no_action and "strategy" in data:
        data = data.copy()
        data["_is_no_action"] = data["strategy"].eq("no_action")
        sort_columns.append("_is_no_action")
        ascending.append(False)
    if "alpha" in data:
        sort_columns.append("alpha")
        ascending.append(True)
    return data.sort_values(sort_columns, ascending=ascending).iloc[0]


def write_report(
    path: str | Path,
    inputs: CompensationInputs,
    selections: list[StrategySelection],
    simulation: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    best_task: pd.DataFrame,
    best_side: pd.DataFrame,
    best_family: pd.DataFrame,
) -> None:
    """Write in silico compensation simulation report."""

    primary = simulation[simulation["reference_scope"].eq("subject_medon_proxy")] if not simulation.empty else pd.DataFrame()
    n_medoff = int(primary[["subject", "task"]].drop_duplicates().shape[0]) if not primary.empty else 0
    n_subjects = inputs.state_vectors["subject"].nunique() if not inputs.state_vectors.empty and "subject" in inputs.state_vectors else 0
    n_tasks = inputs.state_vectors["task_original"].nunique() if not inputs.state_vectors.empty and "task_original" in inputs.state_vectors else 0
    n_sides = inputs.state_vectors["side"].nunique() if not inputs.state_vectors.empty and "side" in inputs.state_vectors else 0
    best_active = best_summary_row(strategy_summary, active_only=True, exclude_no_action=True)
    best_conservative = best_summary_row(strategy_summary, active_only=False, exclude_no_action=False)
    holdl_best = best_task[best_task["task"].astype(str).eq("HoldL")].iloc[0] if not best_task.empty and best_task["task"].astype(str).eq("HoldL").any() else None
    movel_best = best_task[best_task["task"].astype(str).eq("MoveL")].iloc[0] if not best_task.empty and best_task["task"].astype(str).eq("MoveL").any() else None
    reliable = best_summary_row(strategy_summary, "most_reliable_candidate")
    random = best_summary_row(strategy_summary, "random_candidate")
    stn_beta = best_summary_row(strategy_summary, "best_stn_beta_candidate")
    stn_gamma = best_summary_row(strategy_summary, "best_stn_gamma_candidate")
    multi = best_summary_row(strategy_summary, "top_k_reliable_candidates")
    single = best_summary_row(strategy_summary, "most_reliable_candidate")

    def compare(row_a: pd.Series | None, row_b: pd.Series | None) -> str:
        if row_a is None or row_b is None:
            return "not_available"
        a = float(row_a["mean_net_compensation_score"])
        b = float(row_b["mean_net_compensation_score"])
        if a > b:
            return "yes"
        if np.isclose(a, b):
            return "similar"
        return "no"

    lines = [
        "# Real In Silico Compensation Simulation Report",
        "",
        "## Purpose",
        "",
        "This module tests whether simulated feature-level shifts from MedOff toward a dataset-internal MedOn compensated proxy reduce proxy deviation. It is an exploratory in silico compensation simulation, not a real intervention model.",
        "",
        "## Inputs",
        "",
        f"- state_vector_rows: {len(inputs.state_vectors)}",
        f"- subjects: {n_subjects}",
        f"- MedOff states simulated: {n_medoff}",
        f"- tasks: {n_tasks}",
        f"- sides: {n_sides}",
        f"- state_features_used: {len(available_real_feature_columns(inputs.state_vectors)) if not inputs.state_vectors.empty else 0}",
        f"- deviation_score_rows: {len(inputs.deviation_scores)}",
        f"- reliability_rows: {len(inputs.reliability)}",
        f"- compensation_direction_rows: {len(inputs.directions_group)}",
        f"- candidate_target_rows: {len(inputs.candidate_targets)}",
        "",
        "## Candidate Strategies",
        "",
    ]
    for selection in selections:
        lines.append(
            f"- {selection.strategy}: {(';'.join(selection.features) if selection.features else 'none')} "
            f"(source={selection.source})"
        )
    if inputs.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in inputs.warnings])
    lines.extend(["", "## Main Results", ""])
    if best_active is None:
        lines.append("No strategy summary was available.")
    else:
        lines.append(
            f"- best_overall_active_strategy: {best_active['strategy']} at alpha={best_active['alpha']} "
            f"(mean_net_compensation_score={best_active['mean_net_compensation_score']:.4f}, "
            f"mean_deviation_reduction={best_active['mean_absolute_deviation_reduction']:.4f})"
        )
    if best_conservative is not None:
        lines.append(
            f"- best_conservative_strategy_including_no_action: {best_conservative['strategy']} "
            f"at alpha={best_conservative['alpha']} "
            f"(mean_net_compensation_score={best_conservative['mean_net_compensation_score']:.4f})"
        )
    if holdl_best is not None:
        lines.append(
            f"- best_HoldL_active_strategy: {holdl_best['strategy']} at alpha={holdl_best['alpha']} "
            f"(mean_deviation_reduction={holdl_best['mean_absolute_deviation_reduction']:.4f}, "
            f"beats_no_action_by_net_score={holdl_best['beats_no_action_by_net_score']})"
        )
    if movel_best is not None:
        lines.append(
            f"- best_MoveL_active_strategy: {movel_best['strategy']} at alpha={movel_best['alpha']} "
            f"(mean_deviation_reduction={movel_best['mean_absolute_deviation_reduction']:.4f}, "
            f"beats_no_action_by_net_score={movel_best['beats_no_action_by_net_score']})"
        )
    if not best_side.empty:
        for _, row in best_side.iterrows():
            lines.append(
                f"- best_{row['side']}_side_active_strategy: {row['strategy']} at alpha={row['alpha']} "
                f"(mean_deviation_reduction={row['mean_absolute_deviation_reduction']:.4f}, "
                f"beats_no_action_by_net_score={row['beats_no_action_by_net_score']})"
            )
    if not best_family.empty:
        for _, row in best_family.iterrows():
            lines.append(
                f"- best_{row['task_family']}_family_active_strategy: {row['strategy']} at alpha={row['alpha']} "
                f"(mean_deviation_reduction={row['mean_absolute_deviation_reduction']:.4f}, "
                f"beats_no_action_by_net_score={row['beats_no_action_by_net_score']})"
            )
    lines.append(f"- reliable_candidate_outperforms_random: {compare(reliable, random)}")
    lines.append(f"- top_k_reliable_outperforms_single_reliable: {compare(multi, single)}")
    if stn_beta is not None:
        lines.append(
            f"- best_STN_beta_candidate_score: {stn_beta['mean_net_compensation_score']:.4f}, "
            f"mean_deviation_reduction={stn_beta['mean_absolute_deviation_reduction']:.4f}"
        )
    if stn_gamma is not None:
        lines.append(
            f"- best_STN_gamma_candidate_score: {stn_gamma['mean_net_compensation_score']:.4f}, "
            f"mean_deviation_reduction={stn_gamma['mean_absolute_deviation_reduction']:.4f}"
        )
    lines.append(f"- STN_beta_stronger_than_STN_gamma_by_net_score: {compare(stn_beta, stn_gamma)}")

    lines.extend(
        [
            "",
            "## Conservative Interpretation",
            "",
            "- Results are simulated feature-level shifts toward a dataset-internal compensated proxy.",
            "- A positive deviation reduction means the shifted MedOff feature vector is closer to the selected MedOn proxy under the current metric.",
            "- Candidate compensation directions are descriptive MedOff-to-MedOn differences, not intervention instructions.",
            "- Exploratory in silico results require validation on more subjects and with perturbation-response data.",
            "",
            "## Penalty Formula",
            "",
            "net_compensation_score = deviation_reduction - lambda_magnitude * intervention_magnitude - lambda_complexity * n_shifted_features - lambda_instability * unstable_direction_penalty.",
            "",
            "## Limitations",
            "",
            f"- Only {n_subjects} processed subjects are included in the current output subset.",
            "- MedOn is not a healthy state; it is only a dataset-internal compensated proxy.",
            "- Feature-level shifts are not real interventions.",
            "- There are no causal intervention-response data in this module.",
            "- There is no real closed-loop stimulation.",
            "- Validation is required on more subjects and with true perturbation data.",
            "",
            "## Next Steps",
            "",
            "- Expand to more ds004998 subjects.",
            "- Compare subject-specific versus group-level compensation directions.",
            "- Add uncertainty intervals around deviation reduction and net score.",
            "- Test subject-specific compensation policies before pooled policies.",
            "- Later, integrate causal perturbation datasets as future work only.",
            "",
            "## Methodological Guardrails",
            "",
            "- HCP is not used as an electrophysiological reference.",
            "- HCP may only serve as a structural or connectomic prior.",
            "- Candidate outputs are in silico research candidates, not device prescriptions.",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_deviation_reduction(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Plot best mean deviation reduction by strategy."""

    if summary.empty:
        return
    primary = summary[summary["reference_scope"].eq("subject_medon_proxy")]
    data = primary[primary["alpha"] > 0].copy() if not primary.empty else summary.copy()
    best = (
        data.sort_values("mean_net_compensation_score", ascending=False)
        .groupby("strategy", as_index=False)
        .head(1)
        .sort_values("mean_absolute_deviation_reduction", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(best["strategy"], best["mean_absolute_deviation_reduction"], color="#437c90")
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_ylabel("Mean proxy deviation reduction")
    ax.set_title("In silico compensation deviation reduction")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_strategy_comparison(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Plot best net compensation score by strategy."""

    if summary.empty:
        return
    primary = summary[summary["reference_scope"].eq("subject_medon_proxy")]
    data = primary[primary["alpha"] > 0].copy() if not primary.empty else summary.copy()
    best = (
        data.sort_values("mean_net_compensation_score", ascending=False)
        .groupby("strategy", as_index=False)
        .head(1)
        .sort_values("mean_net_compensation_score", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(best["strategy"], best["mean_net_compensation_score"], color="#6f8f4e")
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_ylabel("Mean net compensation score")
    ax.set_title("Strategy comparison with conservative penalties")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_trajectories(trajectories: pd.DataFrame, output_path: str | Path) -> None:
    """Plot mean trajectory by strategy."""

    if trajectories.empty:
        return
    grouped = trajectories.groupby(["strategy", "step"], as_index=False)["deviation"].mean()
    strategy_order = (
        grouped.groupby("strategy")["deviation"].last().sort_values().head(6).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for strategy in strategy_order:
        subset = grouped[grouped["strategy"].eq(strategy)]
        ax.plot(subset["step"], subset["deviation"], linewidth=2, label=strategy)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Mean proxy deviation")
    ax.set_title("Transition and maintenance alpha schedules")
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_compensation_simulation(
    state_vectors_path: str | Path,
    reliability_path: str | Path,
    directions_path: str | Path,
    output_dir: str | Path = "outputs/tables",
    figures_dir: str | Path = "outputs/figures",
    reports_dir: str | Path = "reports",
    seed: int = 42,
    top_k: int = 3,
    alpha_values: list[float] | None = None,
    lambda_magnitude: float = 0.01,
    lambda_complexity: float = 0.005,
    lambda_instability: float = 0.05,
) -> dict[str, Path]:
    """Run full in silico compensation simulation from existing outputs."""

    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    alpha_values = alpha_values or ALPHA_VALUES
    inputs = load_compensation_inputs(state_vectors_path, reliability_path, directions_path, output_dir)
    feature_columns = available_real_feature_columns(inputs.state_vectors) if not inputs.state_vectors.empty else []
    selections, selection_warnings = select_strategies(inputs, feature_columns, top_k=top_k, seed=seed)
    inputs.warnings.extend(selection_warnings)
    direction_lookup = build_direction_lookup(inputs.state_vectors, inputs.directions_group, feature_columns)
    simulation = simulate_rows(
        inputs,
        selections,
        direction_lookup,
        feature_columns,
        alpha_values,
        lambda_magnitude,
        lambda_complexity,
        lambda_instability,
    )
    summary = summarize_strategies(simulation)
    side_summary = summarize_sideaware_strategies(simulation)
    trajectories = simulate_trajectories(inputs, selections, direction_lookup, feature_columns, simulation)
    best_subject = best_by_subject(simulation)
    best_task = summarize_by_task(simulation)
    best_side = best_active_by_group(simulation, "side")
    best_family = best_active_by_group(simulation, "task_family")

    paths = {
        "simulation": output_dir / "real_compensation_simulation.csv",
        "simulation_sideaware": output_dir / "real_compensation_simulation_sideaware.csv",
        "strategy_summary": output_dir / "real_compensation_strategy_summary.csv",
        "strategy_summary_sideaware": output_dir / "real_compensation_strategy_summary_sideaware.csv",
        "trajectories": output_dir / "real_compensation_trajectories.csv",
        "best_strategies_by_subject": output_dir / "real_compensation_best_strategies_by_subject.csv",
        "best_strategies_by_task": output_dir / "real_compensation_best_strategies_by_task.csv",
        "best_strategies_by_side": output_dir / "real_compensation_best_strategies_by_side.csv",
        "best_strategies_by_task_family": output_dir / "real_compensation_best_strategies_by_task_family.csv",
        "deviation_reduction_figure": figures_dir / "real_compensation_deviation_reduction.png",
        "strategy_comparison_figure": figures_dir / "real_compensation_strategy_comparison.png",
        "trajectories_figure": figures_dir / "real_compensation_trajectories.png",
        "report": reports_dir / "real_compensation_model_report.md",
    }
    simulation.to_csv(paths["simulation"], index=False)
    simulation.to_csv(paths["simulation_sideaware"], index=False)
    summary.to_csv(paths["strategy_summary"], index=False)
    side_summary.to_csv(paths["strategy_summary_sideaware"], index=False)
    trajectories.to_csv(paths["trajectories"], index=False)
    best_subject.to_csv(paths["best_strategies_by_subject"], index=False)
    best_task.to_csv(paths["best_strategies_by_task"], index=False)
    best_side.to_csv(paths["best_strategies_by_side"], index=False)
    best_family.to_csv(paths["best_strategies_by_task_family"], index=False)
    plot_deviation_reduction(summary, paths["deviation_reduction_figure"])
    plot_strategy_comparison(summary, paths["strategy_comparison_figure"])
    plot_trajectories(trajectories, paths["trajectories_figure"])
    write_report(paths["report"], inputs, selections, simulation, summary, best_task, best_side, best_family)
    side_summary_path = output_dir / "real_side_aware_summary.csv"
    left_right_path = output_dir / "real_left_right_target_comparison.csv"
    hold_move_path = output_dir / "real_hold_move_family_comparison.csv"
    if side_summary_path.exists() and left_right_path.exists() and hold_move_path.exists():
        write_integrated_summary(
            reports_dir / "real_8_subject_integrated_summary.md",
            inputs.state_vectors,
            pd.read_csv(side_summary_path),
            pd.read_csv(left_right_path),
            pd.read_csv(hold_move_path),
            output_dir,
        )
    return paths
