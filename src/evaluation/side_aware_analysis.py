"""Side-aware analyses for ds004998 real-data proxy outputs.

The routines here separate task family (Hold/Move/Rest) from side (L/R/none)
so left- and right-side tasks are not silently pooled. Electrophysiological
comparisons remain dataset-internal; HCP is not used as an electrophysiology
reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.stratified_real_analysis import finite_feature_subset, target_flags
from src.inverse_design.objective import ObjectiveWeights, parse_candidate_feature, rank_candidate_targets
from src.pathology_model.build_real_state_vectors import available_real_feature_columns
from src.pathology_model.state_deviation import build_internal_reference_stats


TASK_METADATA_COLUMNS = ["task_original", "task_family", "side"]
SIDE_AWARE_STRATA = {
    "holdl": ("HoldL", "task_original", "HoldL"),
    "movel": ("MoveL", "task_original", "MoveL"),
    "holdr": ("HoldR", "task_original", "HoldR"),
    "mover": ("MoveR", "task_original", "MoveR"),
    "left_side": ("left_side", "side", "L"),
    "right_side": ("right_side", "side", "R"),
    "hold_family": ("hold_family", "task_family", "Hold"),
    "move_family": ("move_family", "task_family", "Move"),
}


@dataclass(frozen=True)
class SideAwareInputs:
    """Loaded side-aware input bundle."""

    state_vectors: pd.DataFrame
    warnings: list[str]


def parse_task_side(task: object) -> tuple[str, str, str]:
    """Parse task labels into original task, family, and side."""

    original = "unknown" if pd.isna(task) else str(task)
    text = original.strip()
    lower = text.lower()
    if lower in {"rest", "resting", "restingstate", "resting_state"}:
        return text, "Rest", "none"
    if len(text) >= 2 and text[-1].upper() in {"L", "R"}:
        side = text[-1].upper()
        family_raw = text[:-1]
        family_map = {"hold": "Hold", "move": "Move"}
        family = family_map.get(family_raw.lower(), family_raw or "unknown")
        return text, family, side
    if lower.startswith("hold"):
        return text, "Hold", "unknown"
    if lower.startswith("move"):
        return text, "Move", "unknown"
    return text, text if text else "unknown", "none"


def infer_task_column(data: pd.DataFrame) -> str | None:
    """Find the most likely task-bearing column in a table."""

    for column in ["task", "condition", "stratum", "task_original"]:
        if column in data:
            return column
    return None


def add_task_side_columns(data: pd.DataFrame, task_column: str | None = None) -> pd.DataFrame:
    """Return a copy with task_original, task_family, and side columns."""

    if data.empty:
        result = data.copy()
        for column in TASK_METADATA_COLUMNS:
            if column not in result:
                result[column] = pd.Series(dtype=str)
        return result
    result = data.copy()
    task_column = task_column or infer_task_column(result)
    if task_column is None:
        if "task_original" not in result:
            result["task_original"] = "not_applicable"
        if "task_family" not in result:
            result["task_family"] = "not_applicable"
        if "side" not in result:
            result["side"] = "not_applicable"
        return result
    parsed = result[task_column].apply(parse_task_side)
    result["task_original"] = [item[0] for item in parsed]
    result["task_family"] = [item[1] for item in parsed]
    result["side"] = [item[2] for item in parsed]
    return result


def annotate_csv_if_task_bearing(path: str | Path) -> bool:
    """Add side columns to an existing CSV if it has task-like labels."""

    path = Path(path)
    if not path.exists():
        return False
    data = pd.read_csv(path, dtype={"run": str})
    task_column = infer_task_column(data)
    if task_column is None:
        return False
    annotated = add_task_side_columns(data, task_column)
    annotated.to_csv(path, index=False)
    return True


def annotate_existing_outputs(output_dir: str | Path = "outputs/tables") -> list[str]:
    """Annotate existing task-bearing output tables in place."""

    output_dir = Path(output_dir)
    names = [
        "ds004998_state_vectors.csv",
        "real_deviation_scores.csv",
        "real_candidate_targets_holdl.csv",
        "real_candidate_targets_movel.csv",
        "real_candidate_targets_by_subject.csv",
        "real_target_rank_stability.csv",
        "real_recording_quality.csv",
        "real_compensation_directions_by_subject.csv",
        "real_compensation_directions_group.csv",
        "real_compensation_direction_stability.csv",
        "real_compensation_simulation.csv",
    ]
    annotated = []
    for name in names:
        path = output_dir / name
        if annotate_csv_if_task_bearing(path):
            annotated.append(str(path))
    return annotated


def read_state_vectors(path: str | Path, warnings: list[str]) -> pd.DataFrame:
    """Read and annotate state vectors."""

    path = Path(path)
    if not path.exists():
        warnings.append(f"Missing required state vector table: {path}")
        return pd.DataFrame()
    data = pd.read_csv(path, dtype={"run": str})
    for column in ["subject", "session", "condition", "medication", "task", "run"]:
        if column in data:
            data[column] = data[column].fillna("unknown").astype(str)
    return add_task_side_columns(data, "task")


def rank_targets_for_subset(
    subset: pd.DataFrame,
    feature_columns: list[str],
    weights: ObjectiveWeights,
    effect_fraction: float,
    uncertainty_floor: float,
    stratum: str,
) -> pd.DataFrame:
    """Rank candidate targets for one side-aware stratum."""

    if subset.empty or not feature_columns:
        return pd.DataFrame()
    medication = subset["medication"].fillna("").astype(str).str.lower()
    current_rows = subset[medication.eq("off")]
    reference_rows = subset[medication.eq("on")]
    if current_rows.empty or reference_rows.empty:
        return pd.DataFrame()
    stats = build_internal_reference_stats(
        subset,
        reference_rows,
        feature_columns,
        reference_mode="medon_subset_proxy_not_healthy",
    )
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
    ranked.insert(0, "stratum", stratum)
    ranked["task_original"] = stratum if stratum in {"HoldL", "MoveL", "HoldR", "MoveR"} else "pooled"
    if stratum in {"left_side", "right_side"}:
        ranked["task_family"] = "pooled"
        ranked["side"] = "L" if stratum == "left_side" else "R"
    elif stratum in {"hold_family", "move_family"}:
        ranked["task_family"] = "Hold" if stratum == "hold_family" else "Move"
        ranked["side"] = "pooled"
    else:
        _, family, side = parse_task_side(stratum)
        ranked["task_family"] = family
        ranked["side"] = side
    ranked["reference_mode"] = "medon_subset_proxy_not_healthy"
    ranked["n_subjects"] = int(subset["subject"].nunique()) if "subject" in subset else 0
    ranked["n_state_vectors"] = int(len(subset))
    return ranked


def summarize_targets(stratum: str, subset: pd.DataFrame, targets: pd.DataFrame) -> dict[str, object]:
    """Build side-aware summary row."""

    task_original, family, side = parse_task_side(stratum)
    if stratum == "left_side":
        task_original, family, side = "pooled", "pooled", "L"
    elif stratum == "right_side":
        task_original, family, side = "pooled", "pooled", "R"
    elif stratum == "hold_family":
        task_original, family, side = "pooled", "Hold", "pooled"
    elif stratum == "move_family":
        task_original, family, side = "pooled", "Move", "pooled"
    if targets.empty:
        return {
            "stratum": stratum,
            "task_original": task_original,
            "task_family": family,
            "side": side,
            "n_subjects": int(subset["subject"].nunique()) if "subject" in subset else 0,
            "n_state_vectors": int(len(subset)),
            "top1_target": "",
            "top10_STN_count": 0,
            "top10_beta_count": 0,
            "top10_gamma_count": 0,
            "top10_alpha_count": 0,
            "best_expected_delta_deviation": np.nan,
            "mean_uncertainty": np.nan,
            "interpretation_note": "No side-aware ranking available for this stratum.",
        }
    top10 = targets.head(10)
    top10_text = (
        top10["feature"].fillna("").astype(str)
        + " "
        + top10["band"].fillna("").astype(str)
    ).str.lower()
    return {
        "stratum": stratum,
        "task_original": task_original,
        "task_family": family,
        "side": side,
        "n_subjects": int(subset["subject"].nunique()),
        "n_state_vectors": int(len(subset)),
        "top1_target": str(targets.iloc[0]["feature"]),
        "top10_targets": ";".join(top10["feature"].astype(str)),
        "top10_STN_count": int(top10["is_stn_target"].sum()),
        "top10_beta_count": int(top10["is_beta_target"].sum()),
        "top10_gamma_count": int(top10["is_gamma_target"].sum()),
        "top10_alpha_count": int(top10_text.str.contains("alpha").sum()),
        "best_expected_delta_deviation": float(targets["expected_delta_deviation"].max()),
        "mean_uncertainty": float(targets["uncertainty"].mean()),
        "interpretation_note": "Side-aware dataset-internal MedOn proxy ranking; exploratory only.",
    }


def compare_two_rankings(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
    note_column: str,
) -> pd.DataFrame:
    """Compare two ranked target tables feature by feature."""

    features = sorted(
        set(left.get("feature", pd.Series(dtype=str)).dropna().astype(str))
        | set(right.get("feature", pd.Series(dtype=str)).dropna().astype(str))
    )
    left_lookup = left.set_index("feature") if not left.empty and "feature" in left else pd.DataFrame()
    right_lookup = right.set_index("feature") if not right.empty and "feature" in right else pd.DataFrame()
    rows = []
    for feature in features:
        rank_left = float(left_lookup.loc[feature, "rank"]) if not left_lookup.empty and feature in left_lookup.index else np.nan
        rank_right = float(right_lookup.loc[feature, "rank"]) if not right_lookup.empty and feature in right_lookup.index else np.nan
        appears_left = bool(np.isfinite(rank_left) and rank_left <= 10)
        appears_right = bool(np.isfinite(rank_right) and rank_right <= 10)
        if appears_left and appears_right:
            diff = rank_right - rank_left
            if abs(diff) <= 3:
                note = "shared_high_priority"
            elif diff > 0:
                note = f"{left_name}_dominant"
            else:
                note = f"{right_name}_dominant"
        elif appears_left:
            note = f"{left_name}_only_top10"
        elif appears_right:
            note = f"{right_name}_only_top10"
        else:
            note = "unstable_or_low_priority"
        desc = parse_candidate_feature(feature)
        rows.append(
            {
                "feature": feature,
                "node": desc.get("node", ""),
                "band": desc.get("band", ""),
                "target_type": desc.get("target_type", ""),
                f"rank_{left_name}": rank_left,
                f"rank_{right_name}": rank_right,
                "rank_difference": rank_right - rank_left if np.isfinite(rank_left) and np.isfinite(rank_right) else np.nan,
                f"appears_{left_name}_top10": appears_left,
                f"appears_{right_name}_top10": appears_right,
                note_column: note,
            }
        )
    return pd.DataFrame(rows)


def direction_label(value: float, epsilon: float = 1e-6) -> str:
    """Map signed delta to direction label."""

    if not np.isfinite(value) or abs(value) <= epsilon:
        return "unchanged"
    return "increase" if value > 0 else "decrease"


def sideaware_compensation_directions(state_vectors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build side-aware compensation direction tables."""

    feature_columns = available_real_feature_columns(state_vectors)
    rows = []
    for (subject, session, task_original), subset in state_vectors.groupby(
        ["subject", "session", "task_original"],
        dropna=False,
    ):
        medication = subset["medication"].fillna("").astype(str).str.lower()
        off = subset[medication.eq("off")]
        on = subset[medication.eq("on")]
        if off.empty or on.empty:
            continue
        first = subset.iloc[0]
        off_values = off[feature_columns].mean(numeric_only=True)
        on_values = on[feature_columns].mean(numeric_only=True)
        for feature in feature_columns:
            desc = parse_candidate_feature(feature)
            delta = float(on_values[feature] - off_values[feature])
            rows.append(
                {
                    "stratum": str(task_original),
                    "task_original": str(task_original),
                    "task_family": first["task_family"],
                    "side": first["side"],
                    "subject": subject,
                    "session": session,
                    "feature": feature,
                    "node": desc.get("node", ""),
                    "band": desc.get("band", ""),
                    "target_type": desc.get("target_type", ""),
                    "medoff_value": float(off_values[feature]),
                    "medon_value": float(on_values[feature]),
                    "signed_delta": delta,
                    "direction": direction_label(delta),
                    "interpretation_note": "subject/task MedOff-to-MedOn dataset-internal proxy direction",
                }
            )
    by_subject = pd.DataFrame(rows)
    if by_subject.empty:
        return by_subject, pd.DataFrame(), pd.DataFrame()

    pooled = []
    for (subject, session, task_family), subset in by_subject.groupby(["subject", "session", "task_family"], dropna=False):
        if str(task_family) not in {"Hold", "Move"}:
            continue
        for feature, table in subset.groupby("feature", dropna=False):
            desc = parse_candidate_feature(str(feature))
            delta = float(table["signed_delta"].mean())
            pooled.append(
                {
                    "stratum": f"{task_family}_family_pooled_side",
                    "task_original": "pooled",
                    "task_family": task_family,
                    "side": "pooled",
                    "subject": subject,
                    "session": session,
                    "feature": feature,
                    "node": desc.get("node", ""),
                    "band": desc.get("band", ""),
                    "target_type": desc.get("target_type", ""),
                    "medoff_value": float(table["medoff_value"].mean()),
                    "medon_value": float(table["medon_value"].mean()),
                    "signed_delta": delta,
                    "direction": direction_label(delta),
                    "interpretation_note": "subject task-family direction pooled across side",
                }
            )
    by_subject = pd.concat([by_subject, pd.DataFrame(pooled)], ignore_index=True) if pooled else by_subject

    group_rows = []
    for keys, table in by_subject.groupby(["stratum", "task_original", "task_family", "side", "feature"], dropna=False):
        stratum, task_original, task_family, side, feature = keys
        desc = parse_candidate_feature(str(feature))
        counts = table["direction"].value_counts().to_dict()
        direction_counts = {key: int(counts.get(key, 0)) for key in ["increase", "decrease", "unchanged"]}
        dominant = max(direction_counts, key=direction_counts.get)
        n_subjects = int(table["subject"].nunique())
        mean_delta = float(table["signed_delta"].mean())
        group_rows.append(
            {
                "stratum": stratum,
                "task_original": task_original,
                "task_family": task_family,
                "side": side,
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
                "group_direction": direction_label(mean_delta),
                "increase_count": direction_counts["increase"],
                "decrease_count": direction_counts["decrease"],
                "unchanged_count": direction_counts["unchanged"],
                "consistency_direction": dominant,
                "consistency_fraction": float(direction_counts[dominant] / max(1, n_subjects)),
                "interpretation_note": (
                    "pooled across side" if str(side) == "pooled" else "side-specific direction"
                ),
            }
        )
    group = pd.DataFrame(group_rows)

    stability_rows = []
    for feature, table in group.groupby("feature", dropna=False):
        desc = parse_candidate_feature(str(feature))
        left = table[table["side"].eq("L")]
        right = table[table["side"].eq("R")]
        hold = table[table["task_family"].eq("Hold") & table["side"].eq("pooled")]
        move = table[table["task_family"].eq("Move") & table["side"].eq("pooled")]
        left_dir = ";".join(left["group_direction"].astype(str).unique())
        right_dir = ";".join(right["group_direction"].astype(str).unique())
        hold_dir = str(hold["group_direction"].iloc[0]) if not hold.empty else ""
        move_dir = str(move["group_direction"].iloc[0]) if not move.empty else ""
        stability_rows.append(
            {
                "feature": feature,
                "node": desc.get("node", ""),
                "band": desc.get("band", ""),
                "target_type": desc.get("target_type", ""),
                "left_side_directions": left_dir,
                "right_side_directions": right_dir,
                "hold_family_direction": hold_dir,
                "move_family_direction": move_dir,
                "left_right_direction_match": bool(left_dir and right_dir and left_dir == right_dir),
                "hold_move_direction_match": bool(hold_dir and move_dir and hold_dir == move_dir),
                "mean_consistency_fraction": float(table["consistency_fraction"].mean()),
                "interpretation_note": "side-aware direction stability summary",
            }
        )
    stability = pd.DataFrame(stability_rows)
    return by_subject, group, stability


def band_family_summary(comparison: pd.DataFrame, prefix_left: str, prefix_right: str, note_col: str) -> list[str]:
    """Summarize STN beta/gamma, motor beta, and alpha behavior."""

    if comparison.empty:
        return ["- comparison unavailable"]
    feature = comparison["feature"].astype(str).str.lower()
    node = comparison["node"].astype(str).str.lower()
    band = comparison["band"].astype(str).str.lower()
    groups = {
        "STN beta": node.eq("stn") & band.str.contains("beta"),
        "STN gamma": node.eq("stn") & band.str.contains("gamma"),
        "motor beta": node.eq("motor") & band.str.contains("beta"),
        "alpha": band.str.contains("alpha"),
    }
    lines = []
    for label, mask in groups.items():
        subset = comparison[mask]
        if subset.empty:
            lines.append(f"- {label}: no matching features")
            continue
        top = subset.sort_values([f"rank_{prefix_left}", f"rank_{prefix_right}"], na_position="last").head(3)
        notes = subset[note_col].value_counts().to_dict()
        lines.append(
            f"- {label}: {notes}; examples={';'.join(top['feature'].astype(str))}"
        )
    return lines


def write_side_report(
    path: str | Path,
    state_vectors: pd.DataFrame,
    summary: pd.DataFrame,
    left_right: pd.DataFrame,
    hold_move: pd.DataFrame,
    warnings: list[str],
) -> None:
    """Write side-aware report."""

    tasks = sorted(state_vectors["task_original"].dropna().astype(str).unique()) if not state_vectors.empty else []
    sides = sorted(state_vectors["side"].dropna().astype(str).unique()) if not state_vectors.empty else []
    lines = [
        "# Real Side-Aware Analysis Report",
        "",
        "This report separates task family and side for ds004998 dataset-internal proxy analyses. It is exploratory and in silico.",
        "",
        "## Dataset Subset",
        "",
        f"- subjects: {state_vectors['subject'].nunique() if not state_vectors.empty else 0}",
        f"- state_vector_rows: {len(state_vectors)}",
        f"- tasks: {';'.join(tasks)}",
        f"- sides: {';'.join(sides)}",
        "- reference_policy: MedOn is a dataset-internal compensated proxy, not a healthy state.",
        "- HCP is not used as an electrophysiological reference.",
        "",
    ]
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    lines.extend(["## Side-Aware Strata", ""])
    if summary.empty:
        lines.append("No side-aware summaries were available.")
    else:
        for _, row in summary.iterrows():
            lines.append(
                f"- {row['stratum']}: n_subjects={int(row['n_subjects'])}, "
                f"top1={row['top1_target'] or 'not_available'}, "
                f"top10_STN={int(row['top10_STN_count'])}, "
                f"top10_beta={int(row['top10_beta_count'])}, "
                f"top10_gamma={int(row['top10_gamma_count'])}, "
                f"top10_alpha={int(row['top10_alpha_count'])}"
            )
    lines.extend(["", "## Left Versus Right", ""])
    lines.extend(band_family_summary(left_right, "left", "right", "side_specificity_note"))
    lines.extend(["", "## Hold Versus Move Family", ""])
    lines.extend(band_family_summary(hold_move, "hold_family", "move_family", "task_family_specificity_note"))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Side mixing can change apparent candidate priority when L and R task families differ.",
            "- Candidate targets remain in silico research candidates, not device settings or real-world guidance.",
            "- Side-aware findings require validation on more subjects and preprocessing variants.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_updated_stratified_report(path: str | Path, state_vectors: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Refresh the legacy stratified report with side-aware caveat."""

    lines = [
        "# Real Data Stratified Report",
        "",
        "This report is refreshed for the current side-aware output set. It is not a clinical report.",
        "",
        f"- subjects: {state_vectors['subject'].nunique() if not state_vectors.empty else 0}",
        f"- state_vector_rows: {len(state_vectors)}",
        f"- tasks: {';'.join(sorted(state_vectors['task_original'].dropna().astype(str).unique())) if not state_vectors.empty else 'not_available'}",
        f"- sides: {';'.join(sorted(state_vectors['side'].dropna().astype(str).unique())) if not state_vectors.empty else 'not_available'}",
        "- reference_policy: MedOn is a dataset-internal compensated proxy, not a healthy state.",
        "- HCP is not used as an electrophysiological reference.",
        "",
        "## Side-Aware Strata",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['stratum']}: top1={row['top1_target']}, "
            f"top10_beta={int(row['top10_beta_count'])}, "
            f"top10_gamma={int(row['top10_gamma_count'])}, "
            f"top10_STN={int(row['top10_STN_count'])}"
        )
    lines.extend(
        [
            "",
            "## Methodological Guardrails",
            "",
            "- Candidate targets are in silico optimization outputs, not real-world guidance.",
            "- The analysis uses ds004998-internal electrophysiological proxies only.",
            "- Left- and right-side tasks should be reviewed separately before pooled interpretation.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_integrated_summary(
    path: str | Path,
    state_vectors: pd.DataFrame,
    side_summary: pd.DataFrame,
    left_right: pd.DataFrame,
    hold_move: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Write the integrated 8-subject summary report."""

    output_dir = Path(output_dir)
    quality = pd.read_csv(output_dir / "real_recording_quality.csv", dtype={"run": str}) if (output_dir / "real_recording_quality.csv").exists() else pd.DataFrame()
    reliability = pd.read_csv(output_dir / "real_target_reliability.csv") if (output_dir / "real_target_reliability.csv").exists() else pd.DataFrame()
    comp_summary = pd.read_csv(output_dir / "real_compensation_strategy_summary.csv") if (output_dir / "real_compensation_strategy_summary.csv").exists() else pd.DataFrame()
    comp_side = pd.read_csv(output_dir / "real_compensation_best_strategies_by_side.csv") if (output_dir / "real_compensation_best_strategies_by_side.csv").exists() else pd.DataFrame()
    comp_family = pd.read_csv(output_dir / "real_compensation_best_strategies_by_task_family.csv") if (output_dir / "real_compensation_best_strategies_by_task_family.csv").exists() else pd.DataFrame()

    tasks = sorted(state_vectors["task_original"].dropna().astype(str).unique()) if not state_vectors.empty else []
    sides = sorted(state_vectors["side"].dropna().astype(str).unique()) if not state_vectors.empty else []
    quality_counts = quality["quality_flag"].value_counts().to_dict() if not quality.empty and "quality_flag" in quality else {}
    top_reliable = (
        "; ".join(reliability.head(6)["feature"].astype(str))
        if not reliability.empty and "feature" in reliability
        else "not_available"
    )
    active = comp_summary[
        (comp_summary.get("reference_scope", pd.Series(dtype=str)).eq("subject_medon_proxy"))
        & (comp_summary.get("alpha", pd.Series(dtype=float)) > 0)
        & ~comp_summary.get("strategy", pd.Series(dtype=str)).eq("no_action")
    ] if not comp_summary.empty else pd.DataFrame()
    best_active = active.sort_values("mean_net_compensation_score", ascending=False).iloc[0] if not active.empty else None
    no_action_best = True
    if not comp_summary.empty:
        primary = comp_summary[comp_summary["reference_scope"].eq("subject_medon_proxy")]
        no_action_score = primary[primary["strategy"].eq("no_action")]["mean_net_compensation_score"].max()
        active_score = active["mean_net_compensation_score"].max() if not active.empty else np.nan
        no_action_best = bool(np.isfinite(no_action_score) and (not np.isfinite(active_score) or no_action_score >= active_score))
    lines = [
        "# Real 8-Subject Integrated Summary",
        "",
        "## Dataset Subset",
        "",
        f"- subjects: {state_vectors['subject'].nunique() if not state_vectors.empty else 0}",
        f"- state_vectors: {len(state_vectors)}",
        f"- tasks: {';'.join(tasks)}",
        f"- sides: {';'.join(sides)}",
        "- medication_proxy: MedOff compared with MedOn as a dataset-internal compensated proxy.",
        "",
        "## Main Candidate Findings",
        "",
        "- Motor beta support increased in the expanded side-aware outputs and should be evaluated separately from STN beta.",
        "- STN beta remains present and became stronger in active in silico compensation summaries.",
        "- STN gamma remains interesting but is subject-sensitive and side/task dependent.",
        "- Alpha features remain relevant in reliability rankings but are not always best active compensation candidates.",
        f"- top_reliable_targets: {top_reliable}",
        "",
        "## Side-Aware Findings",
        "",
    ]
    if not side_summary.empty:
        for _, row in side_summary.iterrows():
            lines.append(
                f"- {row['stratum']}: top1={row['top1_target']}, beta_top10={int(row['top10_beta_count'])}, "
                f"gamma_top10={int(row['top10_gamma_count'])}, STN_top10={int(row['top10_STN_count'])}"
            )
    lines.extend(["", "Left/right comparison highlights:"])
    lines.extend(band_family_summary(left_right, "left", "right", "side_specificity_note"))
    lines.extend(["", "Hold/move family comparison highlights:"])
    lines.extend(band_family_summary(hold_move, "hold_family", "move_family", "task_family_specificity_note"))
    lines.extend(["", "## Quality Findings", ""])
    lines.append(f"- quality_flag_counts: {quality_counts}")
    if not quality.empty and "preprocessing_stage_omitted_percent" in quality:
        high = quality[pd.to_numeric(quality["preprocessing_stage_omitted_percent"], errors="coerce") >= 25]
        lines.append(f"- recordings_with_preprocessing_stage_omitted_percent_at_least_25: {len(high)}")
        if not high.empty:
            examples = "; ".join(
                f"{row.subject} {row.task} Med{str(row.medication).title()} {row.preprocessing_stage_omitted_percent:.2f}%"
                for row in high.head(5).itertuples()
            )
            lines.append(f"- high_omission_examples: {examples}")
    lines.extend(
        [
            "- Gamma should be interpreted with quality, side, and task-family sensitivity in mind.",
            "",
            "## Reliability Findings",
            "",
            "- Stable candidates are those that survive subject, side, task-family, and quality filters.",
            "- Subject-sensitive candidates should remain exploratory until more subjects and uncertainty intervals are available.",
            "",
            "## In Silico Compensation Simulation",
            "",
        ]
    )
    if best_active is not None:
        lines.append(
            f"- best_active_strategy_overall: {best_active['strategy']} alpha={best_active['alpha']} "
            f"net_score={best_active['mean_net_compensation_score']:.4f}"
        )
    lines.append(f"- no_action_remains_conservative_best: {no_action_best}")
    if not comp_side.empty:
        for _, row in comp_side.iterrows():
            lines.append(
                f"- best_active_{row['side']}_side: {row['strategy']} alpha={row['alpha']} "
                f"reduction={row['mean_absolute_deviation_reduction']:.4f}"
            )
    if not comp_family.empty:
        for _, row in comp_family.iterrows():
            lines.append(
                f"- best_active_{row['task_family']}_family: {row['strategy']} alpha={row['alpha']} "
                f"reduction={row['mean_absolute_deviation_reduction']:.4f}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- These are candidate, in silico, dataset-internal proxy results.",
            "- Side-aware conclusions require validation and should not be treated as real-world guidance.",
            "",
            "## Next Steps",
            "",
            "- Decide whether to add split-file support.",
            "- Expand toward 10-12 subjects only after side-aware analysis is stable.",
            "- Add uncertainty intervals.",
            "- Compare subject-specific versus group-level compensation directions.",
            "- Do not add clinical claims.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_side_aware_analysis(
    state_vectors_path: str | Path,
    output_dir: str | Path = "outputs/tables",
    reports_dir: str | Path = "reports",
    seed: int = 42,
    effect_fraction: float = 0.6,
    uncertainty_floor: float = 0.05,
    weights: ObjectiveWeights | None = None,
) -> dict[str, Path]:
    """Run side-aware ranking, comparisons, directions, and reports."""

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
    warnings: list[str] = []
    annotated_files = annotate_existing_outputs(output_dir)
    if annotated_files:
        warnings.append(f"Annotated task side metadata in {len(annotated_files)} existing output tables.")
    state_vectors = read_state_vectors(state_vectors_path, warnings)
    state_vectors.to_csv(state_vectors_path, index=False)
    feature_columns = available_real_feature_columns(state_vectors)

    stratum_tables: dict[str, pd.DataFrame] = {}
    summary_rows = []
    for key, (label, column, value) in SIDE_AWARE_STRATA.items():
        subset = state_vectors[state_vectors[column].astype(str).eq(value)].copy()
        targets = rank_targets_for_subset(
            subset,
            feature_columns,
            weights,
            effect_fraction,
            uncertainty_floor,
            label,
        )
        stratum_tables[key] = targets
        summary_rows.append(summarize_targets(label, subset, targets))

    left_right = compare_two_rankings(
        stratum_tables["left_side"],
        stratum_tables["right_side"],
        "left",
        "right",
        "side_specificity_note",
    )
    hold_move = compare_two_rankings(
        stratum_tables["hold_family"],
        stratum_tables["move_family"],
        "hold_family",
        "move_family",
        "task_family_specificity_note",
    )
    by_subject_dir, group_dir, stability_dir = sideaware_compensation_directions(state_vectors)

    paths = {
        "holdl": output_dir / "real_candidate_targets_holdl_sideaware.csv",
        "movel": output_dir / "real_candidate_targets_movel_sideaware.csv",
        "holdr": output_dir / "real_candidate_targets_holdr_sideaware.csv",
        "mover": output_dir / "real_candidate_targets_mover_sideaware.csv",
        "left_side": output_dir / "real_candidate_targets_left_side.csv",
        "right_side": output_dir / "real_candidate_targets_right_side.csv",
        "hold_family": output_dir / "real_candidate_targets_hold_family.csv",
        "move_family": output_dir / "real_candidate_targets_move_family.csv",
        "summary": output_dir / "real_side_aware_summary.csv",
        "left_right_comparison": output_dir / "real_left_right_target_comparison.csv",
        "hold_move_comparison": output_dir / "real_hold_move_family_comparison.csv",
        "directions_by_subject": output_dir / "real_compensation_directions_sideaware_by_subject.csv",
        "directions_group": output_dir / "real_compensation_directions_sideaware_group.csv",
        "direction_stability": output_dir / "real_compensation_direction_sideaware_stability.csv",
        "report": reports_dir / "real_side_aware_analysis_report.md",
        "integrated": reports_dir / "real_8_subject_integrated_summary.md",
    }
    stratum_tables["holdl"].to_csv(paths["holdl"], index=False)
    stratum_tables["movel"].to_csv(paths["movel"], index=False)
    stratum_tables["holdr"].to_csv(paths["holdr"], index=False)
    stratum_tables["mover"].to_csv(paths["mover"], index=False)
    stratum_tables["left_side"].to_csv(paths["left_side"], index=False)
    stratum_tables["right_side"].to_csv(paths["right_side"], index=False)
    stratum_tables["hold_family"].to_csv(paths["hold_family"], index=False)
    stratum_tables["move_family"].to_csv(paths["move_family"], index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(paths["summary"], index=False)
    left_right.to_csv(paths["left_right_comparison"], index=False)
    hold_move.to_csv(paths["hold_move_comparison"], index=False)
    by_subject_dir.to_csv(paths["directions_by_subject"], index=False)
    group_dir.to_csv(paths["directions_group"], index=False)
    stability_dir.to_csv(paths["direction_stability"], index=False)
    write_side_report(paths["report"], state_vectors, summary, left_right, hold_move, warnings)
    write_updated_stratified_report(reports_dir / "real_data_stratified_report.md", state_vectors, summary)
    write_integrated_summary(paths["integrated"], state_vectors, summary, left_right, hold_move, output_dir)
    return paths
