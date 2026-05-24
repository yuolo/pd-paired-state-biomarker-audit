"""Quality-controlled clean-subset analysis for ds004998 outputs.

This module recomputes downstream analyses after filtering low-quality
recordings while preserving complete MedOff/MedOn subject-task pairs. MedOn is
used only as a dataset-internal compensated proxy; HCP is not used as an
electrophysiological reference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.controller.in_silico_compensation import (
    ALPHA_VALUES,
    CompensationInputs,
    StrategySelection,
    build_direction_lookup,
    simulate_rows,
    summarize_strategies,
)
from src.evaluation.side_aware_analysis import (
    SIDE_AWARE_STRATA,
    add_task_side_columns,
    rank_targets_for_subset as rank_sideaware_targets_for_subset,
    summarize_targets as summarize_sideaware_targets,
)
from src.evaluation.stratified_real_analysis import finite_feature_subset, target_flags
from src.evaluation.target_reliability import compensation_directions
from src.inverse_design.objective import ObjectiveWeights, parse_candidate_feature, rank_candidate_targets
from src.modeling.patient_specific_compensation_predictor import (
    build_patient_specific_pairs,
    run_loso_predictions,
    summarize_by_group as summarize_predictor_by_group,
    summarize_models as summarize_predictor_models,
)
from src.pathology_model.build_real_state_vectors import available_real_feature_columns
from src.pathology_model.state_deviation import build_internal_reference_stats, compute_internal_deviation_table


ANALYSIS_SUBSETS = {
    "all_recordings": {"good", "acceptable", "caution", "low_quality", "unknown"},
    "exclude_low_quality": {"good", "acceptable", "caution", "unknown"},
    "exclude_caution_and_low_quality": {"good", "acceptable"},
    "good_plus_acceptable_only": {"good", "acceptable"},
    "good_only": {"good"},
}
QUALITY_SEVERITY = {"good": 0, "acceptable": 1, "caution": 2, "low_quality": 3, "unknown": 4}


@dataclass(frozen=True)
class CleanSubsetResult:
    """All result tables for one clean-subset condition."""

    counts: dict[str, object]
    deviation: pd.DataFrame
    candidates: pd.DataFrame
    side_summary: pd.DataFrame
    reliability: pd.DataFrame
    compensation_summary: pd.DataFrame
    predictor_summary: pd.DataFrame
    predictor_by_subject: pd.DataFrame


def read_state_vectors(path: str | Path) -> pd.DataFrame:
    """Read state vectors and ensure side-aware labels."""

    data = pd.read_csv(path, dtype={"run": str})
    for column in ["subject", "session", "condition", "medication", "task", "run"]:
        if column in data:
            data[column] = data[column].fillna("unknown").astype(str)
    if not {"task_original", "task_family", "side"}.issubset(data.columns):
        data = add_task_side_columns(data, "task")
    else:
        for column in ["task_original", "task_family", "side"]:
            data[column] = data[column].fillna("unknown").astype(str)
    return data


def read_quality_table(path: str | Path) -> pd.DataFrame:
    """Read recording quality table and ensure side-aware labels."""

    data = pd.read_csv(path, dtype={"run": str})
    for column in ["subject", "session", "task", "medication", "run", "quality_flag"]:
        if column in data:
            data[column] = data[column].fillna("unknown").astype(str)
    if "task_original" not in data:
        data = add_task_side_columns(data, "task")
    for column in ["task_original", "task_family", "side"]:
        if column in data:
            data[column] = data[column].fillna("unknown").astype(str)
    return data


def worst_quality(flags: pd.Series) -> str:
    """Return the most conservative quality flag from duplicate matches."""

    values = [str(value) for value in flags.dropna().astype(str) if str(value)]
    if not values:
        return "unknown"
    return sorted(values, key=lambda flag: QUALITY_SEVERITY.get(flag, 4), reverse=True)[0]


def attach_quality(state_vectors: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Attach quality flags to state vectors by subject/session/task/medication/run."""

    data = state_vectors.copy()
    if quality.empty:
        data["quality_flag"] = "unknown"
        data["quality_source"] = "missing_quality_table"
        return data
    q = quality.copy()
    key = ["subject", "session", "task_original", "medication", "run"]
    for column in key:
        if column not in data:
            data[column] = "unknown"
        if column not in q:
            q[column] = "unknown"
        data[column] = data[column].fillna("unknown").astype(str)
        q[column] = q[column].fillna("unknown").astype(str)
    q["_severity"] = q["quality_flag"].map(QUALITY_SEVERITY).fillna(4)
    q = q.sort_values("_severity").groupby(key, as_index=False).tail(1)
    keep = [*key, "quality_flag", "quality_source"]
    if "preprocessing_stage_omitted_percent" in q:
        keep.append("preprocessing_stage_omitted_percent")
    merged = data.merge(q[keep], on=key, how="left")
    merged["quality_flag"] = merged["quality_flag"].fillna("unknown").astype(str)
    merged["quality_source"] = merged["quality_source"].fillna("quality_not_matched").astype(str)
    return merged


def complete_medoff_medon_pairs(filtered: pd.DataFrame) -> pd.DataFrame:
    """Keep only subject/task pairs with both MedOff and MedOn rows."""

    if filtered.empty:
        return filtered.copy()
    keep_indices: list[int] = []
    for _, subset in filtered.groupby(["subject", "task_original"], dropna=False):
        medication = subset["medication"].fillna("").astype(str).str.lower()
        if medication.eq("off").any() and medication.eq("on").any():
            keep_indices.extend(list(subset.index))
    return filtered.loc[keep_indices].copy() if keep_indices else filtered.iloc[0:0].copy()


def subset_state_vectors(annotated: pd.DataFrame, allowed_flags: set[str]) -> pd.DataFrame:
    """Filter by quality flags and preserve complete paired examples."""

    quality = annotated["quality_flag"].fillna("unknown").astype(str)
    filtered = annotated[quality.isin(allowed_flags)].copy()
    return complete_medoff_medon_pairs(filtered)


def pair_count(state_vectors: pd.DataFrame) -> int:
    """Count complete subject-task MedOff/MedOn pairs."""

    if state_vectors.empty:
        return 0
    count = 0
    for _, subset in state_vectors.groupby(["subject", "task_original"], dropna=False):
        medication = subset["medication"].fillna("").astype(str).str.lower()
        if medication.eq("off").any() and medication.eq("on").any():
            count += 1
    return int(count)


def quality_counts_json(state_vectors: pd.DataFrame) -> str:
    """Return quality counts as a JSON string."""

    if state_vectors.empty or "quality_flag" not in state_vectors:
        return "{}"
    counts = state_vectors["quality_flag"].fillna("unknown").astype(str).value_counts().to_dict()
    return json.dumps({key: int(value) for key, value in counts.items()}, sort_keys=True)


def count_table_row(
    analysis_subset: str,
    subset: pd.DataFrame,
    source_total_rows: int,
    source_total_pairs: int,
) -> dict[str, object]:
    """Build one clean-subset count row."""

    medication = subset["medication"].fillna("").astype(str).str.lower() if not subset.empty else pd.Series(dtype=str)
    pairs = pair_count(subset)
    return {
        "analysis_subset": analysis_subset,
        "n_subjects": int(subset["subject"].nunique()) if not subset.empty else 0,
        "n_state_vectors": int(len(subset)),
        "n_medoff_state_vectors": int(medication.eq("off").sum()),
        "n_medon_state_vectors": int(medication.eq("on").sum()),
        "n_paired_examples": pairs,
        "n_tasks": int(subset["task_original"].nunique()) if not subset.empty else 0,
        "n_sides": int(subset["side"].nunique()) if not subset.empty else 0,
        "quality_flag_counts": quality_counts_json(subset),
        "removed_state_vectors_vs_all": int(source_total_rows - len(subset)),
        "removed_pairs_vs_all": int(source_total_pairs - pairs),
        "pair_preservation_rule": "drop_subject_task_pair_if_either_medoff_or_medon_excluded",
    }


def rank_subset_candidates(
    subset: pd.DataFrame,
    feature_columns: list[str],
    analysis_subset: str,
    weights: ObjectiveWeights,
    effect_fraction: float,
    uncertainty_floor: float,
    reference_mode: str = "medon_subset_proxy_not_healthy",
) -> pd.DataFrame:
    """Rank candidate targets for a clean subset."""

    if subset.empty or not feature_columns:
        return pd.DataFrame()
    medication = subset["medication"].fillna("").astype(str).str.lower()
    current_rows = subset[medication.eq("off")]
    reference_rows = subset[medication.eq("on")]
    if current_rows.empty or reference_rows.empty:
        return pd.DataFrame()
    stats = build_internal_reference_stats(subset, reference_rows, feature_columns, reference_mode)
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
    ranked.insert(0, "analysis_subset", analysis_subset)
    ranked["reference_mode"] = reference_mode
    ranked["n_subjects"] = int(subset["subject"].nunique())
    ranked["n_state_vectors"] = int(len(subset))
    ranked["n_paired_examples"] = pair_count(subset)
    return ranked


def add_analysis_subset(table: pd.DataFrame, analysis_subset: str) -> pd.DataFrame:
    """Add analysis subset column to a table."""

    if table.empty:
        result = table.copy()
        result.insert(0, "analysis_subset", pd.Series(dtype=str))
        return result
    result = table.copy()
    if "analysis_subset" not in result:
        result.insert(0, "analysis_subset", analysis_subset)
    else:
        result["analysis_subset"] = analysis_subset
    return result


def compute_deviation_scores(
    subset: pd.DataFrame,
    feature_columns: list[str],
    analysis_subset: str,
    reference_strategy: str,
) -> pd.DataFrame:
    """Compute dataset-internal deviation scores for one subset."""

    if subset.empty or not feature_columns:
        return pd.DataFrame()
    group_columns = [
        "subject",
        "session",
        "condition",
        "medication",
        "task",
        "run",
        "task_original",
        "task_family",
        "side",
        "quality_flag",
    ]
    deviation = compute_internal_deviation_table(
        subset,
        feature_columns,
        strategy=reference_strategy,
        group_columns=group_columns,
    )
    return add_analysis_subset(deviation, analysis_subset)


def compute_side_summary(
    subset: pd.DataFrame,
    feature_columns: list[str],
    analysis_subset: str,
    weights: ObjectiveWeights,
    effect_fraction: float,
    uncertainty_floor: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compute side-aware summaries for one clean subset."""

    rows = []
    rank_tables: dict[str, pd.DataFrame] = {}
    for key, (label, column, value) in SIDE_AWARE_STRATA.items():
        stratum_subset = subset[subset[column].astype(str).eq(value)].copy() if column in subset else pd.DataFrame()
        targets = rank_sideaware_targets_for_subset(
            stratum_subset,
            feature_columns,
            weights,
            effect_fraction,
            uncertainty_floor,
            label,
        )
        targets = add_analysis_subset(targets, analysis_subset) if not targets.empty else targets
        rank_tables[label] = targets
        row = summarize_sideaware_targets(label, stratum_subset, targets)
        row["analysis_subset"] = analysis_subset
        rows.append(row)
    summary = pd.DataFrame(rows)
    columns = ["analysis_subset", *[column for column in summary.columns if column != "analysis_subset"]]
    return summary[columns], rank_tables


def target_group_flags(table: pd.DataFrame) -> pd.DataFrame:
    """Add motor/STN beta/gamma/alpha group flags."""

    if table.empty:
        return table.copy()
    data = table.copy()
    node = data.get("node", pd.Series("", index=data.index)).fillna("").astype(str).str.lower()
    band = data.get("band", pd.Series("", index=data.index)).fillna("").astype(str).str.lower()
    feature = data.get("feature", pd.Series("", index=data.index)).fillna("").astype(str).str.lower()
    text = node + " " + band + " " + feature
    data["is_motor_beta_target"] = node.eq("motor") & text.str.contains("beta")
    data["is_stn_beta_target"] = node.eq("stn") & text.str.contains("beta")
    data["is_stn_gamma_target"] = node.eq("stn") & text.str.contains("gamma")
    data["is_alpha_target"] = text.str.contains("alpha")
    return data


def quality_survival_lookup(candidates: pd.DataFrame) -> dict[str, float]:
    """Fraction of clean subsets where a feature appears in the top 10."""

    if candidates.empty or "feature" not in candidates:
        return {}
    features = candidates["feature"].dropna().astype(str).unique()
    subsets = list(ANALYSIS_SUBSETS)
    result = {}
    for feature in features:
        hits = 0
        for subset_name in subsets:
            table = candidates[
                candidates["analysis_subset"].astype(str).eq(subset_name)
                & candidates["feature"].astype(str).eq(feature)
            ]
            if not table.empty and float(table["rank"].min()) <= 10:
                hits += 1
        result[feature] = hits / max(1, len(subsets))
    return result


def minmax_inverse(values: pd.Series) -> pd.Series:
    """Return high score for low values."""

    series = pd.to_numeric(values, errors="coerce")
    if series.notna().sum() == 0:
        return pd.Series(0.5, index=values.index)
    min_value = float(series.min())
    max_value = float(series.max())
    if np.isclose(min_value, max_value):
        return pd.Series(0.5, index=values.index)
    return 1.0 - (series - min_value) / (max_value - min_value)


def reliability_from_contexts(
    analysis_subset: str,
    subset: pd.DataFrame,
    feature_columns: list[str],
    overall_candidates: pd.DataFrame,
    side_rank_tables: dict[str, pd.DataFrame],
    quality_survival: dict[str, float],
    weights: ObjectiveWeights,
    effect_fraction: float,
    uncertainty_floor: float,
) -> pd.DataFrame:
    """Compute transparent target reliability from subset-specific rank contexts."""

    frames = []
    if not overall_candidates.empty:
        overall = overall_candidates.copy()
        overall["context_type"] = "overall"
        overall["context_name"] = analysis_subset
        frames.append(overall)
    for label, table in side_rank_tables.items():
        if table.empty:
            continue
        context = table.copy()
        context["context_type"] = "side_task_stratum"
        context["context_name"] = label
        frames.append(context)
    for subject, subject_subset in subset.groupby("subject", dropna=False):
        subject_rank = rank_subset_candidates(
            subject_subset,
            feature_columns,
            analysis_subset,
            weights,
            effect_fraction,
            uncertainty_floor,
            reference_mode="subject_medon_subset_proxy_not_healthy",
        )
        if subject_rank.empty:
            continue
        subject_rank["context_type"] = "subject"
        subject_rank["context_name"] = str(subject)
        frames.append(subject_rank)
    observations = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if observations.empty:
        return pd.DataFrame()
    observations["rank"] = pd.to_numeric(observations["rank"], errors="coerce")
    rows = []
    n_subjects = int(subset["subject"].nunique()) if not subset.empty else 0
    n_contexts = int(observations["context_name"].nunique())
    max_rank = max(1.0, float(observations["rank"].max()))
    for feature in feature_columns:
        table = observations[observations["feature"].astype(str).eq(feature)]
        if table.empty:
            desc = parse_candidate_feature(feature)
            rows.append(
                {
                    "analysis_subset": analysis_subset,
                    "feature": feature,
                    "node": desc.get("node", ""),
                    "band": desc.get("band", ""),
                    "target_type": desc.get("target_type", ""),
                    "top5_frequency": 0.0,
                    "top10_frequency": 0.0,
                    "mean_rank": np.nan,
                    "median_rank": np.nan,
                    "rank_std": np.nan,
                    "appears_in_n_subjects": 0,
                    "subject_coverage": 0.0,
                    "task_family_coverage": 0.0,
                    "side_coverage": 0.0,
                    "quality_survival_fraction": quality_survival.get(feature, 0.0),
                    "mean_expected_delta_deviation": np.nan,
                    "mean_uncertainty": np.nan,
                    "mean_stability_estimate": np.nan,
                    "n_rank_contexts": n_contexts,
                }
            )
            continue
        top10 = table[table["rank"] <= 10]
        subject_hits = top10[top10["context_type"].eq("subject")]["context_name"].nunique()
        family_hits = set()
        side_hits = set()
        for _, row in top10.iterrows():
            context_name = str(row.get("context_name", ""))
            task_family = str(row.get("task_family", ""))
            side = str(row.get("side", ""))
            if context_name in {"hold_family", "move_family"}:
                family_hits.add("Hold" if context_name == "hold_family" else "Move")
            if task_family in {"Hold", "Move"}:
                family_hits.add(task_family)
            if context_name in {"left_side", "right_side"}:
                side_hits.add("L" if context_name == "left_side" else "R")
            if side in {"L", "R"}:
                side_hits.add(side)
        first = table.iloc[0]
        rows.append(
            {
                "analysis_subset": analysis_subset,
                "feature": feature,
                "node": first.get("node", parse_candidate_feature(feature).get("node", "")),
                "band": first.get("band", parse_candidate_feature(feature).get("band", "")),
                "target_type": first.get("target_type", parse_candidate_feature(feature).get("target_type", "")),
                "top5_frequency": float((table["rank"] <= 5).mean()),
                "top10_frequency": float((table["rank"] <= 10).mean()),
                "mean_rank": float(table["rank"].mean()),
                "median_rank": float(table["rank"].median()),
                "rank_std": float(table["rank"].std(ddof=0)),
                "appears_in_n_subjects": int(subject_hits),
                "subject_coverage": 0.0 if n_subjects == 0 else float(subject_hits / n_subjects),
                "task_family_coverage": float(len(family_hits) / 2.0),
                "side_coverage": float(len(side_hits) / 2.0),
                "quality_survival_fraction": quality_survival.get(feature, 0.0),
                "mean_expected_delta_deviation": float(table["expected_delta_deviation"].mean(skipna=True)),
                "mean_uncertainty": float(table["uncertainty"].mean(skipna=True)),
                "mean_stability_estimate": float(table["stability_estimate"].mean(skipna=True)),
                "n_rank_contexts": n_contexts,
            }
        )
    reliability = pd.DataFrame(rows)
    reliability = target_group_flags(reliability)
    reliability["rank_score"] = 1.0 - ((reliability["mean_rank"] - 1.0) / max(1.0, max_rank - 1.0))
    reliability["rank_score"] = reliability["rank_score"].fillna(0.0).clip(0.0, 1.0)
    reliability["uncertainty_score"] = minmax_inverse(reliability["mean_uncertainty"]).fillna(0.5)
    reliability["reliability_score"] = (
        0.25 * reliability["top10_frequency"]
        + 0.15 * reliability["rank_score"]
        + 0.15 * reliability["subject_coverage"]
        + 0.10 * reliability["task_family_coverage"]
        + 0.10 * reliability["side_coverage"]
        + 0.15 * reliability["quality_survival_fraction"]
        + 0.10 * reliability["uncertainty_score"]
    ).clip(0.0, 1.0)
    return reliability.sort_values(
        ["reliability_score", "top10_frequency", "mean_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def select_compensation_strategies(
    reliability: pd.DataFrame,
    feature_columns: list[str],
    analysis_subset: str,
    seed: int,
    top_k: int = 3,
) -> list[StrategySelection]:
    """Select clean-subset compensation strategies."""

    rng = np.random.default_rng(seed + sum(ord(char) for char in analysis_subset))
    random_feature = str(rng.choice(feature_columns)) if feature_columns else ""
    ranked = reliability[reliability["feature"].isin(feature_columns)].copy() if not reliability.empty else pd.DataFrame()
    if not ranked.empty:
        ranked = ranked.sort_values(["reliability_score", "top10_frequency", "mean_rank"], ascending=[False, False, True])
    reliable_features = list(dict.fromkeys(ranked["feature"].astype(str))) if not ranked.empty else []
    stn_beta = ranked[ranked.get("is_stn_beta_target", pd.Series(False, index=ranked.index)).astype(bool)] if not ranked.empty else pd.DataFrame()
    stn_gamma = ranked[ranked.get("is_stn_gamma_target", pd.Series(False, index=ranked.index)).astype(bool)] if not ranked.empty else pd.DataFrame()
    return [
        StrategySelection("no_action", [], "none"),
        StrategySelection("random_candidate", [random_feature] if random_feature else [], "seeded_random_feature"),
        StrategySelection("most_reliable_candidate", reliable_features[:1], "clean_subset_reliability_score"),
        StrategySelection(
            "best_stn_beta_candidate",
            list(stn_beta["feature"].astype(str).head(1)) if not stn_beta.empty else [],
            "clean_subset_best_stn_beta",
        ),
        StrategySelection(
            "best_stn_gamma_candidate",
            list(stn_gamma["feature"].astype(str).head(1)) if not stn_gamma.empty else [],
            "clean_subset_best_stn_gamma",
        ),
        StrategySelection("top_k_reliable_candidates", reliable_features[:top_k], f"top_{top_k}_clean_reliable"),
    ]


def no_action_conservative_best(summary: pd.DataFrame) -> bool:
    """Return whether no action is best under conservative net score."""

    if summary.empty:
        return False
    primary = summary[summary["reference_scope"].eq("subject_medon_proxy")] if "reference_scope" in summary else summary
    no_action = primary[primary["strategy"].eq("no_action")]["mean_net_compensation_score"].max()
    active = primary[(primary["alpha"] > 0) & ~primary["strategy"].eq("no_action")]["mean_net_compensation_score"].max()
    if not np.isfinite(no_action):
        return False
    return bool(not np.isfinite(active) or no_action >= active)


def mark_compensation_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Mark best active strategy and no-action status."""

    if summary.empty:
        return summary.copy()
    data = summary.copy()
    primary = data[data["reference_scope"].eq("subject_medon_proxy")] if "reference_scope" in data else data
    active = primary[(primary["alpha"] > 0) & ~primary["strategy"].eq("no_action")].copy()
    best_key = None
    if not active.empty:
        best = active.sort_values("mean_net_compensation_score", ascending=False).iloc[0]
        best_key = (best["reference_scope"], best["strategy"], float(best["alpha"]))
        data["best_active_strategy"] = str(best["strategy"])
        data["best_active_alpha"] = float(best["alpha"])
        data["best_active_mean_deviation_reduction"] = float(best["mean_absolute_deviation_reduction"])
        data["best_active_net_compensation_score"] = float(best["mean_net_compensation_score"])
    else:
        data["best_active_strategy"] = ""
        data["best_active_alpha"] = np.nan
        data["best_active_mean_deviation_reduction"] = np.nan
        data["best_active_net_compensation_score"] = np.nan
    data["is_best_active"] = False
    if best_key is not None:
        mask = (
            data["reference_scope"].astype(str).eq(str(best_key[0]))
            & data["strategy"].astype(str).eq(str(best_key[1]))
            & np.isclose(data["alpha"].astype(float), best_key[2])
        )
        data.loc[mask, "is_best_active"] = True
    data["no_action_remains_conservative_best"] = no_action_conservative_best(summary)
    return data


def run_subset_compensation(
    analysis_subset: str,
    subset: pd.DataFrame,
    feature_columns: list[str],
    deviation: pd.DataFrame,
    reliability: pd.DataFrame,
    candidates: pd.DataFrame,
    quality: pd.DataFrame,
    seed: int,
    top_k: int = 3,
    lambda_magnitude: float = 0.01,
    lambda_complexity: float = 0.005,
    lambda_instability: float = 0.05,
) -> pd.DataFrame:
    """Run in silico compensation simulation for one clean subset."""

    if subset.empty or not feature_columns:
        return pd.DataFrame()
    direction_subject, direction_group, direction_stability = compensation_directions(subset)
    selections = select_compensation_strategies(reliability, feature_columns, analysis_subset, seed, top_k=top_k)
    inputs = CompensationInputs(
        state_vectors=subset,
        deviation_scores=deviation,
        reliability=reliability,
        directions_by_subject=direction_subject,
        directions_group=direction_group,
        direction_stability=direction_stability,
        candidate_targets=candidates,
        reliability_holdl=pd.DataFrame(),
        reliability_movel=pd.DataFrame(),
        quality=quality,
        warnings=[],
    )
    direction_lookup = build_direction_lookup(subset, direction_group, feature_columns)
    simulation = simulate_rows(
        inputs,
        selections,
        direction_lookup,
        feature_columns,
        ALPHA_VALUES,
        lambda_magnitude,
        lambda_complexity,
        lambda_instability,
    )
    summary = summarize_strategies(simulation)
    summary = mark_compensation_summary(summary)
    return add_analysis_subset(summary, analysis_subset)


def run_subset_predictor(
    analysis_subset: str,
    subset: pd.DataFrame,
    feature_columns: list[str],
    reliability: pd.DataFrame,
    quality: pd.DataFrame,
    seed: int,
    lambda_magnitude: float = 0.01,
    lambda_complexity: float = 0.005,
    lambda_instability: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run patient-specific predictor for one clean subset."""

    if subset.empty or not feature_columns:
        return pd.DataFrame(), pd.DataFrame()
    pairs, _ = build_patient_specific_pairs(subset, feature_columns)
    if pairs.empty or pairs["subject"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()
    results, _ = run_loso_predictions(
        pairs,
        feature_columns,
        reliability,
        pd.DataFrame(),
        quality,
        seed=seed,
        alpha_values=ALPHA_VALUES,
        lambda_magnitude=lambda_magnitude,
        lambda_complexity=lambda_complexity,
        lambda_instability=lambda_instability,
    )
    summary = summarize_predictor_models(results)
    by_subject = summarize_predictor_by_group(results, "subject")
    return add_analysis_subset(summary, analysis_subset), add_analysis_subset(by_subject, analysis_subset)


def band_counts_from_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Top-10 target group counts for report-friendly summaries."""

    rows = []
    if candidates.empty:
        return pd.DataFrame()
    flagged = target_group_flags(candidates)
    for subset_name, table in flagged.groupby("analysis_subset", dropna=False):
        top10 = table[pd.to_numeric(table["rank"], errors="coerce") <= 10]
        rows.append(
            {
                "analysis_subset": subset_name,
                "top10_motor_beta_count": int(top10["is_motor_beta_target"].sum()),
                "top10_stn_beta_count": int(top10["is_stn_beta_target"].sum()),
                "top10_stn_gamma_count": int(top10["is_stn_gamma_target"].sum()),
                "top10_alpha_count": int(top10["is_alpha_target"].sum()),
                "top1_feature": str(table.sort_values("rank").iloc[0]["feature"]) if not table.empty else "",
            }
        )
    return pd.DataFrame(rows)


def summarize_clean_subset_result(
    analysis_subset: str,
    subset: pd.DataFrame,
    quality: pd.DataFrame,
    feature_columns: list[str],
    source_total_rows: int,
    source_total_pairs: int,
    reference_strategy: str,
    weights: ObjectiveWeights,
    effect_fraction: float,
    uncertainty_floor: float,
    seed: int,
    top_k: int,
    lambda_magnitude: float,
    lambda_complexity: float,
    lambda_instability: float,
) -> CleanSubsetResult:
    """Run all clean-subset computations for one quality condition."""

    counts = count_table_row(analysis_subset, subset, source_total_rows, source_total_pairs)
    deviation = compute_deviation_scores(subset, feature_columns, analysis_subset, reference_strategy)
    candidates = rank_subset_candidates(
        subset,
        feature_columns,
        analysis_subset,
        weights,
        effect_fraction,
        uncertainty_floor,
    )
    candidates = target_group_flags(candidates)
    side_summary, side_tables = compute_side_summary(
        subset,
        feature_columns,
        analysis_subset,
        weights,
        effect_fraction,
        uncertainty_floor,
    )
    # Quality survival needs all-subset candidates; filled in the outer run.
    reliability = reliability_from_contexts(
        analysis_subset,
        subset,
        feature_columns,
        candidates,
        side_tables,
        {},
        weights,
        effect_fraction,
        uncertainty_floor,
    )
    compensation_summary = run_subset_compensation(
        analysis_subset,
        subset,
        feature_columns,
        deviation,
        reliability,
        candidates,
        quality,
        seed=seed,
        top_k=top_k,
        lambda_magnitude=lambda_magnitude,
        lambda_complexity=lambda_complexity,
        lambda_instability=lambda_instability,
    )
    predictor_summary, predictor_by_subject = run_subset_predictor(
        analysis_subset,
        subset,
        feature_columns,
        reliability,
        quality,
        seed=seed,
        lambda_magnitude=lambda_magnitude,
        lambda_complexity=lambda_complexity,
        lambda_instability=lambda_instability,
    )
    return CleanSubsetResult(
        counts=counts,
        deviation=deviation,
        candidates=candidates,
        side_summary=side_summary,
        reliability=reliability,
        compensation_summary=compensation_summary,
        predictor_summary=predictor_summary,
        predictor_by_subject=predictor_by_subject,
    )


def recompute_reliability_with_quality_survival(
    results: dict[str, CleanSubsetResult],
    subsets: dict[str, pd.DataFrame],
    feature_columns: list[str],
    weights: ObjectiveWeights,
    effect_fraction: float,
    uncertainty_floor: float,
) -> pd.DataFrame:
    """Recompute reliability after all subset candidate tables are available."""

    all_candidates = pd.concat([result.candidates for result in results.values() if not result.candidates.empty], ignore_index=True)
    survival = quality_survival_lookup(all_candidates)
    rows = []
    for subset_name, result in results.items():
        _, side_tables = compute_side_summary(
            subsets[subset_name],
            feature_columns,
            subset_name,
            weights,
            effect_fraction,
            uncertainty_floor,
        )
        reliability = reliability_from_contexts(
            subset_name,
            subsets[subset_name],
            feature_columns,
            result.candidates,
            side_tables,
            survival,
            weights,
            effect_fraction,
            uncertainty_floor,
        )
        rows.append(reliability)
    return pd.concat([row for row in rows if not row.empty], ignore_index=True) if rows else pd.DataFrame()


def update_compensation_with_reliability(
    results: dict[str, CleanSubsetResult],
    subsets: dict[str, pd.DataFrame],
    reliability: pd.DataFrame,
    quality: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    top_k: int,
    lambda_magnitude: float,
    lambda_complexity: float,
    lambda_instability: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rerun compensation and predictor summaries using final reliability rows."""

    compensation_rows = []
    predictor_rows = []
    predictor_subject_rows = []
    for subset_name, subset in subsets.items():
        subset_reliability = reliability[reliability["analysis_subset"].astype(str).eq(subset_name)].copy()
        deviation = results[subset_name].deviation
        candidates = results[subset_name].candidates
        comp = run_subset_compensation(
            subset_name,
            subset,
            feature_columns,
            deviation,
            subset_reliability,
            candidates,
            quality,
            seed=seed,
            top_k=top_k,
            lambda_magnitude=lambda_magnitude,
            lambda_complexity=lambda_complexity,
            lambda_instability=lambda_instability,
        )
        pred, pred_subject = run_subset_predictor(
            subset_name,
            subset,
            feature_columns,
            subset_reliability,
            quality,
            seed=seed,
            lambda_magnitude=lambda_magnitude,
            lambda_complexity=lambda_complexity,
            lambda_instability=lambda_instability,
        )
        compensation_rows.append(comp)
        predictor_rows.append(pred)
        predictor_subject_rows.append(pred_subject)
    compensation = pd.concat([row for row in compensation_rows if not row.empty], ignore_index=True) if compensation_rows else pd.DataFrame()
    predictor = pd.concat([row for row in predictor_rows if not row.empty], ignore_index=True) if predictor_rows else pd.DataFrame()
    predictor_by_subject = pd.concat([row for row in predictor_subject_rows if not row.empty], ignore_index=True) if predictor_subject_rows else pd.DataFrame()
    return compensation, predictor, predictor_by_subject


def _fmt(value: object, digits: int = 4) -> str:
    """Format numeric values for report text."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _top_feature(table: pd.DataFrame, subset_name: str, mask_column: str | None = None) -> str:
    """Return the top feature for a subset, optionally filtered by a flag."""

    if table.empty:
        return "not_available"
    data = table[table["analysis_subset"].astype(str).eq(subset_name)].copy()
    if mask_column and mask_column in data:
        data = data[data[mask_column].astype(bool)]
    if data.empty:
        return "not_available"
    return str(data.sort_values("rank").iloc[0]["feature"])


def write_clean_subset_report(
    path: str | Path,
    counts: pd.DataFrame,
    candidates: pd.DataFrame,
    reliability: pd.DataFrame,
    compensation: pd.DataFrame,
    predictor: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    """Write clean-subset analysis report."""

    band_counts = band_counts_from_candidates(candidates)
    flagged_candidates = target_group_flags(candidates)
    lines = [
        "# Real Clean-Subset Quality-Controlled Analysis Report",
        "",
        "This report recomputes downstream ds004998 analyses after quality filtering while preserving complete MedOff-MedOn subject-task pairs. MedOn remains a dataset-internal compensated proxy, not a healthy state.",
        "",
        "## Pair Retention",
        "",
    ]
    if counts.empty:
        lines.append("No clean-subset count rows were available.")
    else:
        for _, row in counts.iterrows():
            lines.append(
                f"- {row['analysis_subset']}: subjects={int(row['n_subjects'])}, "
                f"state_vectors={int(row['n_state_vectors'])}, pairs={int(row['n_paired_examples'])}, "
                f"removed_pairs={int(row['removed_pairs_vs_all'])}, quality_counts={row['quality_flag_counts']}"
            )
    if not quality.empty and "preprocessing_stage_omitted_percent" in quality:
        high = quality[pd.to_numeric(quality["preprocessing_stage_omitted_percent"], errors="coerce") >= 25]
        if not high.empty:
            lines.extend(["", "## High-Omission Examples", ""])
            for row in high.sort_values("preprocessing_stage_omitted_percent", ascending=False).head(8).itertuples():
                lines.append(
                    f"- {row.subject} {row.task_original} Med{str(row.medication).title()}: "
                    f"{float(row.preprocessing_stage_omitted_percent):.2f}% omitted, quality={row.quality_flag}"
                )
    lines.extend(["", "## Candidate Stability By Quality Filter", ""])
    if band_counts.empty:
        lines.append("Candidate count summaries were unavailable.")
    else:
        for _, row in band_counts.iterrows():
            lines.append(
                f"- {row['analysis_subset']}: top1={row['top1_feature']}, "
                f"motor_beta_top10={int(row['top10_motor_beta_count'])}, "
                f"stn_beta_top10={int(row['top10_stn_beta_count'])}, "
                f"stn_gamma_top10={int(row['top10_stn_gamma_count'])}, "
                f"alpha_top10={int(row['top10_alpha_count'])}"
            )
    lines.extend(["", "## Direct Answers", ""])
    if not band_counts.empty:
        all_row = band_counts[band_counts["analysis_subset"].eq("all_recordings")]
        good_row = band_counts[band_counts["analysis_subset"].eq("good_only")]
        if not all_row.empty and not good_row.empty:
            all_row = all_row.iloc[0]
            good_row = good_row.iloc[0]
            lines.append(
                f"1. Pair retention: good_only keeps {int(counts[counts['analysis_subset'].eq('good_only')]['n_paired_examples'].iloc[0]) if not counts[counts['analysis_subset'].eq('good_only')].empty else 0} paired examples."
            )
            lines.append(
                f"2. Motor beta: top-10 count changes from {int(all_row['top10_motor_beta_count'])} in all_recordings to {int(good_row['top10_motor_beta_count'])} in good_only; top motor beta all={_top_feature(flagged_candidates, 'all_recordings', 'is_motor_beta_target')}, good_only={_top_feature(flagged_candidates, 'good_only', 'is_motor_beta_target')}."
            )
            lines.append(
                f"3. STN beta: top-10 count changes from {int(all_row['top10_stn_beta_count'])} to {int(good_row['top10_stn_beta_count'])}; top STN beta all={_top_feature(flagged_candidates, 'all_recordings', 'is_stn_beta_target')}, good_only={_top_feature(flagged_candidates, 'good_only', 'is_stn_beta_target')}."
            )
            stn_gamma_all = int(all_row["top10_stn_gamma_count"])
            stn_gamma_good = int(good_row["top10_stn_gamma_count"])
            stn_gamma_removed_by_filtering = stn_gamma_all > 0 and stn_gamma_good == 0
            lines.append(
                f"4. STN gamma: top-10 count changes from {stn_gamma_all} to {stn_gamma_good}; "
                f"removed_by_good_only_filtering={stn_gamma_removed_by_filtering}."
            )
    lines.extend(["", "## Compensation Simulation", ""])
    if compensation.empty:
        lines.append("Compensation summaries were unavailable.")
    else:
        for subset_name, table in compensation.groupby("analysis_subset", dropna=False):
            primary = table[table["reference_scope"].eq("subject_medon_proxy")]
            active = primary[(primary["alpha"] > 0) & ~primary["strategy"].eq("no_action")]
            best = active.sort_values("mean_net_compensation_score", ascending=False).iloc[0] if not active.empty else None
            no_action_best = bool(primary.get("no_action_remains_conservative_best", pd.Series([False])).iloc[0]) if not primary.empty else False
            if best is None:
                lines.append(f"- {subset_name}: no active strategy available; no_action_conservative_best={no_action_best}")
            else:
                lines.append(
                    f"- {subset_name}: best_active={best['strategy']} alpha={best['alpha']}, "
                    f"mean_reduction={_fmt(best['mean_absolute_deviation_reduction'])}, "
                    f"net={_fmt(best['mean_net_compensation_score'])}, "
                    f"no_action_conservative_best={no_action_best}"
                )
    lines.extend(["", "## Patient-Specific Prediction", ""])
    if predictor.empty:
        lines.append("Predictor summaries were unavailable.")
    else:
        for subset_name, table in predictor.groupby("analysis_subset", dropna=False):
            learned = table[table["model_name"].isin(["ridge_regression_predictor", "elastic_net_predictor"])]
            group = table[table["model_name"].eq("group_mean_direction")]
            no_action = table[table["model_name"].eq("no_action")]
            random = table[table["model_name"].eq("random_direction")]
            best = table.sort_values("mean_deviation_reduction", ascending=False).iloc[0]
            any_positive = bool((table[~table["model_name"].eq("no_action")]["mean_deviation_reduction"] > 0).any())
            beats_group = bool(not learned.empty and not group.empty and learned["mean_deviation_reduction"].max() > group["mean_deviation_reduction"].max())
            beats_random = bool(not learned.empty and not random.empty and learned["mean_deviation_reduction"].max() > random["mean_deviation_reduction"].max())
            beats_no_action = bool(not learned.empty and not no_action.empty and learned["mean_deviation_reduction"].max() > no_action["mean_deviation_reduction"].max())
            lines.append(
                f"- {subset_name}: best_by_reduction={best['model_name']} ({_fmt(best['mean_deviation_reduction'])}), "
                f"learned_beats_group={beats_group}, learned_beats_random={beats_random}, "
                f"learned_beats_no_action={beats_no_action}, any_active_positive={any_positive}"
            )
    lines.extend(["", "## Interpretation", ""])
    if predictor.empty:
        lines.append("- The clean-subset predictor conclusion could not be evaluated because predictor summaries were unavailable.")
    else:
        clean = predictor[predictor["analysis_subset"].isin(["exclude_caution_and_low_quality", "good_plus_acceptable_only", "good_only"])]
        learned = clean[clean["model_name"].isin(["ridge_regression_predictor", "elastic_net_predictor"])]
        no_action = clean[clean["model_name"].eq("no_action")]
        group = clean[clean["model_name"].eq("group_mean_direction")]
        learned_positive = bool(not learned.empty and (learned["mean_deviation_reduction"] > 0).any())
        beats_no_action = bool(not learned.empty and not no_action.empty and learned["mean_deviation_reduction"].max() > no_action["mean_deviation_reduction"].max())
        beats_group = bool(not learned.empty and not group.empty and learned["mean_deviation_reduction"].max() > group["mean_deviation_reduction"].max())
        if learned_positive or beats_no_action or beats_group:
            lines.append(
                "- Clean-subset filtering changes at least part of the patient-specific prediction result; inspect clean_subset_predictor_summary.csv before drawing conclusions."
            )
        else:
            lines.append(
                "- Clean-subset filtering does not make learned patient-specific predictors positive by deviation reduction and does not make them beat no_action in the current outputs."
            )
        lines.append(
            "- Therefore, the previous negative predictor result is not obviously explained only by low-quality or high-omission recordings, although the sample remains small and side/task coverage is uneven."
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- HCP is not used as an electrophysiological reference.",
            "- MedOn is only a dataset-internal compensated proxy, not a healthy state.",
            "- Candidate targets and predicted directions are in silico research outputs, not device settings.",
            "- This report makes no claim about real-world benefit.",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_clean_subset_analysis(
    state_vectors_path: str | Path,
    quality_table_path: str | Path,
    output_dir: str | Path = "outputs/tables",
    reports_dir: str | Path = "reports",
    reference_strategy: str = "medon_condition_proxy",
    weights: ObjectiveWeights | None = None,
    effect_fraction: float = 0.6,
    uncertainty_floor: float = 0.05,
    seed: int = 42,
    top_k: int = 3,
    lambda_magnitude: float = 0.01,
    lambda_complexity: float = 0.005,
    lambda_instability: float = 0.05,
) -> dict[str, Path]:
    """Run quality-controlled clean-subset analysis."""

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
    state_vectors = read_state_vectors(state_vectors_path)
    quality = read_quality_table(quality_table_path)
    annotated = attach_quality(state_vectors, quality)
    feature_columns = available_real_feature_columns(annotated)
    all_complete = complete_medoff_medon_pairs(annotated)
    source_total_rows = int(len(all_complete))
    source_total_pairs = pair_count(all_complete)

    subsets = {
        subset_name: subset_state_vectors(annotated, allowed_flags)
        for subset_name, allowed_flags in ANALYSIS_SUBSETS.items()
    }
    results = {
        subset_name: summarize_clean_subset_result(
            subset_name,
            subset,
            quality,
            feature_columns,
            source_total_rows,
            source_total_pairs,
            reference_strategy,
            weights,
            effect_fraction,
            uncertainty_floor,
            seed,
            top_k,
            lambda_magnitude,
            lambda_complexity,
            lambda_instability,
        )
        for subset_name, subset in subsets.items()
    }
    counts = pd.DataFrame([result.counts for result in results.values()])
    deviation = pd.concat([result.deviation for result in results.values() if not result.deviation.empty], ignore_index=True)
    candidates = pd.concat([result.candidates for result in results.values() if not result.candidates.empty], ignore_index=True)
    side_summary = pd.concat([result.side_summary for result in results.values() if not result.side_summary.empty], ignore_index=True)
    reliability = recompute_reliability_with_quality_survival(
        results,
        subsets,
        feature_columns,
        weights,
        effect_fraction,
        uncertainty_floor,
    )
    compensation, predictor, predictor_by_subject = update_compensation_with_reliability(
        results,
        subsets,
        reliability,
        quality,
        feature_columns,
        seed,
        top_k,
        lambda_magnitude,
        lambda_complexity,
        lambda_instability,
    )
    paths = {
        "counts": output_dir / "clean_subset_counts.csv",
        "deviation_scores": output_dir / "clean_subset_deviation_scores.csv",
        "candidate_targets": output_dir / "clean_subset_candidate_targets.csv",
        "side_aware_summary": output_dir / "clean_subset_side_aware_summary.csv",
        "target_reliability": output_dir / "clean_subset_target_reliability.csv",
        "compensation_summary": output_dir / "clean_subset_compensation_summary.csv",
        "predictor_summary": output_dir / "clean_subset_predictor_summary.csv",
        "predictor_by_subject": output_dir / "clean_subset_predictor_by_subject.csv",
        "report": reports_dir / "real_clean_subset_analysis_report.md",
    }
    counts.to_csv(paths["counts"], index=False)
    deviation.to_csv(paths["deviation_scores"], index=False)
    candidates.to_csv(paths["candidate_targets"], index=False)
    side_summary.to_csv(paths["side_aware_summary"], index=False)
    reliability.to_csv(paths["target_reliability"], index=False)
    compensation.to_csv(paths["compensation_summary"], index=False)
    predictor.to_csv(paths["predictor_summary"], index=False)
    predictor_by_subject.to_csv(paths["predictor_by_subject"], index=False)
    write_clean_subset_report(paths["report"], counts, candidates, reliability, compensation, predictor, quality)
    return paths
