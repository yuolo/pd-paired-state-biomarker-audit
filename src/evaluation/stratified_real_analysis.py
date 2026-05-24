"""Stratified real-data analysis for ds004998 state vectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.inverse_design.objective import ObjectiveWeights, rank_candidate_targets
from src.pathology_model.build_real_state_vectors import available_real_feature_columns
from src.pathology_model.state_deviation import (
    build_internal_reference_stats,
    compute_internal_deviation_table,
    select_internal_reference_rows,
)


METADATA_COLUMNS = ["subject", "session", "condition", "medication", "task", "run"]


def read_state_vectors(path: str | Path) -> pd.DataFrame:
    """Read real-data state vectors while preserving identifiers."""

    dtype = {column: str for column in METADATA_COLUMNS}
    data = pd.read_csv(path, dtype=dtype)
    for column in METADATA_COLUMNS:
        if column in data:
            data[column] = data[column].fillna("unknown").astype(str)
    return data


def finite_feature_subset(
    current_state: np.ndarray,
    target_state: np.ndarray,
    reference_scale: np.ndarray,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Drop non-finite features before candidate ranking."""

    mask = np.isfinite(current_state) & np.isfinite(target_state) & np.isfinite(reference_scale)
    mask &= reference_scale > 0
    return current_state[mask], target_state[mask], reference_scale[mask], [
        feature for feature, keep in zip(feature_columns, mask, strict=True) if keep
    ]


def target_flags(targets: pd.DataFrame) -> pd.DataFrame:
    """Add beta/STN/gamma/top-5 flags to a target table."""

    data = targets.copy()
    text = (
        data.get("feature", pd.Series("", index=data.index)).fillna("").astype(str)
        + " "
        + data.get("band", pd.Series("", index=data.index)).fillna("").astype(str)
        + " "
        + data.get("node", pd.Series("", index=data.index)).fillna("").astype(str)
    ).str.lower()
    data["is_beta_target"] = text.str.contains("beta")
    data["is_stn_target"] = text.str.contains("stn")
    data["is_gamma_target"] = text.str.contains("gamma")
    data["is_top5"] = data["rank"] <= 5
    return data


def rank_targets_for_subset(
    subset: pd.DataFrame,
    feature_columns: list[str],
    weights: ObjectiveWeights,
    effect_fraction: float,
    uncertainty_floor: float,
    reference_strategy: str,
    label_columns: dict[str, str],
) -> pd.DataFrame:
    """Rank candidate targets for a condition or subject-condition subset."""

    medication = subset["medication"].fillna("unknown").astype(str).str.lower()
    current_rows = subset[medication.eq("off")]
    if current_rows.empty:
        current_rows = subset
    reference_rows, reference_label = select_internal_reference_rows(
        subset,
        current_rows.iloc[0] if not current_rows.empty else None,
        strategy=reference_strategy,
    )
    stats = build_internal_reference_stats(
        subset,
        reference_rows,
        feature_columns,
        reference_mode=reference_label,
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
    for column, value in label_columns.items():
        ranked.insert(0, column, value)
    ranked["reference_mode"] = reference_label
    ranked["n_state_rows"] = len(subset)
    ranked["n_subjects"] = subset["subject"].nunique() if "subject" in subset else 0
    return ranked


def baseline_comparison_for_targets(targets: pd.DataFrame, seed: int, labels: dict[str, str]) -> pd.DataFrame:
    """Compare no action, random, beta heuristic, and full objective."""

    columns = [
        *labels.keys(),
        "method",
        "selected_feature",
        "post_deviation",
        "expected_delta_deviation",
        "intervention_energy",
        "notes",
    ]
    if targets.empty:
        return pd.DataFrame(columns=columns)
    rng = np.random.default_rng(seed)
    baseline = float(targets["baseline_deviation"].iloc[0])
    random_row = targets.iloc[int(rng.integers(0, len(targets)))]
    beta_candidates = targets[targets["is_beta_target"]]
    beta_row = (
        beta_candidates.sort_values("expected_delta_deviation", ascending=False).iloc[0]
        if not beta_candidates.empty
        else random_row
    )
    full_row = targets.sort_values(["objective", "expected_delta_deviation"], ascending=[True, False]).iloc[0]
    rows = [
        {
            **labels,
            "method": "no_action",
            "selected_feature": "",
            "post_deviation": baseline,
            "expected_delta_deviation": 0.0,
            "intervention_energy": 0.0,
            "notes": "Baseline with no abstract action.",
        },
        {
            **labels,
            "method": "random_target_ranking",
            "selected_feature": random_row["feature"],
            "post_deviation": float(random_row["post_deviation"]),
            "expected_delta_deviation": float(random_row["expected_delta_deviation"]),
            "intervention_energy": float(random_row["intervention_energy"]),
            "notes": "Random candidate from available state features.",
        },
        {
            **labels,
            "method": "beta_only_heuristic",
            "selected_feature": beta_row["feature"],
            "post_deviation": float(beta_row["post_deviation"]),
            "expected_delta_deviation": float(beta_row["expected_delta_deviation"]),
            "intervention_energy": float(beta_row["intervention_energy"]),
            "notes": "Best beta-band candidate by expected proxy deviation reduction.",
        },
        {
            **labels,
            "method": "full_inverse_design_objective",
            "selected_feature": full_row["feature"],
            "post_deviation": float(full_row["post_deviation"]),
            "expected_delta_deviation": float(full_row["expected_delta_deviation"]),
            "intervention_energy": float(full_row["intervention_energy"]),
            "notes": "Best candidate by full constrained objective.",
        },
    ]
    return pd.DataFrame(rows, columns=columns)


def rank_stability(
    by_subject_targets: pd.DataFrame,
    group_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize subject-level rank stability for each candidate feature."""

    if by_subject_targets.empty:
        return pd.DataFrame()
    rows = []
    group_lookup = group_targets.set_index(["stratum", "feature"])["rank"].to_dict()
    grouped = by_subject_targets.groupby(["stratum", "feature"], dropna=False)
    for (stratum, feature), table in grouped:
        top5_subjects = sorted(table.loc[table["is_top5"], "subject"].astype(str).unique())
        is_gamma = bool(table["is_gamma_target"].iloc[0])
        top5_frequency = float(table["is_top5"].mean())
        if is_gamma:
            if len(top5_subjects) == 1:
                gamma_pattern = "driven_by_one_subject"
            elif top5_frequency >= 0.6:
                gamma_pattern = "stable_across_subjects"
            else:
                gamma_pattern = "mixed_subject_support"
        else:
            gamma_pattern = ""
        rows.append(
            {
                "stratum": stratum,
                "feature": feature,
                "target_type": table["target_type"].iloc[0],
                "node": table["node"].iloc[0],
                "band": table["band"].iloc[0],
                "group_rank": group_lookup.get((stratum, feature), np.nan),
                "mean_subject_rank": float(table["rank"].mean()),
                "median_subject_rank": float(table["rank"].median()),
                "std_subject_rank": float(table["rank"].std(ddof=0)),
                "n_subjects": int(table["subject"].nunique()),
                "top5_count": int(table["is_top5"].sum()),
                "top5_frequency": top5_frequency,
                "top5_subjects": ";".join(top5_subjects),
                "is_beta_target": bool(table["is_beta_target"].iloc[0]),
                "is_stn_target": bool(table["is_stn_target"].iloc[0]),
                "is_gamma_target": is_gamma,
                "gamma_subject_pattern": gamma_pattern,
            }
        )
    return pd.DataFrame(rows).sort_values(["stratum", "group_rank", "mean_subject_rank"])


def write_stratified_report(
    path: str | Path,
    state_vectors: pd.DataFrame,
    holdl_targets: pd.DataFrame,
    movel_targets: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    """Write a concise stratified real-data report."""

    def summarize_top5(label: str, targets: pd.DataFrame) -> list[str]:
        top5 = targets.head(5)
        beta_count = int(top5["is_beta_target"].sum()) if not top5.empty else 0
        stn_count = int(top5["is_stn_target"].sum()) if not top5.empty else 0
        lines = [f"### {label}", ""]
        lines.append(f"- top5_beta_targets: {beta_count}")
        lines.append(f"- top5_stn_targets: {stn_count}")
        for _, row in top5.iterrows():
            lines.append(
                f"- rank {int(row['rank'])}: {row['feature']} "
                f"(delta={row['expected_delta_deviation']:.4f})"
            )
        return lines

    gamma = stability[stability.get("is_gamma_target", False).astype(bool)] if not stability.empty else pd.DataFrame()
    gamma_summary = (
        gamma.groupby(["stratum", "gamma_subject_pattern"]).size().reset_index(name="n_targets")
        if not gamma.empty
        else pd.DataFrame(columns=["stratum", "gamma_subject_pattern", "n_targets"])
    )
    lines = [
        "# Real Data Stratified Report",
        "",
        "This report stratifies ds004998 real-data proxy outputs. It is not a clinical report and does not estimate therapeutic efficacy.",
        "",
        f"- subjects: {state_vectors['subject'].nunique()}",
        f"- state_vector_rows: {len(state_vectors)}",
        "- reference_policy: MedOn is a dataset-internal compensated proxy, not a healthy state.",
        "- HCP is not used as an electrophysiological reference.",
        "",
        "## Top-5 Group Targets",
        "",
        *summarize_top5("HoldL", holdl_targets),
        "",
        *summarize_top5("MoveL", movel_targets),
        "",
        "## Gamma Stability",
        "",
        "Gamma stability is labeled stable_across_subjects when a gamma target appears in the top 5 for at least 60% of subjects within a stratum.",
        "",
    ]
    if gamma_summary.empty:
        lines.append("No gamma targets were present in the ranked subject tables.")
    else:
        for _, row in gamma_summary.iterrows():
            lines.append(
                f"- {row['stratum']} {row['gamma_subject_pattern']}: {int(row['n_targets'])} targets"
            )
        if not gamma_summary["gamma_subject_pattern"].eq("stable_across_subjects").any():
            lines.append("- No gamma targets met the stable_across_subjects criterion.")
    lines.extend(
        [
            "",
            "## Methodological Guardrails",
            "",
            "- Candidate targets are in silico optimization outputs, not clinical recommendations.",
            "- The analysis uses ds004998-internal electrophysiological proxies only.",
            "- The controller and target rankings do not represent real DBS optimization.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stratified_analysis(
    state_vectors_path: str | Path,
    output_dir: str | Path = "outputs/tables",
    reports_dir: str | Path = "reports",
    seed: int = 42,
    reference_strategy: str = "medon_condition_proxy",
    top_k: int = 12,
    effect_fraction: float = 0.6,
    uncertainty_floor: float = 0.05,
    weights: ObjectiveWeights | None = None,
) -> dict[str, Path]:
    """Run HoldL/MoveL/per-subject/stability analyses."""

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
    feature_columns = available_real_feature_columns(state_vectors)

    deviation_tables = []
    group_targets = []
    baseline_tables = []
    by_subject_targets = []

    for stratum in ["HoldL", "MoveL"]:
        subset = state_vectors[state_vectors["condition"].astype(str).str.lower().eq(stratum.lower())]
        if subset.empty:
            continue
        deviation_tables.append(
            compute_internal_deviation_table(
                subset,
                feature_columns,
                strategy=reference_strategy,
            ).assign(stratum=stratum)
        )
        ranked = rank_targets_for_subset(
            subset,
            feature_columns,
            weights,
            effect_fraction,
            uncertainty_floor,
            reference_strategy,
            {"stratum": stratum},
        )
        ranked = ranked.copy()
        group_targets.append(ranked)
        baseline_tables.append(
            baseline_comparison_for_targets(ranked, seed, {"stratum": stratum})
        )
        for subject, subject_subset in subset.groupby("subject", dropna=False):
            subject_ranked = rank_targets_for_subset(
                subject_subset,
                feature_columns,
                weights,
                effect_fraction,
                uncertainty_floor,
                reference_strategy,
                {"subject": str(subject), "stratum": stratum},
            )
            if not subject_ranked.empty:
                by_subject_targets.append(subject_ranked)

    all_group_targets = pd.concat(group_targets, ignore_index=True) if group_targets else pd.DataFrame()
    by_subject = pd.concat(by_subject_targets, ignore_index=True) if by_subject_targets else pd.DataFrame()
    stability = rank_stability(by_subject, all_group_targets) if not by_subject.empty else pd.DataFrame()
    deviations = pd.concat(deviation_tables, ignore_index=True) if deviation_tables else pd.DataFrame()
    baselines = pd.concat(baseline_tables, ignore_index=True) if baseline_tables else pd.DataFrame()
    top5 = all_group_targets[all_group_targets["rank"] <= 5].copy() if not all_group_targets.empty else pd.DataFrame()

    holdl = all_group_targets[all_group_targets["stratum"].eq("HoldL")]
    movel = all_group_targets[all_group_targets["stratum"].eq("MoveL")]
    paths = {
        "holdl_targets": output_dir / "real_candidate_targets_holdl.csv",
        "movel_targets": output_dir / "real_candidate_targets_movel.csv",
        "by_subject_targets": output_dir / "real_candidate_targets_by_subject.csv",
        "rank_stability": output_dir / "real_target_rank_stability.csv",
        "stratified_deviation": output_dir / "real_deviation_scores_stratified.csv",
        "stratified_baselines": output_dir / "real_stratified_baseline_comparison.csv",
        "stratified_top5": output_dir / "real_stratified_top5_targets.csv",
        "report": reports_dir / "real_data_stratified_report.md",
    }
    holdl.to_csv(paths["holdl_targets"], index=False)
    movel.to_csv(paths["movel_targets"], index=False)
    by_subject.to_csv(paths["by_subject_targets"], index=False)
    stability.to_csv(paths["rank_stability"], index=False)
    deviations.to_csv(paths["stratified_deviation"], index=False)
    baselines.to_csv(paths["stratified_baselines"], index=False)
    top5.to_csv(paths["stratified_top5"], index=False)
    write_stratified_report(paths["report"], state_vectors, holdl, movel, stability)
    return paths
