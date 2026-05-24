"""Controlled distance-geometry improvements for ds004998 paired-state retrieval.

This script tests a small set of predefined scoring-geometry changes for the
paired MedOff/MedOn retrieval task. It is intentionally conservative: it does
not perform exhaustive feature search, does not introduce deep learning, and
does not make clinical, DBS, treatment, or MedOn-as-healthy claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.predictive_rescue_analysis import (  # noqa: E402
    build_base_candidate_sets,
    build_paired_examples,
    fit_train_scaler,
    read_quality_table,
    read_state_vectors,
    read_subspace_definitions,
    transform,
)


STATE_VECTORS_PATH = Path("outputs/tables/ds004998_state_vectors_enhanced_coupling_expanded.csv")
QUALITY_TABLE_PATH = Path("outputs/tables/real_recording_quality.csv")
SUBSPACE_DEFINITIONS_PATH = Path("outputs/tables/subspace_definitions.csv")
OUTPUT_DIR = Path("outputs/retrieval_distance_geometry_improvement")
FIGURES_DIR = OUTPUT_DIR / "figures"

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
N_PERMUTATIONS_RANDOM_LABEL = 1000
PRIMARY_FEATURE_SET = "clean_stable_features"
PRIMARY_DISTANCE = "cosine"
MIN_DISTRACTORS = 2
MRR_TOLERANCE = 0.01
SAME_SUBJECT_TOLERANCE = 0.02


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class VariantSpec:
    """A predefined retrieval-scoring variant."""

    name: str
    category: str
    score_kind: str
    feature_space: str
    distance: str = "cosine"
    weight_scheme: str = "baseline_all_weights_1"
    residualized: bool = False
    note: str = ""


def to_builtin(value: object) -> object:
    """Convert numpy/pandas objects into JSON-safe Python objects."""

    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def write_json(data: dict[str, object], path: Path) -> Path:
    """Write JSON with parent creation and stable indentation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(data), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(data: pd.DataFrame, path: Path) -> Path:
    """Write CSV with parent creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return path


def write_text(lines: Iterable[str], path: Path) -> Path:
    """Write a Markdown/text report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance with a stable zero-vector fallback."""

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with a stable zero-vector fallback."""

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance."""

    return float(np.linalg.norm(a - b))


def pair_value(row: pd.Series, prefix: str, features: list[str]) -> np.ndarray:
    """Extract one pair vector by prefix."""

    return row[[f"{prefix}{feature}" for feature in features]].to_numpy(dtype=float)


def ordered_unique(values: Iterable[str]) -> list[str]:
    """Return ordered unique string values."""

    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def select_feature_space(
    subspaces: dict[str, list[str]],
    name: str,
    all_features: list[str],
) -> list[str]:
    """Select a predefined feature space without searching."""

    if name == "clean_stable_features":
        return list(subspaces.get("clean_stable_features", []))
    if name == "full_28":
        return list(subspaces.get("full_28", all_features))
    if name == "full_28_plus_coupling":
        return ordered_unique([*subspaces.get("full_28", all_features), *subspaces.get("cortico_stn_coupling", [])])
    if name == "clean_stable_plus_coupling":
        return ordered_unique([*subspaces.get("clean_stable_features", []), *subspaces.get("cortico_stn_coupling", [])])
    return list(subspaces.get(name, []))


def is_coupling_feature(feature: str) -> bool:
    """Return whether a feature is a cortico-STN coupling/network feature."""

    text = feature.lower()
    return "coupling" in text or "coherence" in text or "cortico_stn" in text


def is_asymmetry_feature(feature: str) -> bool:
    """Return whether a feature is an asymmetry/laterality feature."""

    text = feature.lower()
    return "asymmetry" in text or "laterality" in text


def is_cortical_beta_gamma_feature(feature: str) -> bool:
    """Return whether a feature is one of the predefined cortical beta/gamma power features."""

    text = feature.lower()
    cortical = text.startswith("meg_") or text.startswith("motor_")
    band = "beta" in text or "gamma" in text
    power_like = "power" in text and not is_coupling_feature(text)
    return cortical and band and power_like


def feature_group_name(feature: str, clean_features: set[str]) -> str:
    """Assign a feature to one conservative group for group-balanced scoring."""

    text = feature.lower()
    if is_coupling_feature(text):
        if is_asymmetry_feature(text):
            return "asymmetry_network_features"
        return "cortico_stn_coupling"
    if is_asymmetry_feature(text):
        return "asymmetry_network_features"
    if text.startswith("meg_") and "power" in text:
        return "meg_power_alpha_beta_gamma"
    if text.startswith("motor_") and "power" in text:
        return "motor_power_alpha_beta_gamma"
    if text.startswith("stn_") and ("beta" in text or "gamma" in text):
        return "stn_beta_gamma"
    if feature in clean_features:
        return "clean_stable_features"
    return "other_features"


def build_feature_groups(features: list[str], clean_features: list[str]) -> dict[str, list[int]]:
    """Build predefined feature-group indices for available features."""

    clean = set(clean_features)
    groups: dict[str, list[int]] = {}
    for idx, feature in enumerate(features):
        groups.setdefault(feature_group_name(feature, clean), []).append(idx)
    return {name: indices for name, indices in groups.items() if indices}


def weighting_vector(features: list[str], scheme: str) -> np.ndarray:
    """Return predefined conservative feature weights."""

    weights = np.ones(len(features), dtype=float)
    if scheme in {"cortical_beta_gamma_weight_0_5", "beta_gamma_weight_0_5_plus_coupling_weight_2"}:
        for idx, feature in enumerate(features):
            if is_cortical_beta_gamma_feature(feature):
                weights[idx] *= 0.5
    if scheme in {"cortical_beta_gamma_weight_0_25", "beta_gamma_weight_0_25_plus_coupling_weight_2"}:
        for idx, feature in enumerate(features):
            if is_cortical_beta_gamma_feature(feature):
                weights[idx] *= 0.25
    if scheme in {
        "coupling_features_weight_2",
        "beta_gamma_weight_0_5_plus_coupling_weight_2",
        "beta_gamma_weight_0_25_plus_coupling_weight_2",
    }:
        for idx, feature in enumerate(features):
            if is_coupling_feature(feature):
                weights[idx] *= 2.0
    return weights


def zscore(values: np.ndarray) -> np.ndarray:
    """Z-score a 1D array with small-sample fallback."""

    arr = np.asarray(values, dtype=float)
    std = float(np.nanstd(arr))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    mean = float(np.nanmean(arr)) if len(arr) else 0.0
    return (arr - mean) / std


def rank_candidate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank candidate scores within each query and variant."""

    if scores.empty:
        return scores
    ranked = scores.sort_values(["variant_name", "query_pair_id", "score"], ascending=True).copy()
    ranked["rank"] = ranked.groupby(["variant_name", "query_pair_id"], dropna=False).cumcount() + 1
    return ranked


def classify_failure(row: pd.Series) -> str:
    """Classify top1 retrieval failures using diagnosed failure categories."""

    if bool(row.get("top_ranked_is_true_pair", False)):
        return "success"
    query_quality = str(row.get("query_quality"))
    top_quality = str(row.get("top_ranked_candidate_quality"))
    if query_quality in {"low_quality", "caution"} or top_quality != query_quality:
        return "quality_related"
    same_subject = str(row.get("top_ranked_candidate_subject")) == str(row.get("query_subject"))
    same_task = str(row.get("top_ranked_candidate_task")) == str(row.get("query_task"))
    same_side = str(row.get("top_ranked_candidate_side")) == str(row.get("query_side"))
    if same_subject and not same_task:
        return "same_subject_wrong_task"
    if same_subject and not same_side:
        return "same_subject_wrong_side"
    if (not same_subject) and same_task and same_side:
        return "other_subject_same_task_side"
    if not same_task:
        return "other_subject_wrong_task"
    return "unclear"


def query_diagnostics_from_scores(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    """Create one diagnostic row per query and variant."""

    if candidate_scores.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["variant_name", "query_pair_id"]
    for (variant_name, query_id), group in candidate_scores.groupby(group_cols, dropna=False):
        true = group[group["is_true_pair"].astype(bool)]
        if true.empty:
            continue
        true_row = true.iloc[0]
        top = group.sort_values("rank").iloc[0]
        distractors = group[~group["is_true_pair"].astype(bool)]
        nearest = distractors.sort_values("score").iloc[0] if not distractors.empty else pd.Series(dtype=object)
        top1 = bool(int(true_row["rank"]) == 1)
        row = {
            "variant_name": variant_name,
            "query_pair_id": query_id,
            "query_subject": true_row["query_subject"],
            "query_task": true_row["query_task_original"],
            "query_side": true_row["query_side"],
            "query_quality": true_row["query_quality_flag"],
            "true_medon_quality": true_row["candidate_quality_flag"],
            "true_medon_rank": int(true_row["rank"]),
            "top_ranked_candidate_pair_id": top["candidate_pair_id"],
            "top_ranked_candidate_subject": top["candidate_subject"],
            "top_ranked_candidate_task": top["candidate_task_original"],
            "top_ranked_candidate_side": top["candidate_side"],
            "top_ranked_candidate_quality": top["candidate_quality_flag"],
            "top_ranked_is_true_pair": top1,
            "distance_to_true_medon": float(true_row["score"]),
            "distance_to_nearest_distractor": float(nearest.get("score", np.nan)),
            "retrieval_margin": float(nearest.get("score", np.nan) - true_row["score"]),
            "candidate_set_size": int(true_row["number_of_candidates"]),
            "chance_top1": 1.0 / float(true_row["number_of_candidates"]),
            "reciprocal_rank": 1.0 / float(true_row["rank"]),
            "percentile_rank": 1.0
            - ((float(true_row["rank"]) - 1.0) / max(1.0, float(true_row["number_of_candidates"]) - 1.0)),
        }
        row["failure_type"] = classify_failure(pd.Series(row))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_diagnostics(diagnostics: pd.DataFrame) -> dict[str, float | int]:
    """Summarize query-level retrieval metrics."""

    if diagnostics.empty:
        return {
            "top1": np.nan,
            "mrr": np.nan,
            "percentile_rank": np.nan,
            "retrieval_margin": np.nan,
            "n_pairs": 0,
            "n_candidates_mean": np.nan,
            "other_subject_same_task_side_failure_count": 0,
            "quality_related_failure_count": 0,
        }
    failures = diagnostics[~diagnostics["top_ranked_is_true_pair"].astype(bool)]
    return {
        "top1": float(diagnostics["top_ranked_is_true_pair"].astype(float).mean()),
        "mrr": float(diagnostics["reciprocal_rank"].mean()),
        "percentile_rank": float(diagnostics["percentile_rank"].mean()),
        "retrieval_margin": float(diagnostics["retrieval_margin"].mean()),
        "n_pairs": int(len(diagnostics)),
        "n_candidates_mean": float(diagnostics["candidate_set_size"].mean()),
        "other_subject_same_task_side_failure_count": int(failures["failure_type"].eq("other_subject_same_task_side").sum()),
        "quality_related_failure_count": int(failures["failure_type"].eq("quality_related").sum()),
    }


def fit_residualizer(train: pd.DataFrame, features: list[str]) -> dict[str, object]:
    """Fit train-subject-only linear residualizer for task/side/quality."""

    rows: list[dict[str, object]] = []
    for _, pair in train.iterrows():
        for prefix in ["x_", "medon_"]:
            record: dict[str, object] = {
                "task_original": pair["task_original"],
                "side": pair["side"],
                "quality_flag": pair.get("quality_flag", "unknown"),
            }
            for feature in features:
                record[feature] = pair[f"{prefix}{feature}"]
            rows.append(record)
    frame = pd.DataFrame(rows)
    design = pd.get_dummies(frame[["task_original", "side", "quality_flag"]], drop_first=False)
    design.insert(0, "intercept", 1.0)
    columns = list(design.columns)
    x = design.to_numpy(dtype=float)
    y = frame[features].to_numpy(dtype=float)
    beta = np.linalg.pinv(x) @ y
    return {"columns": columns, "beta": beta}


def apply_residualizer(row: pd.Series, prefix: str, features: list[str], residualizer: dict[str, object]) -> np.ndarray:
    """Apply a train-only residualizer to one row."""

    design = pd.DataFrame(
        [
            {
                "task_original": row["task_original"],
                "side": row["side"],
                "quality_flag": row.get("quality_flag", "unknown"),
            }
        ]
    )
    design = pd.get_dummies(design, drop_first=False)
    design.insert(0, "intercept", 1.0)
    columns = list(residualizer["columns"])
    for column in columns:
        if column not in design:
            design[column] = 0.0
    x = design[columns].to_numpy(dtype=float)
    observed = pair_value(row, prefix, features)
    predicted = np.asarray(x @ np.asarray(residualizer["beta"], dtype=float)).reshape(-1)
    return observed - predicted


def prepare_context(train: pd.DataFrame, features: list[str], residualized: bool) -> dict[str, object]:
    """Prepare fold-specific scaling, residualization, and transition direction."""

    residualizer: dict[str, object] | None = fit_residualizer(train, features) if residualized else None
    x_rows = []
    m_rows = []
    for _, pair in train.iterrows():
        if residualizer is None:
            x_rows.append(pair_value(pair, "x_", features))
            m_rows.append(pair_value(pair, "medon_", features))
        else:
            x_rows.append(apply_residualizer(pair, "x_", features, residualizer))
            m_rows.append(apply_residualizer(pair, "medon_", features, residualizer))
    values = np.vstack([x_rows, m_rows])
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0, ddof=1)
    std = np.where(~np.isfinite(std) | (std < 1e-8), 1.0, std)
    x_z = (np.vstack(x_rows) - mean) / std
    m_z = (np.vstack(m_rows) - mean) / std
    mean_delta = np.nanmean(m_z - x_z, axis=0)
    return {
        "mean": mean,
        "std": std,
        "residualizer": residualizer,
        "mean_delta": mean_delta,
    }


def vector_for(row: pd.Series, prefix: str, features: list[str], context: dict[str, object]) -> np.ndarray:
    """Return transformed vector for one row under fold context."""

    residualizer = context.get("residualizer")
    if residualizer is None:
        values = pair_value(row, prefix, features)
    else:
        values = apply_residualizer(row, prefix, features, residualizer)  # type: ignore[arg-type]
    return transform(values, np.asarray(context["mean"], dtype=float), np.asarray(context["std"], dtype=float))


def score_candidate_array(
    qx: np.ndarray,
    candidate_vectors: list[np.ndarray],
    features: list[str],
    variant: VariantSpec,
    clean_features: list[str],
    context: dict[str, object],
) -> list[float]:
    """Score all candidates for one query under a variant. Lower is better."""

    if not candidate_vectors:
        return []
    if variant.score_kind == "standard":
        if variant.distance == "euclidean":
            return [euclidean_distance(qx, cm) for cm in candidate_vectors]
        return [cosine_distance(qx, cm) for cm in candidate_vectors]

    if variant.score_kind == "group_balanced":
        groups = build_feature_groups(features, clean_features)
        scores: list[float] = []
        for cm in candidate_vectors:
            group_scores = []
            for indices in groups.values():
                qa = qx[indices]
                ca = cm[indices]
                if variant.distance == "euclidean":
                    group_scores.append(euclidean_distance(qa, ca))
                else:
                    group_scores.append(cosine_distance(qa, ca))
            scores.append(float(np.nanmean(group_scores)) if group_scores else np.nan)
        return scores

    if variant.score_kind == "weighted":
        weights = weighting_vector(features, variant.weight_scheme)
        q_weighted = qx * weights
        return [cosine_distance(q_weighted, cm * weights) for cm in candidate_vectors]

    if variant.score_kind in {"transition_only", "combined_transition"}:
        mean_delta = np.asarray(context["mean_delta"], dtype=float)
        distances = np.asarray([cosine_distance(qx, cm) for cm in candidate_vectors], dtype=float)
        transition = np.asarray([cosine_similarity(cm - qx, mean_delta) for cm in candidate_vectors], dtype=float)
        if variant.score_kind == "transition_only":
            return [float(-value) for value in transition]
        distance_similarity = -distances
        combined = zscore(distance_similarity) + zscore(transition)
        return [float(-value) for value in combined]

    raise ValueError(f"Unsupported score_kind: {variant.score_kind}")


def evaluate_variant(
    pairs: pd.DataFrame,
    base_candidates: pd.DataFrame,
    features: list[str],
    variant: VariantSpec,
    clean_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one predefined variant with subject-held-out train-only transforms."""

    if pairs.empty or base_candidates.empty or not features:
        return pd.DataFrame(), pd.DataFrame()
    pair_lookup = pairs.set_index("pair_id", drop=False)
    rows: list[dict[str, object]] = []
    for heldout_subject in sorted(pairs["subject"].astype(str).unique()):
        train = pairs[~pairs["subject"].astype(str).eq(heldout_subject)].copy()
        test_ids = pairs.loc[pairs["subject"].astype(str).eq(heldout_subject), "pair_id"].astype(str).tolist()
        if train.empty:
            continue
        context = prepare_context(train, features, variant.residualized)
        for query_id in test_ids:
            query = pair_lookup.loc[query_id]
            qx = vector_for(query, "x_", features, context)
            candidates = base_candidates[base_candidates["query_pair_id"].astype(str).eq(str(query_id))]
            candidate_rows = [pair_lookup.loc[str(row["candidate_pair_id"])] for _, row in candidates.iterrows()]
            candidate_vectors = [vector_for(candidate, "medon_", features, context) for candidate in candidate_rows]
            scores = score_candidate_array(qx, candidate_vectors, features, variant, clean_features, context)
            for (_, candidate_meta), score in zip(candidates.iterrows(), scores, strict=False):
                rows.append(
                    {
                        **candidate_meta.to_dict(),
                        "heldout_subject": heldout_subject,
                        "variant_name": variant.name,
                        "variant_category": variant.category,
                        "feature_space": variant.feature_space,
                        "score_kind": variant.score_kind,
                        "distance": variant.distance,
                        "weight_scheme": variant.weight_scheme,
                        "residualized": bool(variant.residualized),
                        "n_features": len(features),
                        "score": float(score),
                    }
                )
    scores = rank_candidate_scores(pd.DataFrame(rows))
    return scores, query_diagnostics_from_scores(scores)


def same_subject_hard_negative(
    pairs: pd.DataFrame,
    features: list[str],
    variant: VariantSpec,
    clean_features: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate whether true MedOn beats same-subject wrong-task/wrong-side MedOn."""

    rows: list[dict[str, object]] = []
    for _, query in pairs.iterrows():
        subject = str(query["subject"])
        negatives = pairs[pairs["subject"].astype(str).eq(subject) & ~pairs["pair_id"].astype(str).eq(str(query["pair_id"]))]
        if negatives.empty:
            continue
        train = pairs[~pairs["subject"].astype(str).eq(subject)].copy()
        if train.empty:
            continue
        context = prepare_context(train, features, variant.residualized)
        qx = vector_for(query, "x_", features, context)
        candidate_rows = [query, *[row for _, row in negatives.iterrows()]]
        candidate_vectors = [vector_for(candidate, "medon_", features, context) for candidate in candidate_rows]
        scores = score_candidate_array(qx, candidate_vectors, features, variant, clean_features, context)
        true_score = scores[0]
        neg_scores = scores[1:]
        nearest_idx = int(np.nanargmin(neg_scores)) if neg_scores else -1
        nearest = negatives.iloc[nearest_idx] if nearest_idx >= 0 else pd.Series(dtype=object)
        rank = 1 + int(np.sum(np.asarray(neg_scores, dtype=float) < true_score))
        rows.append(
            {
                "variant_name": variant.name,
                "query_subject": subject,
                "query_task": query["task_original"],
                "query_side": query["side"],
                "true_rank_among_same_subject_candidates": int(rank),
                "n_same_subject_candidates": int(len(candidate_rows)),
                "true_beats_all_same_subject_negatives": bool(rank == 1),
                "nearest_same_subject_negative_task": nearest.get("task_original", ""),
                "nearest_same_subject_negative_side": nearest.get("side", ""),
                "distance_true": float(true_score),
                "distance_nearest_same_subject_negative": float(neg_scores[nearest_idx]) if nearest_idx >= 0 else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "variant_name": variant.name,
        "queries_with_same_subject_hard_negatives": int(len(table)),
        "true_beats_all_rate": float(table["true_beats_all_same_subject_negatives"].mean()) if not table.empty else np.nan,
        "mean_true_rank": float(table["true_rank_among_same_subject_candidates"].mean()) if not table.empty else np.nan,
        "by_task": table.groupby("query_task")["true_beats_all_same_subject_negatives"].mean().to_dict()
        if not table.empty
        else {},
    }
    return table, summary


def summarize_variant(
    diagnostics: pd.DataFrame,
    hard_summary: dict[str, object],
    variant: VariantSpec,
    baseline_diag: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Create one summary row for a variant."""

    metrics = summarize_diagnostics(diagnostics)
    row: dict[str, object] = {
        "variant_name": variant.name,
        "variant_category": variant.category,
        "score_kind": variant.score_kind,
        "feature_space": variant.feature_space,
        "distance": variant.distance,
        "weight_scheme": variant.weight_scheme,
        "residualized": bool(variant.residualized),
        **metrics,
        "same_subject_hard_negative_success": hard_summary.get("true_beats_all_rate", np.nan),
    }
    if baseline_diag is not None and not baseline_diag.empty:
        change = query_change_log_for_variant(baseline_diag, diagnostics, variant.name)
        row.update(
            {
                "fail_to_success": int(change["change_type"].eq("fail_to_success").sum()),
                "success_to_failure": int(change["change_type"].eq("success_to_failure").sum()),
                "net_query_improvement": int(change["change_type"].eq("fail_to_success").sum())
                - int(change["change_type"].eq("success_to_failure").sum()),
            }
        )
    return row


def query_change_log_for_variant(
    baseline_diag: pd.DataFrame,
    variant_diag: pd.DataFrame,
    variant_name: str,
) -> pd.DataFrame:
    """Compare query-level outcomes for one variant against baseline."""

    b = baseline_diag.set_index("query_pair_id", drop=False)
    v = variant_diag.set_index("query_pair_id", drop=False)
    rows: list[dict[str, object]] = []
    for query_id in sorted(set(b.index).intersection(set(v.index))):
        brow = b.loc[query_id]
        vrow = v.loc[query_id]
        b_success = bool(brow["top_ranked_is_true_pair"])
        v_success = bool(vrow["top_ranked_is_true_pair"])
        if b_success and v_success:
            change_type = "unchanged_success"
        elif (not b_success) and (not v_success):
            change_type = "unchanged_failure"
        elif (not b_success) and v_success:
            change_type = "fail_to_success"
        else:
            change_type = "success_to_failure"
        rows.append(
            {
                "variant_name": variant_name,
                "query_subject": brow["query_subject"],
                "query_task": brow["query_task"],
                "query_side": brow["query_side"],
                "baseline_true_rank": int(brow["true_medon_rank"]),
                "variant_true_rank": int(vrow["true_medon_rank"]),
                "baseline_top1_success": b_success,
                "variant_top1_success": v_success,
                "change_type": change_type,
                "baseline_top_candidate_subject": brow["top_ranked_candidate_subject"],
                "variant_top_candidate_subject": vrow["top_ranked_candidate_subject"],
                "baseline_top_candidate_task": brow["top_ranked_candidate_task"],
                "variant_top_candidate_task": vrow["top_ranked_candidate_task"],
                "baseline_top_candidate_side": brow["top_ranked_candidate_side"],
                "variant_top_candidate_side": vrow["top_ranked_candidate_side"],
                "baseline_distance_true": float(brow["distance_to_true_medon"]),
                "variant_distance_true": float(vrow["distance_to_true_medon"]),
                "baseline_margin": float(brow["retrieval_margin"]),
                "variant_margin": float(vrow["retrieval_margin"]),
            }
        )
    return pd.DataFrame(rows)


def query_change_summary(change_log: pd.DataFrame) -> dict[str, object]:
    """Summarize query-level changes for all variants."""

    if change_log.empty:
        return {}
    summary: dict[str, object] = {
        "fail_to_success_by_variant": {},
        "success_to_failure_by_variant": {},
        "net_improvement_by_variant": {},
    }
    for variant, subset in change_log.groupby("variant_name", dropna=False):
        fail_to_success = int(subset["change_type"].eq("fail_to_success").sum())
        success_to_failure = int(subset["change_type"].eq("success_to_failure").sum())
        summary["fail_to_success_by_variant"][str(variant)] = fail_to_success
        summary["success_to_failure_by_variant"][str(variant)] = success_to_failure
        summary["net_improvement_by_variant"][str(variant)] = fail_to_success - success_to_failure
    changed = change_log[change_log["change_type"].isin(["fail_to_success", "success_to_failure"])]
    if not changed.empty:
        changed["query_id"] = changed["query_subject"].astype(str) + "|" + changed["query_task"].astype(str)
        repeated = changed["query_id"].value_counts()
        summary["queries_changed_repeatedly_across_variants"] = {
            str(k): int(v) for k, v in repeated[repeated > 1].to_dict().items()
        }
    else:
        summary["queries_changed_repeatedly_across_variants"] = {}
    return summary


def evaluate_variant_set(
    pairs: pd.DataFrame,
    base_candidates: pd.DataFrame,
    subspaces: dict[str, list[str]],
    all_features: list[str],
    variants: list[VariantSpec],
    baseline_diag: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, dict[str, object]]]:
    """Evaluate a list of variants and return summary/diagnostics/scores/hard summaries."""

    clean_features = select_feature_space(subspaces, PRIMARY_FEATURE_SET, all_features)
    rows: list[dict[str, object]] = []
    diagnostics: dict[str, pd.DataFrame] = {}
    scores: dict[str, pd.DataFrame] = {}
    hard_summaries: dict[str, dict[str, object]] = {}
    for variant in variants:
        features = select_feature_space(subspaces, variant.feature_space, all_features)
        if not features:
            rows.append(
                {
                    "variant_name": variant.name,
                    "variant_category": variant.category,
                    "status": "skipped_missing_features",
                    "note": variant.note,
                }
            )
            continue
        score_table, diag = evaluate_variant(pairs, base_candidates, features, variant, clean_features)
        _, hard_summary = same_subject_hard_negative(pairs, features, variant, clean_features)
        scores[variant.name] = score_table
        diagnostics[variant.name] = diag
        hard_summaries[variant.name] = hard_summary
        row = summarize_variant(diag, hard_summary, variant, baseline_diag)
        row["status"] = "ok"
        row["note"] = variant.note
        rows.append(row)
    return pd.DataFrame(rows), diagnostics, scores, hard_summaries


def random_label_negative_control(
    candidate_scores: pd.DataFrame,
    diagnostics: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Global MedOn-label shuffle negative control using fixed candidate scores."""

    rng = np.random.default_rng(seed)
    query_ids = diagnostics["query_pair_id"].astype(str).tolist()
    pair_ids = sorted(set(candidate_scores["candidate_pair_id"].astype(str)))
    grouped = {str(query_id): group.copy() for query_id, group in candidate_scores.groupby("query_pair_id", dropna=False)}
    rows: list[dict[str, object]] = []
    for permutation_index in range(n_permutations):
        shuffled_labels = rng.permutation(pair_ids)
        records = []
        for idx, query_id in enumerate(query_ids):
            group = grouped[query_id]
            pseudo_id = str(shuffled_labels[idx % len(shuffled_labels)])
            pseudo = group[group["candidate_pair_id"].astype(str).eq(pseudo_id)]
            if pseudo.empty:
                n_candidates = float(group["number_of_candidates"].iloc[0])
                rank = n_candidates + 1.0
                score = float(group["score"].max()) + 1e-6
                best_other = float(group["score"].min())
                chance = 1.0 / n_candidates
            else:
                row = pseudo.iloc[0]
                rank = float(row["rank"])
                score = float(row["score"])
                other = group[group["candidate_pair_id"].astype(str).ne(pseudo_id)]
                best_other = float(other["score"].min()) if not other.empty else np.nan
                chance = 1.0 / float(row["number_of_candidates"])
            n_for_percentile = max(1.0, float(group["number_of_candidates"].iloc[0]) - 1.0)
            records.append(
                {
                    "top1": float(rank == 1.0),
                    "mrr": 1.0 / rank,
                    "percentile_rank": max(0.0, 1.0 - ((rank - 1.0) / n_for_percentile)),
                    "retrieval_margin": best_other - score if np.isfinite(best_other) else np.nan,
                    "chance_top1": chance,
                }
            )
        frame = pd.DataFrame(records)
        rows.append(
            {
                "permutation_index": permutation_index,
                "top1": float(frame["top1"].mean()),
                "mrr": float(frame["mrr"].mean()),
                "percentile_rank": float(frame["percentile_rank"].mean()),
                "retrieval_margin": float(frame["retrieval_margin"].mean()),
                "mean_chance_top1": float(frame["chance_top1"].mean()),
            }
        )
    null = pd.DataFrame(rows)
    observed = summarize_diagnostics(diagnostics)
    summary = {
        "observed_top1": observed["top1"],
        "null_mean_top1": float(null["top1"].mean()),
        "empirical_p_top1": empirical_p(float(observed["top1"]), null["top1"]),
        "observed_mrr": observed["mrr"],
        "null_mean_mrr": float(null["mrr"].mean()),
        "empirical_p_mrr": empirical_p(float(observed["mrr"]), null["mrr"]),
        "observed_percentile_rank": observed["percentile_rank"],
        "null_mean_percentile_rank": float(null["percentile_rank"].mean()),
        "empirical_p_percentile_rank": empirical_p(float(observed["percentile_rank"]), null["percentile_rank"]),
        "observed_margin": observed["retrieval_margin"],
        "null_mean_margin": float(null["retrieval_margin"].mean()),
        "empirical_p_margin": empirical_p(float(observed["retrieval_margin"]), null["retrieval_margin"]),
        "n_permutations": int(n_permutations),
        "null_method": "global MedOn label shuffle; absent pseudo labels are treated as unretrieved",
    }
    return null, summary


def empirical_p(observed: float, null_values: pd.Series) -> float:
    """One-sided empirical p-value for larger-is-better metrics."""

    values = pd.to_numeric(null_values, errors="coerce").dropna().to_numpy(dtype=float)
    if not np.isfinite(observed) or len(values) == 0:
        return np.nan
    return float((1 + np.sum(values >= observed)) / (1 + len(values)))


def leave_one_subject_out_selected(
    pairs: pd.DataFrame,
    subspaces: dict[str, list[str]],
    all_features: list[str],
    variant: VariantSpec,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Remove each subject from the analysis and rerun the selected variant."""

    rows: list[dict[str, object]] = []
    clean_features = select_feature_space(subspaces, PRIMARY_FEATURE_SET, all_features)
    features = select_feature_space(subspaces, variant.feature_space, all_features)
    for subject in sorted(pairs["subject"].astype(str).unique()):
        subset = pairs[~pairs["subject"].astype(str).eq(subject)].copy()
        base_candidates = build_base_candidate_sets(subset, MIN_DISTRACTORS)
        _, diag = evaluate_variant(subset, base_candidates, features, variant, clean_features)
        metrics = summarize_diagnostics(diag)
        rows.append(
            {
                "removed_subject": subject,
                "n_pairs_remaining": int(len(subset)),
                "top1": metrics["top1"],
                "mrr": metrics["mrr"],
                "percentile_rank": metrics["percentile_rank"],
                "retrieval_margin": metrics["retrieval_margin"],
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "selected_variant": variant.name,
        "min_top1": float(table["top1"].min()) if not table.empty else np.nan,
        "max_top1": float(table["top1"].max()) if not table.empty else np.nan,
        "mean_top1": float(table["top1"].mean()) if not table.empty else np.nan,
        "n_subject_removals": int(len(table)),
    }
    return table, summary


def choose_best_by_category(rows: pd.DataFrame, category: str) -> str:
    """Choose a category representative using conservative fixed ordering."""

    subset = rows[rows["variant_category"].astype(str).eq(category) & rows["status"].astype(str).eq("ok")].copy()
    if subset.empty:
        return ""
    subset = subset.sort_values(
        [
            "top1",
            "other_subject_same_task_side_failure_count",
            "quality_related_failure_count",
            "same_subject_hard_negative_success",
            "mrr",
            "retrieval_margin",
        ],
        ascending=[False, True, True, False, False, False],
    )
    return str(subset.iloc[0]["variant_name"])


def select_conservative_variant(
    all_rows: pd.DataFrame,
    baseline: dict[str, object],
) -> dict[str, object]:
    """Select a conservative variant only when it satisfies required controls."""

    if all_rows.empty:
        return {"selected_variant_name": "", "warning": "No variants were evaluated."}
    candidates = all_rows[
        all_rows["status"].astype(str).eq("ok")
        & ~all_rows["variant_name"].astype(str).eq("baseline_clean_stable_cosine")
        & ~all_rows["variant_name"].astype(str).eq("baseline_distance_only_clean_stable")
        & ~all_rows["variant_name"].astype(str).str.startswith("residualized_")
    ].copy()
    baseline_top1 = float(baseline["top1"])
    baseline_mrr = float(baseline["mrr"])
    baseline_hard = int(baseline["other_subject_same_task_side_failure_count"])
    baseline_quality = int(baseline["quality_related_failure_count"])
    baseline_same = float(baseline["same_subject_hard_negative_success"])
    reasons: dict[str, str] = {}
    eligible_rows = []
    for _, row in candidates.iterrows():
        checks = {
            "top1_not_worse": float(row["top1"]) >= baseline_top1,
            "mrr_not_meaningfully_worse": float(row["mrr"]) >= baseline_mrr - MRR_TOLERANCE,
            "hard_failures_not_higher": int(row["other_subject_same_task_side_failure_count"]) <= baseline_hard,
            "quality_failures_not_higher": int(row["quality_related_failure_count"]) <= baseline_quality,
            "same_subject_not_worse": float(row["same_subject_hard_negative_success"]) >= baseline_same - SAME_SUBJECT_TOLERANCE,
        }
        if all(checks.values()):
            eligible_rows.append(row)
            reasons[str(row["variant_name"])] = "eligible: satisfies required conservative checks"
        else:
            failed = [name for name, ok in checks.items() if not ok]
            reasons[str(row["variant_name"])] = f"not selected: failed {', '.join(failed)}"
    if not eligible_rows:
        return {
            "selected_variant_name": "",
            "reason_selected": "No variant satisfied the conservative selection checks.",
            "reason_not_selected": reasons,
            "baseline_metrics": baseline,
            "warning": "No robust improvement found; keep baseline as the safer default.",
        }
    eligible = pd.DataFrame(eligible_rows)
    eligible = eligible.sort_values(
        [
            "other_subject_same_task_side_failure_count",
            "quality_related_failure_count",
            "same_subject_hard_negative_success",
            "mrr",
            "top1",
            "retrieval_margin",
        ],
        ascending=[True, True, False, False, False, False],
    )
    selected = eligible.iloc[0].to_dict()
    reasons.pop(str(selected["variant_name"]), None)
    for name, reason in list(reasons.items()):
        if reason.startswith("eligible:"):
            reasons[name] = "not selected: eligible, but ranked lower than the selected variant under conservative ordering"
    return {
        "selected_variant_name": str(selected["variant_name"]),
        "reason_selected": (
            "Selected by conservative ordering: reduce other-subject same-task/side failures first, "
            "avoid quality-failure increases, preserve same-subject hard-negative performance, then compare MRR/top1."
        ),
        "reason_not_selected": reasons,
        "baseline_metrics": baseline,
        "selected_variant_metrics": selected,
        "warning": "",
    }


def left_side_check(baseline_diag: pd.DataFrame, selected_diag: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare L/R performance for baseline and selected variant."""

    rows: list[dict[str, object]] = []
    for label, diag in [("baseline", baseline_diag), ("selected_variant", selected_diag)]:
        if diag is None or diag.empty:
            continue
        for side, subset in diag.groupby("query_side", dropna=False):
            metrics = summarize_diagnostics(subset)
            failures = subset[~subset["top_ranked_is_true_pair"].astype(bool)]
            rows.append(
                {
                    "analysis": label,
                    "side": side,
                    "top1": metrics["top1"],
                    "mrr": metrics["mrr"],
                    "n_queries": int(len(subset)),
                    "mean_candidate_set_size": float(subset["candidate_set_size"].mean()),
                    "failure_type_counts": json.dumps(failures["failure_type"].value_counts().to_dict(), sort_keys=True),
                    "quality_counts": json.dumps(subset["query_quality"].value_counts().to_dict(), sort_keys=True),
                }
            )
    table = pd.DataFrame(rows)
    return table, {
        "note": "Side checks are descriptive because side-specific sample sizes are small.",
        "rows": table.to_dict("records"),
    }


def make_figures(
    all_rows: pd.DataFrame,
    change_log: pd.DataFrame,
    random_null: pd.DataFrame,
    selected_summary: dict[str, object],
    loso: pd.DataFrame,
    residualized: pd.DataFrame,
) -> None:
    """Create simple matplotlib figures."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ok = all_rows[all_rows["status"].astype(str).eq("ok")].copy() if not all_rows.empty else pd.DataFrame()
    if not ok.empty:
        plot_rows = ok[~ok["variant_name"].astype(str).str.startswith("residualized_")].head(20)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(plot_rows))
        width = 0.38
        ax.bar(x - width / 2, plot_rows["top1"], width, label="top1")
        ax.bar(x + width / 2, plot_rows["mrr"], width, label="MRR")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_rows["variant_name"], rotation=60, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("Baseline vs Variant Top1/MRR")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "baseline_vs_variant_top1_mrr.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(plot_rows["variant_name"], plot_rows["other_subject_same_task_side_failure_count"])
        ax.set_ylabel("Failure count")
        ax.set_title("Other-Subject Same-Task/Side Failures")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "other_subject_same_task_side_failures.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(plot_rows["variant_name"], plot_rows["same_subject_hard_negative_success"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Success rate")
        ax.set_title("Same-Subject Hard-Negative Success")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "same_subject_hard_negative_success.png", dpi=200)
        plt.close(fig)

    if not change_log.empty:
        counts = (
            change_log[change_log["change_type"].isin(["fail_to_success", "success_to_failure"])]
            .groupby(["variant_name", "change_type"])
            .size()
            .unstack(fill_value=0)
        )
        if not counts.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            counts.plot(kind="bar", ax=ax)
            ax.set_ylabel("Query count")
            ax.set_title("Query Change Counts")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "query_change_counts.png", dpi=200)
            plt.close(fig)

    if not random_null.empty and selected_summary.get("selected_variant_metrics"):
        observed = float(selected_summary["selected_variant_metrics"]["top1"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(random_null["top1"], bins=25)
        ax.axvline(observed, color="black", linestyle="--", label="observed")
        ax.set_xlabel("Top1 under random-label null")
        ax.set_ylabel("Permutations")
        ax.set_title("Selected Variant Random-Label Top1 Null")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "selected_variant_random_label_top1_null.png", dpi=200)
        plt.close(fig)

    if not loso.empty and selected_summary.get("baseline_metrics"):
        baseline_top1 = float(selected_summary["baseline_metrics"]["top1"])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(loso["removed_subject"], loso["top1"], label="selected")
        ax.axhline(baseline_top1, color="black", linestyle="--", label="baseline top1")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Top1")
        ax.set_title("LOSO Top1: Baseline Reference vs Selected Variant")
        ax.tick_params(axis="x", rotation=60)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "loso_top1_baseline_vs_selected.png", dpi=200)
        plt.close(fig)

    if not residualized.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = residualized["feature_space"] + "\n" + residualized["mode"] + "\n" + residualized["scoring_variant"]
        ax.bar(labels, residualized["top1"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Top1")
        ax.set_title("Residualized vs Original Improvement Checks")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "residualized_vs_original_comparison.png", dpi=200)
        plt.close(fig)


def write_report(
    baseline: dict[str, object],
    group_summary: dict[str, object],
    weighting_summary: dict[str, object],
    transition_summary: dict[str, object],
    residual_summary: dict[str, object],
    selection: dict[str, object],
    control_summary: dict[str, object],
    path: Path,
) -> Path:
    """Write concise technical report."""

    selected_name = selection.get("selected_variant_name") or "none"
    selected_metrics = selection.get("selected_variant_metrics", {})
    lines = [
        "# Retrieval Distance Geometry Improvement",
        "",
        "## Purpose",
        "",
        "This controlled improvement run tests predefined retrieval scoring changes against diagnosed hard-negative failures. It does not introduce clinical, DBS, treatment, stimulation-planning, or MedOn-as-healthy claims.",
        "",
        "## Baseline Reproduction",
        "",
        f"- top1: {baseline.get('top1')}",
        f"- MRR: {baseline.get('mrr')}",
        f"- retrieval_margin: {baseline.get('retrieval_margin')}",
        f"- other_subject_same_task_side failures: {baseline.get('other_subject_same_task_side_failure_count')}",
        f"- quality_related failures: {baseline.get('quality_related_failure_count')}",
        f"- same_subject_hard_negative_success: {baseline.get('same_subject_hard_negative_success')}",
        "",
        "## Group-Balanced Distance",
        "",
        f"- best_variant: {group_summary.get('best_variant')}",
        f"- note: {group_summary.get('note')}",
        "",
        "## Predefined Weighting",
        "",
        f"- best_variant: {weighting_summary.get('best_variant')}",
        f"- note: {weighting_summary.get('note')}",
        "",
        "## Transition-Aware Scoring",
        "",
        f"- best_variant: {transition_summary.get('best_variant')}",
        f"- note: {transition_summary.get('note')}",
        "",
        "## Residualized Improvement Checks",
        "",
        f"- note: {residual_summary.get('note')}",
        "",
        "## Selected Conservative Variant",
        "",
        f"- selected_variant_name: {selected_name}",
        f"- reason_selected: {selection.get('reason_selected')}",
        f"- warning: {selection.get('warning')}",
        f"- selected_variant_metrics: {selected_metrics}",
        "",
        "## Safety Controls",
        "",
        f"- random_label_control: {control_summary.get('random_label_control')}",
        f"- loso_control: {control_summary.get('loso_control')}",
        f"- same_subject_hard_negative_control: {control_summary.get('same_subject_hard_negative_control')}",
        "",
        "## Technical Interpretation",
        "",
        "- A variant is treated as safer only if it does not worsen MRR, diagnosed hard failures, quality-related failures, or same-subject hard negatives.",
        "- A one-query top1 gain alone is not interpreted as a robust improvement.",
        "- Retrieval-margin remains a separate caution signal; rank-based gains do not imply confident score separation.",
        "",
        "## Next Pipeline Change",
        "",
        "- Promote the selected variant only if the safety-control summaries remain acceptable.",
        "- If no variant is selected, keep the current baseline and use these outputs to guide a targeted follow-up rather than broad feature search.",
    ]
    return write_text(lines, path)


def best_summary_for(rows: pd.DataFrame, category: str) -> dict[str, object]:
    """Summarize best row for one category."""

    subset = rows[rows["variant_category"].astype(str).eq(category) & rows["status"].astype(str).eq("ok")].copy()
    if subset.empty:
        return {"best_variant": "", "note": "No variants available."}
    best_name = choose_best_by_category(rows, category)
    best = subset[subset["variant_name"].astype(str).eq(best_name)].iloc[0].to_dict()
    return {
        "best_variant": best_name,
        "best_metrics": best,
        "note": "Best is chosen by fixed conservative ordering, not by exhaustive tuning.",
    }


def make_residualized_improvement(
    pairs: pd.DataFrame,
    base_candidates: pd.DataFrame,
    subspaces: dict[str, list[str]],
    all_features: list[str],
    best_group: str,
    best_weight: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate original/residualized feature spaces under selected improvement families."""

    clean_features = select_feature_space(subspaces, PRIMARY_FEATURE_SET, all_features)
    rows: list[dict[str, object]] = []
    variants: list[tuple[str, str, VariantSpec]] = []
    for feature_space in ["clean_stable_features", "full_28"]:
        variants.extend(
            [
                (
                    feature_space,
                    "standard_cosine",
                    VariantSpec(
                        name=f"{feature_space}_standard_cosine",
                        category="residualized_improvement",
                        score_kind="standard",
                        feature_space=feature_space,
                    ),
                ),
                (
                    feature_space,
                    "group_balanced_cosine",
                    VariantSpec(
                        name=f"{feature_space}_group_balanced_cosine",
                        category="residualized_improvement",
                        score_kind="group_balanced",
                        feature_space=feature_space,
                    ),
                ),
                (
                    feature_space,
                    best_weight or "baseline_all_weights_1",
                    VariantSpec(
                        name=f"{feature_space}_{best_weight or 'baseline_all_weights_1'}",
                        category="residualized_improvement",
                        score_kind="weighted",
                        feature_space=feature_space,
                        weight_scheme=best_weight or "baseline_all_weights_1",
                    ),
                ),
                (
                    feature_space,
                    "combined_distance_plus_transition",
                    VariantSpec(
                        name=f"{feature_space}_combined_distance_plus_transition",
                        category="residualized_improvement",
                        score_kind="combined_transition",
                        feature_space=feature_space,
                    ),
                ),
            ]
        )
    for feature_space, scoring_name, variant in variants:
        features = select_feature_space(subspaces, feature_space, all_features)
        if not features:
            continue
        for mode, residualized in [("original", False), ("residualized_task_side_quality", True)]:
            run_variant = VariantSpec(
                name=f"{mode}_{variant.name}",
                category=variant.category,
                score_kind=variant.score_kind,
                feature_space=variant.feature_space,
                distance=variant.distance,
                weight_scheme=variant.weight_scheme,
                residualized=residualized,
            )
            _, diag = evaluate_variant(pairs, base_candidates, features, run_variant, clean_features)
            metrics = summarize_diagnostics(diag)
            rows.append(
                {
                    "feature_space": feature_space,
                    "mode": mode,
                    "scoring_variant": scoring_name,
                    "source_best_group_variant": best_group,
                    **metrics,
                }
            )
    table = pd.DataFrame(rows)
    return table, {
        "note": "Residualization uses train-subject-only task/side/quality regressors inside each held-out-subject fold.",
        "rows": table.to_dict("records"),
    }


def side_failure_counts(diag: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return failure-type counts by side."""

    out: dict[str, dict[str, int]] = {}
    if diag.empty:
        return out
    failures = diag[~diag["top_ranked_is_true_pair"].astype(bool)]
    for side, subset in failures.groupby("query_side", dropna=False):
        out[str(side)] = {str(k): int(v) for k, v in subset["failure_type"].value_counts().to_dict().items()}
    return out


def run(args: argparse.Namespace) -> dict[str, Path]:
    """Run the controlled distance-geometry improvement phase."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    state_vectors = read_state_vectors(args.state_vectors, warnings)
    quality = read_quality_table(args.quality_table, warnings)
    subspaces = read_subspace_definitions(args.subspace_definitions, state_vectors, warnings)
    pairs, _ = build_paired_examples(state_vectors, quality)
    all_features = [column.removeprefix("x_") for column in pairs.columns if column.startswith("x_")]
    clean_features = select_feature_space(subspaces, PRIMARY_FEATURE_SET, all_features)
    if not clean_features:
        raise ValueError(f"Missing required primary feature set: {PRIMARY_FEATURE_SET}")
    base_candidates = build_base_candidate_sets(pairs, MIN_DISTRACTORS)

    baseline_variant = VariantSpec(
        name="baseline_clean_stable_cosine",
        category="baseline",
        score_kind="standard",
        feature_space="clean_stable_features",
        note="Primary baseline reproduction.",
    )
    baseline_scores, baseline_diag = evaluate_variant(pairs, base_candidates, clean_features, baseline_variant, clean_features)
    baseline_hard, baseline_hard_summary = same_subject_hard_negative(pairs, clean_features, baseline_variant, clean_features)
    baseline_summary = summarize_variant(baseline_diag, baseline_hard_summary, baseline_variant)

    group_variants = [
        baseline_variant,
        VariantSpec(
            name="group_balanced_cosine_full28_plus_coupling",
            category="group_balanced",
            score_kind="group_balanced",
            feature_space="full_28_plus_coupling",
            distance="cosine",
            note="Average cosine distance across predefined feature groups.",
        ),
        VariantSpec(
            name="group_balanced_euclidean_full28_plus_coupling",
            category="group_balanced",
            score_kind="group_balanced",
            feature_space="full_28_plus_coupling",
            distance="euclidean",
            note="Average Euclidean distance across predefined feature groups.",
        ),
    ]
    group_rows, group_diag, group_scores, _ = evaluate_variant_set(
        pairs, base_candidates, subspaces, all_features, group_variants, baseline_diag
    )

    weight_schemes = [
        "baseline_all_weights_1",
        "cortical_beta_gamma_weight_0_5",
        "cortical_beta_gamma_weight_0_25",
        "coupling_features_weight_2",
        "beta_gamma_weight_0_5_plus_coupling_weight_2",
        "beta_gamma_weight_0_25_plus_coupling_weight_2",
    ]
    weighting_variants = [
        VariantSpec(
            name=f"weighted_{scheme}",
            category="predefined_weighting",
            score_kind="weighted",
            feature_space="full_28_plus_coupling",
            weight_scheme=scheme,
            note="Predefined weighting in full_28 + cortico-STN coupling feature space.",
        )
        for scheme in weight_schemes
    ]
    weighting_rows, weighting_diag, weighting_scores, _ = evaluate_variant_set(
        pairs, base_candidates, subspaces, all_features, weighting_variants, baseline_diag
    )

    transition_variants = [
        VariantSpec(
            name="transition_score_only_clean_stable",
            category="transition_aware",
            score_kind="transition_only",
            feature_space="clean_stable_features",
            note="Train-subject mean MedOff-to-MedOn direction; lower score is negative transition similarity.",
        ),
        VariantSpec(
            name="baseline_distance_only_clean_stable",
            category="transition_aware",
            score_kind="standard",
            feature_space="clean_stable_features",
            note="Distance-only comparator inside transition analysis.",
        ),
        VariantSpec(
            name="combined_distance_plus_transition_clean_stable",
            category="transition_aware",
            score_kind="combined_transition",
            feature_space="clean_stable_features",
            note="Candidate-set z-scored distance similarity plus transition similarity; train direction is fold-local.",
        ),
    ]
    transition_rows, transition_diag, transition_scores, _ = evaluate_variant_set(
        pairs, base_candidates, subspaces, all_features, transition_variants, baseline_diag
    )

    all_rows = pd.concat([group_rows, weighting_rows, transition_rows], ignore_index=True)
    all_diag: dict[str, pd.DataFrame] = {**group_diag, **weighting_diag, **transition_diag}
    all_scores: dict[str, pd.DataFrame] = {**group_scores, **weighting_scores, **transition_scores}

    best_group = choose_best_by_category(all_rows, "group_balanced")
    best_weight_row = all_rows[
        all_rows["variant_category"].astype(str).eq("predefined_weighting") & all_rows["status"].astype(str).eq("ok")
    ].copy()
    best_weight = ""
    if not best_weight_row.empty:
        best_weight_name = choose_best_by_category(all_rows, "predefined_weighting")
        match = best_weight_row[best_weight_row["variant_name"].astype(str).eq(best_weight_name)]
        if not match.empty:
            best_weight = str(match.iloc[0]["weight_scheme"])

    residualized_table, residualized_summary = make_residualized_improvement(
        pairs, base_candidates, subspaces, all_features, best_group, best_weight
    )

    change_frames = [
        query_change_log_for_variant(baseline_diag, diag, variant_name)
        for variant_name, diag in all_diag.items()
        if variant_name != baseline_variant.name
    ]
    change_log = pd.concat(change_frames, ignore_index=True) if change_frames else pd.DataFrame()
    change_summary = query_change_summary(change_log)

    baseline_summary["same_subject_hard_negative_success"] = baseline_hard_summary.get("true_beats_all_rate", np.nan)
    selection = select_conservative_variant(all_rows, baseline_summary)
    selected_name = str(selection.get("selected_variant_name") or "")
    selected_variant = None
    selected_features: list[str] = []
    for variant in [*group_variants, *weighting_variants, *transition_variants]:
        if variant.name == selected_name:
            selected_variant = variant
            selected_features = select_feature_space(subspaces, variant.feature_space, all_features)
            break

    random_null = pd.DataFrame()
    random_summary: dict[str, object] = {"warning": "No selected variant; random-label control skipped."}
    loso = pd.DataFrame()
    loso_summary: dict[str, object] = {"warning": "No selected variant; LOSO control skipped."}
    selected_hard = pd.DataFrame()
    selected_hard_summary: dict[str, object] = {"warning": "No selected variant; hard-negative control skipped."}
    selected_diag = None
    selected_scores = None
    if selected_variant is not None and selected_name in all_diag and selected_name in all_scores:
        selected_diag = all_diag[selected_name]
        selected_scores = all_scores[selected_name]
        random_null, random_summary = random_label_negative_control(
            selected_scores, selected_diag, args.n_permutations_random_label, args.random_seed
        )
        loso, loso_summary = leave_one_subject_out_selected(pairs, subspaces, all_features, selected_variant)
        selected_hard, selected_hard_summary = same_subject_hard_negative(
            pairs, selected_features, selected_variant, clean_features
        )

    left_side_table, left_side_summary = left_side_check(baseline_diag, selected_diag)
    control_summary = {
        "random_label_control": random_summary,
        "loso_control": loso_summary,
        "same_subject_hard_negative_control": selected_hard_summary,
    }

    paths = {
        "baseline_results": write_csv(baseline_diag, OUTPUT_DIR / "baseline_retrieval_results.csv"),
        "baseline_summary": write_json(baseline_summary, OUTPUT_DIR / "baseline_retrieval_summary.json"),
        "group_balanced_results": write_csv(group_rows, OUTPUT_DIR / "group_balanced_retrieval_results.csv"),
        "group_balanced_summary": write_json(best_summary_for(group_rows, "group_balanced"), OUTPUT_DIR / "group_balanced_retrieval_summary.json"),
        "predefined_weighting_results": write_csv(weighting_rows, OUTPUT_DIR / "predefined_weighting_results.csv"),
        "predefined_weighting_summary": write_json(best_summary_for(weighting_rows, "predefined_weighting"), OUTPUT_DIR / "predefined_weighting_summary.json"),
        "transition_aware_results": write_csv(transition_rows, OUTPUT_DIR / "transition_aware_retrieval_results.csv"),
        "transition_aware_summary": write_json(best_summary_for(transition_rows, "transition_aware"), OUTPUT_DIR / "transition_aware_retrieval_summary.json"),
        "residualized_improvement_results": write_csv(residualized_table, OUTPUT_DIR / "residualized_improvement_results.csv"),
        "residualized_improvement_summary": write_json(residualized_summary, OUTPUT_DIR / "residualized_improvement_summary.json"),
        "variant_selection": write_json(selection, OUTPUT_DIR / "variant_selection_summary.json"),
        "query_change_log": write_csv(change_log, OUTPUT_DIR / "variant_query_change_log.csv"),
        "query_change_summary": write_json(change_summary, OUTPUT_DIR / "variant_query_change_summary.json"),
        "random_label_control": write_csv(random_null, OUTPUT_DIR / "selected_variant_random_label_control.csv"),
        "random_label_control_summary": write_json(random_summary, OUTPUT_DIR / "selected_variant_random_label_control_summary.json"),
        "loso": write_csv(loso, OUTPUT_DIR / "selected_variant_loso.csv"),
        "loso_summary": write_json(loso_summary, OUTPUT_DIR / "selected_variant_loso_summary.json"),
        "same_subject_hard_negative": write_csv(selected_hard, OUTPUT_DIR / "selected_variant_same_subject_hard_negative.csv"),
        "same_subject_hard_negative_summary": write_json(
            selected_hard_summary, OUTPUT_DIR / "selected_variant_same_subject_hard_negative_summary.json"
        ),
        "left_side_check": write_csv(left_side_table, OUTPUT_DIR / "left_side_improvement_check.csv"),
        "left_side_summary": write_json(left_side_summary, OUTPUT_DIR / "left_side_improvement_summary.json"),
    }
    if warnings:
        paths["warnings"] = write_json({"warnings": warnings}, OUTPUT_DIR / "warnings.json")

    group_summary = best_summary_for(group_rows, "group_balanced")
    weighting_summary = best_summary_for(weighting_rows, "predefined_weighting")
    transition_summary = best_summary_for(transition_rows, "transition_aware")
    make_figures(all_rows, change_log, random_null, selection, loso, residualized_table)
    paths["report"] = write_report(
        baseline_summary,
        group_summary,
        weighting_summary,
        transition_summary,
        residualized_summary,
        selection,
        control_summary,
        OUTPUT_DIR / "retrieval_distance_geometry_improvement_report.md",
    )
    print("Retrieval distance-geometry improvement complete.")
    print(f"Baseline top1/MRR: {baseline_summary.get('top1'):.3f} / {baseline_summary.get('mrr'):.3f}")
    if selection.get("selected_variant_metrics"):
        metrics = selection["selected_variant_metrics"]
        print(
            "Selected variant top1/MRR: "
            f"{float(metrics.get('top1', np.nan)):.3f} / {float(metrics.get('mrr', np.nan)):.3f}"
        )
        hard_delta = int(metrics.get("other_subject_same_task_side_failure_count", 0)) - int(
            baseline_summary.get("other_subject_same_task_side_failure_count", 0)
        )
        same_delta = float(metrics.get("same_subject_hard_negative_success", np.nan)) - float(
            baseline_summary.get("same_subject_hard_negative_success", np.nan)
        )
        print(f"Hard-failure count change: {hard_delta}")
        print(f"Same-subject hard-negative success change: {same_delta:.3f}")
        print(
            "Controls passed: "
            f"random-label p_top1={random_summary.get('empirical_p_top1')}, "
            f"LOSO min_top1={loso_summary.get('min_top1')}"
        )
    else:
        print("Selected variant top1/MRR: none selected")
        print("Hard-failure count change: not applicable")
        print("Same-subject hard-negative success change: not applicable")
        print("Controls passed: no selected variant")
    print(f"Output path: {OUTPUT_DIR}")
    return paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run controlled retrieval distance-geometry improvement.")
    parser.add_argument("--state-vectors", default=str(STATE_VECTORS_PATH))
    parser.add_argument("--quality-table", default=str(QUALITY_TABLE_PATH))
    parser.add_argument("--subspace-definitions", default=str(SUBSPACE_DEFINITIONS_PATH))
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-permutations-random-label", type=int, default=N_PERMUTATIONS_RANDOM_LABEL)
    return parser.parse_args()


def main() -> None:
    """Entry point."""

    run(parse_args())


if __name__ == "__main__":
    main()
