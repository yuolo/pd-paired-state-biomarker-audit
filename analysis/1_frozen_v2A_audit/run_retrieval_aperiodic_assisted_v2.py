"""Controlled aperiodic-assisted v2 retrieval analysis.

This script tests whether compact PSD-derived aperiodic/residual features can
assist the current v2 paired-state retrieval scorer without becoming the main
distance geometry. It is intentionally conservative: it does not download data,
does not introduce deep learning, does not perform exhaustive feature search,
and does not make clinical, DBS, treatment, stimulation-planning, or
MedOn-as-healthy claims.
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
    read_quality_table,
    read_state_vectors,
    read_subspace_definitions,
    transform,
)
from scripts.run_retrieval_distance_geometry_improvement import (  # noqa: E402
    VariantSpec,
    cosine_distance,
    empirical_p,
    evaluate_variant as evaluate_v2_variant,
    pair_value,
    query_change_log_for_variant,
    query_change_summary,
    query_diagnostics_from_scores,
    random_label_negative_control,
    same_subject_hard_negative as same_subject_hard_negative_v2,
    summarize_diagnostics,
    to_builtin,
    write_csv,
    write_text,
)
from scripts.run_retrieval_signal_level_compact_v3 import (  # noqa: E402
    COMPACT_APERIODIC_FEATURES,
    MRR_TOLERANCE,
    MIN_DISTRACTORS,
    build_compact_subspaces,
    build_paired_examples_all_features,
    evaluate_compact_group_balanced,
    fit_scaler,
    group_indices,
    same_subject_compact,
    summary_row,
)

STATE_VECTORS_PATH = Path("outputs/retrieval_signal_level_compact_v3/compact_v3_state_vectors.csv")
QUALITY_TABLE_PATH = Path("outputs/tables/real_recording_quality.csv")
SUBSPACE_DEFINITIONS_PATH = Path("outputs/tables/subspace_definitions.csv")
APERIODIC_INVENTORY_PATH = Path("outputs/retrieval_signal_level_compact_v3/compact_signal_level_aperiodic_feature_inventory.json")
OUTPUT_DIR = Path("outputs/retrieval_aperiodic_assisted_v2")
FIGURES_DIR = OUTPUT_DIR / "figures"

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
N_RANDOM_LABEL_PERMUTATIONS = 1000
V2_SCORER_NAME = "group_balanced_cosine_full28_plus_coupling"
TOP_K_VALUES = [3, 5]
APERIODIC_PENALTY_LAMBDAS = [0.25, 0.5]
RERANK_ALPHAS = [0.25, 0.5]
DO_NOT_DOWNLOAD_DATA = True
SAME_SUBJECT_TOLERANCE = 0.0

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class AssistedVariantSpec:
    """Predefined assisted v2 scorer."""

    name: str
    mode: str
    lambda_penalty: float = 0.0
    top_k: int = 0
    alpha: float = 0.0
    note: str = ""


def write_json(data: dict[str, object], path: Path) -> Path:
    """Write JSON with parent creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(data), indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, object]:
    """Read a JSON file or return an empty dict."""

    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def zscore(values: np.ndarray) -> np.ndarray:
    """Candidate-set z-score with stable small-sample fallback."""

    arr = np.asarray(values, dtype=float)
    mean = float(np.nanmean(arr)) if len(arr) else 0.0
    std = float(np.nanstd(arr))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return (arr - mean) / std


def available_aperiodic_features(state_vectors: pd.DataFrame) -> list[str]:
    """Return compact aperiodic/residual features available in state vectors."""

    inventory = read_json(APERIODIC_INVENTORY_PATH)
    selected = inventory.get("selected_features", [])
    if isinstance(selected, list):
        features = [str(feature) for feature in selected if str(feature) in state_vectors.columns]
    else:
        features = []
    if not features:
        features = [feature for feature in COMPACT_APERIODIC_FEATURES if feature in state_vectors.columns]
    return features


def state_vectors_have_required_columns(state_vectors: pd.DataFrame) -> list[str]:
    """Return missing required columns."""

    required = ["subject", "medication", "task_original", "task_family", "side"]
    return [column for column in required if column not in state_vectors.columns]


def build_v2_context(
    train: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit train-only scaler for v2 features."""

    return fit_scaler(train, features)


def aper_distance_scores(
    pairs: pd.DataFrame,
    base_candidates: pd.DataFrame,
    aperiodic_features: list[str],
    variant_name: str = "aperiodic_distance_only",
) -> pd.DataFrame:
    """Compute LOSO train-scaled compact aperiodic cosine distances."""

    if not aperiodic_features:
        return pd.DataFrame()
    pair_lookup = pairs.set_index("pair_id", drop=False)
    rows: list[dict[str, object]] = []
    for heldout_subject in sorted(pairs["subject"].astype(str).unique()):
        train = pairs[~pairs["subject"].astype(str).eq(heldout_subject)].copy()
        test_ids = pairs.loc[pairs["subject"].astype(str).eq(heldout_subject), "pair_id"].astype(str).tolist()
        if train.empty:
            continue
        mean, std = fit_scaler(train, aperiodic_features)
        for query_id in test_ids:
            query = pair_lookup.loc[str(query_id)]
            qx = transform(pair_value(query, "x_", aperiodic_features), mean, std)
            candidates = base_candidates[base_candidates["query_pair_id"].astype(str).eq(str(query_id))]
            for _, candidate_meta in candidates.iterrows():
                candidate = pair_lookup.loc[str(candidate_meta["candidate_pair_id"])]
                cm = transform(pair_value(candidate, "medon_", aperiodic_features), mean, std)
                rows.append(
                    {
                        **candidate_meta.to_dict(),
                        "heldout_subject": heldout_subject,
                        "variant_name": variant_name,
                        "score": float(cosine_distance(qx, cm)),
                    }
                )
    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores
    scores = scores.sort_values(["variant_name", "query_pair_id", "score"], ascending=True).copy()
    scores["rank"] = scores.groupby(["variant_name", "query_pair_id"], dropna=False).cumcount() + 1
    return scores


def combine_scores(
    v2_scores: pd.DataFrame,
    aper_scores: pd.DataFrame,
    spec: AssistedVariantSpec,
) -> pd.DataFrame:
    """Combine v2 candidate distances with compact aperiodic distances."""

    if v2_scores.empty or aper_scores.empty:
        return pd.DataFrame()
    key_cols = ["query_pair_id", "candidate_pair_id"]
    aper = aper_scores[key_cols + ["score"]].rename(columns={"score": "aperiodic_score"})
    merged = v2_scores.drop(columns=["rank"], errors="ignore").merge(aper, on=key_cols, how="left")
    merged["v2_score"] = merged["score"].astype(float)
    merged["aperiodic_score"] = pd.to_numeric(merged["aperiodic_score"], errors="coerce")
    merged["aperiodic_score"] = merged["aperiodic_score"].fillna(merged["aperiodic_score"].max())
    rows: list[pd.DataFrame] = []
    for query_id, group in merged.groupby("query_pair_id", dropna=False):
        group = group.copy()
        if spec.mode == "penalty":
            group["score"] = group["v2_score"] + spec.lambda_penalty * group["aperiodic_score"]
        elif spec.mode == "two_stage":
            group = group.sort_values("v2_score", ascending=True).copy()
            group["v2_rank_for_stage"] = np.arange(1, len(group) + 1)
            group["true_in_v2_top_k"] = bool(group.loc[group["is_true_pair"].astype(bool), "v2_rank_for_stage"].iloc[0] <= spec.top_k)
            inside = group["v2_rank_for_stage"] <= spec.top_k
            group["score"] = 1000.0 + group["v2_rank_for_stage"].astype(float)
            if inside.any():
                stage = group.loc[inside].copy()
                distance_similarity = -stage["v2_score"].to_numpy(dtype=float)
                aper_similarity = -stage["aperiodic_score"].to_numpy(dtype=float)
                combined_similarity = zscore(distance_similarity) + spec.alpha * zscore(aper_similarity)
                stage["score"] = -combined_similarity
                group.loc[stage.index, "score"] = stage["score"]
        else:
            raise ValueError(f"Unsupported assisted mode: {spec.mode}")
        group["variant_name"] = spec.name
        rows.append(group)
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["variant_name", "query_pair_id", "score"], ascending=True).copy()
    out["rank"] = out.groupby(["variant_name", "query_pair_id"], dropna=False).cumcount() + 1
    return out


def evaluate_assisted_variant(
    v2_scores: pd.DataFrame,
    aper_scores: pd.DataFrame,
    spec: AssistedVariantSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one assisted variant from already computed candidate distances."""

    scores = combine_scores(v2_scores, aper_scores, spec)
    return scores, query_diagnostics_from_scores(scores)


def same_subject_assisted(
    pairs: pd.DataFrame,
    v2_features: list[str],
    aperiodic_features: list[str],
    clean_features: list[str],
    spec: AssistedVariantSpec,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Same-subject hard-negative check for assisted v2 scorers."""

    if not aperiodic_features:
        return pd.DataFrame(), {"variant_name": spec.name, "true_beats_all_rate": np.nan}
    v2_groups_func = None
    try:
        from scripts.run_retrieval_distance_geometry_improvement import build_feature_groups

        v2_groups_func = build_feature_groups
    except Exception:  # noqa: BLE001
        v2_groups_func = None
    rows: list[dict[str, object]] = []
    for _, query in pairs.iterrows():
        subject = str(query["subject"])
        negatives = pairs[pairs["subject"].astype(str).eq(subject) & ~pairs["pair_id"].astype(str).eq(str(query["pair_id"]))]
        train = pairs[~pairs["subject"].astype(str).eq(subject)].copy()
        if negatives.empty or train.empty:
            continue
        v2_mean, v2_std = fit_scaler(train, v2_features)
        aper_mean, aper_std = fit_scaler(train, aperiodic_features)
        q_v2 = transform(pair_value(query, "x_", v2_features), v2_mean, v2_std)
        q_aper = transform(pair_value(query, "x_", aperiodic_features), aper_mean, aper_std)
        v2_groups = v2_groups_func(v2_features, clean_features) if v2_groups_func is not None else {"all": list(range(len(v2_features)))}
        candidate_rows = [query, *[row for _, row in negatives.iterrows()]]
        records: list[dict[str, object]] = []
        for idx, candidate in enumerate(candidate_rows):
            c_v2 = transform(pair_value(candidate, "medon_", v2_features), v2_mean, v2_std)
            c_aper = transform(pair_value(candidate, "medon_", aperiodic_features), aper_mean, aper_std)
            v2_score = float(np.nanmean([cosine_distance(q_v2[indices], c_v2[indices]) for indices in v2_groups.values()]))
            aper_score = float(cosine_distance(q_aper, c_aper))
            records.append(
                {
                    "candidate_index": idx,
                    "is_true": idx == 0,
                    "v2_score": v2_score,
                    "aperiodic_score": aper_score,
                }
            )
        frame = pd.DataFrame(records).sort_values("v2_score", ascending=True).copy()
        if spec.mode == "penalty":
            frame["score"] = frame["v2_score"] + spec.lambda_penalty * frame["aperiodic_score"]
        elif spec.mode == "two_stage":
            frame["v2_rank_for_stage"] = np.arange(1, len(frame) + 1)
            frame["score"] = 1000.0 + frame["v2_rank_for_stage"].astype(float)
            inside = frame["v2_rank_for_stage"] <= spec.top_k
            if inside.any():
                distance_similarity = -frame.loc[inside, "v2_score"].to_numpy(dtype=float)
                aper_similarity = -frame.loc[inside, "aperiodic_score"].to_numpy(dtype=float)
                combined = zscore(distance_similarity) + spec.alpha * zscore(aper_similarity)
                frame.loc[inside, "score"] = -combined
        else:
            frame["score"] = frame["v2_score"]
        frame = frame.sort_values("score", ascending=True).copy()
        frame["rank"] = np.arange(1, len(frame) + 1)
        true = frame[frame["is_true"]].iloc[0]
        nearest = frame[~frame["is_true"]].iloc[0] if (~frame["is_true"]).any() else pd.Series(dtype=object)
        nearest_idx = int(nearest.get("candidate_index", -1))
        nearest_row = negatives.iloc[nearest_idx - 1] if nearest_idx > 0 and nearest_idx - 1 < len(negatives) else pd.Series(dtype=object)
        rows.append(
            {
                "variant_name": spec.name,
                "query_subject": subject,
                "query_task": query["task_original"],
                "query_side": query["side"],
                "true_rank_among_same_subject_candidates": int(true["rank"]),
                "n_same_subject_candidates": int(len(frame)),
                "true_beats_all_same_subject_negatives": bool(int(true["rank"]) == 1),
                "nearest_same_subject_negative_task": nearest_row.get("task_original", ""),
                "nearest_same_subject_negative_side": nearest_row.get("side", ""),
                "distance_true": float(true["score"]),
                "distance_nearest_same_subject_negative": float(nearest.get("score", np.nan)),
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "variant_name": spec.name,
        "queries_with_same_subject_hard_negatives": int(len(table)),
        "true_beats_all_rate": float(table["true_beats_all_same_subject_negatives"].mean()) if not table.empty else np.nan,
        "mean_true_rank": float(table["true_rank_among_same_subject_candidates"].mean()) if not table.empty else np.nan,
        "by_task": table.groupby("query_task")["true_beats_all_same_subject_negatives"].mean().to_dict() if not table.empty else {},
        "by_side": table.groupby("query_side")["true_beats_all_same_subject_negatives"].mean().to_dict() if not table.empty else {},
    }
    return table, summary


def summarize_assisted(
    spec_name: str,
    diag: pd.DataFrame,
    hard_summary: dict[str, object],
    baseline_diag: pd.DataFrame,
    mode: str,
) -> dict[str, object]:
    """Build summary row for assisted scorer."""

    metrics = summarize_diagnostics(diag)
    change = query_change_log_for_variant(baseline_diag, diag, spec_name)
    return {
        "scorer": spec_name,
        "mode": mode,
        **metrics,
        "same_subject_hard_negative_success": hard_summary.get("true_beats_all_rate", np.nan),
        "fail_to_success_vs_v2": int(change["change_type"].eq("fail_to_success").sum()) if not change.empty else 0,
        "success_to_failure_vs_v2": int(change["change_type"].eq("success_to_failure").sum()) if not change.empty else 0,
        "net_query_improvement_vs_v2": int(change["change_type"].eq("fail_to_success").sum()) - int(change["change_type"].eq("success_to_failure").sum()) if not change.empty else 0,
    }


def compare_query_level(v2_diag: pd.DataFrame, aper_diag: pd.DataFrame, v2_hard: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare v2 and compact aperiodic-only scorer at query level."""

    v2 = v2_diag.set_index("query_pair_id", drop=False)
    aper = aper_diag.set_index("query_pair_id", drop=False)
    hard_fail_queries = set()
    if v2_hard is not None and not v2_hard.empty:
        failed_hard = v2_hard[~v2_hard["true_beats_all_same_subject_negatives"].astype(bool)]
        hard_fail_queries = set(failed_hard["query_subject"].astype(str) + "|" + failed_hard["query_task"].astype(str))
    rows: list[dict[str, object]] = []
    for query_id in sorted(set(v2.index).intersection(set(aper.index))):
        vrow = v2.loc[query_id]
        arow = aper.loc[query_id]
        v_success = bool(vrow["top_ranked_is_true_pair"])
        a_success = bool(arow["top_ranked_is_true_pair"])
        if v_success and a_success:
            change_type = "both_success"
        elif (not v_success) and (not a_success):
            change_type = "both_failure"
        elif v_success and (not a_success):
            change_type = "v2_success_aperiodic_failure"
        else:
            change_type = "v2_failure_aperiodic_success"
        query_key = f"{vrow['query_subject']}|{vrow['query_task']}"
        rows.append(
            {
                "query_id": query_id,
                "subject": vrow["query_subject"],
                "task": vrow["query_task"],
                "side": vrow["query_side"],
                "quality": vrow["query_quality"],
                "v2_true_rank": int(vrow["true_medon_rank"]),
                "aperiodic_true_rank": int(arow["true_medon_rank"]),
                "v2_top1_success": v_success,
                "aperiodic_top1_success": a_success,
                "change_type": change_type,
                "v2_top_candidate_subject": vrow["top_ranked_candidate_subject"],
                "aperiodic_top_candidate_subject": arow["top_ranked_candidate_subject"],
                "v2_top_candidate_task": vrow["top_ranked_candidate_task"],
                "aperiodic_top_candidate_task": arow["top_ranked_candidate_task"],
                "v2_top_candidate_side": vrow["top_ranked_candidate_side"],
                "aperiodic_top_candidate_side": arow["top_ranked_candidate_side"],
                "v2_distance_or_score_true": float(vrow["distance_to_true_medon"]),
                "aperiodic_distance_or_score_true": float(arow["distance_to_true_medon"]),
                "v2_margin": float(vrow["retrieval_margin"]),
                "aperiodic_margin": float(arow["retrieval_margin"]),
                "whether_same_subject_hard_negative_failure": bool(query_key in hard_fail_queries),
                "whether_other_subject_same_task_side_failure": bool(vrow["failure_type"] == "other_subject_same_task_side" or arow["failure_type"] == "other_subject_same_task_side"),
            }
        )
    table = pd.DataFrame(rows)
    fixes = table[table["change_type"].eq("v2_failure_aperiodic_success")] if not table.empty else pd.DataFrame()
    breaks = table[table["change_type"].eq("v2_success_aperiodic_failure")] if not table.empty else pd.DataFrame()
    summary = {
        "n_queries_fixed_by_aperiodic": int(len(fixes)),
        "n_queries_broken_by_aperiodic": int(len(breaks)),
        "fixes_by_subject": fixes["subject"].value_counts().to_dict() if not fixes.empty else {},
        "breaks_by_subject": breaks["subject"].value_counts().to_dict() if not breaks.empty else {},
        "fixes_by_task": fixes["task"].value_counts().to_dict() if not fixes.empty else {},
        "breaks_by_task": breaks["task"].value_counts().to_dict() if not breaks.empty else {},
        "fixes_by_side": fixes["side"].value_counts().to_dict() if not fixes.empty else {},
        "breaks_by_side": breaks["side"].value_counts().to_dict() if not breaks.empty else {},
        "breaks_same_subject_hard_negative_flags": int(breaks["whether_same_subject_hard_negative_failure"].sum()) if not breaks.empty else 0,
        "aperiodic_fixes_other_subject_same_task_side_flags": int(fixes["whether_other_subject_same_task_side_failure"].sum()) if not fixes.empty else 0,
    }
    return table, summary


def factor_predictability(
    state_vectors: pd.DataFrame,
    aperiodic_features: list[str],
    quality: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Diagnose whether compact aperiodic features encode broad factors."""

    if not aperiodic_features:
        return pd.DataFrame(), {"warning": "No aperiodic features available."}
    try:
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), {"warning": f"sklearn unavailable for factor predictability: {type(exc).__name__}: {exc}"}

    data = state_vectors.copy()
    if "quality_flag" not in data:
        data["quality_flag"] = "unknown"
        if not quality.empty and {"subject", "task_original", "quality_flag"}.issubset(quality.columns):
            lookup = quality.groupby(["subject", "task_original"], dropna=False)["quality_flag"].agg(lambda x: str(list(x)[0])).to_dict()
            data["quality_flag"] = [lookup.get((str(row.subject), str(row.task_original)), "unknown") for row in data.itertuples()]
    x = data[aperiodic_features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for label_name, label_col, cv_kind in [
        ("subject_id", "subject", "stratified_cv"),
        ("task", "task_original", "subject_held_out"),
        ("side", "side", "subject_held_out"),
        ("medication_state", "medication", "subject_held_out"),
        ("quality_bin", "quality_flag", "subject_held_out"),
    ]:
        if label_col not in data:
            rows.append({"label": label_name, "status": "skipped_missing_label"})
            continue
        y = data[label_col].fillna("unknown").astype(str).to_numpy()
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or len(data) < 4:
            rows.append({"label": label_name, "status": "skipped_insufficient_classes", "n_classes": int(len(classes)), "n_samples": int(len(data))})
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
        dummy = DummyClassifier(strategy="most_frequent")
        groups = data["subject"].fillna("unknown").astype(str).to_numpy()
        try:
            if cv_kind == "subject_held_out" and len(np.unique(groups)) > 2:
                logo = LeaveOneGroupOut()
                preds = np.empty_like(y, dtype=object)
                dpreds = np.empty_like(y, dtype=object)
                for train_idx, test_idx in logo.split(x, y, groups):
                    if len(np.unique(y[train_idx])) < 2:
                        preds[test_idx] = classes[0]
                        dpreds[test_idx] = classes[0]
                        continue
                    model.fit(x[train_idx], y[train_idx])
                    dummy.fit(x[train_idx], y[train_idx])
                    preds[test_idx] = model.predict(x[test_idx])
                    dpreds[test_idx] = dummy.predict(x[test_idx])
            else:
                n_splits = min(5, int(counts.min()))
                if n_splits < 2:
                    rows.append({"label": label_name, "status": "skipped_insufficient_fold_counts", "n_classes": int(len(classes)), "n_samples": int(len(data))})
                    continue
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                preds = cross_val_predict(model, x, y, cv=cv)
                dpreds = cross_val_predict(dummy, x, y, cv=cv)
            acc = float(accuracy_score(y, preds))
            null_acc = []
            for _ in range(200):
                shuffled = rng.permutation(y)
                null_acc.append(float(accuracy_score(shuffled, preds)))
            rows.append(
                {
                    "label": label_name,
                    "model": "logistic_regression_balanced",
                    "cv_kind": cv_kind,
                    "accuracy": acc,
                    "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
                    "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
                    "majority_dummy_accuracy": float(accuracy_score(y, dpreds)),
                    "random_label_baseline_mean_accuracy": float(np.mean(null_acc)),
                    "empirical_p_accuracy_vs_random_label": empirical_p(acc, pd.Series(null_acc)),
                    "n_classes": int(len(classes)),
                    "n_samples": int(len(y)),
                    "status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"label": label_name, "status": f"failed:{type(exc).__name__}", "error_message": str(exc)})
    table = pd.DataFrame(rows)
    summary = {
        "n_features": int(len(aperiodic_features)),
        "rows": table.to_dict("records"),
        "interpretation_note": (
            "High subject-ID predictability with weaker subject-held-out task/side/medication predictability "
            "supports using compact aperiodic features as assistive signals rather than as a primary scorer."
        ),
    }
    return table, summary


def select_assisted_variant(
    candidates: pd.DataFrame,
    v2_row: dict[str, object],
    change_log: pd.DataFrame,
) -> dict[str, object]:
    """Select an assisted variant only when all conservative gates pass."""

    if candidates.empty:
        return {"selected_variant_name": "", "warning": "No assisted variants were evaluated.", "v2_remains_default": True}
    reasons: dict[str, str] = {}
    eligible = []
    v2_top1 = float(v2_row["top1"])
    v2_mrr = float(v2_row["mrr"])
    v2_same = float(v2_row["same_subject_hard_negative_success"])
    v2_hard = int(v2_row["other_subject_same_task_side_failure_count"])
    v2_quality = int(v2_row["quality_related_failure_count"])
    for _, row in candidates.iterrows():
        name = str(row["scorer"])
        changes = change_log[change_log["variant_name"].astype(str).eq(name)]
        fail_to_success = int(row.get("fail_to_success_vs_v2", 0))
        success_to_failure = int(row.get("success_to_failure_vs_v2", 0))
        improved_subjects = changes.loc[changes["change_type"].eq("fail_to_success"), "query_subject"].astype(str)
        concentrated = bool(not improved_subjects.empty and improved_subjects.value_counts().iloc[0] > max(1, int(np.ceil(0.6 * fail_to_success))))
        checks = {
            "top1_not_worse": float(row["top1"]) >= v2_top1,
            "mrr_not_meaningfully_worse": float(row["mrr"]) >= v2_mrr - MRR_TOLERANCE,
            "same_subject_not_worse": float(row["same_subject_hard_negative_success"]) >= v2_same - SAME_SUBJECT_TOLERANCE,
            "hard_failures_not_higher": int(row["other_subject_same_task_side_failure_count"]) <= v2_hard,
            "quality_failures_not_higher": int(row["quality_related_failure_count"]) <= v2_quality,
            "net_positive_query_change": fail_to_success > success_to_failure,
            "improvements_not_one_subject": not concentrated,
            "same_subject_wrong_task_not_obviously_higher": float(row["same_subject_hard_negative_success"]) >= v2_same - SAME_SUBJECT_TOLERANCE,
        }
        if all(checks.values()):
            eligible.append(row)
            reasons[name] = "eligible"
        else:
            reasons[name] = "not selected: failed " + ", ".join([key for key, ok in checks.items() if not ok])
    if not eligible:
        return {
            "selected_variant_name": "",
            "reason_selected": "No assisted variant passed all required conservative gates.",
            "reason_not_selected": reasons,
            "warning": "No assisted aperiodic variant clearly improves over v2; v2 remains default.",
            "v2_remains_default": True,
            "v2_reference_metrics": v2_row,
        }
    frame = pd.DataFrame(eligible).sort_values(
        ["other_subject_same_task_side_failure_count", "quality_related_failure_count", "same_subject_hard_negative_success", "mrr", "top1"],
        ascending=[True, True, False, False, False],
    )
    selected = frame.iloc[0].to_dict()
    for key, value in list(reasons.items()):
        if key == selected["scorer"]:
            reasons.pop(key)
        elif value == "eligible":
            reasons[key] = "not selected: eligible, but ranked lower by conservative ordering"
    return {
        "selected_variant_name": str(selected["scorer"]),
        "reason_selected": "Selected because all required assisted-v2 gates passed.",
        "reason_not_selected": reasons,
        "selected_variant_metrics": selected,
        "v2_reference_metrics": v2_row,
        "warning": "",
        "v2_remains_default": False,
    }


def bootstrap_ci(diag: pd.DataFrame, hard: pd.DataFrame, n_bootstrap: int, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """Bootstrap query-level retrieval and same-subject hard-negative metrics."""

    rng = np.random.default_rng(seed)
    metric_columns = {
        "top1": "top_ranked_is_true_pair",
        "mrr": "reciprocal_rank",
        "percentile_rank": "percentile_rank",
        "retrieval_margin": "retrieval_margin",
    }
    observed = {metric: float(diag[column].astype(float).mean()) for metric, column in metric_columns.items()}
    if not hard.empty:
        observed["same_subject_hard_negative_success"] = float(hard["true_beats_all_same_subject_negatives"].astype(float).mean())
    samples: dict[str, list[float]] = {metric: [] for metric in observed}
    values = {metric: diag[column].astype(float).to_numpy() for metric, column in metric_columns.items()}
    hard_values = hard["true_beats_all_same_subject_negatives"].astype(float).to_numpy() if not hard.empty else np.asarray([])
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(diag), size=len(diag))
        for metric in metric_columns:
            samples[metric].append(float(np.nanmean(values[metric][idx])))
        if len(hard_values):
            hidx = rng.integers(0, len(hard_values), size=len(hard_values))
            samples["same_subject_hard_negative_success"].append(float(np.nanmean(hard_values[hidx])))
    rows = []
    for metric, arr_values in samples.items():
        arr = np.asarray(arr_values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "observed": observed[metric],
                "bootstrap_mean": float(np.nanmean(arr)),
                "ci_lower_95": float(np.nanpercentile(arr, 2.5)),
                "ci_upper_95": float(np.nanpercentile(arr, 97.5)),
                "n_bootstrap": int(n_bootstrap),
            }
        )
    table = pd.DataFrame(rows)
    return table, {
        "rows": table.to_dict("records"),
        "margin_ci_crosses_zero": bool(
            not table[table["metric"].eq("retrieval_margin")].empty
            and table.loc[table["metric"].eq("retrieval_margin"), "ci_lower_95"].iloc[0] <= 0
            <= table.loc[table["metric"].eq("retrieval_margin"), "ci_upper_95"].iloc[0]
        ),
    }


def loso_selected(
    pairs: pd.DataFrame,
    v2_features: list[str],
    aperiodic_features: list[str],
    clean_features: list[str],
    spec: AssistedVariantSpec,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Remove each subject and rerun selected assisted variant."""

    rows: list[dict[str, object]] = []
    v2_variant = VariantSpec(
        name="v2_reference",
        category="v2_reproduction",
        score_kind="group_balanced",
        feature_space="full_28_plus_coupling",
        distance="cosine",
    )
    for subject in sorted(pairs["subject"].astype(str).unique()):
        subset = pairs[~pairs["subject"].astype(str).eq(subject)].copy()
        base = build_base_candidate_sets(subset, MIN_DISTRACTORS)
        v2_scores, _ = evaluate_v2_variant(subset, base, v2_features, v2_variant, clean_features)
        aper_scores = aper_distance_scores(subset, base, aperiodic_features, "aperiodic_distance")
        scores, diag = evaluate_assisted_variant(v2_scores, aper_scores, spec)
        metrics = summarize_diagnostics(diag)
        rows.append({"removed_subject": subject, "n_pairs_remaining": int(len(subset)), **metrics})
    table = pd.DataFrame(rows)
    return table, {
        "selected_variant": spec.name,
        "min_top1": float(table["top1"].min()) if not table.empty else np.nan,
        "max_top1": float(table["top1"].max()) if not table.empty else np.nan,
        "mean_top1": float(table["top1"].mean()) if not table.empty else np.nan,
        "n_subject_removals": int(len(table)),
    }


def two_stage_recovery_counts(scores: pd.DataFrame, v2_diag: pd.DataFrame, spec: AssistedVariantSpec) -> dict[str, int]:
    """Summarize unrecoverable/recovered v2 failures for a two-stage variant."""

    if scores.empty or spec.mode != "two_stage":
        return {"v2_failures_unrecoverable_true_outside_top_k": 0, "v2_failures_recovered_by_reranking": 0}
    v2_failures = set(v2_diag.loc[~v2_diag["top_ranked_is_true_pair"].astype(bool), "query_pair_id"].astype(str))
    recovered = 0
    unrecoverable = 0
    for query_id in v2_failures:
        group = scores[scores["query_pair_id"].astype(str).eq(query_id)]
        true = group[group["is_true_pair"].astype(bool)]
        if true.empty:
            continue
        if not bool(true.get("true_in_v2_top_k", pd.Series([False])).iloc[0]):
            unrecoverable += 1
        if int(true["rank"].iloc[0]) == 1:
            recovered += 1
    return {
        "v2_failures_unrecoverable_true_outside_top_k": int(unrecoverable),
        "v2_failures_recovered_by_reranking": int(recovered),
    }


def failure_comparison(
    v2_diag: pd.DataFrame,
    aper_diag: pd.DataFrame,
    assisted_diag: pd.DataFrame,
    assisted_name: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare failures under v2, compact aperiodic-only, and an assisted variant."""

    v2 = v2_diag.set_index("query_pair_id", drop=False)
    aper = aper_diag.set_index("query_pair_id", drop=False)
    assisted = assisted_diag.set_index("query_pair_id", drop=False) if not assisted_diag.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for query_id in sorted(v2.index):
        vrow = v2.loc[query_id]
        arow = aper.loc[query_id] if query_id in aper.index else None
        srow = assisted.loc[query_id] if not assisted.empty and query_id in assisted.index else None
        v_success = bool(vrow["top_ranked_is_true_pair"])
        s_success = bool(srow["top_ranked_is_true_pair"]) if srow is not None else np.nan
        if srow is None:
            change = "no_assisted_variant"
        elif v_success and s_success:
            change = "unchanged_success"
        elif (not v_success) and (not s_success):
            change = "unchanged_failure"
        elif (not v_success) and s_success:
            change = "fail_to_success"
        else:
            change = "success_to_failure"
        rows.append(
            {
                "query_id": query_id,
                "subject": vrow["query_subject"],
                "task": vrow["query_task"],
                "side": vrow["query_side"],
                "quality": vrow["query_quality"],
                "v2_true_rank": int(vrow["true_medon_rank"]),
                "aperiodic_only_true_rank": int(arow["true_medon_rank"]) if arow is not None else np.nan,
                "assisted_true_rank": int(srow["true_medon_rank"]) if srow is not None else np.nan,
                "v2_top_candidate": vrow["top_ranked_candidate_subject"],
                "aperiodic_top_candidate": arow["top_ranked_candidate_subject"] if arow is not None else "",
                "assisted_top_candidate": srow["top_ranked_candidate_subject"] if srow is not None else "",
                "assisted_variant": assisted_name,
                "change_type_vs_v2": change,
                "failure_type": srow["failure_type"] if srow is not None else vrow["failure_type"],
                "whether_same_subject_hard_negative_worsened": bool(change == "success_to_failure"),
                "whether_other_subject_same_task_side_failure_was_fixed": bool(vrow["failure_type"] == "other_subject_same_task_side" and change == "fail_to_success"),
            }
        )
    table = pd.DataFrame(rows)
    return table, {
        "assisted_variant": assisted_name,
        "change_type_counts": table["change_type_vs_v2"].value_counts().to_dict(),
        "other_subject_same_task_side_fixes": int(table["whether_other_subject_same_task_side_failure_was_fixed"].sum()),
        "n_queries": int(len(table)),
    }


def make_figures(
    comparison: pd.DataFrame,
    change_log: pd.DataFrame,
    factor_table: pd.DataFrame,
    selected_random: pd.DataFrame,
    selected_loso: pd.DataFrame,
    selected_metrics: dict[str, object] | None,
) -> None:
    """Create simple matplotlib figures."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not comparison.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(comparison))
        width = 0.38
        ax.bar(x - width / 2, comparison["top1"], width, label="top1")
        ax.bar(x + width / 2, comparison["mrr"], width, label="MRR")
        ax.set_xticks(x)
        ax.set_xticklabels(comparison["scorer"], rotation=55, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("V2, Aperiodic-Only, and Assisted Variants")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "v2_aperiodic_assisted_top1_mrr.png", dpi=200)
        plt.close(fig)

        for column, title, fname in [
            ("same_subject_hard_negative_success", "Same-Subject Hard-Negative Success", "same_subject_hard_negative_success.png"),
            ("other_subject_same_task_side_failure_count", "Other-Subject Same-Task/Side Failures", "other_subject_same_task_side_failures.png"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(comparison["scorer"], comparison[column])
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=55)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / fname, dpi=200)
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
            ax.set_title("Fail-to-Success vs Success-to-Failure")
            ax.set_ylabel("Query count")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "fail_success_change_counts.png", dpi=200)
            plt.close(fig)

        pivot = change_log.pivot_table(index="query_subject", columns="variant_name", values="variant_top1_success", aggfunc="first")
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=55, ha="right")
            ax.set_title("Query-Level Success by Assisted Variant")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "query_level_change_heatmap.png", dpi=200)
            plt.close(fig)

    if not factor_table.empty and "accuracy" in factor_table:
        ok = factor_table[factor_table["status"].astype(str).eq("ok")]
        if not ok.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(ok))
            width = 0.38
            ax.bar(x - width / 2, ok["accuracy"], width, label="accuracy")
            ax.bar(x + width / 2, ok["random_label_baseline_mean_accuracy"], width, label="random label")
            ax.set_xticks(x)
            ax.set_xticklabels(ok["label"], rotation=45, ha="right")
            ax.set_ylim(0, 1)
            ax.set_title("Aperiodic Factor Predictability")
            ax.legend()
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "aperiodic_factor_predictability.png", dpi=200)
            plt.close(fig)

    if not selected_random.empty and selected_metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(selected_random["top1"], bins=25)
        ax.axvline(float(selected_metrics["top1"]), color="black", linestyle="--", label="observed")
        ax.set_title("Selected Assisted Random-Label Top1 Null")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "selected_assisted_random_label_top1_null.png", dpi=200)
        plt.close(fig)

    if not selected_loso.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(selected_loso["removed_subject"], selected_loso["top1"])
        ax.set_ylim(0, 1)
        ax.set_title("Selected Assisted LOSO Top1")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "selected_assisted_loso_top1.png", dpi=200)
        plt.close(fig)


def write_report(
    v2_row: dict[str, object],
    aper_row: dict[str, object],
    query_summary: dict[str, object],
    factor_summary: dict[str, object],
    penalty_summary: dict[str, object],
    rerank_summary: dict[str, object],
    selection: dict[str, object],
    failure_summary: dict[str, object],
) -> Path:
    """Write final technical report."""

    selected = selection.get("selected_variant_name") or "none"
    selected_metrics = selection.get("selected_variant_metrics", {}) if isinstance(selection.get("selected_variant_metrics"), dict) else {}
    lines = [
        "# Retrieval Aperiodic-Assisted V2",
        "",
        "## Purpose",
        "",
        "This controlled analysis tests whether compact aperiodic/residual spectral features can assist the v2 paired-state retrieval scorer without becoming the primary distance geometry. It does not introduce clinical, DBS, treatment, stimulation-planning, or MedOn-as-healthy claims.",
        "",
        "## Direct Answers",
        "",
        f"1. Was v2 reproduced? Yes: top1={v2_row.get('top1')}, MRR={v2_row.get('mrr')}.",
        f"2. Was compact aperiodic-only reproduced? Yes: top1={aper_row.get('top1')}, MRR={aper_row.get('mrr')}.",
        f"3. Queries fixed by aperiodic-only: {query_summary.get('n_queries_fixed_by_aperiodic')}; by task={query_summary.get('fixes_by_task')}.",
        f"4. Queries broken by aperiodic-only: {query_summary.get('n_queries_broken_by_aperiodic')}; by task={query_summary.get('breaks_by_task')}.",
        f"5. Aperiodic factor predictability summary: {factor_summary.get('interpretation_note')}",
        f"6. Did aperiodic penalty improve over v2? best={penalty_summary.get('best_variant')}; note={penalty_summary.get('note')}.",
        f"7. Did two-stage v2-to-aperiodic reranking improve over v2? best={rerank_summary.get('best_variant')}; note={rerank_summary.get('note')}.",
        f"8. Did any assisted variant preserve or improve same-subject hard-negative success? selected={selected}.",
        f"9. Did any assisted variant reduce other_subject_same_task_side failures? selected={selected}; see aperiodic_assisted_selection_summary.json.",
        f"10. Did any assisted variant pass selection gates? {'yes' if selected != 'none' else 'no'}.",
        f"11. Should assisted aperiodic scoring replace v2? {'no, v2 remains default' if selection.get('v2_remains_default') else 'yes, the selected two-stage assisted variant passed the predefined gates'}.",
        f"12. If no variant passed, what was learned? {selection.get('warning')}",
        "13. Next technical fix: inspect aperiodic subject-signature behavior and same-subject hard negatives before adding more feature groups.",
        "",
        "## Selected Variant Metrics",
        "",
        f"- selected_variant: {selected}",
        f"- top1: {selected_metrics.get('top1')}",
        f"- MRR: {selected_metrics.get('mrr')}",
        f"- other_subject_same_task_side failures: {selected_metrics.get('other_subject_same_task_side_failure_count')}",
        f"- quality_related failures: {selected_metrics.get('quality_related_failure_count')}",
        f"- same_subject_hard_negative_success: {selected_metrics.get('same_subject_hard_negative_success')}",
        f"- fail_to_success vs v2: {selected_metrics.get('fail_to_success_vs_v2')}",
        f"- success_to_failure vs v2: {selected_metrics.get('success_to_failure_vs_v2')}",
        "",
        "For two-stage reranking variants, retrieval margin is a rank-diagnostic score rather than the same physical distance scale as v2. Selection therefore uses top1, MRR, diagnosed failure counts, and hard-negative preservation rather than treating the two-stage margin as confident score separation.",
        "",
        "## Failure Comparison",
        "",
        f"- Assisted comparison summary: {failure_summary}",
        "",
        "## Guardrail Notes",
        "",
        "- Compact aperiodic/residual features are treated as technical assistive retrieval features only.",
        "- Better top1 alone is not enough for selection; hard-negative specificity must not degrade.",
        "- The selected two-stage result supports rank improvement only; it is not a confident score-margin claim.",
        "- No HCP electrophysiology reference, clinical inference, device setting, or MedOn-as-healthy interpretation is used.",
    ]
    return write_text(lines, OUTPUT_DIR / "retrieval_aperiodic_assisted_v2_report.md")


def best_variant_summary(rows: pd.DataFrame, mode: str) -> dict[str, object]:
    """Return fixed-order best summary for one assisted family."""

    subset = rows[rows["mode"].astype(str).eq(mode)].copy()
    if subset.empty:
        return {"best_variant": "", "note": "No variants available."}
    subset = subset.sort_values(
        ["top1", "mrr", "other_subject_same_task_side_failure_count", "same_subject_hard_negative_success"],
        ascending=[False, False, True, False],
    )
    best = subset.iloc[0].to_dict()
    return {"best_variant": str(best["scorer"]), "best_metrics": best, "note": "Best is descriptive only; selection uses conservative gates."}


def run(args: argparse.Namespace) -> dict[str, Path]:
    """Run controlled aperiodic-assisted v2 analysis."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    state_vectors = read_state_vectors(args.state_vectors, warnings)
    missing = state_vectors_have_required_columns(state_vectors)
    if missing:
        raise SystemExit(f"State-vector table is missing required columns: {missing}. Available columns: {list(state_vectors.columns)}")
    quality = read_quality_table(args.quality_table, warnings)
    existing_subspaces = read_subspace_definitions(args.subspace_definitions, state_vectors, warnings)
    aperiodic_features = available_aperiodic_features(state_vectors)

    pairs, pair_log = build_paired_examples_all_features(state_vectors, quality)
    base_candidates = build_base_candidate_sets(pairs, MIN_DISTRACTORS)
    compact_subspaces = build_compact_subspaces(existing_subspaces, state_vectors, [], aperiodic_features)
    clean_features = [feature for feature in existing_subspaces.get("clean_stable_features", []) if f"x_{feature}" in pairs.columns]
    v2_features = compact_subspaces.get("v2_reference", [])
    aper_addition_features = compact_subspaces.get("compact_v3_aperiodic_only_addition", [])

    if not aperiodic_features:
        write_json(
            {"warning": "Compact aperiodic/residual feature files or columns were unavailable; stopped after v2 reproduction.", "warnings": warnings},
            OUTPUT_DIR / "aperiodic_feature_unavailable_warning.json",
        )

    v2_variant = VariantSpec(
        name="v2_reference",
        category="v2_reproduction",
        score_kind="group_balanced",
        feature_space="full_28_plus_coupling",
        distance="cosine",
    )
    v2_scores, v2_diag = evaluate_v2_variant(pairs, base_candidates, v2_features, v2_variant, clean_features)
    v2_hard, v2_hard_summary = same_subject_hard_negative_v2(pairs, v2_features, v2_variant, clean_features)
    v2_row = {
        "scorer": "v2_reference",
        "mode": "reference",
        "n_features": int(len(v2_features)),
        "n_pairs": int(len(v2_diag)),
        **summarize_diagnostics(v2_diag),
        "same_subject_hard_negative_success": v2_hard_summary.get("true_beats_all_rate", np.nan),
    }

    aper_scores_compact = pd.DataFrame()
    aper_diag = pd.DataFrame()
    aper_hard = pd.DataFrame()
    aper_hard_summary: dict[str, object] = {"true_beats_all_rate": np.nan}
    aper_row: dict[str, object] = {
        "scorer": "compact_aperiodic_only",
        "mode": "aperiodic_diagnostic",
        "status": "skipped_missing_aperiodic_features",
        "n_features": 0,
    }
    if aperiodic_features and aper_addition_features:
        aper_scores_compact, aper_diag = evaluate_compact_group_balanced(
            pairs,
            base_candidates,
            aper_addition_features,
            "compact_aperiodic_only",
        )
        aper_hard, aper_hard_summary = same_subject_compact(pairs, aper_addition_features, "compact_aperiodic_only")
        aper_row = {
            "scorer": "compact_aperiodic_only",
            "mode": "aperiodic_diagnostic",
            "n_features": int(len(aper_addition_features)),
            "n_pairs": int(len(aper_diag)),
            **summarize_diagnostics(aper_diag),
            "same_subject_hard_negative_success": aper_hard_summary.get("true_beats_all_rate", np.nan),
            "status": "ok",
        }

    reproduction = pd.DataFrame([v2_row, aper_row])
    paths: dict[str, Path] = {
        "reproduction": write_csv(reproduction, OUTPUT_DIR / "v2_and_aperiodic_reproduction.csv"),
        "reproduction_summary": write_json({"rows": reproduction.to_dict("records"), "warnings": warnings}, OUTPUT_DIR / "v2_and_aperiodic_reproduction_summary.json"),
        "pairing_log": write_csv(pair_log, OUTPUT_DIR / "aperiodic_assisted_pairing_log.csv"),
    }

    if not aperiodic_features:
        empty_summary = {"warning": "Compact aperiodic features unavailable; assisted variants skipped.", "v2_remains_default": True}
        for name in [
            "query_level_v2_vs_aperiodic_summary",
            "aperiodic_factor_predictability_summary",
            "aperiodic_penalty_summary",
            "two_stage_aperiodic_reranking_summary",
            "aperiodic_assisted_selection_summary",
            "v2_aperiodic_assisted_comparison_summary",
            "assisted_failure_comparison_summary",
        ]:
            paths[name] = write_json(empty_summary, OUTPUT_DIR / f"{name}.json")
        paths["report"] = write_report(v2_row, aper_row, {}, {}, {}, {}, empty_summary, {})
        return paths

    query_cmp, query_cmp_summary = compare_query_level(v2_diag, aper_diag, v2_hard)
    paths["query_cmp"] = write_csv(query_cmp, OUTPUT_DIR / "query_level_v2_vs_aperiodic.csv")
    paths["query_cmp_summary"] = write_json(query_cmp_summary, OUTPUT_DIR / "query_level_v2_vs_aperiodic_summary.json")

    factor_table, factor_summary = factor_predictability(state_vectors, aperiodic_features, quality, args.random_seed)
    paths["factor"] = write_csv(factor_table, OUTPUT_DIR / "aperiodic_factor_predictability.csv")
    paths["factor_summary"] = write_json(factor_summary, OUTPUT_DIR / "aperiodic_factor_predictability_summary.json")

    aper_distance = aper_distance_scores(pairs, base_candidates, aperiodic_features, "aperiodic_distance")
    assisted_specs = [
        *[
            AssistedVariantSpec(
                name=f"v2_plus_aperiodic_penalty_lambda_{str(lam).replace('.', '_')}",
                mode="penalty",
                lambda_penalty=lam,
                note="final_distance = v2_distance + lambda * compact_aperiodic_distance",
            )
            for lam in APERIODIC_PENALTY_LAMBDAS
        ],
        *[
            AssistedVariantSpec(
                name=f"top{k}_alpha_{str(alpha).replace('.', '_')}",
                mode="two_stage",
                top_k=k,
                alpha=alpha,
                note="v2 generates top-k candidates; compact aperiodic similarity reranks inside top-k only",
            )
            for k in TOP_K_VALUES
            for alpha in RERANK_ALPHAS
        ],
    ]

    result_rows: list[dict[str, object]] = []
    diagnostics_by_name = {"v2_reference": v2_diag, "compact_aperiodic_only": aper_diag}
    scores_by_name = {"v2_reference": v2_scores, "compact_aperiodic_only": aper_scores_compact}
    hard_by_name = {"v2_reference": v2_hard, "compact_aperiodic_only": aper_hard}
    hard_summary_by_name = {"v2_reference": v2_hard_summary, "compact_aperiodic_only": aper_hard_summary}
    penalty_change_logs: list[pd.DataFrame] = []
    two_stage_change_logs: list[pd.DataFrame] = []
    two_stage_extra: dict[str, dict[str, int]] = {}

    for spec in assisted_specs:
        scores, diag = evaluate_assisted_variant(v2_scores, aper_distance, spec)
        hard, hard_summary = same_subject_assisted(pairs, v2_features, aperiodic_features, clean_features, spec)
        row = summarize_assisted(spec.name, diag, hard_summary, v2_diag, spec.mode)
        row.update({"lambda_penalty": spec.lambda_penalty, "top_k": spec.top_k, "alpha": spec.alpha, "note": spec.note})
        if spec.mode == "two_stage":
            row.update(two_stage_recovery_counts(scores, v2_diag, spec))
        result_rows.append(row)
        diagnostics_by_name[spec.name] = diag
        scores_by_name[spec.name] = scores
        hard_by_name[spec.name] = hard
        hard_summary_by_name[spec.name] = hard_summary
        change = query_change_log_for_variant(v2_diag, diag, spec.name)
        if spec.mode == "penalty":
            penalty_change_logs.append(change)
        else:
            two_stage_change_logs.append(change)
            two_stage_extra[spec.name] = two_stage_recovery_counts(scores, v2_diag, spec)

    assisted_results = pd.DataFrame(result_rows)
    penalty_results = assisted_results[assisted_results["mode"].eq("penalty")].copy()
    two_stage_results = assisted_results[assisted_results["mode"].eq("two_stage")].copy()
    penalty_change_log = pd.concat(penalty_change_logs, ignore_index=True) if penalty_change_logs else pd.DataFrame()
    two_stage_change_log = pd.concat(two_stage_change_logs, ignore_index=True) if two_stage_change_logs else pd.DataFrame()
    all_change_log = pd.concat([penalty_change_log, two_stage_change_log], ignore_index=True) if not penalty_change_log.empty or not two_stage_change_log.empty else pd.DataFrame()

    paths["penalty_results"] = write_csv(penalty_results, OUTPUT_DIR / "aperiodic_penalty_results.csv")
    paths["penalty_summary"] = write_json(
        {"rows": penalty_results.to_dict("records"), **best_variant_summary(assisted_results, "penalty")},
        OUTPUT_DIR / "aperiodic_penalty_summary.json",
    )
    paths["penalty_change_log"] = write_csv(penalty_change_log, OUTPUT_DIR / "aperiodic_penalty_query_change_log.csv")
    paths["two_stage_results"] = write_csv(two_stage_results, OUTPUT_DIR / "two_stage_aperiodic_reranking_results.csv")
    paths["two_stage_summary"] = write_json(
        {"rows": two_stage_results.to_dict("records"), "two_stage_recovery_counts": two_stage_extra, **best_variant_summary(assisted_results, "two_stage")},
        OUTPUT_DIR / "two_stage_aperiodic_reranking_summary.json",
    )
    paths["two_stage_change_log"] = write_csv(two_stage_change_log, OUTPUT_DIR / "two_stage_aperiodic_query_change_log.csv")

    selection = select_assisted_variant(assisted_results, v2_row, all_change_log)
    selected_name = str(selection.get("selected_variant_name") or "")
    selected_diag = diagnostics_by_name.get(selected_name, pd.DataFrame())
    selected_scores = scores_by_name.get(selected_name, pd.DataFrame())
    selected_hard = hard_by_name.get(selected_name, pd.DataFrame())
    selected_spec = next((spec for spec in assisted_specs if spec.name == selected_name), None)

    selected_boot = pd.DataFrame()
    selected_boot_summary: dict[str, object] = {"warning": "No selected assisted variant; bootstrap skipped because v2 remains default."}
    selected_random = pd.DataFrame()
    selected_random_summary: dict[str, object] = {"warning": "No selected assisted variant; random-label control skipped because v2 remains default."}
    selected_loso = pd.DataFrame()
    selected_loso_summary: dict[str, object] = {"warning": "No selected assisted variant; LOSO skipped because v2 remains default."}
    selected_hard_summary: dict[str, object] = {"warning": "No selected assisted variant; same-subject hard-negative control skipped because v2 remains default."}
    if selected_name and selected_spec is not None and not selected_diag.empty and not selected_scores.empty:
        selected_boot, selected_boot_summary = bootstrap_ci(selected_diag, selected_hard, args.n_bootstrap, args.random_seed)
        selected_random, selected_random_summary = random_label_negative_control(selected_scores, selected_diag, args.n_random_label_permutations, args.random_seed)
        selected_loso, selected_loso_summary = loso_selected(pairs, v2_features, aperiodic_features, clean_features, selected_spec)
        _, selected_hard_summary = same_subject_assisted(pairs, v2_features, aperiodic_features, clean_features, selected_spec)

    paths["selection"] = write_json(selection, OUTPUT_DIR / "aperiodic_assisted_selection_summary.json")
    paths["selected_boot"] = write_csv(selected_boot, OUTPUT_DIR / "selected_assisted_bootstrap_ci.csv")
    paths["selected_boot_summary"] = write_json(selected_boot_summary, OUTPUT_DIR / "selected_assisted_bootstrap_ci_summary.json")
    paths["selected_random"] = write_csv(selected_random, OUTPUT_DIR / "selected_assisted_random_label_control.csv")
    paths["selected_random_summary"] = write_json(selected_random_summary, OUTPUT_DIR / "selected_assisted_random_label_control_summary.json")
    paths["selected_loso"] = write_csv(selected_loso, OUTPUT_DIR / "selected_assisted_loso.csv")
    paths["selected_loso_summary"] = write_json(selected_loso_summary, OUTPUT_DIR / "selected_assisted_loso_summary.json")
    paths["selected_hard"] = write_csv(selected_hard if selected_name else pd.DataFrame(), OUTPUT_DIR / "selected_assisted_same_subject_hard_negative.csv")
    paths["selected_hard_summary"] = write_json(selected_hard_summary, OUTPUT_DIR / "selected_assisted_same_subject_hard_negative_summary.json")

    comparison_rows = [v2_row, aper_row, *assisted_results.to_dict("records")]
    comparison = pd.DataFrame(comparison_rows)
    comparison["selected_yes_no"] = comparison["scorer"].astype(str).eq(selected_name)
    paths["comparison"] = write_csv(comparison, OUTPUT_DIR / "v2_aperiodic_assisted_comparison.csv")
    paths["comparison_summary"] = write_json({"rows": comparison.to_dict("records"), "selected_variant_name": selected_name}, OUTPUT_DIR / "v2_aperiodic_assisted_comparison_summary.json")

    best_raw = assisted_results.sort_values(["top1", "mrr"], ascending=False).iloc[0]["scorer"] if not assisted_results.empty else ""
    diag_for_failure = selected_diag if selected_name else diagnostics_by_name.get(str(best_raw), pd.DataFrame())
    failure_variant = selected_name if selected_name else f"best_raw_diagnostic:{best_raw}"
    failure_table, failure_summary = failure_comparison(v2_diag, aper_diag, diag_for_failure, failure_variant)
    paths["failure"] = write_csv(failure_table, OUTPUT_DIR / "assisted_failure_comparison.csv")
    paths["failure_summary"] = write_json(failure_summary, OUTPUT_DIR / "assisted_failure_comparison_summary.json")
    paths["change_summary"] = write_json(query_change_summary(all_change_log), OUTPUT_DIR / "aperiodic_assisted_query_change_summary.json")
    paths["change_log"] = write_csv(all_change_log, OUTPUT_DIR / "aperiodic_assisted_query_change_log.csv")

    penalty_summary = best_variant_summary(assisted_results, "penalty")
    rerank_summary = best_variant_summary(assisted_results, "two_stage")
    selected_metrics = selection.get("selected_variant_metrics") if isinstance(selection.get("selected_variant_metrics"), dict) else None
    make_figures(comparison, all_change_log, factor_table, selected_random, selected_loso, selected_metrics)
    paths["report"] = write_report(v2_row, aper_row, query_cmp_summary, factor_summary, penalty_summary, rerank_summary, selection, failure_summary)
    if warnings:
        paths["warnings"] = write_json({"warnings": warnings}, OUTPUT_DIR / "warnings.json")

    best_assisted = assisted_results.sort_values(["top1", "mrr"], ascending=False).iloc[0] if not assisted_results.empty else pd.Series(dtype=object)
    selected_metrics = selection.get("selected_variant_metrics") if isinstance(selection.get("selected_variant_metrics"), dict) else None
    print("Retrieval aperiodic-assisted v2 complete.")
    print(f"v2 top1/MRR: {float(v2_row['top1']):.3f} / {float(v2_row['mrr']):.3f}")
    print(f"aperiodic-only top1/MRR: {float(aper_row.get('top1', np.nan)):.3f} / {float(aper_row.get('mrr', np.nan)):.3f}")
    if not best_assisted.empty:
        print(f"best assisted variant: {best_assisted['scorer']} top1/MRR {float(best_assisted['top1']):.3f} / {float(best_assisted['mrr']):.3f}")
    if selected_metrics:
        print(f"selected assisted variant: {selected_metrics['scorer']} top1/MRR {float(selected_metrics['top1']):.3f} / {float(selected_metrics['mrr']):.3f}")
        same_delta = float(selected_metrics["same_subject_hard_negative_success"]) - float(v2_row["same_subject_hard_negative_success"])
        hard_delta = int(selected_metrics["other_subject_same_task_side_failure_count"]) - int(v2_row["other_subject_same_task_side_failure_count"])
        print(f"same-subject hard-negative success change: {same_delta:.3f}")
        print(f"other_subject_same_task_side failure change: {hard_delta}")
    else:
        print("selected assisted variant: none")
        print("same-subject hard-negative success change: no selected assisted variant")
        print("other_subject_same_task_side failure change: no selected assisted variant")
    print(f"assisted variant passed selection: {bool(selected_metrics)}")
    print(f"v2 remains default: {bool(selection.get('v2_remains_default', True))}")
    print(f"output path: {OUTPUT_DIR}")
    return paths


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run controlled aperiodic-assisted v2 retrieval analysis.")
    parser.add_argument("--state-vectors", default=str(STATE_VECTORS_PATH))
    parser.add_argument("--quality-table", default=str(QUALITY_TABLE_PATH))
    parser.add_argument("--subspace-definitions", default=str(SUBSPACE_DEFINITIONS_PATH))
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--n-random-label-permutations", type=int, default=N_RANDOM_LABEL_PERMUTATIONS)
    return parser.parse_args()


def main() -> None:
    """Entry point."""

    run(parse_args())


if __name__ == "__main__":
    main()
