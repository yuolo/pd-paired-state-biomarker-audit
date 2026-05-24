#!/usr/bin/env python3
"""Try raw-signal OXF montage reconstruction before median fallback.

This is an exploratory transferability audit. It keeps all 30 complete OXF
hemisphere pairs, but attempts a fixed, label-derived reconstruction of the
missing ON bipolar channel before using the median-channel fallback.

Guardrails:
- No ds004998 frozen output is modified.
- top_k=5 and aperiodic_alpha=0.5 are preserved.
- Exact contacts are used first.
- Reconstructed rows are flagged by provenance.
- Numeric-token repair remains sensitivity only.
- Raw reconstruction is only accepted for two-contact bipolar targets and
  uses fixed contact-label algebra, not metric-based selection.
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from scripts import run_oxf_external_stn_retrieval_validation as oxf_base  # noqa: E402
from scripts import run_oxf_montage_harmonization_audit as montage  # noqa: E402
from scripts import run_oxf_stn_physiology_locked_v2e as v2e  # noqa: E402
from scripts.run_v2a_scientific_audit_17subjects_33pairs import (  # noqa: E402
    APERIODIC_ALPHA,
    RANDOM_SEED,
    TOP_K,
)
from scripts.run_v2b_gated_scientific_audit_17subjects_33pairs import ALL_MEDON_CONDITION  # noqa: E402
from scripts.run_v2c_deconfounded_transition_retrieval_experiment_17subjects_33pairs import (  # noqa: E402
    V2C_DECONFOUNDED_NAME,
)
from scripts.run_v2d_kreciprocal_quality_rerank_experiment_17subjects_33pairs import (  # noqa: E402
    V2D_QUALITY_DECONFOUNDED_NAME,
)


OUTPUT_DIR = Path("outputs/oxf_raw_montage_reconstruction_audit")
CHANNEL_FEATURES_PATH = montage.CHANNEL_FEATURES_PATH
V2E_PAIRS_PATH = montage.V2E_PAIRS_PATH
MAX_DURATION_SEC = 120.0
FEATURES = list(v2e.PHYSIOLOGY_FEATURES)

BRANCH_EXACT_RECON_MEDIAN = "all_pairs_exact_raw_reconstruct_else_median_compact_residual_7"
BRANCH_EXACT_RECON_NUMERIC_MEDIAN = "all_pairs_exact_raw_reconstruct_numeric_else_median_compact_residual_7"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    return str(path)


def write_json(data: dict[str, object], path: Path) -> str:
    path.write_text(json.dumps(oxf_base.to_builtin(data), indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def safe_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:  # noqa: BLE001
        return default
    return out if np.isfinite(out) else default


def load_raw_selected_channels(file_path: str, side: str) -> tuple[pd.DataFrame, float, dict[str, object]]:
    """Load side-selected raw channels and annotate contact geometry."""

    titles, signals, sfreq, meta = oxf_base.read_record_channels(Path(file_path), MAX_DURATION_SEC)
    rows = []
    for title, signal in zip(titles, signals, strict=False):
        geom = montage.geometry_from_channel(title, side)
        rows.append(
            {
                "channel": title,
                "signal": np.asarray(signal, dtype=float),
                **geom,
            }
        )
    return pd.DataFrame(rows), float(sfreq), meta


def feature_row_from_signal(signal: np.ndarray, sfreq: float) -> dict[str, float]:
    """Compute the same cached feature family for a reconstructed signal."""

    spectral = oxf_base.channel_feature_row(signal, sfreq)
    bursts = v2e.beta_burst_feature_row(signal, sfreq)
    row = {feature: np.nan for feature in FEATURES}
    row.update(spectral)
    row.update(bursts)
    return row


def row_for_reconstructed_signal(
    base_on_row: pd.Series,
    channel: str,
    signal: np.ndarray,
    sfreq: float,
    provenance: dict[str, object],
) -> pd.Series:
    row = base_on_row.to_dict()
    row.update(feature_row_from_signal(signal, sfreq))
    row["channel"] = channel
    row["sampling_frequency"] = sfreq
    row["n_samples_used"] = int(len(signal))
    row["duration_sec_used"] = float(len(signal) / sfreq) if np.isfinite(sfreq) and sfreq > 0 else np.nan
    row["feature_status"] = "ok"
    row.update(provenance)
    return pd.Series(row)


def find_single_contact_signals(raw: pd.DataFrame) -> dict[int, np.ndarray]:
    singles: dict[int, np.ndarray] = {}
    for _, row in raw.iterrows():
        if str(row.get("channel_geometry_class")) != "single_or_monopolar_label":
            continue
        digits = str(row.get("contact_digits", ""))
        if len(digits) != 1:
            continue
        singles[int(digits)] = np.asarray(row["signal"], dtype=float)
    return singles


def graph_edges(raw: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    edges: dict[tuple[int, int], np.ndarray] = {}
    for _, row in raw.iterrows():
        if not bool(row.get("valid_bipolar_geometry", False)):
            continue
        digits = [int(char) for char in str(row.get("contact_digits", "")) if char.isdigit()]
        if len(digits) != 2:
            continue
        a, b = sorted(digits)
        edges[(a, b)] = np.asarray(row["signal"], dtype=float)
    return edges


def reconstruct_from_single_contacts(raw: pd.DataFrame, target: tuple[int, int]) -> tuple[np.ndarray | None, dict[str, object]]:
    singles = find_single_contact_signals(raw)
    a, b = target
    if a not in singles or b not in singles:
        return None, {"reconstruction_mode": "single_contact_difference_unavailable"}
    n = min(len(singles[a]), len(singles[b]))
    if n < 128:
        return None, {"reconstruction_mode": "single_contact_difference_too_short"}
    return singles[a][:n] - singles[b][:n], {
        "reconstruction_mode": "single_contact_difference",
        "reconstruction_source_channels": f"{a};{b}",
    }


def shortest_edge_path(edges: dict[tuple[int, int], np.ndarray], target: tuple[int, int]) -> list[tuple[int, int, int]] | None:
    """Return path as directed edge triples (from, to, sign)."""

    a, b = target
    adjacency: dict[int, list[tuple[int, tuple[int, int], int]]] = {}
    for edge in edges:
        lo, hi = edge
        adjacency.setdefault(lo, []).append((hi, edge, 1))
        adjacency.setdefault(hi, []).append((lo, edge, -1))
    queue: deque[tuple[int, list[tuple[int, int, int]]]] = deque([(a, [])])
    seen = {a}
    while queue:
        node, path = queue.popleft()
        if node == b:
            return path
        for nxt, edge, sign in adjacency.get(node, []):
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, [*path, (edge[0], edge[1], sign)]))
    return None


def reconstruct_from_bipolar_path(raw: pd.DataFrame, target: tuple[int, int]) -> tuple[np.ndarray | None, dict[str, object]]:
    edges = graph_edges(raw)
    path = shortest_edge_path(edges, target)
    if not path:
        return None, {"reconstruction_mode": "bipolar_path_unavailable"}
    lengths = [len(edges[(lo, hi)]) for lo, hi, _sign in path]
    n = min(lengths) if lengths else 0
    if n < 128:
        return None, {"reconstruction_mode": "bipolar_path_too_short"}
    signal = np.zeros(n, dtype=float)
    labels = []
    for lo, hi, sign in path:
        signal += sign * edges[(lo, hi)][:n]
        labels.append(f"{lo}{hi}:{sign:+d}")
    return signal, {
        "reconstruction_mode": "bipolar_path_sum",
        "reconstruction_source_channels": ";".join(labels),
        "reconstruction_path_length": int(len(path)),
    }


def reconstruct_on_channel(
    on_group: pd.DataFrame,
    target_digits: str,
    side: str,
) -> tuple[pd.Series | None, dict[str, object]]:
    """Try fixed raw reconstruction of the ON target bipolar channel."""

    digits = [int(char) for char in str(target_digits) if char.isdigit()]
    if len(digits) != 2:
        return None, {
            "reconstruction_status": "not_attempted",
            "reconstruction_mode": "target_not_two_contact_bipolar",
        }
    target = tuple(sorted(digits))
    first = on_group.sort_values("channel").iloc[0]
    try:
        raw, sfreq, meta = load_raw_selected_channels(str(first["file_path"]), side)
    except Exception as exc:  # noqa: BLE001
        return None, {
            "reconstruction_status": "failed",
            "reconstruction_mode": "raw_load_failed",
            "reconstruction_error": str(exc),
        }

    signal, provenance = reconstruct_from_single_contacts(raw, target)
    if signal is None:
        signal, provenance = reconstruct_from_bipolar_path(raw, target)
    if signal is None:
        return None, {
            "reconstruction_status": "failed",
            **provenance,
            "available_raw_channels": ";".join(raw["channel"].astype(str).tolist()),
        }

    channel = f"{str(side).lower()[0].upper()}recon{target[0]}{target[1]}"
    provenance = {
        "reconstruction_status": "reconstructed",
        "raw_loader": meta.get("loader", ""),
        "available_raw_channels": ";".join(raw["channel"].astype(str).tolist()),
        **provenance,
    }
    return row_for_reconstructed_signal(first, channel, signal, sfreq, provenance), provenance


def source_maps(v2e_pairs: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    out = {}
    for feature_set in [
        "off_beta_peak_contact_exact",
        "off_beta_peak_contact_numeric",
        "median_channel_reference_with_bursts",
    ]:
        frame = v2e_pairs[v2e_pairs["feature_set"].astype(str).eq(feature_set)].copy()
        out[feature_set] = {str(row["pair_id"]): row for _, row in frame.iterrows()}
    return out


def selected_off_row(inventory: pd.DataFrame, subject: str, side: str, selected_channel: str) -> pd.Series | None:
    rows = inventory[
        inventory["subject"].astype(str).eq(subject)
        & inventory["side"].astype(str).eq(side)
        & inventory["medication_state"].astype(str).eq("OFF")
        & inventory["channel"].astype(str).eq(str(selected_channel))
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def build_raw_reconstruction_branches(
    inventory: pd.DataFrame,
    coverage: pd.DataFrame,
    v2e_pairs: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    maps = source_maps(v2e_pairs)
    branch_rows = {BRANCH_EXACT_RECON_MEDIAN: [], BRANCH_EXACT_RECON_NUMERIC_MEDIAN: []}
    cases = []
    ok_inventory = inventory[inventory["feature_status"].astype(str).eq("ok")].copy()

    for _, case in coverage.sort_values(["subject", "side"]).iterrows():
        pair_id = str(case["pair_id"])
        subject = str(case["subject"])
        side = str(case["side"])
        repair_class = str(case.get("repair_class", ""))
        exact_row = maps["off_beta_peak_contact_exact"].get(pair_id)
        numeric_row = maps["off_beta_peak_contact_numeric"].get(pair_id)
        median_row = maps["median_channel_reference_with_bursts"].get(pair_id)

        selected_channel = str(case.get("off_selected_channel", ""))
        off_row = selected_off_row(ok_inventory, subject, side, selected_channel)
        on_group = ok_inventory[
            ok_inventory["subject"].astype(str).eq(subject)
            & ok_inventory["side"].astype(str).eq(side)
            & ok_inventory["medication_state"].astype(str).eq("ON")
        ].copy()

        reconstructed_pair = None
        recon_info = {
            "pair_id": pair_id,
            "subject": subject,
            "side": side,
            "repair_class": repair_class,
            "off_selected_channel": selected_channel,
            "off_contact_digits": case.get("off_contact_digits", ""),
        }
        if exact_row is None and off_row is not None and not on_group.empty:
            on_recon, provenance = reconstruct_on_channel(on_group, str(case.get("off_contact_digits", "")), side)
            recon_info.update(provenance)
            if on_recon is not None:
                selection = {
                    "selection_rule": "off_beta_peak_raw_reconstruct_if_exact_missing",
                    "match_mode": "raw_reconstructed_same_contact_pair",
                    "selection_status": "paired",
                    "off_contact_token": case.get("off_exact_contact_token", ""),
                    "on_contact_token": on_recon.get("channel", ""),
                    "montage_hierarchy_source": "raw_reconstructed_contact",
                    "montage_repair_class": repair_class,
                    "off_spacing_class": case.get("off_spacing_class", ""),
                    "same_intercontact_distance": False,
                    **{key: value for key, value in provenance.items() if key != "available_raw_channels"},
                }
                reconstructed_pair = montage.build_pair_row_from_rows(
                    subject,
                    side,
                    "raw_reconstructed_contact",
                    off_row,
                    on_recon,
                    selection,
                )
        elif exact_row is not None:
            recon_info.update({"reconstruction_status": "not_needed", "reconstruction_mode": "exact_contact_available"})
        else:
            recon_info.update({"reconstruction_status": "not_attempted", "reconstruction_mode": "missing_off_or_on_group"})

        for branch_id, include_numeric in [
            (BRANCH_EXACT_RECON_MEDIAN, False),
            (BRANCH_EXACT_RECON_NUMERIC_MEDIAN, True),
        ]:
            source = "missing"
            row = None
            if exact_row is not None:
                row = exact_row.copy().to_dict()
                source = "exact_contact"
            elif reconstructed_pair is not None:
                row = dict(reconstructed_pair)
                source = "raw_reconstructed_contact"
            elif include_numeric and numeric_row is not None and repair_class == "numeric_only_recovered":
                row = numeric_row.copy().to_dict()
                source = "numeric_contact_repair"
            elif median_row is not None:
                row = median_row.copy().to_dict()
                source = "median_channel_fallback"
            if row is None:
                continue
            row["feature_set"] = branch_id
            row["montage_hierarchy_source"] = source
            row["montage_repair_class"] = repair_class
            row["raw_reconstruction_status"] = recon_info.get("reconstruction_status", "")
            row["raw_reconstruction_mode"] = recon_info.get("reconstruction_mode", "")
            branch_rows[branch_id].append(row)

        cases.append(recon_info)

    branch_tables = {
        branch_id: pd.DataFrame(rows).sort_values(["subject", "side"]).reset_index(drop=True)
        for branch_id, rows in branch_rows.items()
    }
    case_table = pd.DataFrame(cases).sort_values(["reconstruction_status", "subject", "side"]).reset_index(drop=True)
    source_summary = []
    for branch_id, table in branch_tables.items():
        for keys, group in table.groupby(["montage_hierarchy_source", "montage_repair_class"], dropna=False):
            source, repair = keys
            source_summary.append(
                {
                    "branch_id": branch_id,
                    "montage_hierarchy_source": source,
                    "montage_repair_class": repair,
                    "n_pairs": int(len(group)),
                    "n_subjects": int(group["subject"].nunique()),
                }
            )
    return branch_tables, case_table, pd.DataFrame(source_summary)


def evaluate_reconstruction_branches(branch_tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    results = []
    branch_specs = [
        {
            "branch_id": BRANCH_EXACT_RECON_MEDIAN,
            "branch_role": "coverage_complete_exploratory",
            "interpretation": "All 30 pairs: exact contact, then raw reconstructed contact, then median fallback.",
        },
        {
            "branch_id": BRANCH_EXACT_RECON_NUMERIC_MEDIAN,
            "branch_role": "coverage_complete_numeric_sensitivity",
            "interpretation": "All 30 pairs: exact contact, then raw reconstructed contact, then numeric repair, then median fallback.",
        },
    ]
    for spec in branch_specs:
        results.append(montage.evaluate_branch(spec, branch_tables[spec["branch_id"]]))
    return {
        "candidate_scores": montage.concat_result(results, "candidate_scores"),
        "observed_metrics": montage.concat_result(results, "observed_metrics"),
        "query_diagnostics": montage.concat_result(results, "query_diagnostics"),
        "failure_cases": montage.concat_result(results, "failure_cases"),
        "hard_negative_summary": montage.concat_result(results, "hard_negative_summary"),
        "hard_negative_cases": montage.concat_result(results, "hard_negative_cases"),
    }


def focus_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    focus = metrics[
        metrics["candidate_pool_condition"].astype(str).eq(ALL_MEDON_CONDITION)
        & metrics["variant_name"].astype(str).isin(
            [oxf_base.V2A_NAME, V2C_DECONFOUNDED_NAME, V2D_QUALITY_DECONFOUNDED_NAME]
        )
    ].copy()
    return focus.sort_values(["top1", "mrr"], ascending=False).reset_index(drop=True)


def metric_value(focus: pd.DataFrame, branch_id: str, variant: str, metric: str) -> float:
    rows = focus[
        focus["branch_id"].astype(str).eq(branch_id)
        & focus["variant_name"].astype(str).eq(variant)
    ]
    return safe_float(rows.iloc[0].get(metric)) if not rows.empty else np.nan


def write_report(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# OXF Raw Montage Reconstruction Audit",
        "",
        "Exploratory all-pairs audit. Fixed top_k=5 and alpha=0.5. No ds004998 frozen output changed.",
        "",
        "## Key Results",
    ]
    for key, value in summary["key_results"].items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.3f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings"])
    for warning in summary["warnings"]:
        lines.append(f"- {warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    assert TOP_K == 5
    assert APERIODIC_ALPHA == 0.5

    channel_features = read_csv(CHANNEL_FEATURES_PATH)
    v2e_pairs = read_csv(V2E_PAIRS_PATH)
    inventory = montage.annotate_channel_inventory(channel_features)
    coverage = montage.build_montage_coverage_cases(inventory)
    branch_tables, reconstruction_cases, source_summary = build_raw_reconstruction_branches(
        inventory,
        coverage,
        v2e_pairs,
    )
    evaluated = evaluate_reconstruction_branches(branch_tables)
    focus = focus_metrics(evaluated["observed_metrics"])
    bootstrap = montage.bootstrap_by_branch(evaluated["query_diagnostics"])

    outputs = {
        "reconstruction_cases": write_csv(reconstruction_cases, OUTPUT_DIR / "raw_reconstruction_cases.csv"),
        "source_summary": write_csv(source_summary, OUTPUT_DIR / "raw_reconstruction_source_summary.csv"),
        "candidate_scores": write_csv(evaluated["candidate_scores"], OUTPUT_DIR / "raw_reconstruction_candidate_scores.csv"),
        "observed_metrics": write_csv(evaluated["observed_metrics"], OUTPUT_DIR / "raw_reconstruction_observed_metrics.csv"),
        "focus_metrics": write_csv(focus, OUTPUT_DIR / "raw_reconstruction_focus_metrics.csv"),
        "query_diagnostics": write_csv(evaluated["query_diagnostics"], OUTPUT_DIR / "raw_reconstruction_query_diagnostics.csv"),
        "bootstrap_ci": write_csv(bootstrap, OUTPUT_DIR / "raw_reconstruction_bootstrap_ci.csv"),
        "failure_cases": write_csv(evaluated["failure_cases"], OUTPUT_DIR / "raw_reconstruction_failure_cases.csv"),
        "hard_negative_summary": write_csv(evaluated["hard_negative_summary"], OUTPUT_DIR / "raw_reconstruction_hard_negative_summary.csv"),
        "hard_negative_cases": write_csv(evaluated["hard_negative_cases"], OUTPUT_DIR / "raw_reconstruction_hard_negative_cases.csv"),
    }
    for branch_id, table in branch_tables.items():
        outputs[f"{branch_id}_pairs"] = write_csv(table, OUTPUT_DIR / f"{branch_id}_pairs.csv")

    key_results = {
        "seed": RANDOM_SEED,
        "top_k": TOP_K,
        "aperiodic_alpha": APERIODIC_ALPHA,
        "n_complete_pairs_per_branch": int(len(next(iter(branch_tables.values())))) if branch_tables else 0,
        "reconstructed_pairs": int((reconstruction_cases["reconstruction_status"].astype(str) == "reconstructed").sum())
        if not reconstruction_cases.empty
        else 0,
        "failed_reconstruction_attempts": int((reconstruction_cases["reconstruction_status"].astype(str) == "failed").sum())
        if not reconstruction_cases.empty
        else 0,
        "exact_raw_reconstruct_else_median_v2c_top1": metric_value(
            focus,
            BRANCH_EXACT_RECON_MEDIAN,
            V2C_DECONFOUNDED_NAME,
            "top1",
        ),
        "exact_raw_reconstruct_else_median_v2c_mrr": metric_value(
            focus,
            BRANCH_EXACT_RECON_MEDIAN,
            V2C_DECONFOUNDED_NAME,
            "mrr",
        ),
        "exact_raw_reconstruct_else_median_v2d_top1": metric_value(
            focus,
            BRANCH_EXACT_RECON_MEDIAN,
            V2D_QUALITY_DECONFOUNDED_NAME,
            "top1",
        ),
        "exact_raw_reconstruct_else_median_v2d_mrr": metric_value(
            focus,
            BRANCH_EXACT_RECON_MEDIAN,
            V2D_QUALITY_DECONFOUNDED_NAME,
            "mrr",
        ),
        "exact_raw_reconstruct_numeric_else_median_v2d_top1": metric_value(
            focus,
            BRANCH_EXACT_RECON_NUMERIC_MEDIAN,
            V2D_QUALITY_DECONFOUNDED_NAME,
            "top1",
        ),
        "exact_raw_reconstruct_numeric_else_median_v2d_mrr": metric_value(
            focus,
            BRANCH_EXACT_RECON_NUMERIC_MEDIAN,
            V2D_QUALITY_DECONFOUNDED_NAME,
            "mrr",
        ),
    }
    warnings = [
        "Raw reconstruction is exploratory and depends on fixed label-derived sign assumptions.",
        "Single-contact labels are treated as monopolar-like only for sensitivity; this must be metadata-verified before primary use.",
        "Bipolar path reconstruction assumes consistent ascending contact sign convention; no metric-based sign choice is used.",
        "Numeric repair remains sensitivity only.",
        "Frozen ds004998 v2A outputs are not modified.",
    ]
    summary = {"key_results": key_results, "outputs": outputs, "warnings": warnings}
    write_json(summary, OUTPUT_DIR / "raw_montage_reconstruction_summary.json")
    write_report(summary, OUTPUT_DIR / "raw_montage_reconstruction_summary.md")

    print("OXF raw montage reconstruction audit complete.")
    print(f"Output folder: {OUTPUT_DIR}")
    for key, value in key_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
