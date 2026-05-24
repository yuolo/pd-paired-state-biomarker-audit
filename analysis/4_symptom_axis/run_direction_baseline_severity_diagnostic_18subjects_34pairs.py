"""Direction and baseline-severity diagnostic for symptom transition axes.

This lightweight diagnostic explains the directed negative symptom-axis result
without changing frozen retrieval logic or refitting the fixed-plan models.
It reads cached outputs from the 18-subject symptom-axis analysis.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402

from scripts.run_retrieval_distance_geometry_improvement import to_builtin  # noqa: E402


OUTPUT_DIR = Path("outputs/dopaminergic_symptom_transition_axes_18subjects_34pairs")
COMPONENT_SCORES = OUTPUT_DIR / "component_transition_scores.csv"
AXIS_PREDICTIONS = OUTPUT_DIR / "symptom_axis_loso_predictions.csv"
CLINICAL_TARGETS = OUTPUT_DIR / "clinical_target_inventory.csv"
AXIS_SUMMARY = OUTPUT_DIR / "symptom_axis_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "symptom_transition_axes_summary.json"

COMPONENTS = [
    "aperiodic_normalization",
    "beta_power_residual_suppression",
    "cortico_stn_beta_desynchronization",
    "gamma_restoration",
    "coupling_asymmetry_transition",
    "compact_v2a_transition",
]

TARGET_SPECS = [
    {
        "symptom": "AR",
        "laterality": "contralateral",
        "raw": "contralateral_ar_response",
        "percent": "contralateral_ar_percent_response",
        "off": "contralateral_ar_off",
        "primary": True,
    },
    {
        "symptom": "AR",
        "laterality": "ipsilateral",
        "raw": "ipsilateral_ar_response",
        "percent": "ipsilateral_ar_percent_response",
        "off": "ipsilateral_ar_off",
        "primary": False,
    },
    {
        "symptom": "tremor",
        "laterality": "contralateral",
        "raw": "contralateral_trem_response",
        "percent": "contralateral_trem_percent_response",
        "off": "contralateral_trem_off",
        "primary": True,
    },
    {
        "symptom": "tremor",
        "laterality": "ipsilateral",
        "raw": "ipsilateral_trem_response",
        "percent": "ipsilateral_trem_percent_response",
        "off": "ipsilateral_trem_off",
        "primary": False,
    },
    {
        "symptom": "total_updrs",
        "laterality": "nonlateralized",
        "raw": "sum_response",
        "percent": "sum_percent_response",
        "off": "updrs_off_sum",
        "primary": False,
    },
    {
        "symptom": "axial",
        "laterality": "nonlateralized",
        "raw": "axial_response",
        "percent": "axial_percent_response",
        "off": "updrs_off_axial",
        "primary": False,
    },
]

AXIS_TARGET_MAP = {
    "AR_contralateral": {
        "raw": "contralateral_ar_response",
        "percent": "contralateral_ar_percent_response",
        "off": "contralateral_ar_off",
    },
    "tremor_contralateral": {
        "raw": "contralateral_trem_response",
        "percent": "contralateral_trem_percent_response",
        "off": "contralateral_trem_off",
    },
    "total_updrs": {
        "raw": "sum_response",
        "percent": "sum_percent_response",
        "off": "updrs_off_sum",
    },
    "axial": {
        "raw": "axial_response",
        "percent": "axial_percent_response",
        "off": "updrs_off_axial",
    },
}


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_json(data: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(data), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_text(lines: Iterable[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def finite_spearman(x: pd.Series, y: pd.Series) -> tuple[int, float, float]:
    data = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(data) < 4 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return int(len(data)), float("nan"), float("nan")
    result = spearmanr(data["x"], data["y"])
    return int(len(data)), float(result.statistic), float(result.pvalue)


def partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> tuple[int, float]:
    data = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
            "z": pd.to_numeric(z, errors="coerce"),
        }
    ).dropna()
    if len(data) < 6 or data["x"].nunique() < 2 or data["y"].nunique() < 2 or data["z"].nunique() < 2:
        return int(len(data)), float("nan")
    xr = rankdata(data["x"], method="average")
    yr = rankdata(data["y"], method="average")
    zr = rankdata(data["z"], method="average")
    design = np.column_stack([np.ones(len(zr)), zr])
    x_res = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    y_res = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    if np.std(x_res) == 0 or np.std(y_res) == 0:
        return int(len(data)), float("nan")
    return int(len(data)), float(np.corrcoef(x_res, y_res)[0, 1])


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return pd.Series(q, index=p_values.index)
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    adjusted = np.empty(m)
    running = 1.0
    for pos in range(m - 1, -1, -1):
        running = min(running, ranked[pos] * m / (pos + 1))
        adjusted[pos] = running
    q[order] = adjusted
    return pd.Series(np.minimum(q, 1.0), index=p_values.index)


def sign_label(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-12:
        return "zero_or_nan"
    return "positive" if value > 0 else "negative"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    missing = [path for path in [COMPONENT_SCORES, AXIS_PREDICTIONS, CLINICAL_TARGETS, AXIS_SUMMARY, SUMMARY_JSON] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required diagnostic inputs: " + "; ".join(str(path) for path in missing))
    scores = pd.read_csv(COMPONENT_SCORES)
    clinical = pd.read_csv(CLINICAL_TARGETS)
    axis_pred = pd.read_csv(AXIS_PREDICTIONS)
    axis_summary = pd.read_csv(AXIS_SUMMARY)
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    scores = scores.merge(
        clinical[["subject", "updrs_off_sum", "updrs_off_axial"]].drop_duplicates("subject"),
        on="subject",
        how="left",
    )
    return scores, axis_pred, axis_summary, clinical, summary


def component_diagnostic(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for spec in TARGET_SPECS:
        for component in COMPONENTS:
            for feature_kind, feature_col in [
                ("oriented_mean", f"component__{component}__oriented_mean"),
                ("norm", f"component__{component}__norm"),
            ]:
                if feature_col not in scores.columns:
                    continue
                n_raw, r_raw, p_raw = finite_spearman(scores[feature_col], scores[spec["raw"]])
                n_percent, r_percent, p_percent = finite_spearman(scores[feature_col], scores[spec["percent"]])
                n_off, r_off, p_off = finite_spearman(scores[feature_col], scores[spec["off"]])
                n_pr_raw, partial_raw = partial_spearman(scores[feature_col], scores[spec["raw"]], scores[spec["off"]])
                n_pr_percent, partial_percent = partial_spearman(
                    scores[feature_col], scores[spec["percent"]], scores[spec["off"]]
                )
                raw_sign = sign_label(r_raw)
                percent_sign = sign_label(r_percent)
                off_sign = sign_label(r_off)
                sign_flip = raw_sign in {"positive", "negative"} and percent_sign in {"positive", "negative"} and raw_sign != percent_sign
                severity_pattern = (
                    np.isfinite(r_off)
                    and np.isfinite(r_percent)
                    and abs(r_off) >= 0.30
                    and abs(r_percent) >= 0.30
                    and np.sign(r_off) != np.sign(r_percent)
                )
                rows.append(
                    {
                        "symptom": spec["symptom"],
                        "laterality": spec["laterality"],
                        "primary_target": bool(spec["primary"]),
                        "component": component,
                        "feature_kind": feature_kind,
                        "feature_column": feature_col,
                        "raw_response_column": spec["raw"],
                        "percent_response_column": spec["percent"],
                        "off_severity_column": spec["off"],
                        "n_raw": n_raw,
                        "raw_response_spearman_r": r_raw,
                        "raw_response_spearman_p": p_raw,
                        "n_percent": n_percent,
                        "percent_response_spearman_r": r_percent,
                        "percent_response_spearman_p": p_percent,
                        "n_off": n_off,
                        "off_severity_spearman_r": r_off,
                        "off_severity_spearman_p": p_off,
                        "n_partial_raw": n_pr_raw,
                        "partial_raw_response_controlling_off_r": partial_raw,
                        "n_partial_percent": n_pr_percent,
                        "partial_percent_response_controlling_off_r": partial_percent,
                        "raw_response_sign": raw_sign,
                        "percent_response_sign": percent_sign,
                        "off_severity_sign": off_sign,
                        "raw_vs_percent_sign_flip": bool(sign_flip),
                        "baseline_severity_opposite_percent_pattern": bool(severity_pattern),
                    }
                )
    out = pd.DataFrame(rows)
    for col in ["raw_response_spearman_p", "percent_response_spearman_p", "off_severity_spearman_p"]:
        out[f"{col}_fdr_q"] = bh_fdr(out[col])
    summary_rows = []
    for symptom, group in out.groupby(["symptom", "laterality"], dropna=False):
        summary_rows.append(
            {
                "symptom": symptom[0],
                "laterality": symptom[1],
                "n_tests": int(len(group)),
                "n_raw_percent_sign_flips": int(group["raw_vs_percent_sign_flip"].sum()),
                "n_baseline_severity_opposite_percent_patterns": int(
                    group["baseline_severity_opposite_percent_pattern"].sum()
                ),
                "strongest_percent_component": str(
                    group.loc[group["percent_response_spearman_r"].abs().idxmax(), "component"]
                )
                if group["percent_response_spearman_r"].notna().any()
                else "",
                "strongest_percent_abs_r": float(group["percent_response_spearman_r"].abs().max())
                if group["percent_response_spearman_r"].notna().any()
                else float("nan"),
                "strongest_off_component": str(group.loc[group["off_severity_spearman_r"].abs().idxmax(), "component"])
                if group["off_severity_spearman_r"].notna().any()
                else "",
                "strongest_off_abs_r": float(group["off_severity_spearman_r"].abs().max())
                if group["off_severity_spearman_r"].notna().any()
                else float("nan"),
            }
        )
    return out, pd.DataFrame(summary_rows)


def axis_diagnostic(scores: pd.DataFrame, axis_pred: pd.DataFrame, axis_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target_lookup = scores[
        [
            "subject",
            "hemisphere",
            "contralateral_ar_response",
            "contralateral_ar_percent_response",
            "contralateral_ar_off",
            "contralateral_trem_response",
            "contralateral_trem_percent_response",
            "contralateral_trem_off",
            "sum_response",
            "sum_percent_response",
            "updrs_off_sum",
            "axial_response",
            "axial_percent_response",
            "updrs_off_axial",
        ]
    ].drop_duplicates(["subject", "hemisphere"])
    merged = axis_pred.merge(target_lookup, on=["subject", "hemisphere"], how="left", suffixes=("", "_target"))
    for axis_name, spec in AXIS_TARGET_MAP.items():
        sub = merged[merged["axis_name"].eq(axis_name)].copy()
        if sub.empty:
            continue
        _, r_raw, p_raw = finite_spearman(sub["y_pred"], sub[spec["raw"]])
        _, r_percent, p_percent = finite_spearman(sub["y_pred"], sub[spec["percent"]])
        _, r_off, p_off = finite_spearman(sub["y_pred"], sub[spec["off"]])
        _, r_true_off, p_true_off = finite_spearman(sub[spec["raw"]], sub[spec["off"]])
        axis_row = axis_summary[axis_summary["axis_name"].eq(axis_name)]
        rows.append(
            {
                "axis_name": axis_name,
                "raw_response_column": spec["raw"],
                "percent_response_column": spec["percent"],
                "off_severity_column": spec["off"],
                "n_predictions": int(sub["y_pred"].notna().sum()),
                "pred_vs_raw_response_spearman_r": r_raw,
                "pred_vs_raw_response_spearman_p": p_raw,
                "pred_vs_percent_response_spearman_r": r_percent,
                "pred_vs_percent_response_spearman_p": p_percent,
                "pred_vs_off_severity_spearman_r": r_off,
                "pred_vs_off_severity_spearman_p": p_off,
                "raw_response_vs_off_severity_spearman_r": r_true_off,
                "raw_response_vs_off_severity_spearman_p": p_true_off,
                "axis_original_spearman_r": float(axis_row["spearman_r"].iloc[0]) if not axis_row.empty else r_raw,
                "axis_directional_permutation_p": float(axis_row["permutation_p_spearman_directional"].iloc[0])
                if not axis_row.empty
                else float("nan"),
                "axis_weight_stable": bool(axis_row["axis_weight_stable"].iloc[0]) if not axis_row.empty else False,
                "raw_vs_percent_sign_flip": bool(
                    sign_label(r_raw) in {"positive", "negative"}
                    and sign_label(r_percent) in {"positive", "negative"}
                    and sign_label(r_raw) != sign_label(r_percent)
                ),
                "prediction_tracks_off_more_than_raw_response": bool(
                    np.isfinite(r_off) and np.isfinite(r_raw) and abs(r_off) > abs(r_raw)
                ),
            }
        )
    out = pd.DataFrame(rows)
    for col in [
        "pred_vs_raw_response_spearman_p",
        "pred_vs_percent_response_spearman_p",
        "pred_vs_off_severity_spearman_p",
    ]:
        out[f"{col}_fdr_q"] = bh_fdr(out[col])
    return out


def plot_diagnostics(component_df: pd.DataFrame, axis_df: pd.DataFrame) -> None:
    primary = component_df[
        component_df["primary_target"].astype(bool)
        & component_df["laterality"].eq("contralateral")
        & component_df["feature_kind"].eq("oriented_mean")
    ].copy()
    if not primary.empty:
        for symptom, filename in [
            ("AR", "direction_raw_vs_percent_ar_components.png"),
            ("tremor", "direction_raw_vs_percent_tremor_components.png"),
        ]:
            sub = primary[primary["symptom"].eq(symptom)]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            x = np.arange(len(sub))
            ax.bar(x - 0.2, sub["raw_response_spearman_r"], width=0.4, label="raw response")
            ax.bar(x + 0.2, sub["percent_response_spearman_r"], width=0.4, label="percent response")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(x, [c.replace("_", "\n") for c in sub["component"]], fontsize=7)
            ax.set_ylabel("Spearman r")
            ax.set_title(f"{symptom}: raw vs percent response direction")
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / filename, dpi=180)
            plt.close(fig)

        heat = primary.pivot_table(index="component", columns="symptom", values="off_severity_spearman_r")
        fig, ax = plt.subplots(figsize=(6, 4.8))
        im = ax.imshow(heat.fillna(0).to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(len(heat.columns)), heat.columns)
        ax.set_yticks(range(len(heat.index)), heat.index, fontsize=7)
        ax.set_title("Component association with baseline OFF severity")
        fig.colorbar(im, ax=ax, label="Spearman r")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "baseline_severity_component_heatmap.png", dpi=180)
        plt.close(fig)

    if not axis_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(axis_df))
        ax.bar(x - 0.25, axis_df["pred_vs_raw_response_spearman_r"], width=0.25, label="pred vs raw")
        ax.bar(x, axis_df["pred_vs_percent_response_spearman_r"], width=0.25, label="pred vs percent")
        ax.bar(x + 0.25, axis_df["pred_vs_off_severity_spearman_r"], width=0.25, label="pred vs OFF")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, axis_df["axis_name"], rotation=30, ha="right")
        ax.set_ylabel("Spearman r")
        ax.set_title("Axis predictions: response vs percent vs baseline severity")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "axis_prediction_raw_percent_severity.png", dpi=180)
        plt.close(fig)


def write_report(component_df: pd.DataFrame, component_summary: pd.DataFrame, axis_df: pd.DataFrame) -> dict[str, object]:
    primary = component_df[
        component_df["primary_target"].astype(bool)
        & component_df["laterality"].eq("contralateral")
        & component_df["feature_kind"].eq("oriented_mean")
    ].copy()
    notable = primary[
        (primary["percent_response_spearman_r"].abs() >= 0.30)
        | (primary["off_severity_spearman_r"].abs() >= 0.30)
        | primary["baseline_severity_opposite_percent_pattern"].astype(bool)
    ].copy()
    summary = {
        "diagnostic": "direction_and_baseline_severity",
        "input_folder": str(OUTPUT_DIR),
        "n_component_tests": int(len(component_df)),
        "n_axis_tests": int(len(axis_df)),
        "n_primary_oriented_notable_rows": int(len(notable)),
        "notable_primary_oriented_rows": notable[
            [
                "symptom",
                "component",
                "raw_response_spearman_r",
                "percent_response_spearman_r",
                "off_severity_spearman_r",
                "baseline_severity_opposite_percent_pattern",
            ]
        ].to_dict(orient="records"),
        "axis_direction_rows": axis_df.to_dict(orient="records"),
        "interpretation_boundary": [
            "Diagnostic only; does not rescue or replace the fixed-plan primary result.",
            "Percent-response and baseline-severity patterns are exploratory explanations for the directed negative result.",
            "No clinical prediction, treatment recommendation, DBS optimization, or causal medication-effect claim is made.",
        ],
    }
    write_json(summary, OUTPUT_DIR / "direction_baseline_severity_summary.json")

    lines = [
        "# Direction and Baseline-Severity Diagnostic",
        "",
        "Scope: diagnostic explanation of the directed negative symptom-axis result.",
        "",
        "## Summary",
        f"- component diagnostic rows: {len(component_df)}",
        f"- axis diagnostic rows: {len(axis_df)}",
        f"- notable primary oriented rows: {len(notable)}",
        "",
        "## Notable Primary Patterns",
    ]
    if notable.empty:
        lines.append("- No notable primary oriented component rows under the diagnostic thresholds.")
    else:
        for _, row in notable.iterrows():
            lines.append(
                "- {symptom} / {component}: raw r={raw:.3f}, percent r={percent:.3f}, OFF r={off:.3f}, severity-opposite-percent={flag}".format(
                    symptom=row["symptom"],
                    component=row["component"],
                    raw=row["raw_response_spearman_r"],
                    percent=row["percent_response_spearman_r"],
                    off=row["off_severity_spearman_r"],
                    flag=row["baseline_severity_opposite_percent_pattern"],
                )
            )
    lines.extend(
        [
            "",
            "## Axis Direction",
        ]
    )
    for _, row in axis_df.iterrows():
        lines.append(
            "- {axis}: pred vs raw r={raw:.3f}, pred vs percent r={percent:.3f}, pred vs OFF r={off:.3f}".format(
                axis=row["axis_name"],
                raw=row["pred_vs_raw_response_spearman_r"],
                percent=row["pred_vs_percent_response_spearman_r"],
                off=row["pred_vs_off_severity_spearman_r"],
            )
        )
    lines.extend(["", "## Boundary"])
    lines.extend(f"- {line}" for line in summary["interpretation_boundary"])
    write_text(lines, OUTPUT_DIR / "direction_baseline_severity_report.md")
    return summary


def main() -> None:
    scores, axis_pred, axis_summary, _clinical, _summary = load_inputs()
    component_df, component_summary = component_diagnostic(scores)
    axis_df = axis_diagnostic(scores, axis_pred, axis_summary)

    write_csv(component_df, OUTPUT_DIR / "direction_baseline_severity_component_diagnostic.csv")
    write_csv(component_summary, OUTPUT_DIR / "direction_baseline_severity_component_summary.csv")
    write_csv(axis_df, OUTPUT_DIR / "direction_baseline_severity_axis_diagnostic.csv")
    plot_diagnostics(component_df, axis_df)
    summary = write_report(component_df, component_summary, axis_df)

    print("direction/baseline severity diagnostic completed")
    print(f"component diagnostic rows: {summary['n_component_tests']}")
    print(f"axis diagnostic rows: {summary['n_axis_tests']}")
    print(f"notable primary oriented rows: {summary['n_primary_oriented_notable_rows']}")
    print(f"output path: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
