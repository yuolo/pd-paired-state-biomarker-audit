"""DBS vs levodopa STN transition-geometry convergence (Wiest/OXF cohort).

Question
--------
Does 130 Hz STN-DBS move subthalamic field-potential physiology in the *same
direction* as levodopa? We compare the medication transition vector
(Off-meds -> On-meds) with the stimulation transition vector
(baseline -> On-DBS) in the transportable STN feature space used elsewhere in
the manuscript, and test their alignment with bootstrap CIs and two nulls.

Data
----
* Medication transition: outputs/oxf_external_stn_retrieval_validation/
  oxf_onoff_pairs.csv  -> per-hemisphere Off (x_*) and On (medon_*) features
  (30 hemispheres / 17 patients), computed by the locked STN extractor.
* Stimulation transition: data/oxf/DBS/MATRIX_DBS.mat -> 26 hemispheres with
  paired baseline (signal_base) and 130 Hz DBS (signal_dbs) time series.
  Features here are computed with the SAME feature function as the meds
  pipeline (replicated verbatim from run_oxf_external_stn_retrieval_validation).

Notes / boundaries
------------------
* The 26 DBS hemispheres in MATRIX_DBS carry no patient identifiers, so the
  primary analysis is a *group-level* direction comparison (no cross-set
  per-hemisphere pairing is asserted).
* On-DBS LFP contains a 130 Hz stimulation artefact; the aperiodic fit band is
  5-95 Hz (excludes 130 Hz). We diagnose 65 Hz subharmonic leakage into the
  beta/gamma bands and report a stim-robust feature subset as sensitivity.
* This is a neurophysiological geometry comparison, not a clinical-prediction
  or DBS-programming claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as scipy_signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dbs_levodopa_transition_convergence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MATRIX_DBS = PROJECT_ROOT / "data" / "oxf" / "DBS" / "MATRIX_DBS.mat"
MEDS_PAIRS = (PROJECT_ROOT / "outputs" / "oxf_external_stn_retrieval_validation"
              / "oxf_onoff_pairs.csv")
MAX_DURATION_SEC = 120.0
EPS = 1e-30
SEED = 42
N_BOOT = 5000
N_PERM = 5000

# ---- feature definitions (verbatim from the locked OXF STN extractor) -------
SPECTRAL_FEATURES = [
    "stn_alpha_log_power", "stn_low_beta_log_power", "stn_high_beta_log_power",
    "stn_broad_beta_log_power", "stn_gamma_log_power",
    "stn_low_beta_high_beta_log_ratio", "stn_beta_gamma_log_ratio",
]
COMPACT_FEATURES = [
    "aperiodic_offset", "aperiodic_slope", "low_beta_residual_power",
    "high_beta_residual_power", "broad_beta_residual_power",
    "gamma_residual_power", "beta_peak_frequency", "beta_peak_amplitude",
    "gamma_peak_amplitude",
]
ALL_FEATURES = [*SPECTRAL_FEATURES, *COMPACT_FEATURES]

# Principled subspaces. The periodic (oscillation-specific) features are
# residuals over the fitted aperiodic 1/f and are bounded / amplitude-robust;
# the aperiodic+absolute-power features are sensitive to recording amplitude
# and to DBS broadband effects (and to the EPS floor on dead channels).
PERIODIC_FEATURES = [
    "low_beta_residual_power", "high_beta_residual_power",
    "broad_beta_residual_power", "gamma_residual_power",
    "beta_peak_frequency", "beta_peak_amplitude", "gamma_peak_amplitude",
]
# Stim-robust periodic core: drop gamma-touching features (65 Hz subharmonic /
# >55 Hz) to be conservative about the 130 Hz stimulation artefact.
BETA_PERIODIC_FEATURES = [
    "low_beta_residual_power", "high_beta_residual_power",
    "broad_beta_residual_power", "beta_peak_frequency", "beta_peak_amplitude",
]
APERIODIC_BROADBAND_FEATURES = [
    "aperiodic_offset", "aperiodic_slope",
    "stn_alpha_log_power", "stn_low_beta_log_power", "stn_high_beta_log_power",
    "stn_broad_beta_log_power", "stn_gamma_log_power",
    "stn_low_beta_high_beta_log_ratio", "stn_beta_gamma_log_ratio",
]

# Amplitude/gain de-confound. A global recording-gain change multiplies the PSD
# by a constant -> adds a constant to every log-power. That constant CANCELS in
# band ratios, the aperiodic slope, aperiodic residuals and peak features
# (gain-invariant), but SHIFTS absolute log-powers and the aperiodic offset
# (gain-sensitive). Convergence measured on the gain-invariant set therefore
# cannot be explained by a meds-vs-DBS recording-amplitude difference.
RATIO_FEATURES = ["stn_low_beta_high_beta_log_ratio", "stn_beta_gamma_log_ratio"]
GAIN_INVARIANT_FEATURES = [*PERIODIC_FEATURES, *RATIO_FEATURES, "aperiodic_slope"]
GAIN_SENSITIVE_FEATURES = [
    "aperiodic_offset", "stn_alpha_log_power", "stn_low_beta_log_power",
    "stn_high_beta_log_power", "stn_broad_beta_log_power", "stn_gamma_log_power",
]
SUBSPACES = {
    "gain_invariant": GAIN_INVARIANT_FEATURES,            # primary de-confounded
    "periodic": PERIODIC_FEATURES,
    "beta_periodic_stim_robust": BETA_PERIODIC_FEATURES,
    "gain_sensitive_amplitude": GAIN_SENSITIVE_FEATURES,  # confounded contrast
    "aperiodic_broadband": APERIODIC_BROADBAND_FEATURES,
    "all_features": ALL_FEATURES,
}
# Dead-channel detection: a hemisphere is invalid if any absolute STN log-power
# is at the EPS floor (< -8) in either state. (Peak-frequency shifts of many Hz
# are physiological and must NOT be treated as floor artefacts.)
POWER_FEATURES = ["stn_alpha_log_power", "stn_low_beta_log_power",
                  "stn_high_beta_log_power", "stn_broad_beta_log_power",
                  "stn_gamma_log_power"]
FLOOR_DB = -8.0


def clean_signal(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return arr
    arr = arr - float(np.nanmean(arr))
    if len(arr) > 100:
        low, high = np.nanpercentile(arr, [0.1, 99.9])
        if np.isfinite(low) and np.isfinite(high) and high > low:
            arr = np.clip(arr, low, high)
    return arr


def compute_psd(signal: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(sfreq) or sfreq <= 0 or len(signal) < 128:
        return np.asarray([]), np.asarray([])
    nperseg = min(4096, max(256, len(signal) // 8), len(signal))
    freqs, psd = scipy_signal.welch(signal, fs=sfreq, nperseg=nperseg, detrend="constant")
    return freqs, psd


def band_power(freqs, psd, band):
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if mask.sum() < 2:
        return np.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))


def gamma_band_for_sfreq(sfreq):
    if not np.isfinite(sfreq) or sfreq <= 0:
        return None
    nyq = sfreq / 2.0
    if nyq > 95:
        return (60.0, 90.0)
    if nyq > 45:
        return (35.0, min(55.0, nyq * 0.9))
    return None


def channel_feature_row(signal, sfreq):
    """Spectral + compact aperiodic features (verbatim from OXF extractor)."""
    freqs, psd = compute_psd(signal, sfreq)
    if len(freqs) == 0:
        return {f: np.nan for f in ALL_FEATURES}
    bands = {"alpha": (8.0, 12.0), "low_beta": (13.0, 20.0),
             "high_beta": (20.0, 35.0), "broad_beta": (13.0, 35.0)}
    gamma = gamma_band_for_sfreq(sfreq)
    if gamma is not None:
        bands["gamma"] = gamma
    powers = {name: band_power(freqs, psd, band) for name, band in bands.items()}
    row = {
        "stn_alpha_log_power": float(np.log10(max(powers.get("alpha", np.nan), EPS))),
        "stn_low_beta_log_power": float(np.log10(max(powers.get("low_beta", np.nan), EPS))),
        "stn_high_beta_log_power": float(np.log10(max(powers.get("high_beta", np.nan), EPS))),
        "stn_broad_beta_log_power": float(np.log10(max(powers.get("broad_beta", np.nan), EPS))),
        "stn_gamma_log_power": float(np.log10(max(powers.get("gamma", np.nan), EPS))),
    }
    row["stn_low_beta_high_beta_log_ratio"] = row["stn_low_beta_log_power"] - row["stn_high_beta_log_power"]
    row["stn_beta_gamma_log_ratio"] = row["stn_broad_beta_log_power"] - row["stn_gamma_log_power"]
    max_fit_freq = min(95.0, float(np.nanmax(freqs)))
    fit_mask = (freqs >= 5.0) & (freqs <= max_fit_freq) & (psd > 0)
    fit_mask &= ~((freqs >= 45.0) & (freqs <= 55.0))
    if fit_mask.sum() < 8:
        for f in COMPACT_FEATURES:
            row[f] = np.nan
        return row
    x = np.log10(freqs[fit_mask])
    y = np.log10(psd[fit_mask])
    slope, offset = np.polyfit(x, y, 1)
    predicted = offset + slope * np.log10(np.maximum(freqs, 1e-6))
    residual = np.log10(np.maximum(psd, EPS)) - predicted
    row["aperiodic_offset"] = float(offset)
    row["aperiodic_slope"] = float(slope)
    for name, band in bands.items():
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if name in {"low_beta", "high_beta", "broad_beta", "gamma"}:
            row[f"{name}_residual_power"] = float(np.nanmean(residual[mask])) if mask.sum() else np.nan
    beta_mask = (freqs >= 13.0) & (freqs <= 35.0)
    if beta_mask.sum():
        bf, br = freqs[beta_mask], residual[beta_mask]
        idx = int(np.nanargmax(br))
        row["beta_peak_frequency"] = float(bf[idx])
        row["beta_peak_amplitude"] = float(br[idx])
    else:
        row["beta_peak_frequency"] = np.nan
        row["beta_peak_amplitude"] = np.nan
    if gamma is not None:
        gm = (freqs >= gamma[0]) & (freqs <= gamma[1])
        row["gamma_peak_amplitude"] = float(np.nanmax(residual[gm])) if gm.sum() else np.nan
    else:
        row["gamma_peak_amplitude"] = np.nan
    return row


def stim_leakage_ratio(signal, sfreq, lo, hi, peak_lo, peak_hi):
    """Peak-to-neighbour power ratio in [peak_lo, peak_hi] vs band [lo, hi]."""
    freqs, psd = compute_psd(signal, sfreq)
    if len(freqs) == 0:
        return np.nan
    peak = (freqs >= peak_lo) & (freqs <= peak_hi)
    side = (freqs >= lo) & (freqs <= hi) & ~peak
    if peak.sum() < 1 or side.sum() < 2:
        return np.nan
    return float(psd[peak].max() / (np.median(psd[side]) + EPS))


def cosine(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def main():
    rng = np.random.default_rng(SEED)

    # ---- 1. Stimulation transition features from MATRIX_DBS -----------------
    mat = sio.loadmat(MATRIX_DBS, squeeze_me=True, struct_as_record=False)["MATRIX_DBS"]
    fs_all = np.asarray(mat.fs, dtype=float)
    base_sig, dbs_sig = mat.signal_base, mat.signal_dbs
    n_dbs = len(base_sig)

    dbs_rows = []
    leak_rows = []
    for i in range(n_dbs):
        fs = float(fs_all[i])
        b = clean_signal(np.asarray(base_sig[i], float)[: int(MAX_DURATION_SEC * fs)])
        d = clean_signal(np.asarray(dbs_sig[i], float)[: int(MAX_DURATION_SEC * fs)])
        fb = channel_feature_row(b, fs)
        fd = channel_feature_row(d, fs)
        dbs_rows.append({"hemi": i, "fs": fs,
                         **{f"base_{k}": v for k, v in fb.items()},
                         **{f"dbs_{k}": v for k, v in fd.items()}})
        leak_rows.append({
            "hemi": i, "fs": fs,
            "ratio_130Hz": stim_leakage_ratio(d, fs, 110, 150, 128, 132),
            "ratio_65Hz_subharm": stim_leakage_ratio(d, fs, 55, 75, 63.5, 66.5),
        })
    dbs_df = pd.DataFrame(dbs_rows)
    leak_df = pd.DataFrame(leak_rows)
    leak_df.to_csv(OUTPUT_DIR / "dbs_stim_artifact_diagnostic.csv", index=False)

    # ---- empirical gain-invariance check: scale one signal x3, compare -----
    fs0 = float(fs_all[0])
    s0 = clean_signal(np.asarray(base_sig[0], float)[: int(MAX_DURATION_SEC * fs0)])
    f_orig = channel_feature_row(s0, fs0)
    f_scaled = channel_feature_row(s0 * 3.0, fs0)
    gain_check = {f: float(f_scaled[f] - f_orig[f]) for f in ALL_FEATURES}
    # expected shift for gain-sensitive features = 2*log10(3) ≈ 0.954
    gain_check_summary = {
        "scale_factor": 3.0,
        "expected_gain_sensitive_shift": float(2 * np.log10(3.0)),
        "max_abs_change_gain_invariant": float(max(
            abs(gain_check[f]) for f in GAIN_INVARIANT_FEATURES)),
        "median_abs_change_gain_sensitive": float(np.median(
            [abs(gain_check[f]) for f in GAIN_SENSITIVE_FEATURES])),
        "per_feature_change": gain_check,
    }

    delta_dbs = np.column_stack([
        dbs_df[f"dbs_{f}"].to_numpy() - dbs_df[f"base_{f}"].to_numpy()
        for f in ALL_FEATURES
    ])  # (26, F)

    # ---- 2. Medication transition features from cached OXF pairs ------------
    pairs = pd.read_csv(MEDS_PAIRS)
    delta_med = np.column_stack([
        pairs[f"medon_{f}"].to_numpy() - pairs[f"x_{f}"].to_numpy()
        for f in ALL_FEATURES
    ])  # (30, F)
    med_subject = pairs["subject"].astype(str).to_numpy()

    # ---- 3. Dead-channel validity masks (hemisphere-level, both states) -----
    med_off_pow = pairs[[f"x_{f}" for f in POWER_FEATURES]].to_numpy()
    med_on_pow = pairs[[f"medon_{f}" for f in POWER_FEATURES]].to_numpy()
    med_valid = (np.all(med_off_pow > FLOOR_DB, axis=1)
                 & np.all(med_on_pow > FLOOR_DB, axis=1))
    dbs_base_pow = dbs_df[[f"base_{f}" for f in POWER_FEATURES]].to_numpy()
    dbs_dbs_pow = dbs_df[[f"dbs_{f}" for f in POWER_FEATURES]].to_numpy()
    dbs_valid = (np.all(dbs_base_pow > FLOOR_DB, axis=1)
                 & np.all(dbs_dbs_pow > FLOOR_DB, axis=1))

    # ---- 4. Per-subspace evaluation ----------------------------------------
    def evaluate_subspace(feats):
        idx = np.array([ALL_FEATURES.index(f) for f in feats])
        dm_all, dd_all = delta_med[:, idx], delta_dbs[:, idx]
        m_ok = med_valid & np.all(np.isfinite(dm_all), axis=1)
        d_ok = dbs_valid & np.all(np.isfinite(dd_all), axis=1)
        dm, dd = dm_all[m_ok], dd_all[d_ok]
        subj = med_subject[m_ok]
        pooled = np.vstack([dm, dd])
        mad = np.median(np.abs(pooled - np.median(pooled, axis=0)), axis=0)
        scale = 1.4826 * mad
        scale[scale == 0] = 1.0

        def gcos(dm_, dd_):
            mu_m = np.mean(dm_ / scale, axis=0)
            mu_d = np.mean(dd_ / scale, axis=0)
            return cosine(mu_m, mu_d), mu_m, mu_d

        obs_cos, mu_m, mu_d = gcos(dm, dd)
        subjects = np.unique(subj)
        boot = []
        for _ in range(N_BOOT):
            drawn = rng.choice(subjects, size=len(subjects), replace=True)
            rows = np.concatenate([np.where(subj == s)[0] for s in drawn])
            dd_b = dd[rng.integers(0, len(dd), len(dd))]
            c, _, _ = gcos(dm[rows], dd_b)
            boot.append(c)
        ci = np.nanpercentile(boot, [2.5, 97.5])
        nullA = []
        for _ in range(N_PERM):
            c, _, _ = gcos(dm * rng.choice([-1.0, 1.0], len(dm))[:, None],
                           dd * rng.choice([-1.0, 1.0], len(dd))[:, None])
            nullA.append(c)
        nullA = np.asarray(nullA)
        pA = (np.sum(nullA >= obs_cos) + 1) / (N_PERM + 1)
        nullB = np.asarray([cosine(mu_m, rng.permutation(mu_d)) for _ in range(N_PERM)])
        pB = (np.sum(nullB >= obs_cos) + 1) / (N_PERM + 1)
        return {
            "features": feats, "n_features": len(feats),
            "n_med_used": int(m_ok.sum()), "n_dbs_used": int(d_ok.sum()),
            "n_med_subjects": int(len(subjects)),
            "observed_cosine": obs_cos,
            "ci_lower_95": float(ci[0]), "ci_upper_95": float(ci[1]),
            "null_signflip_mean": float(np.nanmean(nullA)), "p_signflip": float(pA),
            "null_featshuffle_mean": float(np.nanmean(nullB)), "p_featshuffle": float(pB),
        }

    results = {name: evaluate_subspace(feats) for name, feats in SUBSPACES.items()}

    # ---- 5. Per-feature direction-agreement table (valid hemispheres) -------
    dm_v, dd_v = delta_med[med_valid], delta_dbs[dbs_valid]
    pooled_all = np.vstack([dm_v, dd_v])
    mad_all = np.nanmedian(np.abs(pooled_all - np.nanmedian(pooled_all, axis=0)), axis=0)
    scale_all = 1.4826 * mad_all
    scale_all[scale_all == 0] = 1.0
    mu_m_all = np.nanmean(dm_v / scale_all, axis=0)
    mu_d_all = np.nanmean(dd_v / scale_all, axis=0)
    feat_table = pd.DataFrame({
        "feature": ALL_FEATURES,
        "mean_delta_med_z": mu_m_all,
        "mean_delta_dbs_z": mu_d_all,
        "same_direction": np.sign(mu_m_all) == np.sign(mu_d_all),
        "subspace": ["periodic" if f in PERIODIC_FEATURES else "aperiodic_broadband"
                     for f in ALL_FEATURES],
        "gain_class": ["gain_sensitive" if f in GAIN_SENSITIVE_FEATURES
                       else "gain_invariant" for f in ALL_FEATURES],
    })
    feat_table.to_csv(OUTPUT_DIR / "per_feature_direction_agreement.csv", index=False)

    summary = {
        "n_dbs_hemispheres_total": int(n_dbs),
        "n_med_hemispheres_total": int(delta_med.shape[0]),
        "stim_artifact_130Hz_median_ratio": float(np.nanmedian(leak_df["ratio_130Hz"])),
        "stim_artifact_65Hz_median_ratio": float(np.nanmedian(leak_df["ratio_65Hz_subharm"])),
        "gain_invariance_check": gain_check_summary,
        "n_boot": N_BOOT, "n_perm": N_PERM, "seed": SEED,
        "results": results,
    }
    (OUTPUT_DIR / "convergence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(OUTPUT_DIR / "transition_vectors.npz",
             delta_med=delta_med, delta_dbs=delta_dbs,
             features=np.array(ALL_FEATURES), scale=scale_all)

    # ---- console report ----
    print("=" * 70)
    print("DBS vs LEVODOPA STN TRANSITION-GEOMETRY CONVERGENCE")
    print("=" * 70)
    print(f"DBS hemispheres: {n_dbs} total | Meds: {delta_med.shape[0]} hemispheres")
    print(f"Stim artifact: 130 Hz peak/neighbour median = "
          f"{summary['stim_artifact_130Hz_median_ratio']:.0f}x ; "
          f"65 Hz subharmonic = {summary['stim_artifact_65Hz_median_ratio']:.1f}x")
    gc = gain_check_summary
    print(f"Gain-invariance check (signal x3): gain-invariant max |Δ| = "
          f"{gc['max_abs_change_gain_invariant']:.4f} (expect ~0); "
          f"gain-sensitive median |Δ| = {gc['median_abs_change_gain_sensitive']:.3f} "
          f"(expect ~{gc['expected_gain_sensitive_shift']:.3f})")
    for name, r in results.items():
        print(f"\n[{name}] ({r['n_features']} feat | meds {r['n_med_used']} hem/"
              f"{r['n_med_subjects']} pts, DBS {r['n_dbs_used']} hem)")
        print(f"  cosine(Δ_med, Δ_DBS) = {r['observed_cosine']:+.3f}  "
              f"95% CI [{r['ci_lower_95']:+.3f}, {r['ci_upper_95']:+.3f}]")
        print(f"  sign-flip null mean  = {r['null_signflip_mean']:+.3f}  p = {r['p_signflip']:.4f}")
        print(f"  feat-shuffle null    = {r['null_featshuffle_mean']:+.3f}  p = {r['p_featshuffle']:.4f}")
    print("\nPer-feature direction agreement (med vs DBS):")
    for _, row in feat_table.iterrows():
        flag = "same" if row["same_direction"] else "OPP "
        print(f"  {flag}  {row['feature']:<34} med {row['mean_delta_med_z']:+.3f}  "
              f"dbs {row['mean_delta_dbs_z']:+.3f}")
    print(f"\nSaved to {OUTPUT_DIR}")
    return summary


if __name__ == "__main__":
    main()
