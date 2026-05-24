"""
Figures for the paired-state retrieval / MAGE-G manuscript.

Produces draft versions of Figures 2-6 (data-driven figures).
Figure 1 (conceptual diagram) is best made in Inkscape/Figma.

Each figure saved as both PDF (vector, for submission) and PNG (for previews).

Design system (2026 redesign)
------------------------------
* Typography: Times New Roman (manuscript requirement).
* Colour: Paul Tol "vibrant" qualitative palette - colourblind-safe and
  print-safe (https://personal.sron.nl/~pault/), warmer/cleaner than the
  previous Okabe-Ito mapping.
* Chart idioms chosen per message rather than defaulting to bars/histograms:
  - permutation nulls -> filled kernel-density curves (not spiky histograms);
  - paired before/after -> dumbbell plots (not diagonal arrows);
  - "effect reduced to zero" -> lollipops sitting on the zero line;
  - ordered method ladders -> stepped dot-with-CI;
  - effect-direction summaries -> horizontal forest / lollipop of rho.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# =========================================================================
# Paths
# =========================================================================

# Self-contained layout: every figure-input table lives in ./data, figures are
# written to ./output. (In the full research repo these came from manuscript/ and
# outputs/; here they are collected into one folder so the figures reproduce alone.)
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "output"
PLOT_DATA_DIR = DATA_DIR
TABLE_DIR = DATA_DIR
V2A_AUDIT_DIR = DATA_DIR
V2A_BOOTSTRAP_SAMPLES = DATA_DIR / "v2A_bootstrap_samples.csv"
SYMPTOM_AXIS_DIR = DATA_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================================
# Typography
# =========================================================================

# Times New Roman is proprietary and not redistributed here. If the .ttf/.ttc
# files are available locally, drop them in figures/fonts/ (or point TIMES_FONT_DIR
# at a folder that contains them); otherwise the script falls back to the default
# serif font and the figures still render.
FONT_DIRS = [HERE / "fonts"]
_env_font_dir = os.environ.get("TIMES_FONT_DIR")
if _env_font_dir:
    FONT_DIRS.append(Path(_env_font_dir))
for font_dir in FONT_DIRS:
    if font_dir.is_dir():
        for font_path in sorted(font_dir.glob("*.tt[fc]")):
            font_manager.fontManager.addfont(str(font_path))

INK = "#23262B"        # near-black for text
SPINE = "#5C636B"      # soft grey for spines / ticks
GRID = "#E8EAEE"       # very light grid

mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman", "Times"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.titlepad": 7.0,
    "axes.titlecolor": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": SPINE,
    "text.color": INK,
    "xtick.color": SPINE,
    "ytick.color": SPINE,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    # mathtext (italic p-values etc.) must render in Times too - the default
    # mathtext fontset is DejaVu, which would silently mix fonts into the figures
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "pdf.fonttype": 42,  # TrueType embedding, JNE requirement
    "ps.fonttype": 42,
})

PANEL_TITLE_SIZE = 9.0


def panel_title(ax, text, size=PANEL_TITLE_SIZE, color=INK):
    """Left-aligned bold panel title at a consistent size."""
    ax.set_title(text, loc="left", fontweight="bold", fontsize=size, color=color)


# =========================================================================
# Colour palette - Paul Tol "vibrant" (+ neutrals)
# =========================================================================

COLOURS = {
    "baseline": "#9499A0",        # neutral grey for baselines
    "baseline_dark": "#5A5F66",
    "v2A": "#0077BB",             # vibrant blue, primary frozen method
    "v2B": "#33BBEE",             # vibrant cyan
    "v2C": "#009988",             # teal
    "v2D": "#EE7733",             # orange, strongest internal
    "mage_g": "#EE3377",          # magenta, geometry harmonization (centerpiece)
    "null": "#D9DDE3",            # cool light grey for null mass
    "null_edge": "#AEB4BD",
    "observed": "#23262B",        # near-black for observed values
    "negative": "#CC3311",        # vibrant red for negative results
    "positive": "#009988",        # teal for confirmations
    "connector": "#C7CCD2",       # dumbbell / step connectors
}

# Sequential ramp for the ordered OXF geometry ladder (cool -> accent).
LADDER_RAMP = ["#BFC6CE", "#7FB0D6", "#2E78B6", "#EE3377"]


def mm_to_in(mm):
    return mm / 25.4


# JNE-style two-column width is ~174mm; single column ~84mm
FULL_WIDTH = mm_to_in(174)
SINGLE_COL = mm_to_in(84)


def save_figure(fig, name):
    """Save both PDF (vector for submission) and PNG (for preview)."""
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    png_path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"  saved {name}.pdf and {name}.png")


def filled_density(ax, samples, colour, label=None, x_grid=None,
                   alpha=0.32, lw=1.6, zorder=2, bw=None):
    """Filled kernel-density curve - a cleaner read than a spiky histogram
    for permutation-null distributions. Returns the peak density (for y-scaling)."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[~np.isnan(samples)]
    if x_grid is None:
        x_grid = np.linspace(0, 1, 512)
    if np.allclose(samples.std(), 0.0):
        # Degenerate: draw a thin spike at the single value.
        x0 = float(samples[0]) if samples.size else 0.0
        ax.axvline(x0, color=colour, lw=lw, alpha=0.9, zorder=zorder, label=label)
        return 1.0
    kde = gaussian_kde(samples, bw_method=bw)
    dens = kde(x_grid)
    ax.fill_between(x_grid, dens, color=colour, alpha=alpha, lw=0, zorder=zorder)
    ax.plot(x_grid, dens, color=colour, lw=lw, label=label, zorder=zorder + 1,
            solid_capstyle="round")
    return float(dens.max())


def style_grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# =========================================================================
# Figure 1 -- study overview / three validity-boundary framework (schematic)
# =========================================================================

def make_figure1():
    """Graphical-abstract overview: a process flow from the two cohorts and the
    paired-state assay, through the three validity audits, to the bottom line."""
    print("Figure 1: study overview / validity-boundary framework")
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

    W, H = 178.0, 178.0
    fig = plt.figure(figsize=(mm_to_in(W), mm_to_in(H)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    def tint(hexc, a):
        c = np.array(mcolors.to_rgb(hexc))
        return tuple(c * a + np.ones(3) * (1 - a))

    def card(x, y, w, h, fc, ec, lw=1.2, r=2.4):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fc, edgecolor=ec, linewidth=lw, mutation_aspect=1.0, zorder=2))

    def txt(x, y, s, size=6.5, color=INK, weight="normal", ha="left",
            va="center", style="normal"):
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                ha=ha, va=va, style=style, zorder=4, linespacing=1.28)

    def seclabel(x, y, s, color):
        txt(x, y, s.upper(), size=9.2, color=color, weight="bold")

    def darrow(x1, y1, x2, y2, color, lw=1.1, ms=10):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
            mutation_scale=ms, color=color, lw=lw, zorder=3))

    B1, B2, B3 = COLOURS["v2A"], COLOURS["mage_g"], COLOURS["v2D"]
    G, POS, GREY = COLOURS["v2C"], COLOURS["positive"], COLOURS["baseline_dark"]

    # mini PSD sketches: MedOff has a beta peak, MedOn is suppressed (same scale)
    fx = np.linspace(0, 1, 200)

    def _psd(bump):
        return 0.5 * np.exp(-2.7 * fx) + 0.06 + bump * np.exp(-((fx - 0.42) / 0.07) ** 2)

    denom = _psd(0.95).max()

    def sketch(bx, by, bw, bh, bump, color, label):
        yv = _psd(bump) / denom
        xs, ys = bx + fx * bw, by + yv * bh
        ax.fill_between(xs, by, ys, color=color, alpha=0.12, zorder=3)
        ax.plot(xs, ys, color=color, lw=1.9, zorder=4, solid_capstyle="round")
        txt(bx + bw / 2, by - 4.5, label, 8.8, color, weight="bold", ha="center")

    NEU = COLOURS["baseline"]

    # ============ INPUTS: data + paired-state assay in one container ============
    # A dedicated accent (NOT a dataset hue): OFF and ON are two states of the SAME
    # subject in the SAME cohort - the assay itself, not a second dataset.
    ASSAY = "#7B4FBF"
    card(6, 125, 166, 46, "white", ASSAY, 1.4, r=2.8)
    ax.plot([74, 74], [129, 161], color=tint(NEU, 0.55), lw=1.0, zorder=3,
            solid_capstyle="butt")
    seclabel(10, 165, "Data", INK)
    seclabel(80, 165, "Paired-state assay", INK)

    # -- left: the two cohorts --
    ax.add_patch(Rectangle((10, 154.8), 3.4, 3.4, facecolor=ASSAY, edgecolor="none", zorder=4))
    txt(16.0, 156.1, "ds004998", 9.8, ASSAY, "bold")
    txt(10, 150.5, "MEG + STN-LFP · MedOFF/MedON · UPDRS", 8.3)
    ax.plot([10, 72], [144.5, 144.5], color=tint(NEU, 0.55), lw=1.0, zorder=3,
            solid_capstyle="butt")
    ax.add_patch(Rectangle((10, 135.8), 3.4, 3.4, facecolor=ASSAY, edgecolor="none", zorder=4))
    txt(16.0, 137.1, "OXF / Wiest", 9.8, ASSAY, "bold")
    txt(10, 131.5, "STN-LFP · paired Off/On · 130 Hz DBS", 8.3)

    # -- right: a MedOFF query is ranked against MedON candidates --
    sketch(80, 145, 30, 15, 0.95, ASSAY, "Med OFF")
    sketch(128, 145, 30, 15, 0.18, ASSAY, "Med ON")
    darrow(112, 148, 126, 148, ASSAY, lw=1.6, ms=14)
    txt(118, 151.5, "retrieve", 8.6, ASSAY, style="italic", ha="center")
    txt(119, 132, "MedOFF query → MedON ranking",
        8.6, INK, "bold", ha="center")
    for xa in (32, 89, 146):
        darrow(xa, 124.5, xa, 119.0, COLOURS["baseline"], lw=1.1, ms=10)

    # ---- the question the three audits answer (paired with the conclusion below) ----
    card(6, 108.0, 166, 11.0, tint(POS, 0.06), POS, 1.2, r=2.4)
    txt(W / 2, 113.5, "High accuracy is necessary — but is it sufficient?",
        10.6, INK, "bold", ha="center")
    for xa in (32, 89, 146):
        darrow(xa, 108.0, xa, 102.0, COLOURS["baseline"], lw=1.1, ms=10)

    # ============ THREE VALIDITY AUDITS ============
    cw, ctop, cbot = 52.0, 102.0, 44.0
    xs0 = [6, 63, 120]

    def icon_identity(cx, cy, col):
        ax.add_patch(Circle((cx - 2.6, cy), 4.6, facecolor=tint(col, 0.22), edgecolor=col, lw=1.4, zorder=4))
        ax.add_patch(Circle((cx + 2.6, cy), 4.6, facecolor="none", edgecolor=col, lw=1.4, zorder=4))

    def icon_geometry(cx, cy, col):
        # two leads, four contacts each; a DIFFERENT bipolar pair is selected on
        # each lead (lower pair vs upper pair) -> the montage differs by recording
        sel = {-7.5: (2, 3), 7.5: (0, 1)}
        for dx in (-7.5, 7.5):
            card(cx + dx - 0.95, cy - 6.5, 1.9, 13, tint(col, 0.28), tint(col, 0.5), lw=0, r=0.6)
            for i in range(4):
                on = i in sel[dx]
                ax.add_patch(Circle((cx + dx, cy + 4.5 - i * 3.0), 1.4,
                    facecolor=col if on else "white", edgecolor=col, lw=1.2, zorder=5))

    def icon_severity(cx, cy, col):
        n = 24
        hw = 13.0
        seg = 2 * hw / n
        for i in range(n):
            ax.add_patch(Rectangle((cx - hw + i * seg, cy - 2.2), seg + 0.3, 4.4,
                facecolor=tint(col, 0.15 + 0.8 * i / n), edgecolor="none", zorder=4))
        darrow(cx - hw, cy - 6.0, cx + hw, cy - 6.0, col, lw=1.0, ms=9)

    audits = [
        (B1, icon_identity, "Subject identity", "state, or the patient?",
         "subject fingerprinting\ninflates accuracy"),
        (B2, icon_geometry, "Recording geometry", "physiology, or contact montage?",
         "transfer is\ngeometry-gated"),
        (B3, icon_severity, "Baseline severity", "treatment response, or\nbaseline severity?",
         "state accuracy does not\npredict response"),
    ]

    # one construction for both lines: a labelled chip (header + body), same style
    def chip(x0, ytop, header, body, col):
        card(x0 + 3, ytop - 14, cw - 6, 14, tint(col, 0.10), tint(col, 0.45), 0.8, r=1.5)
        txt(x0 + cw / 2, ytop - 3.2, header, 8.0, col, "bold", ha="center")
        txt(x0 + cw / 2, ytop - 8.8, body, 8.6, INK, ha="center")

    for x0, (col, icon, name, q, res) in zip(xs0, audits):
        card(x0, cbot, cw, ctop - cbot, "white", col, 1.4)
        icon(x0 + cw / 2, ctop - 12, col)
        txt(x0 + cw / 2, ctop - 23, name, 10.0, col, "bold", ha="center")
        chip(x0, 75, "QUESTION", q, col)
        chip(x0, 59, "FINDING", res, col)
        darrow(x0 + cw / 2, cbot - 0.5, x0 + cw / 2, 38.0, COLOURS["baseline"], lw=1.0, ms=9)

    # ============ CONCLUSION: the answer, then its three distinct forms ============
    card(6, 27.0, 166, 11.0, tint(POS, 0.12), POS, 1.5)
    txt(W / 2, 32.5, "No — high paired-state accuracy is not sufficient: validity has three distinct forms",
        10.4, INK, "bold", ha="center")
    for xa in (32, 89, 146):
        darrow(xa, 27.0, xa, 21.0, COLOURS["baseline"], lw=1.1, ms=10)

    # the three forms - one per audit above, matched by colour and column
    forms = [(B1, "Audited identifiability"), (B2, "External transfer"), (B3, "Clinical prediction")]
    for x0, (col, name) in zip(xs0, forms):
        card(x0, 6.0, cw, 15.0, tint(col, 0.10), col, 1.2, r=2.0)
        txt(x0 + cw / 2, 13.5, name, 10.0, col, "bold", ha="center")

    save_figure(fig, "figure1_overview")
    plt.close(fig)


# =========================================================================
# Figure 2 -- Frozen retrieval performance vs null distributions
# =========================================================================

def make_figure2():
    """
    Panel A: filled-density nulls vs observed v2A (top-1).
    Panel B: dumbbell of v2 -> v2A for Top-1 and MRR, with subject-bootstrap CIs.
    """
    print("Figure 2: frozen retrieval vs nulls")
    import pandas as pd

    fig, axes = plt.subplots(
        1, 2,
        figsize=(FULL_WIDTH, mm_to_in(82)),
        gridspec_kw={"width_ratios": [1.1, 1.0]},
    )

    # ---- Panel A: observed vs nulls (filled densities) ----
    ax = axes[0]
    null_csv_v2a = PLOT_DATA_DIR / "figure2_v2a_null_samples_compact.csv"
    if not null_csv_v2a.exists():
        raise FileNotFoundError(null_csv_v2a)
    df = pd.read_csv(null_csv_v2a)
    random_null = df["random_label_top1"].values
    matched_null = df["matched_task_side_top1"].values
    observed_csv = V2A_AUDIT_DIR / "observed_v2_v2A_metrics.csv"
    observed_top1 = 0.8484848484848485
    if observed_csv.exists():
        observed = pd.read_csv(observed_csv).set_index("variant_name")
        observed_top1 = float(observed.loc["v2A_top5_aperiodic_rerank", "top1"])

    x_grid = np.linspace(0, 1, 512)
    # Both nulls pile up near low top-1, so use light fills (visible through each
    # other) and a darker grey for the random-label null so its outline reads
    # against the bright cyan matched null.
    p1 = filled_density(ax, random_null, COLOURS["baseline_dark"], x_grid=x_grid,
                        label="Random-label null (5000×)", alpha=0.16, lw=1.9)
    p2 = filled_density(ax, matched_null, COLOURS["v2B"], x_grid=x_grid,
                        label="Matched task/side null (5000×)", alpha=0.16, lw=1.9)
    peak = max(p1, p2)
    ax.set_ylim(0, peak * 1.30)

    ax.axvline(observed_top1, color=COLOURS["v2A"], linewidth=2.4, zorder=5)
    ax.scatter([observed_top1], [peak * 1.30], marker="v", s=46,
               color=COLOURS["v2A"], zorder=6, clip_on=False)
    ax.text(observed_top1 - 0.03, peak * 0.62,
            f"v2A observed = {observed_top1:.3f}\np ≤ 0.0002 vs both nulls",
            ha="right", va="center", fontsize=7.8, color=COLOURS["v2A"],
            fontweight="bold")

    ax.set_xlabel("Top-1 accuracy")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1.0)
    panel_title(ax, "A. v2A vs null distributions (n = 33)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.99),
              frameon=False, fontsize=7.6, handlelength=1.4)

    # ---- Panel B: dumbbell v2 -> v2A (Top-1 and MRR) ----
    ax = axes[1]
    top1 = [0.727, 0.848]
    ci_low = [0.530, 0.667]
    ci_high = [0.882, 1.000]
    mrr = [0.826, 0.893]
    mrr_ci_low = [0.700, 0.759]
    mrr_ci_high = [0.930, 1.000]
    delta_top1, delta_top1_low, delta_top1_high = 0.121, 0.030, 0.235

    if observed_csv.exists() and V2A_BOOTSTRAP_SAMPLES.exists():
        obs = pd.read_csv(observed_csv).set_index("variant_name")
        boot = pd.read_csv(V2A_BOOTSTRAP_SAMPLES)
        top1 = [float(obs.loc["v2_reference", "top1"]),
                float(obs.loc["v2A_top5_aperiodic_rerank", "top1"])]
        mrr = [float(obs.loc["v2_reference", "mrr"]),
               float(obs.loc["v2A_top5_aperiodic_rerank", "mrr"])]
        ci_low = [float(np.percentile(boot["v2_top1"], 2.5)),
                  float(np.percentile(boot["v2A_top1"], 2.5))]
        ci_high = [float(np.percentile(boot["v2_top1"], 97.5)),
                   float(np.percentile(boot["v2A_top1"], 97.5))]
        mrr_ci_low = [float(np.percentile(boot["v2_mrr"], 2.5)),
                      float(np.percentile(boot["v2A_mrr"], 2.5))]
        mrr_ci_high = [float(np.percentile(boot["v2_mrr"], 97.5)),
                       float(np.percentile(boot["v2A_mrr"], 97.5))]
        delta_top1 = float(obs.loc["v2A_top5_aperiodic_rerank", "top1"]
                           - obs.loc["v2_reference", "top1"])
        delta_top1_low = float(np.percentile(boot["delta_top1_v2A_minus_v2"], 2.5))
        delta_top1_high = float(np.percentile(boot["delta_top1_v2A_minus_v2"], 97.5))

    rows = [
        ("Top-1", 1.0, top1, ci_low, ci_high),
        ("MRR", 0.0, mrr, mrr_ci_low, mrr_ci_high),
    ]
    for _, y, vals, los, his in rows:
        ax.plot([vals[0], vals[1]], [y, y], color=COLOURS["connector"],
                lw=3.0, solid_capstyle="round", zorder=1)
        ax.errorbar(vals[0], y, xerr=[[vals[0] - los[0]], [his[0] - vals[0]]],
                    fmt="o", color=COLOURS["baseline_dark"], markersize=8.5,
                    capsize=3, elinewidth=1.1, markeredgecolor="white",
                    markeredgewidth=0.9, zorder=3)
        ax.errorbar(vals[1], y, xerr=[[vals[1] - los[1]], [his[1] - vals[1]]],
                    fmt="o", color=COLOURS["v2A"], markersize=8.5,
                    capsize=3, elinewidth=1.1, markeredgecolor="white",
                    markeredgewidth=0.9, zorder=3)

    # Paired delta annotation (no diagonal arrow).
    ax.text((top1[0] + top1[1]) / 2, 1.34,
            f"paired Δ top-1 = +{delta_top1:.3f}  [{delta_top1_low:+.3f}, {delta_top1_high:+.3f}]",
            ha="center", va="center", fontsize=7.4, style="italic",
            color=COLOURS["observed"])

    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Top-1", "MRR"])
    ax.set_ylim(-0.7, 1.75)
    ax.set_xlim(0.45, 1.03)
    xt = np.arange(0.45, 1.001, 0.05)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{v:.2f}" for v in xt])
    ax.tick_params(axis="x", labelsize=6.4)
    ax.set_xlabel("Retrieval performance (95% subject-bootstrap CI)")
    style_grid(ax, axis="x")
    panel_title(ax, "B. Subject-aware bootstrap CIs (5000 iter)")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=COLOURS["baseline_dark"], markeredgecolor="white",
               markersize=8.5, label="v2 reference"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=COLOURS["v2A"], markeredgecolor="white",
               markersize=8.5, label="v2A frozen"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.8)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.87, bottom=0.17, wspace=0.24)
    save_figure(fig, "figure2_frozen_audit")
    plt.close(fig)


# =========================================================================
# Figure 3 -- Candidate-pool severity ladder + fingerprint gap
# =========================================================================

def make_figure3():
    """
    Panel A: top-1 across 4 candidate pools, 4 methods (x-dodged lines).
    Panel B: fingerprint gap as horizontal lollipops (zeros sit on the line).
    """
    print("Figure 3: severity ladder + fingerprint gap")
    import pandas as pd

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, mm_to_in(86)),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    pool_labels = ["Matched\ntask/side", "+same-subj\nwrong task/side",
                   "all-MedOn", "true + all\nother-subj MedOn"]
    pool_conditions = [
        "original_frozen_matched_pool",
        "original_plus_same_subject_wrong_task_side",
        "all_medon_candidates",
        "true_plus_all_other_subject_medon",
    ]
    variant_map = {
        "v2A": "v2A_top5_aperiodic_rerank",
        "v2B": "v2B_gated_aperiodic_top5",
        "v2C": "v2C_deconfounded_cycle",
        "v2D": "v2D_quality_deconfounded_cycle",
    }
    method_colours = {
        "v2A": COLOURS["v2A"], "v2B": COLOURS["v2B"],
        "v2C": COLOURS["v2C"], "v2D": COLOURS["v2D"],
    }
    method_markers = {"v2A": "o", "v2B": "s", "v2C": "^", "v2D": "D"}
    method_dodge = {"v2A": -0.12, "v2B": -0.04, "v2C": 0.04, "v2D": 0.12}

    metrics_path = DATA_DIR / "v2d_18subjects_metrics.csv"
    diagnostics_path = DATA_DIR / "v2d_18subjects_query_diagnostics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not diagnostics_path.exists():
        raise FileNotFoundError(diagnostics_path)
    metrics = pd.read_csv(metrics_path)
    diagnostics = pd.read_csv(diagnostics_path)

    def subject_clustered_top1_ci(rows, n_boot=5000):
        rng = np.random.default_rng(42)
        subjects = sorted(rows["query_subject"].astype(str).unique())
        per_subject = {}
        for subject in subjects:
            subject_rows = rows[rows["query_subject"].astype(str) == subject]
            per_subject[subject] = (
                int(len(subject_rows)),
                float(subject_rows["top_ranked_is_true_pair"].astype(float).sum()),
            )
        samples = []
        for _ in range(n_boot):
            draw = rng.choice(subjects, size=len(subjects), replace=True)
            n = 0
            successes = 0.0
            for subject in draw:
                item_n, item_successes = per_subject[str(subject)]
                n += item_n
                successes += item_successes
            samples.append(successes / n if n else np.nan)
        return np.nanpercentile(samples, [2.5, 97.5])

    methods_data = {}
    method_cis = {}
    fingerprint_gap = {}
    for method, variant in variant_map.items():
        values = []
        ci_values = []
        for condition in pool_conditions:
            row = metrics[
                (metrics["candidate_pool_condition"] == condition)
                & (metrics["variant_name"] == variant)
            ].iloc[0]
            values.append(float(row["top1"]))
            fingerprint_gap[(condition, method)] = float(row["subject_fingerprint_gap"])
            rows = diagnostics[
                (diagnostics["candidate_pool_condition"] == condition)
                & (diagnostics["variant_name"] == variant)
            ]
            ci_values.append(tuple(float(v) for v in subject_clustered_top1_ci(rows)))
        methods_data[method] = values
        method_cis[method] = ci_values

    # ---- Panel A: dodged lines ----
    ax = axes[0]
    x = np.arange(len(pool_labels))
    for method, values in methods_data.items():
        dx = method_dodge[method]
        vals = np.asarray(values)
        ci = method_cis[method]
        yerr = [vals - np.asarray([lo for lo, _ in ci]),
                np.asarray([hi for _, hi in ci]) - vals]
        ax.plot(x + dx, vals, color=method_colours[method], linewidth=1.8,
                zorder=2, alpha=0.9)
        ax.errorbar(x + dx, vals, yerr=yerr, fmt=method_markers[method],
                    color=method_colours[method], markersize=6.5,
                    markeredgecolor="white", markeredgewidth=0.8,
                    capsize=2.5, elinewidth=0.9, zorder=3, label=method)

    ax.axhline(0.05, color=COLOURS["baseline"], linestyle=(0, (1, 2)),
               linewidth=0.8, alpha=0.7, zorder=1)
    ax.text(3.32, 0.06, "random-label\nnull", fontsize=7.2, va="center",
            ha="left", color=COLOURS["baseline_dark"])
    ax.set_xticks(x)
    ax.set_xticklabels(pool_labels, fontsize=7.5)
    ax.set_xlim(-0.45, 3.95)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1.05)
    yt = np.arange(0.0, 1.001, 0.05)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:.2f}" for v in yt])
    ax.tick_params(axis="y", labelsize=6.4)
    style_grid(ax, axis="y")
    panel_title(ax, "A. Candidate-pool stress test")
    handles, labels = ax.get_legend_handles_labels()

    # ---- Panel B: fingerprint-gap lollipops ----
    ax = axes[1]
    fp_methods = ["v2A", "v2B", "v2C", "v2D"]
    y_pos = {m: i for i, m in enumerate(reversed(fp_methods))}  # v2A at top
    series = [
        ("+same-subj wrong task/side", "original_plus_same_subject_wrong_task_side",
         +0.16, COLOURS["v2A"]),
        ("all-MedOn", "all_medon_candidates", -0.16, COLOURS["v2D"]),
    ]

    # Highlight the "gap eliminated" zone (v2B/v2C/v2D).
    ax.axhspan(-0.5, 2.5, color=COLOURS["positive"], alpha=0.06, zorder=0)
    ax.text(0.265, 1.5, "fingerprint gap eliminated (= 0)\nin v2B–v2D",
            fontsize=7.1, color=COLOURS["positive"], va="center", ha="center",
            fontweight="bold")

    for label, condition, dy, colour in series:
        for m in fp_methods:
            gap = fingerprint_gap[(condition, m)]
            y = y_pos[m] + dy
            ax.plot([0, gap], [y, y], color=colour, linewidth=1.6,
                    alpha=0.55, zorder=2, solid_capstyle="round")
            ax.scatter([gap], [y], s=46, color=colour, edgecolor="white",
                       linewidth=0.8, zorder=3,
                       label=label if m == "v2A" else None)
            if np.isclose(gap, 0.0):
                ax.scatter([0], [y], s=46, facecolor="white",
                           edgecolor=colour, linewidth=1.2, zorder=4)

    # Value labels on the v2A lollipops (data-driven, so they track the CSV).
    _gap_hard = fingerprint_gap[("original_plus_same_subject_wrong_task_side", "v2A")]
    _gap_allmedon = fingerprint_gap[("all_medon_candidates", "v2A")]
    ax.text(_gap_hard + 0.012,
            y_pos["v2A"] + 0.16, f"{_gap_hard:.2f}", va="center", ha="left",
            fontsize=7.4, color=COLOURS["v2A"], fontweight="bold")
    ax.text(_gap_allmedon + 0.012,
            y_pos["v2A"] - 0.16, f"{_gap_allmedon:.2f}", va="center", ha="left",
            fontsize=7.4, color=COLOURS["v2D"], fontweight="bold")

    ax.axvline(0, color=COLOURS["baseline_dark"], linewidth=0.9, zorder=1)
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()))
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlim(-0.02, 0.46)
    ax.set_xlabel("Subject-fingerprint gap")
    style_grid(ax, axis="x")
    panel_title(ax, "B. Subject-fingerprint gap")
    ax.legend(loc="lower right", frameon=False, fontsize=7.2)

    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.30, 0.0),
               frameon=False, ncol=4, fontsize=7.6, handlelength=1.6,
               columnspacing=1.4)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.24, wspace=0.28)
    save_figure(fig, "figure3_severity_ladder_fingerprint")
    plt.close(fig)


# =========================================================================
# Figure 4 -- OXF geometry ladder + reverse transfer + permutation null
# =========================================================================

def make_figure4():
    """
    Panel A: OXF 4-step geometry ladder (stepped dot-with-CI, sequential ramp).
    Panel B: reverse transfer on ds (dot-with-CI, no shaded box).
    Panel C: geometry permutation null (filled density).
    """
    print("Figure 4: OXF geometry ladder + reverse transfer + permutation null")
    import pandas as pd

    fig = plt.figure(figsize=(FULL_WIDTH, mm_to_in(124)))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], width_ratios=[1.25, 1, 1],
                          hspace=0.46, wspace=0.42,
                          top=0.92, bottom=0.11, left=0.085, right=0.97)

    ci_path = TABLE_DIR / "journal_subject_clustered_bootstrap_ci_summary.csv"
    delta_path = TABLE_DIR / "journal_median_to_mage_g_paired_delta_ci.csv"
    if not ci_path.exists():
        raise FileNotFoundError(ci_path)
    if not delta_path.exists():
        raise FileNotFoundError(delta_path)
    ci = pd.read_csv(ci_path)
    delta = pd.read_csv(delta_path)

    def top1_ci(label):
        row = ci[(ci["label"] == label) & (ci["metric"] == "top1")].iloc[0]
        return float(row["observed"]), float(row["ci_lower_95"]), float(row["ci_upper_95"])

    # ---- Panel A: geometry ladder ----
    ax = fig.add_subplot(gs[0, 0:2])
    ladder_methods = ["median\nv2D", "exact else\nmedian", "current\nMAGE",
                      "MAGE-G\nsource span"]
    ladder_labels = [
        "OXF_locked_STN_only_v2D",
        "OXF_exact_raw_else_median_v2D",
        "OXF_current_MAGE_v2D",
        "OXF_MAGE_G_source_span_soft_v2D",
    ]
    ladder_stats = [top1_ci(label) for label in ladder_labels]
    top1 = [s[0] for s in ladder_stats]
    ci_low = [s[1] for s in ladder_stats]
    ci_high = [s[2] for s in ladder_stats]
    x = np.arange(len(ladder_methods))

    # Stepped connector between rungs.
    for i in range(len(x) - 1):
        ax.plot([x[i], x[i + 1], x[i + 1]], [top1[i], top1[i], top1[i + 1]],
                color=COLOURS["connector"], linewidth=1.6, zorder=1,
                solid_capstyle="round")
    for i in range(len(ladder_methods)):
        ax.errorbar(x[i], top1[i],
                    yerr=[[top1[i] - ci_low[i]], [ci_high[i] - top1[i]]],
                    fmt="o", color=LADDER_RAMP[i], markersize=11, capsize=5,
                    markeredgecolor="white", markeredgewidth=1.2, elinewidth=1.5,
                    zorder=3)
        ax.text(x[i], ci_high[i] + 0.03, f"{top1[i]:.2f}", ha="center",
                va="bottom", fontsize=7.4, color=LADDER_RAMP[i], fontweight="bold")

    ax.axhline(0.05, color=COLOURS["baseline"], linestyle=(0, (1, 2)),
               linewidth=0.8, alpha=0.7, zorder=1)
    ax.text(3.05, 0.06, "random-label null", fontsize=7.2, va="bottom",
            ha="right", color=COLOURS["baseline_dark"])
    ax.set_xticks(x)
    ax.set_xticklabels(ladder_methods, fontsize=7.5)
    ax.set_xlim(-0.4, 3.4)
    ax.set_ylabel("Top-1 accuracy (OXF, n = 30)")
    ax.set_ylim(0.0, 1.05)
    style_grid(ax, axis="y")
    top1_delta = delta[delta["metric"].eq("delta_top1")].iloc[0]
    panel_title(
        ax,
        f"A. OXF geometry ladder (cumulative Δ = {float(top1_delta['observed_delta']):+.3f}, "
        f"CI [{float(top1_delta['ci_lower_95']):+.3f}, {float(top1_delta['ci_upper_95']):+.3f}])",
        size=8.6,
    )

    # ---- Panel B: reverse transfer ----
    ax = fig.add_subplot(gs[0, 2])
    rt_methods = ["native", "no G", "MAGE-G"]
    rt_labels = [
        "reverse_reference_ds_native_STN_v2D",
        "reverse_ds_frozen_no_MAGE_G",
        "reverse_ds_frozen_MAGE_G",
    ]
    rt_stats = [top1_ci(label) for label in rt_labels]
    rt_top1 = [s[0] for s in rt_stats]
    rt_ci_low = [s[1] for s in rt_stats]
    rt_ci_high = [s[2] for s in rt_stats]
    x_rt = np.arange(len(rt_methods))
    colours_rt = [COLOURS["baseline"], COLOURS["v2D"], COLOURS["mage_g"]]

    # Faint reference line at the native value (replaces the grey box).
    ax.axhline(rt_top1[0], color=COLOURS["baseline"], linestyle=(0, (4, 3)),
               linewidth=0.8, alpha=0.6, zorder=1)
    for i in range(len(rt_methods)):
        ax.errorbar(x_rt[i], rt_top1[i],
                    yerr=[[rt_top1[i] - rt_ci_low[i]], [rt_ci_high[i] - rt_top1[i]]],
                    fmt="o", color=colours_rt[i], markersize=10, capsize=5,
                    markeredgecolor="white", markeredgewidth=1.2, elinewidth=1.5,
                    zorder=3)
        # Value label above each point (matches Panel A).
        ax.text(x_rt[i], rt_ci_high[i] + 0.035, f"{rt_top1[i]:.2f}",
                ha="center", va="bottom", fontsize=7.8, fontweight="bold",
                color=colours_rt[i])
    rt_delta = rt_top1[2] - rt_top1[1]
    ax.text(0.5, 0.07, "CIs overlap — no benefit on\nhomogeneous geometry",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.4,
            style="italic", color=COLOURS["baseline_dark"])
    ax.set_xticks(x_rt)
    ax.set_xticklabels(rt_methods, fontsize=8.5)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("Top-1 (ds, n = 34)")
    ax.set_ylim(0.0, 1.10)
    style_grid(ax, axis="y")
    panel_title(ax, f"B. Reverse transfer (Δ = {rt_delta:+.3f}, n.s.)", size=8.6)

    # ---- Panel C: geometry permutation null (filled density) ----
    ax = fig.add_subplot(gs[1, 0:3])
    null_csv = PLOT_DATA_DIR / "figure4_geometry_permutation_null_compact.csv"
    if not null_csv.exists():
        raise FileNotFoundError(null_csv)
    null_samples = pd.read_csv(null_csv)["top1"].values
    null_mean = float(np.nanmean(null_samples))
    observed_mage_g = top1[-1]
    empirical_p = (float(np.sum(null_samples >= observed_mage_g)) + 1.0) / (len(null_samples) + 1.0)

    x_grid = np.linspace(0, 1, 512)
    peak = filled_density(ax, null_samples, COLOURS["baseline"], x_grid=x_grid,
                          label="Geometry-permutation null (5000×, exact)")
    ax.set_ylim(0, peak * 1.28)
    ax.axvline(null_mean, color=COLOURS["baseline_dark"], linewidth=1.2,
               linestyle=(0, (4, 3)), zorder=4, label=f"null mean = {null_mean:.3f}")
    ax.axvline(observed_mage_g, color=COLOURS["mage_g"], linewidth=2.4,
               zorder=5, label=f"MAGE-G observed = {observed_mage_g:.3f}")
    ax.scatter([observed_mage_g], [peak * 1.28], marker="v", s=46,
               color=COLOURS["mage_g"], zorder=6, clip_on=False)
    ax.annotate("", xy=(observed_mage_g - 0.01, peak * 0.5),
                xytext=(null_mean + 0.01, peak * 0.5),
                arrowprops=dict(arrowstyle="-|>", color=COLOURS["mage_g"],
                                lw=1.0, alpha=0.7))
    ax.set_xlabel("Top-1 accuracy under permuted geometry labels")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1.0)
    panel_title(ax, f"C. Geometry-permutation null (empirical p ≈ {empirical_p:.4f})", size=8.6)
    ax.legend(loc="upper right", frameon=False, fontsize=7.5)

    save_figure(fig, "figure4_oxf_ladder_reverse_permutation")
    plt.close(fig)


# =========================================================================
# Figure 5 -- Single-feature baselines vs proposed methods
# =========================================================================

def make_figure5():
    """Grouped bars: baselines vs proposed methods across 3 evaluation regimes."""
    print("Figure 5: baselines vs proposed methods")
    import pandas as pd

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, mm_to_in(92)))

    baseline_path = TABLE_DIR / "journal_single_feature_baseline_key_metrics.csv"
    baseline_ci_path = TABLE_DIR / "journal_single_feature_baseline_subject_bootstrap_ci.csv"
    method_ci_path = TABLE_DIR / "journal_subject_clustered_bootstrap_ci_summary.csv"
    diagnostics_path = DATA_DIR / "v2d_18subjects_query_diagnostics.csv"
    for required in (baseline_path, baseline_ci_path, method_ci_path, diagnostics_path):
        if not required.exists():
            raise FileNotFoundError(required)
    baseline = pd.read_csv(baseline_path)
    baseline_ci = pd.read_csv(baseline_ci_path)
    method_ci = pd.read_csv(method_ci_path)
    diagnostics = pd.read_csv(diagnostics_path)

    def subject_clustered_top1_ci(rows, n_boot=5000):
        rng = np.random.default_rng(42)
        subjects = sorted(rows["query_subject"].astype(str).unique())
        per_subject = {}
        for subject in subjects:
            subject_rows = rows[rows["query_subject"].astype(str) == subject]
            per_subject[subject] = (
                int(len(subject_rows)),
                float(subject_rows["top_ranked_is_true_pair"].astype(float).sum()),
            )
        samples = []
        for _ in range(n_boot):
            draw = rng.choice(subjects, size=len(subjects), replace=True)
            n = 0
            successes = 0.0
            for subject in draw:
                item_n, item_successes = per_subject[str(subject)]
                n += item_n
                successes += item_successes
            samples.append(successes / n if n else np.nan)
        return tuple(float(v) for v in np.nanpercentile(samples, [2.5, 97.5]))

    def baseline_top1_ci(dataset, condition, variant):
        row = baseline_ci[
            baseline_ci["dataset"].astype(str).eq(dataset)
            & baseline_ci["candidate_pool_condition"].astype(str).eq(condition)
            & baseline_ci["variant_name"].astype(str).eq(variant)
            & baseline_ci["metric"].astype(str).eq("top1")
        ].iloc[0]
        return float(row["ci_lower_95"]), float(row["ci_upper_95"])

    def proposed_top1_ci(dataset, condition):
        if dataset == "OXF":
            row = method_ci[
                method_ci["label"].astype(str).eq("OXF_MAGE_G_source_span_soft_v2D")
                & method_ci["metric"].astype(str).eq("top1")
            ].iloc[0]
            return float(row["ci_lower_95"]), float(row["ci_upper_95"])
        variant = (
            "v2A_top5_aperiodic_rerank"
            if condition == "original_frozen_matched_pool"
            else "v2D_quality_deconfounded_cycle"
        )
        rows = diagnostics[
            diagnostics["candidate_pool_condition"].astype(str).eq(condition)
            & diagnostics["variant_name"].astype(str).eq(variant)
        ]
        return subject_clustered_top1_ci(rows)

    regimes = ["ds matched\n(n = 34)", "ds all-MedOn\n(n = 34)", "OXF all-MedOn\n(n = 30)"]
    group_specs = [
        (regimes[0], "ds004998", "original_frozen_matched_pool"),
        (regimes[1], "ds004998", "all_medon_candidates"),
        (regimes[2], "OXF", "all_medon_candidates"),
    ]
    variant_to_method = {
        "single_stn_beta_band_power": "STN beta power",
        "single_stn_aperiodic_slope": "STN aperiodic slope",
        "two_feature_beta_plus_slope": "Beta + slope (2D)",
    }
    methods = ["STN beta power", "STN aperiodic slope", "Beta + slope (2D)", "Best audited method"]
    method_colours = {
        "STN beta power": "#4A4F56",
        "STN aperiodic slope": "#8B919A",
        "Beta + slope (2D)": "#C7CCD2",
        "Best audited method": COLOURS["mage_g"],
    }
    method_hatch = {
        "STN beta power": "", "STN aperiodic slope": "",
        "Beta + slope (2D)": "////", "Best audited method": "",
    }

    method_data = {m: [] for m in methods}
    method_ci_data = {m: [] for m in methods}
    for _, dataset, condition in group_specs:
        rows = baseline[
            baseline["dataset"].astype(str).eq(dataset)
            & baseline["candidate_pool_condition"].astype(str).eq(condition)
        ]
        for variant, method in variant_to_method.items():
            row = rows[rows["variant_name"].astype(str).eq(variant)].iloc[0]
            method_data[method].append(float(row["top1"]))
            method_ci_data[method].append(baseline_top1_ci(dataset, condition, variant))
        method_data["Best audited method"].append(float(rows["reference_top1"].iloc[0]))
        method_ci_data["Best audited method"].append(proposed_top1_ci(dataset, condition))

    n_methods = len(method_data)
    n_regimes = len(regimes)
    width = 0.8 / n_methods
    x = np.arange(n_regimes)

    mpl.rcParams["hatch.linewidth"] = 0.6
    for i, method in enumerate(methods):
        values = np.asarray(method_data[method], dtype=float)
        offset = (i - n_methods / 2 + 0.5) * width
        ci_values = method_ci_data[method]
        yerr = [values - np.asarray([lo for lo, _ in ci_values]),
                np.asarray([hi for _, hi in ci_values]) - values]
        bars = ax.bar(x + offset, values, width, color=method_colours[method],
                      label=method, edgecolor="#33373D", linewidth=0.6,
                      hatch=method_hatch[method], yerr=yerr, capsize=2.5,
                      error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#555B62"})
        is_best = method == "Best audited method"
        for rect, value, (_, hi) in zip(bars, values, ci_values):
            if not np.isfinite(value):
                continue
            ax.text(rect.get_x() + rect.get_width() / 2, hi + 0.02,
                    f"{value:.2f}", ha="center", va="bottom",
                    fontsize=7.4 if is_best else 6.6,
                    fontweight="bold" if is_best else "normal",
                    color=COLOURS["mage_g"] if is_best else INK)

    ax.axhline(0.05, color=COLOURS["baseline"], linestyle=(0, (1, 2)),
               linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_xlim(-0.55, 3.25)
    ax.text(2.52, 0.075, "random-label null", fontsize=7.2, va="bottom",
            ha="left", color=COLOURS["baseline_dark"])
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1.13)
    style_grid(ax, axis="y")
    panel_title(ax, "Single-feature baselines")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.50, 0.01),
               frameon=False, ncol=4, fontsize=7.6, handlelength=1.5,
               columnspacing=1.4)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.24)
    save_figure(fig, "figure5_baselines_vs_methods")
    plt.close(fig)


# =========================================================================
# Figure 6 -- Symptom-axis dissociation + severity confound diagnostic
# =========================================================================

def make_figure6():
    """
    Panel A: forest / lollipop of the four LOSO ridge axes' Spearman rho
             (all negative -> opposite the predicted positive direction).
    Panels B-C: tremor severity-confound scatters (beta-suppression vs OFF
             severity and vs % improvement).
    """
    print("Figure 6: symptom-axis dissociation + severity diagnostic")
    import pandas as pd

    loso_csv = PLOT_DATA_DIR / "figure6_loso_scatter_compact.csv"
    severity_csv = PLOT_DATA_DIR / "figure6_tremor_severity_diagnostic_compact.csv"
    axis_summary_csv = SYMPTOM_AXIS_DIR / "symptom_axis_summary.csv"
    severity_summary_csv = SYMPTOM_AXIS_DIR / "direction_baseline_severity_component_diagnostic.csv"
    for required in (loso_csv, severity_csv, axis_summary_csv, severity_summary_csv):
        if not required.exists():
            raise FileNotFoundError(required)

    fig = plt.figure(figsize=(FULL_WIDTH, mm_to_in(132)))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], hspace=0.46,
                          wspace=0.26, top=0.90, bottom=0.10, left=0.10, right=0.97)

    # ---- Panel A: forest / lollipop of LOSO rho ----
    ax = fig.add_subplot(gs[0, :])
    axis_summary = pd.read_csv(axis_summary_csv).set_index("axis_name")
    axes_specs = [
        ("AR_contralateral", "Contralateral AR"),
        ("tremor_contralateral", "Contralateral tremor"),
        ("total_updrs", "Total UPDRS"),
        ("axial", "Axial"),
    ]
    y_positions = np.arange(len(axes_specs))[::-1]  # first axis at top

    # Shade the predicted-direction half-plane (rho > 0).
    ax.axvspan(0, 1, color=COLOURS["positive"], alpha=0.06, zorder=0)
    ax.axvline(0, color=COLOURS["baseline_dark"], linewidth=0.9, zorder=2)

    for (axis_key, axis_label), y in zip(axes_specs, y_positions):
        summary = axis_summary.loc[axis_key]
        rho = float(summary["spearman_r"])
        perm_p = float(summary["permutation_p_spearman_directional"])
        stable = str(summary["axis_weight_stable"]).strip().lower() in {"true", "1", "yes"}
        colour = COLOURS["negative"]
        ax.plot([0, rho], [y, y], color=colour, linewidth=2.0, alpha=0.55,
                zorder=2, solid_capstyle="round")
        ax.scatter([rho], [y], s=70, color=colour, edgecolor="white",
                   linewidth=1.0, zorder=3,
                   marker="o" if stable else "D")
        ax.text(rho - 0.02, y + 0.32, f"ρ = {rho:.2f}", ha="right", va="center",
                fontsize=8.0, color=colour, fontweight="bold")
        ax.text(0.03, y, f"{'stable' if stable else 'unstable'} weights · "
                         f"$p_{{dir}}$ = {perm_p:.2f}", ha="left", va="center",
                fontsize=7.4, color=COLOURS["baseline_dark"])

    ax.annotate("predicted direction (ρ > 0)", xy=(0.20, len(axes_specs) - 0.5),
                ha="left", va="center", fontsize=7.8, color=COLOURS["positive"],
                fontweight="bold")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label in axes_specs])
    ax.set_ylim(-0.6, len(axes_specs) - 0.4)
    ax.set_xlim(-0.75, 0.45)
    ax.set_xlabel("Out-of-sample LOSO Spearman ρ (predicted vs true response)")
    style_grid(ax, axis="x")
    panel_title(ax, "A. Symptom-axis LOSO predictions")

    # ---- shared severity data ----
    df_sev = pd.read_csv(severity_csv)
    severity_summary = pd.read_csv(severity_summary_csv)
    tremor_beta = severity_summary[
        severity_summary["symptom"].astype(str).eq("tremor")
        & severity_summary["laterality"].astype(str).eq("contralateral")
        & severity_summary["component"].astype(str).eq("beta_power_residual_suppression")
        & severity_summary["feature_kind"].astype(str).eq("oriented_mean")
    ].iloc[0]
    def severity_scatter(ax, xcol, ycol, valid, colour, xlabel, ylabel, title):
        x_data = df_sev.loc[valid, xcol].values
        y_data = df_sev.loc[valid, ycol].values
        ax.scatter(x_data, y_data, color=colour, s=40, alpha=0.8,
                   edgecolor="white", linewidth=0.6, zorder=3)
        if len(x_data) > 2:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_data.min(), x_data.max(), 50)
            ax.plot(x_line, p(x_line), color=colour, linewidth=1.8,
                    linestyle=(0, (4, 3)), alpha=0.85, zorder=2)
        ax.margins(x=0.10, y=0.16)
        ax.set_xlabel(xlabel, labelpad=5)
        ax.set_ylabel(ylabel, labelpad=5)
        style_grid(ax, axis="y")
        panel_title(ax, title, size=8.6, color=INK)

    ax = fig.add_subplot(gs[1, 0])
    severity_scatter(
        ax, "beta_suppression_delta", "contralateral_tremor_off_severity",
        ~df_sev["contralateral_tremor_off_severity"].isna(), COLOURS["negative"],
        "Beta-suppression component (Δ)", "Contralateral tremor OFF severity",
        "B. OFF severity")

    ax = fig.add_subplot(gs[1, 1])
    severity_scatter(
        ax, "beta_suppression_delta", "contralateral_tremor_percent_response",
        ~df_sev["contralateral_tremor_percent_response"].isna(), COLOURS["v2D"],
        "Beta-suppression component (Δ)", "Contralateral tremor % response",
        "C. Percent improvement")

    save_figure(fig, "figure6_symptom_dissociation")
    plt.close(fig)


# =========================================================================
# Figure 7 -- DBS vs levodopa STN transition-geometry convergence
# =========================================================================

def make_figure7():
    """
    Panel A: cosine(Δ_med, Δ_DBS) by feature subspace, with bootstrap CIs
             (periodic converges; aperiodic/broadband does not).
    Panel B: per-feature mean transition direction, medication vs DBS, with
             the same-direction agreement quadrants shaded.
    """
    print("Figure 7: DBS vs levodopa transition convergence")
    import json
    import pandas as pd
    from matplotlib.patches import Rectangle

    conv_dir = DATA_DIR
    summ_path = conv_dir / "convergence_summary.json"
    feat_path = conv_dir / "per_feature_direction_agreement.csv"
    if not summ_path.exists():
        raise FileNotFoundError(summ_path)
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)
    summary = json.loads(summ_path.read_text())
    res = summary["results"]
    feat = pd.read_csv(feat_path)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, mm_to_in(84)),
                             gridspec_kw={"width_ratios": [1.0, 1.05]})

    # ---- Panel A: subspace cosine forest ----
    ax = axes[0]
    rows = [
        ("gain_invariant", "Gain-invariant\n(de-confounded, 10)", COLOURS["v2C"]),
        ("beta_periodic_stim_robust", "Beta-periodic\n(stim-robust, 5)", COLOURS["v2A"]),
        ("gain_sensitive_amplitude", "Gain-sensitive\n(amplitude, 6)", COLOURS["baseline"]),
        ("all_features", "All\nfeatures (16)", COLOURS["v2D"]),
    ]
    y = np.arange(len(rows))[::-1]
    ax.axvspan(0, 1.05, color=COLOURS["positive"], alpha=0.06, zorder=0)
    ax.axvline(0, color=COLOURS["baseline_dark"], linewidth=0.9, zorder=2)
    for (key, _, colour), yi in zip(rows, y):
        r = res[key]
        c = r["observed_cosine"]
        lo, hi = r["ci_lower_95"], r["ci_upper_95"]
        ax.errorbar(c, yi, xerr=[[c - lo], [hi - c]], fmt="o", color=colour,
                    markersize=10, capsize=4, elinewidth=1.6,
                    markeredgecolor="white", markeredgewidth=1.1, zorder=3)
        ax.text(hi + 0.05, yi, f"{c:+.2f}  ($p$ = {r['p_signflip']:.4f})",
                va="center", ha="left", fontsize=7.2, color=colour, fontweight="bold")
    ax.text(0.52, len(rows) - 0.42, "convergent →", fontsize=7.4,
            color=COLOURS["positive"], fontweight="bold", ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab, _ in rows], fontsize=7.6)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-0.85, 1.75)
    ax.set_xlabel("cosine(Δ levodopa, Δ DBS)  ·  95% bootstrap CI")
    style_grid(ax, axis="x")
    panel_title(ax, "A. Subspace alignment")

    # ---- Panel B: per-feature direction scatter ----
    ax = axes[1]
    lim = 2.6
    # shade same-direction (agreement) quadrants
    for x0, y0 in [(0, 0), (-lim, -lim)]:
        ax.add_patch(Rectangle((x0, y0), lim, lim, color=COLOURS["positive"],
                               alpha=0.06, zorder=0, linewidth=0))
    ax.axhline(0, color=COLOURS["baseline_dark"], linewidth=0.8, zorder=1)
    ax.axvline(0, color=COLOURS["baseline_dark"], linewidth=0.8, zorder=1)
    ax.plot([-lim, lim], [-lim, lim], color=COLOURS["baseline"], linewidth=0.8,
            linestyle=(0, (4, 3)), alpha=0.6, zorder=1)
    gain_colour = {"gain_invariant": COLOURS["v2C"], "gain_sensitive": COLOURS["baseline_dark"]}
    for _, r in feat.iterrows():
        ax.scatter(r["mean_delta_med_z"], r["mean_delta_dbs_z"],
                   color=gain_colour[r["gain_class"]], s=42, alpha=0.9,
                   edgecolor="white", linewidth=0.6, zorder=3)
    # label a few key features (offsets chosen to avoid the diagonal cluster)
    key_labels = {
        "broad_beta_residual_power": ("β residual", (-42, -2), "left"),
        "gamma_residual_power": ("γ residual", (6, 3), "left"),
        "stn_broad_beta_log_power": ("β abs. power", (6, -10), "left"),
        "aperiodic_offset": ("aperiodic offset", (6, 4), "left"),
    }
    for _, r in feat.iterrows():
        if r["feature"] in key_labels:
            txt, off, ha = key_labels[r["feature"]]
            ax.annotate(txt, (r["mean_delta_med_z"], r["mean_delta_dbs_z"]),
                        textcoords="offset points", xytext=off,
                        fontsize=6.8, color=gain_colour[r["gain_class"]], ha=ha)
    ax.text(2.45, 0.6, "same\ndirection", fontsize=7.2, ha="right", va="center",
            color=COLOURS["positive"], fontweight="bold")
    ax.text(-2.45, 2.25, "opposite\n(amplitude only)", fontsize=7.0, ha="left",
            va="center", color=COLOURS["baseline_dark"])
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=8,
               markerfacecolor=COLOURS["v2C"], markeredgecolor="white",
               label="gain-invariant"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=8,
               markerfacecolor=COLOURS["baseline_dark"], markeredgecolor="white",
               label="gain-sensitive"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.025),
                    frameon=True, fontsize=7.0, handletextpad=0.5, labelspacing=0.5,
                    borderpad=0.6, borderaxespad=0.0, handlelength=1.0,
                    edgecolor=SPINE, facecolor="white", framealpha=1.0)
    leg.get_frame().set_linewidth(0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Mean Δ under levodopa (z)")
    ax.set_ylabel("Mean Δ under DBS (z)")
    panel_title(ax, "B. Per-feature directions")

    fig.subplots_adjust(left=0.16, right=0.985, top=0.88, bottom=0.17, wspace=0.32)
    save_figure(fig, "figure7_dbs_levodopa_convergence")
    plt.close(fig)


# =========================================================================
# Run all
# =========================================================================

if __name__ == "__main__":
    print("Producing figures (redesigned)...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    make_figure5()
    make_figure6()
    make_figure7()
    print("\nDone.")
