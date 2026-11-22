"""Figures for the teaching notebook (matplotlib, one consistent style)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import TARGETS, TARGET_NAMES
from .data import SUBSTRATE_THRESHOLD, FEEDS

# Validated categorical palette (fixed order) and neutral tones
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID, CONTEXT = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10, "legend.frameon": False,
        "lines.linewidth": 2, "font.size": 10,
    })


def _measured(raw):
    return raw.dropna(subset=TARGETS)


def plot_batches(raw, highlight=("Batch1", "Batch14", "Batch29")):
    """All training trajectories in grey, three batches highlighted."""
    m = _measured(raw)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, t in zip(axes, TARGETS):
        for b, g in m.groupby("batch_id"):
            ax.plot(g.elapsed_time, g[t], color=CONTEXT, lw=0.7, alpha=0.6)
        for b, c in zip(highlight, (BLUE, ORANGE, AQUA)):
            g = m[m.batch_id == b]
            ax.plot(g.elapsed_time, g[t], color=c, marker="o", ms=3, label=b)
        ax.set_title(TARGET_NAMES[t]); ax.set_xlabel("day")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_feeds(raw, batch="Batch12"):
    """Feed schedule and volume of one batch: what we are given for a test batch."""
    g = raw[raw.batch_id == batch].sort_values("elapsed_time")
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.2), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    for i, (f, c) in enumerate(zip(FEEDS, (BLUE, ORANGE, AQUA, MAGENTA))):
        e = g[g[f].notna()]
        ax.scatter(e.elapsed_time, np.full(len(e), i), s=20 + 400 * e[f] / raw[f].max(), color=c)
    s = g[g.to_predict | g.vcd.notna()]
    for t in s.elapsed_time:
        ax.axvline(t, color=GRID, lw=1, zorder=0)
    ax.set_yticks(range(4), ["substrate feed", "feed B", "feed C", "feed D"]); ax.set_xlabel("day")
    ax.set_title(f"{batch}: feed events (dot size = dose), grey lines = sample times")
    v = g.dropna(subset=["volume"])
    axes[1].plot(v.elapsed_time, v.volume, color=BLUE, marker="o", ms=3)
    axes[1].set_title("culture volume"); axes[1].set_xlabel("day")
    fig.tight_layout()
    return fig


def plot_mape_weight(raw):
    """Why MAPE is dominated by small values."""
    m = _measured(raw).copy(); m = m[m.elapsed_time > 0.5]; m["day"] = m.elapsed_time.round().astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.4))
    y = np.linspace(0.2, 4, 200)
    axes[0].plot(y, 100 * 0.1 / y, color=BLUE)
    for v in (0.5, 3):
        axes[0].plot(v, 10 / v, "o", color=ORANGE, ms=8)
        axes[0].annotate(f"{10 / v:.0f}% at y = {v}", (v, 10 / v), xytext=(8, 6), textcoords="offset points")
    axes[0].set_xlabel("true value y"); axes[0].set_ylabel("APE of a 0.1 error (%)")
    axes[0].set_title("The same absolute error costs more when y is small")
    med = m.groupby("day")[TARGETS].median()
    for t, c in zip(TARGETS, (BLUE, ORANGE, AQUA, MAGENTA)):
        axes[1].plot(med.index, med[t] / med[t].max(), color=c, marker="o", ms=3, label=t)
    axes[1].set_xlabel("day"); axes[1].set_ylabel("median value / its maximum")
    axes[1].set_title("Where each target is small (and MAPE is decided)"); axes[1].legend(ncol=2)
    fig.tight_layout()
    return fig


def plot_controller(ev):
    """Substrate measured before each bolus vs dose per volume: a perfect threshold."""
    L = ev[~ev.test]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for hi, c, lab in ((True, ORANGE, "high dose"), (False, BLUE, "low dose")):
        g = L[L.high_dose == hi]
        ax.scatter(g.substrate_at_sample, 1000 * g.dose_per_volume, color=c, s=22, label=f"{lab} ({len(g)})")
    ax.axvline(SUBSTRATE_THRESHOLD, color=INK, lw=1, ls="--")
    ax.annotate(f"substrate = {SUBSTRATE_THRESHOLD}", (SUBSTRATE_THRESHOLD, 5), xytext=(6, 0), textcoords="offset points")
    ax.set_xlabel("substrate at the preceding sample"); ax.set_ylabel("dose per volume (×10⁻³)")
    ax.set_title("Substrate-feed controller, training batches"); ax.legend(loc="center right")
    fig.tight_layout()
    return fig


def plot_base_offset(base, offsets, cols=("interval 3", "interval 5")):
    """Base added on days 3-5 vs the batch's byproduct over-prediction."""
    fig, axes = plt.subplots(1, len(cols), figsize=(11, 3.4), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), cols):
        x = base.loc[offsets.index, c]
        ax.scatter(x, 100 * offsets, color=BLUE, s=22)
        k, b0 = np.polyfit(x, 100 * offsets, 1); xs = np.linspace(x.min(), x.max(), 10)
        ax.plot(xs, k * xs + b0, color=ORANGE)
        ax.set_title(f"{c}: r = {np.corrcoef(x, offsets)[0, 1]:.2f}")
        ax.set_xlabel("unexplained volume vs typical (≈ base added)")
    np.atleast_1d(axes)[0].set_ylabel("batch over-prediction (%)")
    fig.suptitle("More base (more acid) → the ensemble under-predicts byproduct", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_members(byp, raw, batch, members=("B1", "B2", "C", "K4", "K6", "A6")):
    """Byproduct members and the corrected forecast for one test batch, over the training envelope."""
    m = _measured(raw)
    g = byp[byp.batch_id == batch]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for _, h in m.groupby("batch_id"):
        ax.plot(h.elapsed_time, h.byproduct, color=GRID, lw=0.7)
    for mem in members:
        ax.plot(g.elapsed_time, g[mem], color=CONTEXT, lw=1)
    ax.plot([], [], color=CONTEXT, lw=1, label="6 members")
    ax.plot(g.elapsed_time, g.byproduct, color=BLUE, marker="o", ms=4, label="final forecast")
    ax.set_xlabel("day"); ax.set_ylabel("byproduct"); ax.legend(loc="upper left")
    ax.set_title(f"{batch}: byproduct members and final forecast (grey background = training batches)")
    fig.tight_layout()
    return fig


def plot_daily_bias(T, pred, title):
    """Median log error per day: systematic over/under-prediction."""
    e = 100 * (np.exp(np.log(pred / T.y.to_numpy())) - 1)
    med = pd.Series(e).groupby(T.day.values).median()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(med.index, med.values, color=[ORANGE if d == 13 else BLUE for d in med.index], width=0.6)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("day"); ax.set_ylabel("median error (%)"); ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_ablation(table):
    t = table.copy()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(len(t))
    ax.plot(t["CV MAPE"], y, color=CONTEXT, lw=1, zorder=1)
    ax.scatter(t["CV MAPE"], y, color=BLUE, s=60, zorder=2)
    for i, v in enumerate(t["CV MAPE"]):
        ax.annotate(f"{v:.2f}", (v, i), xytext=(10, 0), textcoords="offset points", va="center", color=INK)
    ax.set_yticks(y, t.index); ax.invert_yaxis()
    ax.set_xlim(t["CV MAPE"].min() - 0.1, t["CV MAPE"].max() + 0.15); ax.set_xlabel("byproduct MAPE, team CV (%)")
    ax.set_title("Byproduct: each step, validated out of fold")
    fig.tight_layout()
    return fig


def plot_leaderboard(lb):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    x = np.arange(len(lb))
    ax.plot(x, lb["leaderboard MAPE"], color=BLUE, marker="o", ms=6)
    for i, v in enumerate(lb["leaderboard MAPE"]):
        ax.annotate(f"{v:.3f}", (i, v), xytext=(0, 8), textcoords="offset points", ha="center")
    ax.set_xticks(x, lb.version, rotation=20, ha="right")
    ax.set_ylabel("mean MAPE (%)"); ax.set_title("Team 6 leaderboard history (12 test batches)")
    ax.set_ylim(lb["leaderboard MAPE"].min() - 0.2, lb["leaderboard MAPE"].max() + 0.25)
    fig.tight_layout()
    return fig


def plot_test_forecasts(final, raw):
    """Final forecasts for the 12 test batches (colour) over the training trajectories (grey)."""
    m = _measured(raw)
    f = final.copy(); f["elapsed_time"] = f.elapsed_time.astype(float)
    day0 = raw[(raw.batch_id.isin(f.batch_id)) & raw.vcd.notna()]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, t in zip(axes, TARGETS):
        for _, g in m.groupby("batch_id"):
            ax.plot(g.elapsed_time, g[t], color=GRID, lw=0.7)
        for b, g in f.groupby("batch_id"):
            d0 = day0[day0.batch_id == b]
            ax.plot(np.r_[d0.elapsed_time, g.elapsed_time], np.r_[d0[t], g[t].astype(float)], color=BLUE, lw=1, alpha=0.8)
        ax.set_title(TARGET_NAMES[t]); ax.set_xlabel("day")
    fig.suptitle("Final forecasts for the 12 test batches (blue) over the 49 training batches (grey)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_cells_blend(c21, v4, final, batch):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
    for ax, t in zip(axes, ("vcd", "tcd")):
        for df, c, lab in ((c21, CONTEXT, "C21"), (v4, ORANGE, "kinetic v4"), (final, BLUE, "blend (C21² × v4)^⅓")):
            g = df[df.batch_id == batch]
            ax.plot(g.elapsed_time.astype(float), g[t].astype(float), color=c, label=lab, lw=2 if c == BLUE else 1.5)
        ax.set_title(f"{batch}: {TARGET_NAMES[t]}"); ax.set_xlabel("day")
    axes[0].legend()
    fig.tight_layout()
    return fig
