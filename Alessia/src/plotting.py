"""Plotting: common style, axis labels with units, interactive model-vs-measurement viewer and summary charts."""

import matplotlib.pyplot as plt # static plots
import numpy as np # numerics
import pandas as pd # tables
import plotly.graph_objects as go # interactive plots
from plotly.subplots import make_subplots # grid of interactive plots

from .data_handling import TARGET_COLS # target names
from .evaluate import TEST_BATCHES # benchmark test batches

LABELS = {"vcd": "VCD [10⁶ cells/mL]", "tcd": "TCD [10⁶ cells/mL]", "substrate": "Substrate [g/L]", "byproduct": "Byproduct [g/L]",
          "elapsed_time": "Time [days]", "ivcd": "IVCD [10⁶ cells·day/mL]"} # axis labels with units
TITLES = {"vcd": "Viable cell density", "tcd": "Total cell density", "substrate": "Substrate", "byproduct": "Byproduct"} # plot titles
PALETTE = ["#8D99AE", "#F2A541", "#E4572E", "#B23A48", "#6C757D", "#0B4F5C", "#2A9D8F", "#7B3FA0", "#3A86FF"] # one color per model, in order


# common matplotlib style for all static plots
def setup():
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, # clean axes
                         "grid.alpha": 0.3, "axes.titlesize": 12, "axes.labelsize": 11, "legend.frameon": False, # light grid, sizes
                         "axes.prop_cycle": plt.cycler(color=["#0B4F5C", "#E4572E", "#2A9D8F", "#7B3FA0", "#F2A541"])}) # palette


# scatter of a rate (y) against candidate variables (x), one panel each
def rate_panels(df, ycol, xcols, ylabel, color_by="elapsed_time"):
    fig, axes = plt.subplots(1, len(xcols), figsize=(4.2 * len(xcols), 3.8), constrained_layout=True, sharey=True) # one panel per candidate
    for ax, x in zip(np.atleast_1d(axes), xcols): # each candidate
        sc = ax.scatter(df[x], df[ycol], c=df[color_by], cmap="viridis", s=12, alpha=0.8) # points colored by time
        ax.axhline(0, color="black", lw=0.8) # zero line
        ax.set_xlabel(LABELS.get(x, x)) # x label with units
        ax.set_title(f"r = {df[[x, ycol]].corr().iloc[0, 1]:.2f}") # correlation
    np.atleast_1d(axes)[0].set_ylabel(ylabel) # y label
    fig.colorbar(sc, ax=axes, label=LABELS["elapsed_time"]) # color legend
    plt.show() # draw


# interactive viewer: measured points and the curves of several models, one batch at a time (menu)
def model_viewer(data, preds, colors=None, height=720):
    colors = colors or {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(preds)} # one color per model
    ids = data.labeled # batches with measurements
    fig = make_subplots(rows=2, cols=2, subplot_titles=[TITLES[c] for c in TARGET_COLS], vertical_spacing=0.14, horizontal_spacing=0.09) # 2x2 grid
    for k, bid in enumerate(ids): # one group of traces per batch
        m = data.samples([bid]) # measured samples of this batch
        for j, c in enumerate(TARGET_COLS): # one panel per target
            row, col = j // 2 + 1, j % 2 + 1 # panel position
            fig.add_trace(go.Scatter(x=m["elapsed_time"], y=m[c], mode="markers", marker=dict(color="black", size=7), name="measured", legendgroup="measured", visible=(k == 0), showlegend=(j == 0)), row=row, col=col) # measured
            for name, pred in preds.items(): # each model
                p = pd.concat([m[m["elapsed_time"] == 0][["elapsed_time", c]], pred[pred["batch_id"] == bid][["elapsed_time", c]]]) # day-0 value + predictions
                fig.add_trace(go.Scatter(x=p["elapsed_time"], y=p[c], mode="lines", line=dict(color=colors[name], width=2.2), name=name, legendgroup=name, visible=(k == 0), showlegend=(j == 0)), row=row, col=col) # model curve
    n = (1 + len(preds)) * len(TARGET_COLS) # traces per batch
    buttons = [dict(label=bid + ("  (test)" if bid in TEST_BATCHES else ""), method="update", args=[{"visible": [i // n == k for i in range(n * len(ids))]}, {"title.text": bid + ("  (benchmark test batch)" if bid in TEST_BATCHES else "  (training batch)")}]) for k, bid in enumerate(ids)] # menu entries
    fig.update_layout(template="plotly_white", height=height, width=1000, margin=dict(t=120, l=60, r=20, b=50), # layout
                      title=dict(text=ids[0] + ("  (benchmark test batch)" if ids[0] in TEST_BATCHES else "  (training batch)"), x=0.5, xanchor="center"), # title
                      updatemenus=[dict(buttons=buttons, x=0, y=1.17, xanchor="left", yanchor="top")], # batch menu
                      legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center")) # legend below (click to hide a model)
    for j, c in enumerate(TARGET_COLS): # axis labels
        fig.update_xaxes(title_text=LABELS["elapsed_time"], row=j // 2 + 1, col=j % 2 + 1) # x label
        fig.update_yaxes(title_text=LABELS[c], row=j // 2 + 1, col=j % 2 + 1) # y label
    fig.show() # draw


# grouped bars: per-target and overall benchmark score of each model
def score_bars(table):
    cols = TARGET_COLS + ["score"] # bar groups
    x = np.arange(len(cols)) # group positions
    w = 0.8 / len(table) # bar width
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True) # figure
    for i, (name, row) in enumerate(table.iterrows()): # one bar series per model
        ax.bar(x + i * w - 0.4 + w / 2, row[cols], width=w, label=name, color=PALETTE[i % len(PALETTE)]) # bars
    ax.set_xticks(x, ["VCD", "TCD", "Substrate", "Byproduct", "Overall score"]) # group labels
    ax.set_ylabel("normalized RMSE (lower is better)") # y label
    ax.legend(ncol=3, fontsize=9, loc="upper left") # legend
    plt.show() # draw


# horizontal bars of the correction coefficients, one panel per target
def coefficient_bars(coef, units):
    fig, axes = plt.subplots(1, len(coef.columns), figsize=(4.2 * len(coef.columns), 4.2), constrained_layout=True) # one panel per target
    for ax, c in zip(axes, coef.columns): # each target
        s = coef[c].dropna().sort_values() # coefficients of this target
        ax.barh(s.index, s.values, color=["#E4572E" if v < 0 else "#0B4F5C" for v in s.values]) # red = lowers, teal = raises
        ax.axvline(0, color="black", lw=0.8) # zero effect
        ax.set_title(TITLES[c]) # target
        ax.set_xlabel(f"effect of +1 std [{units[c]}]") # x label with units
    plt.show() # draw


# all labeled batches over time: training batches in grey, benchmark test batches in color
def overview(data):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True) # 2x2 grid
    for bid in data.labeled: # each batch
        m = data.samples([bid]) # its samples
        test = bid in TEST_BATCHES # benchmark test batch?
        for ax, c in zip(axes.flat, TARGET_COLS): # each target
            ax.plot(m["elapsed_time"], m[c], color="#E4572E" if test else "#8D99AE", lw=1.4 if test else 0.8, alpha=0.9 if test else 0.5) # curve
    for ax, c in zip(axes.flat, TARGET_COLS): # labels
        ax.set_title(TITLES[c]) # target
        ax.set_xlabel(LABELS["elapsed_time"]) # x label
        ax.set_ylabel(LABELS[c]) # y label
    axes[0, 0].plot([], [], color="#8D99AE", label="training batches (39)") # legend entry
    axes[0, 0].plot([], [], color="#E4572E", label="benchmark test batches (10)") # legend entry
    axes[0, 0].legend(loc="upper left") # legend
    plt.show() # draw


# heat map of the per-batch, per-target error of one model (benchmark test batches)
def batch_heatmap(per_batch, title):
    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True) # figure
    per_batch = per_batch.loc[sorted(per_batch.index, key=lambda b: int(b.replace("Batch", "")))] # batches in numeric order
    im = ax.imshow(per_batch.to_numpy(), cmap="Reds", vmin=0, aspect="auto") # colored cells
    top = per_batch.to_numpy().max() # darkest value
    ax.set_xticks(range(len(per_batch.columns)), ["VCD", "TCD", "Substrate", "Byproduct"]) # targets
    ax.set_yticks(range(len(per_batch.index)), per_batch.index) # batches
    for i in range(per_batch.shape[0]): # write values
        for j in range(per_batch.shape[1]): # each cell
            ax.text(j, i, f"{per_batch.iat[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white" if per_batch.iat[i, j] > 0.6 * top else "black") # value, readable on dark cells
    ax.grid(False) # no grid on images
    ax.set_title(title) # title
    fig.colorbar(im, ax=ax, label="normalized RMSE") # color scale
    plt.show() # draw
