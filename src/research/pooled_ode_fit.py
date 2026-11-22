"""Fit shared smooth amount-change rates and plot measured batch trajectories."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear

HERE = Path(__file__).resolve().parent
DATA = Path("input-data/Project 1 - DTB/bioreaction_remaining.csv")
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]


def observed_amounts(rows):
    vcd, tcd, substrate, byproduct = rows[TARGETS].to_numpy().T
    return np.column_stack([vcd, tcd - vcd, substrate, byproduct]) * rows.volume.to_numpy()[:, None]


def feed_totals(rows, times):
    event_times = rows.elapsed_time.to_numpy()
    doses = rows[FEEDS].fillna(0).to_numpy()
    cumulative = np.vstack([np.zeros(4), doses.cumsum(axis=0)])
    return cumulative[np.searchsorted(event_times, times, side="right")]


def fit_shared_ode(groups):
    end = max(13.0, max(measured.elapsed_time.iloc[-1] for _, measured in groups))
    # A common cubic spline defines the rate for each amount. Integrating it
    # analytically gives a linear fit on the measured amount changes.
    degree = 3
    knots = np.r_[np.zeros(degree + 1), [2, 4, 6, 8, 10], np.full(degree + 1, end)]
    size = len(knots) - degree - 1
    primitive = BSpline(knots, np.eye(size), degree).antiderivative()

    def integrated_rates(times, start):
        return primitive(times) - primitive(start)

    design, changes, accumulated_doses = [], [], []
    for rows, measured in groups:
        times = measured.elapsed_time.to_numpy()
        amounts = observed_amounts(measured)
        design.append(integrated_rates(times[1:], times[0]))
        changes.append(amounts[1:] - amounts[0])
        feed = feed_totals(rows, times)
        accumulated_doses.append(feed[1:, :2] - feed[0, :2])
    design = np.vstack(design)
    changes = np.vstack(changes)
    doses = np.vstack(accumulated_doses)
    # Penalize rapid changes between neighboring rate coefficients.
    penalty = 2 * np.diff(np.eye(size), n=2, axis=0)
    regularized = np.vstack([design, penalty])
    targets = np.vstack([changes, np.zeros((len(penalty), 4))])
    coefficients = np.linalg.lstsq(regularized, targets, rcond=None)[0]
    # Substrate and Feed B have unknown effective substrate strengths.
    substrate_design = np.column_stack([regularized, np.vstack([doses, np.zeros((len(penalty), 2))])])
    substrate_fit = lsq_linear(
        substrate_design, targets[:, 2],
        bounds=(np.r_[np.full(size, -np.inf), 0, 0], np.inf),
    )
    if not substrate_fit.success:
        raise RuntimeError(substrate_fit.message)
    coefficients[:, 2] = substrate_fit.x[:size]
    strength = substrate_fit.x[size:]

    def simulate(rows, measured, times):
        first = measured.iloc[0]
        co = feed_totals(rows, np.r_[first.elapsed_time, times])
        feed = co[1:] - co[0]
        amounts = observed_amounts(measured.iloc[:1])[0] + integrated_rates(times, first.elapsed_time) @ coefficients
        amounts[:, 2] += feed[:, :2] @ strength
        # Volume at prediction time is reconstructed from the initial volume and doses.
        concentration = amounts / (first.volume + feed.sum(axis=1))[:, None]
        concentration[:, 1] += concentration[:, 0]
        return concentration

    return simulate


def main():
    source = next(root / DATA for root in HERE.parents if (root / DATA).is_file())
    data = pd.read_csv(source)
    test_ids = set(data.loc[data.to_predict, "batch_id"])
    test_points = data.loc[data.batch_id.isin(test_ids)].dropna(subset=TARGETS)
    groups = []
    for _, rows in data.loc[~data.batch_id.isin(test_ids)].groupby("batch_id"):
        rows = rows.sort_values("elapsed_time")
        measured = rows.dropna(subset=[*TARGETS, "volume"])
        groups.append((rows, measured))
    simulate = fit_shared_ode(groups)
    end = max(measured.elapsed_time.iloc[-1] for _, measured in groups)
    times = np.linspace(0, end, 500)
    forecasts = np.stack([simulate(rows, measured, times) for rows, measured in groups])
    overall = np.median(forecasts, axis=0)
    residual_rows = []
    for rows, measured in groups:
        later = measured.iloc[1:]
        predicted = simulate(rows, measured, later.elapsed_time.to_numpy())
        residual = pd.DataFrame(later[TARGETS].to_numpy() - predicted, columns=TARGETS)
        residual.insert(0, "elapsed_time", later.elapsed_time.to_numpy())
        residual.insert(0, "batch_id", rows.batch_id.iloc[0])
        residual_rows.append(residual)
    residuals = pd.concat(residual_rows, ignore_index=True)
    residuals.to_csv(HERE / "pooled_ode_residuals.csv", index=False)
    errors = residuals[TARGETS].to_numpy()
    print("In-sample RMSE:", dict(zip(TARGETS, np.sqrt(np.mean(errors**2, axis=0)).round(3))))
    print("Minimum fitted concentrations:", dict(zip(TARGETS, overall.min(axis=0).round(3))))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    fig.subplots_adjust(left=.085, right=.975, bottom=.105, top=.835, hspace=.36, wspace=.23)
    fig.text(.085, .955, "Shared bioreactor trajectory", fontsize=21, fontweight="bold", color="#122d42")
    fig.text(.085, .905, f"{len(groups)} training batches + {len(test_points)} test starting points  ·  solid line: median shared-ODE fit",
             fontsize=11, color="#455b6d")
    titles = ["Viable cell density", "Total cell density", "Substrate", "Byproduct"]
    for j, ax in enumerate(axes.flat):
        for _, measured in groups:
            ax.plot(measured.elapsed_time, measured[TARGETS[j]], color="#587991", alpha=.13,
                    linewidth=.9, marker=".", markersize=2.4, zorder=2)
        ax.scatter(test_points.elapsed_time, test_points[TARGETS[j]], s=7,
                   color="#587991", alpha=.13, zorder=2)
        ax.plot(times, overall[:, j], color="#b63c23", linewidth=2.8, zorder=4)
        ax.set_title(titles[j], loc="left", fontweight="bold", color="#122d42")
        ax.set_ylabel("Density" if j < 2 else "Concentration")
        ax.set_xlim(0, end)
        ax.grid(color="#e5edf1", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#b3c1cc")
    fig.supxlabel("Elapsed time (days)", y=.035, color="#455b6d")
    fig.savefig(HERE / "pooled_ode_fit.png", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
