"""Trial I: specific-rate kinetic ODE and feed look-ahead features.

Compares, on the same repeated whole-batch folds as repeated_forecast_trials.py:
- shared ODE (time-only absolute rates) vs a kinetic ODE whose rates scale with viable cells
- ExtraTrees residuals on either base, with standard features or standard + look-ahead feed features
- 50/50 blends with a direct random forest
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

from direct_ml_benchmark import features
from pooled_ode_fit import FEEDS, TARGETS, feed_totals, fit_shared_ode
from validate_feed_residuals import load_batches

HERE = Path(__file__).resolve().parent
DEGREE = 3
END = 13.5
KNOTS = np.r_[np.zeros(DEGREE + 1), [2, 4, 6, 8, 10], np.full(DEGREE + 1, END)]
SIZE = len(KNOTS) - DEGREE - 1
BASIS = BSpline(KNOTS, np.eye(SIZE), DEGREE)
PRIMITIVE = BASIS.antiderivative()
STEP = 0.02
SHRINK = True
SHRINK_ALL = False
FEEDS_EVERYWHERE = True
GROWTH_FEEDS = (1, 2)
RIDGE_BETA = 0.1
FEED_UNITS = np.array([0.16, 6.5, 0.021, 0.044])  # typical batch totals, inputs only


def lookahead(rows, times):
    """Next and previous dose of each feed around each time, plus time gaps."""
    columns = []
    for feed in FEEDS:
        ts, dose = rows.loc[rows[feed] > 0, ["elapsed_time", feed]].to_numpy().T
        nxt = np.searchsorted(ts, times, side="left")
        has = nxt < len(ts)
        next_dose = np.where(has, dose[np.minimum(nxt, len(ts) - 1)], 0)
        gap = np.where(has, ts[np.minimum(nxt, len(ts) - 1)] - times, 14.0)
        within = (ts[None, :] > times[:, None]) & (ts[None, :] <= times[:, None] + 1)
        columns.extend([next_dose, gap, within @ dose, (within * (ts[None, :] - times[:, None] < 1)).sum(1)])
    return np.column_stack(columns)


class Batch:
    """Quadrature grid and feed bookkeeping for one batch."""

    def __init__(self, rows, measured, times):
        self.t0 = measured.elapsed_time.iloc[0]
        first = measured.iloc[0]
        self.x0 = np.array([first.vcd, first.tcd - first.vcd, first.substrate, first.byproduct]) * first.volume
        self.v0 = first.volume
        end = max(times.max(), self.t0 + STEP)
        self.grid = np.arange(self.t0, end + STEP, STEP)
        self.times = times
        self.at = np.minimum(np.searchsorted(self.grid, times), len(self.grid) - 1)
        self.basis = BASIS(self.grid)
        co = feed_totals(rows, np.r_[self.t0, times])
        feed = co[1:] - co[0]
        self.feed = feed[:, :2]
        self.all_feeds = feed
        self.volume = self.v0 + feed.sum(axis=1)
        self.time_integral = PRIMITIVE(times) - PRIMITIVE(self.t0)
        on_grid = feed_totals(rows, self.grid)
        self.grid_feed = (on_grid - on_grid[0]) / FEED_UNITS

    def viable(self, cg, shrink=(0.0, 1.0), beta=None):
        # Effective starting cells: log x = a + b log(measured VCD), times volume.
        start = np.exp(shrink[0] + shrink[1] * np.log(self.x0[0] / self.v0)) * self.v0
        growth = (PRIMITIVE(self.grid) - PRIMITIVE(self.t0)) @ cg
        if beta is not None and len(beta):
            # Cumulative feed exposure shifts log viable cells (integrated growth effect).
            growth = growth + self.grid_feed[:, list(GROWTH_FEEDS)] @ beta
        return start * np.exp(growth)

    def cell_integral(self, xv):
        weighted = self.basis * xv[:, None]
        cumulative = np.vstack([np.zeros(SIZE), np.cumsum((weighted[1:] + weighted[:-1]) / 2 * STEP, axis=0)])
        return cumulative[self.at]


def fit_kinetic(batches, y, scale, relative=False):
    """Viable cells grow at a shared specific rate g(t); dead cells, substrate and
    byproduct change through a specific (per viable cell) and an absolute time rate."""
    weights = [1 / np.sqrt(len(b.times)) for b in batches]
    # relative=True divides each error by the measured value (fits percentage error, as MAPE scores).
    denominators = [np.where(relative, yy, scale) for yy in y]
    penalty = np.diff(np.eye(SIZE), n=2, axis=0)

    log_start = np.log([b.x0[0] / b.v0 for b in batches])
    centre = log_start.mean()

    nb = len(GROWTH_FEEDS)

    def unpack(theta):
        cg, beta, (a, b) = theta[:SIZE], theta[SIZE:SIZE + nb], theta[SIZE + nb:]
        if not SHRINK:
            a, b = 0.0, 1.0
        return cg, beta, (centre * (1 - b) + a, b)

    def growth_residuals(theta):
        cg, beta, shrink = unpack(theta)
        out = [w * (b.viable(cg, shrink, beta)[b.at] / b.volume - yy[:, 0]) / d[:, 0]
               for b, yy, w, d in zip(batches, y, weights, denominators)]
        return np.concatenate(out + [0.3 * penalty @ cg, RIDGE_BETA * beta])

    theta = least_squares(growth_residuals, np.r_[np.full(SIZE, .3), np.zeros(nb), 0, 1], method="trf").x
    cg, beta, shrink = unpack(theta)
    fit_kinetic.last_shrink, fit_kinetic.last_beta = shrink, beta
    cell = [b.cell_integral(b.viable(cg, shrink, beta)) for b in batches]
    def linear_design(b, c, k):
        # Specific (per viable cell) rate, absolute time rate, feed strengths for substrate,
        # and a learned blend of the measured start amount with a population start.
        if FEEDS_EVERYWHERE:
            columns = [c, b.time_integral, b.all_feeds]
        else:
            columns = [c, b.time_integral] + ([b.feed] if k == 2 else [])
        start = [np.full((len(b.times), 1), b.x0[k]), np.full((len(b.times), 1), b.v0)]
        if not SHRINK_ALL:
            start = []
        return np.column_stack(columns + start)

    coefficients = []
    for k in (1, 2, 3):
        rows, targets = [], []
        for b, c, yy, w, d in zip(batches, cell, y, weights, denominators):
            design = linear_design(b, c, k) / b.volume[:, None]
            observed = yy[:, 1] - yy[:, 0] if k == 1 else yy[:, k]
            offset = 0 if SHRINK_ALL else b.x0[k] / b.volume
            dk = d[:, k][:, None]
            rows.append(w * design / dk); targets.append(w * (observed - offset) / d[:, k])
        width = rows[0].shape[1]
        smooth = np.zeros((2 * len(penalty), width))
        smooth[:len(penalty), :SIZE] = penalty * .3
        smooth[len(penalty):, SIZE:2 * SIZE] = penalty * .3
        design = np.vstack(rows + [smooth])
        target = np.concatenate(targets + [np.zeros(len(smooth))])
        coefficients.append(np.linalg.lstsq(design, target, rcond=None)[0])
    fit_kinetic.last_start = [coef[-2] for coef in coefficients] if SHRINK_ALL else None

    def predict(b):
        xv = b.viable(cg, shrink, beta)
        c = b.cell_integral(xv)
        amounts = [xv[b.at]]
        for k, coef in zip((1, 2, 3), coefficients):
            amounts.append(linear_design(b, c, k) @ coef + (0 if SHRINK_ALL else b.x0[k]))
        conc = np.column_stack(amounts) / b.volume[:, None]
        conc[:, 1] += conc[:, 0]
        return conc

    # Exposed for warm-starting the Monod and PINN models in pinn_trials.py.
    predict.params = {"cg": cg, "beta": beta, "shrink": shrink, "coefficients": coefficients}
    return predict


def physical(values):
    values = np.maximum(values, 0)
    values[:, 1] = np.maximum(values[:, 1], values[:, 0])
    return values


def metrics(prediction, indices, y, scale):
    errors = [prediction[i] - y[i] for i in indices]
    mse = np.mean([np.mean((error / scale)**2) for error in errors])
    return np.sqrt(mse), np.sqrt(np.mean(np.vstack(errors)**2, axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    groups, _ = load_batches()
    split = pd.read_csv(HERE / "team_benchmark_split.csv").set_index("batch_id").split
    ids = [rows.batch_id.iloc[0] for rows, _ in groups]
    fit = np.array([i for i, batch in enumerate(ids) if split[batch] == "fit"])
    benchmark = np.array([i for i, batch in enumerate(ids) if split[batch] == "benchmark"])
    times = [measured.elapsed_time.to_numpy()[1:] for _, measured in groups]
    y = [measured[TARGETS].to_numpy()[1:] for _, measured in groups]
    batches = [Batch(*group, times[i]) for i, group in enumerate(groups)]
    standard = [features(*group, times[i]) for i, group in enumerate(groups)]
    ahead = [np.column_stack([standard[i], lookahead(groups[i][0], times[i])]) for i in range(len(groups))]

    def run(train, evaluate):
        scale = np.vstack([y[i] for i in train]).std(axis=0)
        ode = fit_shared_ode([groups[i] for i in train])
        kinetic = fit_kinetic([batches[i] for i in train], [y[i] for i in train], scale)
        bases = {
            "ODE": {i: ode(*groups[i], times[i]) for i in [*train, *evaluate]},
            "Kinetic": {i: kinetic(batches[i]) for i in [*train, *evaluate]},
        }
        predictions = {"Shared ODE": bases["ODE"], "Kinetic ODE": bases["Kinetic"]}
        for title, matrix in [("standard", standard), ("look-ahead", ahead)]:
            x_train = np.vstack([matrix[i] for i in train])
            forest = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, max_features=.75,
                                           random_state=21, n_jobs=4)
            forest.fit(x_train, np.vstack([y[i] for i in train]) / scale)
            predictions[f"Forest · {title}"] = {i: forest.predict(matrix[i]) * scale for i in evaluate}
            for base_name, base in bases.items():
                residual = np.vstack([y[i] - base[i] for i in train]) / scale
                tree = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=2, max_features=.5,
                                           random_state=21, n_jobs=4).fit(x_train, residual)
                name = f"{base_name} + ExtraTrees · {title}"
                predictions[name] = {i: base[i] + tree.predict(matrix[i]) * scale for i in evaluate}
                predictions[f"Blend: forest + {name}"] = {
                    i: .5 * (predictions[name][i] + predictions[f"Forest · {title}"][i]) for i in evaluate}
        predictions["Blend: both bases · look-ahead"] = {
            i: (predictions["Forest · look-ahead"][i] + predictions["ODE + ExtraTrees · look-ahead"][i]
                + predictions["Kinetic + ExtraTrees · look-ahead"][i]) / 3 for i in evaluate}
        predictions["Blend: both bases · standard"] = {
            i: (predictions["Forest · standard"][i] + predictions["ODE + ExtraTrees · standard"][i]
                + predictions["Kinetic + ExtraTrees · standard"][i]) / 3 for i in evaluate}
        return scale, {name: {i: physical(values[i].copy()) for i in evaluate}
                       for name, values in predictions.items()}

    rows = []
    for repeat in range(args.repeats):
        for fold, (a, b) in enumerate(KFold(5, shuffle=True, random_state=72 + repeat).split(fit), 1):
            scale, predicted = run(fit[a], fit[b])
            for name, values in predicted.items():
                score, per_target = metrics(values, fit[b], y, scale)
                rows.append({"repeat": repeat + 1, "fold": fold, "model": name,
                             "normalized_rmse": score, **dict(zip(TARGETS, per_target))})
        pd.DataFrame(rows).to_csv(HERE / "kinetic_cv_folds.csv", index=False)
        print(f"Completed repeat {repeat + 1}/{args.repeats}", flush=True)
    cv = pd.DataFrame(rows)
    summary = cv.groupby("model").normalized_rmse.agg(cv_mean="mean", cv_fold_sd="std")
    scale, predictions = run(fit, benchmark)
    export = []
    for model, values in predictions.items():
        score, per_target = metrics(values, benchmark, y, scale)
        summary.loc[model, "team_benchmark"] = score
        summary.loc[model, TARGETS] = per_target
        for i, estimate in values.items():
            export.extend({"model": model, "batch_id": ids[i], "elapsed_time": t,
                           **dict(zip(TARGETS, p))} for t, p in zip(times[i], estimate))
    summary.sort_values("cv_mean").to_csv(HERE / "kinetic_summary.csv")
    pd.DataFrame(export).to_csv(HERE / "kinetic_benchmark_predictions.csv", index=False)
    print(summary.sort_values("cv_mean").round(4).to_string())


if __name__ == "__main__":
    main()
