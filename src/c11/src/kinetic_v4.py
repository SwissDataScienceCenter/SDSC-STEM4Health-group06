"""Kinetic model v4 by Julio (Group 06), reproduced unchanged: kinetic ODE with per-cell rates + MAPE-trained trees, and a substrate mass-balance model conditioned on the substrate-feed controller. A small wrapper gives it the fit/predict interface of the other models."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


KEY = ['batch_id', 'elapsed_time']
TARGETS = ['vcd', 'tcd', 'substrate', 'byproduct']
FEEDS = ['feed_substrate', 'feed_b', 'feed_c', 'feed_d']
class Timeline:
    def __init__(self, source):
        self.source = source
        raw = pd.read_csv(source, dtype={"batch_id": str})
        self.to_predict = raw.loc[raw.to_predict, KEY].copy()
        test = set(self.to_predict.batch_id)
        self.groups = {}
        for batch, rows in raw.groupby("batch_id"):
            rows = rows.sort_values("elapsed_time")
            self.groups[batch] = rows, rows.dropna(subset=[*TARGETS, "volume"])
        self.labeled = [batch for batch in self.groups if batch not in test]


def features(rows, measured, times):
    initial = measured.iloc[0][[*TARGETS, "volume"]].to_numpy(dtype=float)
    columns = [times, *np.broadcast_to(initial[:, None], (len(initial), len(times)))]
    for feed in FEEDS:
        events = rows.loc[rows[feed] > 0, ["elapsed_time", feed]]
        ts, dose = events.to_numpy().T
        cumulative = np.r_[0, dose.cumsum()]
        latest = np.searchsorted(ts, times, side="left")
        columns.extend([latest, cumulative[latest]])
        for window in (.25, 1, 2):
            previous = np.searchsorted(ts, times - window, side="left")
            columns.append(cumulative[latest] - cumulative[previous])
        had_feed = latest > 0
        last_dose = np.zeros(len(times))
        age = np.full(len(times), 14.0)
        last_dose[had_feed] = dose[latest[had_feed]-1]
        age[had_feed] = times[had_feed] - ts[latest[had_feed]-1]
        columns.extend([last_dose, age])
    return np.column_stack(columns)


def lookahead(rows, times):
    """Encode each feed’s next dose and near-future exposure."""
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



def feed_totals(rows, times):
    event_times = rows.elapsed_time.to_numpy()
    doses = rows[FEEDS].fillna(0).to_numpy()
    cumulative = np.vstack([np.zeros(4), doses.cumsum(axis=0)])
    return cumulative[np.searchsorted(event_times, times, side="right")]


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
FLOOR = 1e-3


def physical(values):
    values = np.maximum(values, 0)
    values[:, 1] = np.maximum(values[:, 1], values[:, 0])
    return values


def design(group, times):
    return np.column_stack([features(*group, times), lookahead(group[0], times)])


def fit_v3(groups):
    times = [m.elapsed_time.to_numpy()[1:] for _, m in groups]
    y = [m[TARGETS].to_numpy()[1:] for _, m in groups]
    scale = np.vstack(y).std(axis=0)
    kinetic = fit_kinetic([Batch(*g, t) for g, t in zip(groups, times)], y, scale, relative=True)
    x = np.vstack([design(g, t) for g, t in zip(groups, times)])
    yt = np.vstack(y)
    if not (yt > 0).all():
        raise ValueError("Training MAPE weights require strictly positive targets.")
    base = np.vstack([np.maximum(kinetic(Batch(*g, t)), FLOOR) for g, t in zip(groups, times)])
    forests, trees = [], []
    for k in range(4):
        w = 1 / yt[:, k]
        forests.append(RandomForestRegressor(n_estimators=150, min_samples_leaf=3, max_features=.75, random_state=21,
                                             n_jobs=4, criterion="absolute_error").fit(x, yt[:, k], sample_weight=w))
        trees.append(ExtraTreesRegressor(n_estimators=150, min_samples_leaf=2, max_features=.5, random_state=21,
                                         n_jobs=4, criterion="absolute_error").fit(x, yt[:, k] - base[:, k], sample_weight=w))

    def predict(group, query_times):
        xq = design(group, query_times)
        kinetic_base = np.maximum(kinetic(Batch(*group, query_times)), FLOOR)
        forest_part = np.column_stack([f.predict(xq) for f in forests])
        kinetic_part = kinetic_base + np.column_stack([t.predict(xq) for t in trees])
        return physical((forest_part + kinetic_part) / 2)

    return predict


def predict_rows(predict, data, requested):
    predictions = []
    for batch, keys in requested.sort_values(KEY).groupby("batch_id"):
        rows, measured = data.groups[batch]
        output = keys.copy()
        output[TARGETS] = predict((rows, measured.iloc[:1]), keys.elapsed_time.to_numpy())
        predictions.append(output)
    return pd.concat(predictions, ignore_index=True)

SUB_STEP = 0.02
SUB_GRID = np.arange(0, 13.5 + SUB_STEP / 2, SUB_STEP)
THRESHOLD = 0.966          # substrate below this at the preceding sample -> high substrate-feed dose
HIGH_DOSE_PER_V = 0.005    # dose / volume separating the two dose levels
U_GRID = np.linspace(-0.6, 0.6, 61)
SUB_BASIS = BASIS(SUB_GRID)
S_COL = TARGETS.index("substrate")


class SubBatch:
    """Feed bookkeeping, controller bits and batch descriptors for one batch on the substrate grid."""

    def __init__(self, rows, measured, times):
        first = measured.iloc[0]
        self.F = feed_totals(rows, SUB_GRID)
        self.dF = np.diff(self.F, axis=0)
        self.v0, self.vcd0 = first.volume, first.vcd
        self.s0 = first.substrate * first.volume
        self.V = self.v0 + self.F.sum(axis=1)
        self.idx = np.minimum(np.floor(times / SUB_STEP + 1e-9).astype(int), len(SUB_GRID) - 1)
        self.bits = []
        for te, dose in rows.loc[rows.feed_substrate > 0, ["elapsed_time", "feed_substrate"]].to_numpy():
            before = np.flatnonzero((times < te) & (times > te - 0.75))
            if len(before):
                k = before[-1]
                self.bits.append((k, dose / self.V[self.idx[k]] > HIGH_DOSE_PER_V))
        level = {}
        for feed in ("feed_c", "feed_d"):
            ev = rows.loc[rows[feed] > 0, ["elapsed_time", feed]].to_numpy()
            level[feed] = np.median(ev[:, 1] / np.interp(ev[:, 0], SUB_GRID, self.V)) if len(ev) else 0
        self.z = np.array([np.log(first.vcd), np.log(first.tcd / first.vcd), first.byproduct, first.substrate,
                           first.volume, rows.feed_b.fillna(0).sum(), rows.feed_c.fillna(0).sum(),
                           (rows.feed_c > 0).sum(), level["feed_c"] * 1e4, level["feed_d"] * 1e3])

    def viable(self, params):
        """Viable-cell amount on the grid from the fitted kinetic model."""
        cg, beta, shrink = params["cg"], params["beta"], params["shrink"]
        start = np.exp(shrink[0] + shrink[1] * np.log(self.vcd0)) * self.v0
        growth = (PRIMITIVE(SUB_GRID) - PRIMITIVE(0)) @ cg
        gf = list(GROWTH_FEEDS)
        growth = growth + (self.F[:, gf] - self.F[0, gf]) / FEED_UNITS[gf] @ beta
        return start * np.exp(growth)


class SubstrateModel:
    """Substrate mass balance with Monod uptake, a batch uptake factor and controller conditioning."""

    def __init__(self, kinetic_params, train):
        self.params = kinetic_params
        z = np.stack([b.z for b in train])
        self.zm, self.zs = z.mean(0), z.std(0) + 1e-9
        self.nz = z.shape[1]

    def unpack(self, th):
        n = SIZE
        return th[:n], np.exp(th[n:n + 3]), np.exp(th[n + 3]), th[n + 4:n + 4 + self.nz]

    def simulate(self, th, batches, u=0.0):
        """Substrate concentration on the grid for a list of SubBatch objects, [B, G]."""
        a, conc, ku, w = self.unpack(th)
        q = np.exp(SUB_BASIS @ a)
        mult = np.exp(((np.stack([b.z for b in batches]) - self.zm) / self.zs) @ w + u)
        xv = np.stack([b.xv for b in batches])
        V = np.stack([b.V for b in batches])
        add = np.stack([b.dF[:, [0, 1, 3]] for b in batches]) @ conc
        s = np.array([b.s0 for b in batches], dtype=float)
        out = np.empty((len(batches), len(SUB_GRID)))
        out[:, 0] = s / V[:, 0]
        for n in range(len(SUB_GRID) - 1):
            c = np.maximum(s / V[:, n], FLOOR)
            s = s - SUB_STEP * q[n] * mult * xv[:, n] * c / (ku + c) + add[:, n]
            s = np.maximum(s, FLOOR * V[:, n + 1])
            out[:, n + 1] = s / V[:, n + 1]
        return out

    def residuals(self, th, batches, ys):
        sim = self.simulate(th, batches)
        res = [(np.log(sim[j, b.idx]) - np.log(y)) / np.sqrt(len(y)) for j, (b, y) in enumerate(zip(batches, ys))]
        a, _, _, w = self.unpack(th)
        return np.concatenate(res + [0.3 * np.diff(a, n=2), 0.05 * w])

    def fit(self, batches, ys):
        for b in batches:
            b.xv = b.viable(self.params)
        th0 = np.r_[np.full(SIZE, np.log(0.12)), np.log([264, 54, 40]), np.log(0.3), np.zeros(self.nz)]
        self.th = least_squares(self.residuals, th0, args=(batches, ys), method="trf", x_scale="jac").x
        # tau: spread of each training batch's best-fitting u; sigma: log noise at controller samples.
        table = self.u_table(batches)
        best, bit_resid = [], []
        for j, (b, y) in enumerate(zip(batches, ys)):
            k = int(np.argmin([np.sum((np.log(table[k][j]) - np.log(y))**2) for k in range(len(U_GRID))]))
            best.append(U_GRID[k])
            bit_resid += [np.log(y[i]) - np.log(table[k][j][i]) for i, _ in b.bits]
        self.tau, self.sigma = max(np.std(best), 0.02), max(np.std(bit_resid), 0.02)
        return self

    def u_table(self, batches):
        table = []
        for u in U_GRID:
            sim = self.simulate(self.th, batches, u)
            table.append([sim[j, b.idx] for j, b in enumerate(batches)])
        return table

    def predict(self, batches):
        """1/S-weighted posterior median after conditioning u on each batch's controller bits."""
        for b in batches:
            if not hasattr(b, "xv"):
                b.xv = b.viable(self.params)
        table = self.u_table(batches)
        prior = norm.logpdf(U_GRID / self.tau)
        out = []
        for j, b in enumerate(batches):
            paths = np.stack([table[k][j] for k in range(len(U_GRID))])
            logp = prior.copy()
            for i, low in b.bits:
                margin = (np.log(THRESHOLD) - np.log(paths[:, i])) / self.sigma
                logp += norm.logcdf(margin) if low else norm.logcdf(-margin)
            p = np.exp(logp - logp.max()); p /= p.sum()
            est = np.empty(paths.shape[1])
            for i in range(paths.shape[1]):
                weight = p / paths[:, i]; order = np.argsort(paths[:, i]); cw = np.cumsum(weight[order])
                est[i] = paths[order, i][np.searchsorted(cw, cw[-1] / 2)]
            out.append(est)
        return out
def fit_v4(groups):
    times = [m.elapsed_time.to_numpy()[1:] for _, m in groups]
    y = [m[TARGETS].to_numpy()[1:] for _, m in groups]
    v3_predict = fit_v3(groups)
    scale = np.vstack(y).std(axis=0)
    kinetic = fit_kinetic([Batch(*g, t) for g, t in zip(groups, times)], y, scale, relative=True)
    sub_batches = [SubBatch(*g, t) for g, t in zip(groups, times)]
    sub_y = [yy[:, S_COL] for yy in y]
    sub_model = SubstrateModel(kinetic.params, sub_batches).fit(sub_batches, sub_y)
    fitted = sub_model.predict(sub_batches)
    x = np.vstack([design(g, t) for g, t in zip(groups, times)])
    ys = np.concatenate(sub_y)
    tree = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=2, max_features=.5, random_state=21, n_jobs=4,
                               criterion="absolute_error").fit(x, ys - np.concatenate(fitted), sample_weight=1 / ys)

    def predict(group, query_times):
        values = v3_predict(group, query_times)
        physics = sub_model.predict([SubBatch(*group, query_times)])[0]
        physics = np.maximum(physics + tree.predict(design(group, query_times)), FLOOR)
        values[:, S_COL] = (2 * values[:, S_COL] + physics) / 3
        return physical(values)

    return predict


# wrapper: the v4 model with the same fit(data, ids) / predict(data, ids) interface as the other models
class V4Model:
    name = "Julio v4" # display name

    # read the raw timeline once (v4 uses its own data structure)
    def __init__(self):
        from .data_handling import DATA_DIR # data folder
        self.tl = Timeline(DATA_DIR / "bioreaction_remaining.csv") # v4 data structure

    # fit v4 on the given batches
    def fit(self, data, ids):
        self.p = fit_v4([self.tl.groups[b] for b in ids]) # fitted predictor
        return self

    # predict the requested times of the given batches
    def predict(self, data, ids):
        req = pd.concat([pd.DataFrame({"batch_id": b, "elapsed_time": data.pk[b]["t_pred"]}) for b in ids], ignore_index=True) # requested rows
        return predict_rows(self.p, self.tl, req) # predictions
