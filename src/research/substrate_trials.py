"""Trial M: a physics model for substrate, conditioned on the substrate-feed controller.

Mass balance (amounts = concentration x volume, explicit Euler on a 0.02-day grid):
    dS/dt = - q(t) * m_b * Xv(t) * S_c / (Ku + S_c)  +  c_sub dF_sub/dt + c_B dF_B/dt + c_D dF_D/dt
q(t) > 0 is a shared spline (specific uptake per viable cell), m_b = exp(w . z_b + u_b) a batch
uptake multiplier, z_b the batch's starting state and designed feed levels, u_b the part the inputs
do not explain. Xv(t) comes from the relative-error kinetic ODE (kinetic_trials.py). Fitted on
log error, which approximates relative error.

Controller bits: the dedicated substrate feed (about 12 h after the day-11 and day-12 samples, and a
few earlier ones) is dosed at a high level when the preceding sample had S < 0.966 and a low level
otherwise. Each event therefore reveals one bit about the batch. For each batch the posterior over
u_b is prior N(0, tau^2) times a probit likelihood per bit; the forecast is the MAPE-optimal
(1/S-weighted) posterior median. tau and the bit noise sigma are estimated on the training batches.

Compared with v3's substrate (exact-MAPE forest + kinetic ODE with exact-MAPE ExtraTrees) on the
same repeated whole-batch folds, scored with substrate MAPE and the overall mean MAPE.
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

import kinetic_trials as K
import mape_trials as M
from pooled_ode_fit import FEEDS, TARGETS, feed_totals

HERE = Path(__file__).resolve().parent
STEP = 0.02
GRID = np.arange(0, 13.5 + STEP / 2, STEP)
THRESHOLD = 0.966            # S below this at the preceding sample -> high substrate-feed dose
HIGH_DOSE_PER_V = 0.005      # dose / volume separating the two dose levels
U_GRID = np.linspace(-0.6, 0.6, 61)
FLOOR = 1e-3
TAUS = (0.05, 0.1, 0.2)
S = 2                        # substrate column


# --------------------------------------------------------------------------- batch data
class SubData:
    def __init__(self, groups, times):
        n = len(groups)
        self.n = n
        self.F = np.stack([feed_totals(rows, GRID) for rows, _ in groups])            # [n, G, 4]
        self.dF = np.diff(self.F, axis=1)
        first = [m.iloc[0] for _, m in groups]
        self.v0 = np.array([f.volume for f in first])
        self.s0 = np.array([f.substrate for f in first]) * self.v0
        self.V = self.v0[:, None] + self.F.sum(axis=2)
        self.idx = [np.minimum(np.floor(t / STEP + 1e-9).astype(int), len(GRID) - 1) for t in times]
        self.basis = K.BASIS(GRID)
        self.bits = []                                                              # (sample index, low?)
        self.z = []
        for b, (rows, measured) in enumerate(groups):
            t = times[b]
            events = rows.loc[rows.feed_substrate > 0, ["elapsed_time", "feed_substrate"]].to_numpy()
            bits = []
            for te, dose in events:
                before = np.flatnonzero((t < te) & (t > te - 0.75))
                if len(before):
                    k = before[-1]
                    v = self.V[b, self.idx[b][k]]
                    bits.append((k, dose / v > HIGH_DOSE_PER_V))
            self.bits.append(bits)
            f = first[b]
            level = {}
            for feed in ("feed_c", "feed_d"):
                ev = rows.loc[rows[feed] > 0, ["elapsed_time", feed]].to_numpy()
                vol = np.interp(ev[:, 0], GRID, self.V[b]) if len(ev) else np.ones(1)
                level[feed] = np.median(ev[:, 1] / vol) if len(ev) else 0
            fb = rows.feed_b.fillna(0)
            self.z.append([np.log(f.vcd), np.log(f.tcd / f.vcd), f.byproduct, f.substrate, f.volume,
                           fb.sum(), rows.feed_c.fillna(0).sum(), (rows.feed_c > 0).sum(),
                           level["feed_c"] * 1e4, level["feed_d"] * 1e3])
        self.z = np.array(self.z)

    def viable(self, kin_params, b):
        cg, beta, shrink = kin_params["cg"], kin_params["beta"], kin_params["shrink"]
        start = np.exp(shrink[0] + shrink[1] * np.log(groups_x0(self, b))) * self.v0[b]
        growth = (K.PRIMITIVE(GRID) - K.PRIMITIVE(0)) @ cg
        gf = list(K.GROWTH_FEEDS)
        growth = growth + (self.F[b][:, gf] - self.F[b][0, gf]) / K.FEED_UNITS[gf] @ beta
        return start * np.exp(growth)


def groups_x0(data, b):
    return data.vcd0[b]


# --------------------------------------------------------------------------- substrate model
class SubstrateModel:
    """Parameters: log q spline (SIZE), log c_sub, log c_B, log c_D, log Ku, w (z-dim)."""

    def __init__(self, data, xv, train):
        self.d, self.xv = data, xv
        self.zm = data.z[train].mean(0); self.zs = data.z[train].std(0) + 1e-9
        self.nz = data.z.shape[1]

    def unpack(self, th):
        n = K.SIZE
        return th[:n], np.exp(th[n:n + 3]), np.exp(th[n + 3]), th[n + 4:n + 4 + self.nz]

    def simulate(self, th, batches, u=None):
        """Substrate concentration on the grid for the given batches, [len(batches), G]."""
        a, conc, ku, w = self.unpack(th)
        d = self.d
        q = np.exp(d.basis @ a)                                             # [G]
        zz = (d.z[batches] - self.zm) / self.zs
        mult = np.exp(zz @ w + (0 if u is None else u))                     # [B]
        xv = self.xv[batches]                                               # [B, G]
        V = d.V[batches]
        add = d.dF[batches][:, :, [0, 1, 3]] @ conc                         # [B, G-1]
        s = d.s0[batches].astype(float).copy()
        out = np.empty((len(batches), len(GRID)))
        out[:, 0] = s / V[:, 0]
        for n in range(len(GRID) - 1):
            c = np.maximum(s / V[:, n], FLOOR)
            s = s - STEP * q[n] * mult * xv[:, n] * c / (ku + c) + add[:, n]
            s = np.maximum(s, FLOOR * V[:, n + 1])
            out[:, n + 1] = s / V[:, n + 1]
        return out

    def residuals(self, th, train, y):
        sim = self.simulate(th, train)
        res = [np.log(sim[j, self.d.idx[b]]) - np.log(y[b][:, S]) for j, b in enumerate(train)]
        res = [r / np.sqrt(len(r)) for r in res]
        a, _, _, w = self.unpack(th)
        return np.concatenate(res + [0.3 * np.diff(a, n=2), 0.05 * w])

    def fit(self, train, y):
        th0 = np.r_[np.full(K.SIZE, np.log(0.12)), np.log([264, 54, 40]), np.log(0.3), np.zeros(self.nz)]
        self.th = least_squares(self.residuals, th0, args=(train, y), method="trf", x_scale="jac").x
        return self

    def at_samples(self, sim_row, b):
        return sim_row[self.d.idx[b]]

    def u_table(self, batches):
        """Substrate at sample times for every u on U_GRID: [len(U), B] list of arrays (cached)."""
        key = tuple(batches)
        self._cache = getattr(self, "_cache", {})
        if key in self._cache:
            return self._cache[key]
        table = []
        for u in U_GRID:
            sim = self.simulate(self.th, batches, u=u)
            table.append([self.at_samples(sim[j], b) for j, b in enumerate(batches)])
        self._cache[key] = table
        return table

    def calibrate(self, train, y):
        """tau: spread of per-batch best u; sigma: log noise of the bit-carrying samples."""
        table = self.u_table(train)
        best, bit_resid = [], []
        for j, b in enumerate(train):
            sse = [np.sum((np.log(table[k][j]) - np.log(y[b][:, S]))**2) for k in range(len(U_GRID))]
            k = int(np.argmin(sse)); best.append(U_GRID[k])
            for i, _ in self.d.bits[b]:
                bit_resid.append(np.log(y[b][i, S]) - np.log(table[k][j][i]))
        self.tau_raw = float(np.std(best))
        self.tau = max(self.tau_raw, 0.02)
        self.sigma = max(np.std(bit_resid), 0.02)
        return self

    def conditioned(self, batches, tau=None):
        """MAPE-optimal forecast after conditioning u on the controller bits."""
        table = self.u_table(batches)
        out, plain = {}, {}
        prior = norm.logpdf(U_GRID / (tau or self.tau))
        k0 = int(np.argmin(np.abs(U_GRID)))
        for j, b in enumerate(batches):
            paths = np.stack([table[k][j] for k in range(len(U_GRID))])     # [U, T]
            logp = prior.copy()
            for i, low in self.d.bits[b]:
                margin = (np.log(THRESHOLD) - np.log(paths[:, i])) / self.sigma
                logp += norm.logcdf(margin) if low else norm.logcdf(-margin)
            p = np.exp(logp - logp.max()); p /= p.sum()
            est = np.empty(paths.shape[1])
            for i in range(paths.shape[1]):                                 # 1/S-weighted median
                w = p / paths[:, i]; order = np.argsort(paths[:, i]); cw = np.cumsum(w[order])
                est[i] = paths[order, i][np.searchsorted(cw, cw[-1] / 2)]
            out[b] = est
            plain[b] = paths[k0]
        return out, plain


# --------------------------------------------------------------------------- harness
def bit_features(data, b, n):
    """Batch-level controller bits as features: low-substrate flags at the day-11 and day-12 events."""
    flags = [float(low) for _, low in data.bits[b]][-2:]
    flags = [0.5] * (2 - len(flags)) + flags
    return np.tile(flags + [sum(flags)], (n, 1))


def run(train, evaluate):
    St = M.STATE
    y, ahead, batches, times = St["y"], St["ahead"], St["batches"], St["times"]
    data = St["sub"]
    scale = np.vstack([y[i] for i in train]).std(axis=0)
    kin = K.fit_kinetic([batches[i] for i in train], [y[i] for i in train], scale, relative=True)
    xv = np.stack([data.viable(kin.params, b) for b in range(data.n)])
    model = SubstrateModel(data, xv, train).fit(list(train), y).calibrate(list(train), y)
    cond_eval, plain_eval = model.conditioned(list(evaluate))
    cond_train, plain_train = model.conditioned(list(train))
    x = np.vstack([ahead[i] for i in train]); yt = np.vstack([y[i] for i in train])
    xb = np.vstack([np.column_stack([ahead[i], bit_features(data, i, len(y[i]))]) for i in train])
    w = 1 / yt[:, S]

    def forest():
        return RandomForestRegressor(n_estimators=150, min_samples_leaf=3, max_features=.75, random_state=21,
                                     n_jobs=1, criterion="absolute_error")

    def extra():
        return ExtraTreesRegressor(n_estimators=150, min_samples_leaf=2, max_features=.5, random_state=21,
                                   n_jobs=1, criterion="absolute_error")

    kin_base = {i: np.maximum(kin(batches[i]), FLOOR) for i in [*train, *evaluate]}
    P = {}
    # v3 (all four targets, for the overall score) -------------------------------------------
    full_f, full_k = {i: np.zeros((len(times[i]), 4)) for i in evaluate}, {i: np.zeros((len(times[i]), 4)) for i in evaluate}
    for k in range(4):
        wk = 1 / yt[:, k]
        f = forest().fit(x, yt[:, k], sample_weight=wk)
        e = extra().fit(x, yt[:, k] - np.vstack([kin_base[i] for i in train])[:, k], sample_weight=wk)
        for i in evaluate:
            full_f[i][:, k] = f.predict(ahead[i]); full_k[i][:, k] = kin_base[i][:, k] + e.predict(ahead[i])
    v3 = {i: (full_f[i] + full_k[i]) / 2 for i in evaluate}
    sub = lambda d: {i: d[i][:, S] for i in evaluate}
    P["v3 substrate (baseline)"] = sub(v3)
    P["member: v3 forest"] = sub(full_f)
    P["member: v3 kinetic-ET"] = sub(full_k)
    # physics substrate ------------------------------------------------------------------------
    P["physics substrate"] = plain_eval
    P["physics substrate + controller bits"] = cond_eval
    for name, tr, ev in [("physics", plain_train, plain_eval), ("physics+bits", cond_train, cond_eval)]:
        e = extra().fit(x, yt[:, S] - np.concatenate([tr[i] for i in train]), sample_weight=w)
        P[f"{name} + ET"] = {i: ev[i] + e.predict(ahead[i]) for i in evaluate}
        eb = extra().fit(xb, yt[:, S] - np.concatenate([tr[i] for i in train]), sample_weight=w)
        P[f"{name} + ET (bit features)"] = {
            i: ev[i] + eb.predict(np.column_stack([ahead[i], bit_features(data, i, len(times[i]))])) for i in evaluate}
    fb = forest().fit(xb, yt[:, S], sample_weight=w)
    P["forest (bit features)"] = {i: fb.predict(np.column_stack([ahead[i], bit_features(data, i, len(times[i]))]))
                                  for i in evaluate}
    f = P["member: v3 forest"]; k = P["member: v3 kinetic-ET"]
    for name in ["physics+bits + ET", "physics+bits + ET (bit features)", "physics + ET"]:
        p = P[name]
        P[f"blend: forest + {name}"] = {i: (f[i] + p[i]) / 2 for i in evaluate}
        P[f"blend: forest + kinetic-ET + {name}"] = {i: (f[i] + k[i] + p[i]) / 3 for i in evaluate}
    p = P["physics+bits + ET (bit features)"]; fbb = P["forest (bit features)"]
    P["blend: forest(bits) + kinetic-ET + physics+bits+ET(bits)"] = {i: (fbb[i] + k[i] + p[i]) / 3 for i in evaluate}
    P["blend: v3 + physics substrate + bits"] = {i: (v3[i][:, S] + cond_eval[i]) / 2 for i in evaluate}
    for tau in TAUS:
        ce, _ = model.conditioned(list(evaluate), tau)
        ct, _ = model.conditioned(list(train), tau)
        e = extra().fit(x, yt[:, S] - np.concatenate([ct[i] for i in train]), sample_weight=w)
        pe = {i: ce[i] + e.predict(ahead[i]) for i in evaluate}
        P[f"physics+bits tau={tau:g}"] = ce
        P[f"physics+bits tau={tau:g} + ET"] = pe
        P[f"blend: forest + kinetic-ET + physics+bits tau={tau:g} + ET"] = {i: (f[i] + k[i] + pe[i]) / 3 for i in evaluate}
    info = {"tau_raw": model.tau_raw, "tau": model.tau, "sigma": model.sigma, "Ku": model.unpack(model.th)[2],
            **dict(zip(["c_sub", "c_B", "c_D"], model.unpack(model.th)[1]))}
    out = {n: {i: np.maximum(v[i], FLOOR) for i in evaluate} for n, v in P.items()}
    return out, v3, info


def job(args):
    repeat, fold, train, evaluate = args
    predicted, v3, info = run(np.array(train), np.array(evaluate))
    y = M.STATE["y"]
    rows = []
    for name, values in predicted.items():
        ape = 100 * np.abs(np.concatenate([values[i] for i in evaluate]) - np.concatenate([y[i][:, S] for i in evaluate])) \
            / np.concatenate([y[i][:, S] for i in evaluate])
        full = {i: v3[i].copy() for i in evaluate}
        for i in evaluate:
            full[i][:, S] = values[i]
        overall, _ = M.mape_score(full, evaluate, y)
        rows.append({"repeat": repeat, "fold": fold, "model": name, "substrate_mape": ape.mean(),
                     "mean_mape_with_v3_others": overall, **info})
    return rows, predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    M.setup()
    groups, times = M.STATE["groups"], M.STATE["times"]
    data = SubData(groups, times)
    data.vcd0 = np.array([m.iloc[0].vcd for _, m in groups])
    M.STATE["sub"] = data
    split = pd.read_csv(HERE / "team_benchmark_split.csv").set_index("batch_id").split
    ids = [rows.batch_id.iloc[0] for rows, _ in groups]
    fit = np.array([i for i, b in enumerate(ids) if split[b] == "fit"])
    benchmark = np.array([i for i, b in enumerate(ids) if split[b] == "benchmark"])
    jobs = [("benchmark", 0, list(fit), list(benchmark))]
    jobs += [(r + 1, f, list(fit[a]), list(fit[b])) for r in range(args.repeats)
             for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=72 + r).split(fit), 1)]
    rows, bench = [], None
    with mp.get_context("fork").Pool(args.workers) as pool:
        for n, (r, predicted) in enumerate(pool.imap(job, jobs, chunksize=1)):
            if n == 0:
                bench = r
                pd.DataFrame([{"model": m, "batch_id": ids[i], "elapsed_time": t, "substrate": p}
                              for m, v in predicted.items() for i, est in v.items()
                              for t, p in zip(times[i], est)]).to_csv(HERE / "substrate_benchmark_predictions.csv", index=False)
            else:
                rows.extend(r)
                pd.DataFrame(rows).to_csv(HERE / "substrate_cv_folds.csv", index=False)
                print(f"completed {n}/{len(jobs) - 1} folds", flush=True)
    cv = pd.DataFrame(rows)
    summary = cv.groupby("model").agg(cv_sub_mape=("substrate_mape", "mean"),
                                      cv_mean_mape=("mean_mape_with_v3_others", "mean"))
    piv = cv.pivot_table(index=["repeat", "fold"], columns="model", values="substrate_mape")
    for m in summary.index:
        diff = piv[m] - piv["v3 substrate (baseline)"]
        summary.loc[m, "vs_v3"] = diff.mean(); summary.loc[m, "folds_better"] = int((diff < 0).sum())
    for row in bench:
        summary.loc[row["model"], "bench_sub_mape"] = row["substrate_mape"]
        summary.loc[row["model"], "bench_mean_mape"] = row["mean_mape_with_v3_others"]
    summary.sort_values("cv_sub_mape").to_csv(HERE / "substrate_summary.csv")
    print(cv[["tau_raw", "tau", "sigma", "Ku", "c_sub", "c_B", "c_D"]].describe().loc[["mean", "std"]].round(3).to_string())
    print(summary.sort_values("cv_sub_mape").round(3).to_string())


if __name__ == "__main__":
    main()
