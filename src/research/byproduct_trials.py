"""Trial O: a byproduct mass-balance model driven by the pH-controller volume sensor.

Physics (amounts = concentration x volume, explicit Euler on a 0.02-day grid):
    dP/dt = alpha * dBase/dt + pi(t) Xv - k(t) Xv * P_c * S_c/(Ks + S_c) + c_D dF_D/dt
- Base(t): base volume added by the pH controller. Measured volume minus initial volume minus feeds,
  plus a constant sample volume per sample taken, made non-decreasing by isotonic regression
  (base can only be added). The sample volume is estimated on the training batches from intervals
  after day 7, when acid production has stopped. Volume is part of the official test inputs.
- pi(t): production not seen by the controller (shared spline); k(t) > 0: consumption per viable
  cell (shared log-spline), first order in byproduct (a saturating form fitted to zero order and
  collapsed late values) and modulated by substrate.
- Xv: viable cells from the relative-error kinetic ODE. S_c: substrate concentration from the
  controller-conditioned substrate model of Trial M (posterior-mean uptake deviation).
Fitted on log error. Exact-MAPE ExtraTrees with volume features then correct the remaining error.

Evaluated with the team's C11 protocol (5 folds x 2 random splits over all 49 labeled batches) and on
the fixed benchmark, against v4's byproduct (exact-MAPE forest + kinetic ODE with exact-MAPE ExtraTrees).
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

import kinetic_trials as K
import mape_trials as M
import substrate_trials as ST
from pooled_ode_fit import TARGETS, feed_totals

HERE = Path(__file__).resolve().parent
STRIDE = 2                         # simulate byproduct on every 2nd substrate-grid point (0.04 d)
GRID, STEP = ST.GRID[::STRIDE], ST.STEP * STRIDE
P = 3
FLOOR = 1e-3


# --------------------------------------------------------------------------- volume sensor
def volume_residual(rows, sample_times, sample_volumes):
    """Measured volume minus initial volume minus feeds added before each sample."""
    fed = feed_totals(rows, sample_times - 1e-9).sum(axis=1)
    return sample_volumes - sample_volumes[0] - fed


def sample_volume(train_vres, train_times):
    """Volume removed per sample, from intervals after day 7 (no base added then)."""
    drops = [-np.diff(v)[t[:-1] >= 7] for v, t in zip(train_vres, train_times)]
    return float(np.median(np.concatenate(drops)))


def clean_base(vres, times, per_sample):
    """Base volume added: residual + sampling losses, made non-decreasing and anchored at 0."""
    raw = vres + per_sample * np.arange(len(vres))
    w = np.ones(len(raw)); w[0] = 1e3
    iso = IsotonicRegression(increasing=True).fit(times, raw, sample_weight=w)
    return iso.predict(times) - iso.predict(times[:1])


def volume_features(base_at, vres_at, times):
    """Per-query volume features: cleaned base, raw residual, base added in the last day, final base."""
    recent = base_at - np.interp(times - 1, times, base_at, left=0)
    return np.column_stack([base_at, vres_at, recent, np.full(len(times), base_at[-1])])


# --------------------------------------------------------------------------- byproduct model
class ByproductModel:
    """Params: pi spline (SIZE), log k spline (SIZE), alpha, c_D, log Kp, log Ks."""

    def __init__(self, xv, sc, V, dbase, dFD, p0, idx, use_base=True, use_substrate=True):
        self.xv, self.sc, self.V, self.dbase, self.dFD, self.p0, self.idx = xv, sc, V, dbase, dFD, p0, idx
        self.use_base, self.use_substrate = use_base, use_substrate
        self.basis = K.BASIS(GRID)

    def unpack(self, th):
        n = K.SIZE
        return th[:n], th[n:2 * n], th[2 * n], th[2 * n + 1], np.exp(th[2 * n + 2]), np.exp(th[2 * n + 3])

    def simulate(self, th, batches):
        pi, logk, alpha, c_d, kp, ks = self.unpack(th)
        prod = self.basis @ pi
        k = np.exp(self.basis @ logk)
        xv, sc, V = self.xv[batches], self.sc[batches], self.V[batches]
        add = (alpha * self.dbase[batches] if self.use_base else 0) + c_d * self.dFD[batches]
        s_term = sc / (ks + sc) if self.use_substrate else np.ones_like(sc)
        p = self.p0[batches].astype(float).copy()
        out = np.empty((len(batches), len(GRID)))
        out[:, 0] = p / V[:, 0]
        for n in range(len(GRID) - 1):
            c = np.maximum(p / V[:, n], FLOOR)
            p = p + STEP * xv[:, n] * (prod[n] - k[n] * c * s_term[:, n]) + add[:, n]
            p = np.maximum(p, FLOOR * V[:, n + 1])
            out[:, n + 1] = p / V[:, n + 1]
        return out

    def residuals(self, th, batches, y):
        sim = self.simulate(th, batches)
        res = [(np.log(sim[j, self.idx[b]]) - np.log(y[b][:, P])) / np.sqrt(len(y[b])) for j, b in enumerate(batches)]
        pi, logk, *_ = self.unpack(th)
        return np.concatenate(res + [0.3 * np.diff(pi, n=2) * 10, 0.3 * np.diff(logk, n=2)])

    def fit(self, batches, y):
        n = K.SIZE
        th0 = np.r_[np.full(n, 0.05), np.full(n, np.log(0.01)), 30.0 if self.use_base else 0.0, 0.0, np.log(1.0), np.log(0.5)]
        lower = np.r_[np.full(2 * n, -np.inf), 0.0 if self.use_base else -1e-9, -np.inf, -1e-9, np.log(1e-3)]
        upper = np.r_[np.full(2 * n, np.inf), np.inf if self.use_base else 1e-9, np.inf, 1e-9, np.log(50)]
        self.th = least_squares(self.residuals, th0, args=(batches, y), bounds=(lower, upper), method="trf",
                                x_scale="jac").x
        return self

    def predict(self, batches):
        sim = self.simulate(self.th, batches)
        return {b: sim[j, self.idx[b]] for j, b in enumerate(batches)}


def substrate_grid(model, batches):
    """Substrate concentration on the grid with the posterior-mean uptake deviation."""
    table = model.u_table(batches)
    prior = norm.logpdf(ST.U_GRID / model.tau)
    u = np.zeros(len(batches))
    for j, b in enumerate(batches):
        paths = np.stack([table[k][j] for k in range(len(ST.U_GRID))])
        logp = prior.copy()
        for i, low in model.d.bits[b]:
            margin = (np.log(ST.THRESHOLD) - np.log(paths[:, i])) / model.sigma
            logp += norm.logcdf(margin) if low else norm.logcdf(-margin)
        p = np.exp(logp - logp.max()); p /= p.sum()
        u[j] = p @ ST.U_GRID
    return model.simulate(model.th, batches, u=u)


# --------------------------------------------------------------------------- harness
STATE = {}


def setup():
    M.setup()
    groups, times, y = M.STATE["groups"], M.STATE["times"], M.STATE["y"]
    data = ST.SubData(groups, times)
    data.vcd0 = np.array([m.iloc[0].vcd for _, m in groups])
    all_times = [m.elapsed_time.to_numpy() for _, m in groups]
    vres = [volume_residual(r, t, m.volume.to_numpy()) for (r, m), t in zip(groups, all_times)]
    STATE.update(data=data, all_times=all_times, vres=vres)


def run(train, evaluate):
    St, S2 = M.STATE, STATE
    groups, times, y, ahead, batches = St["groups"], St["times"], St["y"], St["ahead"], St["batches"]
    data, all_times, vres = S2["data"], S2["all_times"], S2["vres"]
    both = [*train, *evaluate]
    scale = np.vstack([y[i] for i in train]).std(axis=0)
    kin = K.fit_kinetic([batches[i] for i in train], [y[i] for i in train], scale, relative=True)
    xv = np.stack([data.viable(kin.params, b) for b in range(data.n)])
    sub = ST.SubstrateModel(data, xv, train).fit(list(train), y).calibrate(list(train), y)
    sc = np.zeros_like(xv); sc[both] = substrate_grid(sub, both)
    xv, sc = xv[:, ::STRIDE], sc[:, ::STRIDE]
    per_sample = sample_volume([vres[i] for i in train], [all_times[i] for i in train])
    base = {i: clean_base(vres[i], all_times[i], per_sample) for i in both}
    base_grid = np.zeros_like(xv)
    for i in both:
        base_grid[i] = np.interp(GRID, all_times[i], base[i])
    dbase = np.diff(base_grid, axis=1)
    dFD = np.diff(data.F[:, ::STRIDE, 3], axis=1)
    V = data.V[:, ::STRIDE]
    idx = [np.minimum(np.floor(t / STEP + 1e-9).astype(int), len(GRID) - 1) for t in times]
    p0 = np.array([m.iloc[0].byproduct * m.iloc[0].volume for _, m in groups])
    vol_feat = {i: volume_features(base[i][1:], vres[i][1:], times[i]) for i in both}
    xa = {i: np.column_stack([ahead[i], vol_feat[i]]) for i in both}

    variants = {"physics": dict(), "physics, no base sensor": dict(use_base=False),
                "physics, no substrate coupling": dict(use_substrate=False)}
    phys, info = {}, {}
    for name, kw in variants.items():
        m = ByproductModel(xv, sc, V, dbase, dFD, p0, idx, **kw).fit(list(train), y)
        phys[name] = m.predict(both)
        if name == "physics":
            _, _, alpha, c_d, kp, ks = m.unpack(m.th)
            info = {"alpha": alpha, "c_D": c_d, "Ks": ks, "sample_volume": per_sample}

    yt = np.vstack([y[i] for i in train]); yp = yt[:, P]; w = 1 / yp
    x_old = np.vstack([ahead[i] for i in train]); x_vol = np.vstack([xa[i] for i in train])

    def forest():
        return RandomForestRegressor(n_estimators=150, min_samples_leaf=3, max_features=.75, random_state=21,
                                     n_jobs=1, criterion="absolute_error")

    def extra():
        return ExtraTreesRegressor(n_estimators=150, min_samples_leaf=2, max_features=.5, random_state=21,
                                   n_jobs=1, criterion="absolute_error")

    kin_base = {i: np.maximum(kin(batches[i])[:, P], FLOOR) for i in both}
    R = {}
    f = forest().fit(x_old, yp, sample_weight=w)
    R["member: v4 forest"] = {i: f.predict(ahead[i]) for i in evaluate}
    e = extra().fit(x_old, yp - np.concatenate([kin_base[i] for i in train]), sample_weight=w)
    R["member: v4 kinetic-ET"] = {i: kin_base[i] + e.predict(ahead[i]) for i in evaluate}
    R["v4 byproduct (baseline)"] = {i: (R["member: v4 forest"][i] + R["member: v4 kinetic-ET"][i]) / 2 for i in evaluate}
    fv = forest().fit(x_vol, yp, sample_weight=w)
    R["forest + volume features"] = {i: fv.predict(xa[i]) for i in evaluate}
    ev = extra().fit(x_vol, yp - np.concatenate([kin_base[i] for i in train]), sample_weight=w)
    R["kinetic-ET + volume features"] = {i: kin_base[i] + ev.predict(xa[i]) for i in evaluate}
    for name in variants:
        R[f"{name} alone"] = {i: phys[name][i] for i in evaluate}
        et = extra().fit(x_vol, yp - np.concatenate([phys[name][i] for i in train]), sample_weight=w)
        R[f"{name} + ET"] = {i: phys[name][i] + et.predict(xa[i]) for i in evaluate}
    vf, vk = R["member: v4 forest"], R["member: v4 kinetic-ET"]
    pe = R["physics + ET"]; fvv, kvv = R["forest + volume features"], R["kinetic-ET + volume features"]
    R["blend: v4 + physics-ET (3-way)"] = {i: (vf[i] + vk[i] + pe[i]) / 3 for i in evaluate}
    R["blend: volume forest + volume kinetic-ET"] = {i: (fvv[i] + kvv[i]) / 2 for i in evaluate}
    R["blend: volume forest + volume kinetic-ET + physics-ET"] = {i: (fvv[i] + kvv[i] + pe[i]) / 3 for i in evaluate}
    R["blend: volume forest + physics-ET"] = {i: (fvv[i] + pe[i]) / 2 for i in evaluate}
    R["blend: geometric, volume forest + volume kinetic-ET + physics-ET"] = {
        i: np.exp((np.log(np.maximum(fvv[i], FLOOR)) + np.log(np.maximum(kvv[i], FLOOR)) + np.log(np.maximum(pe[i], FLOOR))) / 3)
        for i in evaluate}
    return {n: {i: np.maximum(v[i], FLOOR) for i in evaluate} for n, v in R.items()}, info


def job(args):
    split, fold, train, evaluate = args
    predicted, info = run(np.array(train), np.array(evaluate))
    y = M.STATE["y"]
    actual = np.concatenate([y[i][:, P] for i in evaluate])
    day = np.concatenate([np.round(M.STATE["times"][i]) for i in evaluate])
    rows = []
    for name, values in predicted.items():
        ape = 100 * np.abs(np.concatenate([values[i] for i in evaluate]) - actual) / actual
        rows.append({"split": split, "fold": fold, "model": name, "byproduct_mape": ape.mean(),
                     "late_mape": ape[day >= 8].mean(), **info})
    return rows, predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=2)
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    setup()
    groups = M.STATE["groups"]
    ids = [r.batch_id.iloc[0] for r, _ in groups]
    split = pd.read_csv(HERE / "team_benchmark_split.csv").set_index("batch_id").split
    fit = [i for i, b in enumerate(ids) if split[b] == "fit"]
    bench = [i for i, b in enumerate(ids) if split[b] == "benchmark"]
    everything = np.arange(len(groups))
    jobs = [("benchmark", 0, fit, bench)]
    jobs += [(s, f, list(everything[a]), list(everything[b])) for s in range(args.splits)
             for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=s).split(everything), 1)]
    rows, bench_rows = [], []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for n, (r, predicted) in enumerate(pool.imap(job, jobs, chunksize=1)):
            if n == 0:
                bench_rows = r
                pd.DataFrame([{"model": m, "batch_id": ids[i], "elapsed_time": t, "byproduct": p}
                              for m, v in predicted.items() for i, est in v.items()
                              for t, p in zip(M.STATE["times"][i], est)]).to_csv(HERE / "byproduct_benchmark_predictions.csv", index=False)
            else:
                rows.extend(r)
                pd.DataFrame(rows).to_csv(HERE / "byproduct_cv_folds.csv", index=False)
                print(f"completed {n}/{len(jobs) - 1} folds", flush=True)
    cv = pd.DataFrame(rows)
    summary = cv.groupby("model").agg(cv_mape=("byproduct_mape", "mean"), cv_late=("late_mape", "mean"))
    piv = cv.pivot_table(index=["split", "fold"], columns="model", values="byproduct_mape")
    for m in summary.index:
        d = piv[m] - piv["v4 byproduct (baseline)"]
        summary.loc[m, "vs_v4"] = d.mean(); summary.loc[m, "folds_better"] = int((d < 0).sum())
    for r in bench_rows:
        summary.loc[r["model"], "bench_mape"] = r["byproduct_mape"]; summary.loc[r["model"], "bench_late"] = r["late_mape"]
    summary.sort_values("cv_mape").to_csv(HERE / "byproduct_summary.csv")
    print(cv[["alpha", "c_D", "Ks", "sample_volume"]].describe().loc[["mean", "std"]].round(3).to_string())
    print(summary.sort_values("cv_mape").round(3).to_string())


if __name__ == "__main__":
    main()
