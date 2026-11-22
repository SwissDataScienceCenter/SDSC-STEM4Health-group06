"""Physics-based byproduct members, evaluated inside the submitted byproduct ensemble (C11 + PINN).

Step 1 (this file, variant "deviation"): byproduct mass balance with the pH-controller base sensor,
    dP/dt = pi(t) Xv + alpha d(delta_b)/dt - k(t) Xv P_c + c_D dF_D/dt
  pi(t): typical production per viable cell; k(t) > 0: first-order consumption per viable cell;
  delta_b(t): the batch's cleaned base volume minus the median base trajectory of the training
  batches, so alpha converts extra base (extra acid neutralized) into extra byproduct. delta_b averages
  to zero across batches, so pi(t) cannot absorb it. Variant "alpha35" fixes alpha at the mass-balance
  estimate; "no sensor" drops it.
Step 2 (variant "switch"): consumption additionally switched by the controller-conditioned substrate.

Evaluation: team protocol (5 folds x 2 random splits over all 49 labeled batches) plus the fixed
benchmark. C11's byproduct members (B1, B2, C) and the PINN are fitted once per fold and cached in
byproduct_cache/, so later steps only fit their new member. Every combination is a geometric mean.
"""
import argparse
import copy
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from sklearn.model_selection import KFold

import byproduct_trials as BT
import kinetic_trials as K
import mape_trials as M
import substrate_trials as ST
from pooled_ode_fit import TARGETS

HERE = Path(__file__).resolve().parent
CACHE = HERE / "byproduct_cache"
CACHE.mkdir(exist_ok=True)
GRID, STEP, STRIDE = BT.GRID, BT.STEP, BT.STRIDE
P = 3
FLOOR = 1e-3


# --------------------------------------------------------------------------- physics member
class DeviationModel:
    """Params: pi spline (SIZE), log k spline (SIZE), alpha, c_D, log Ks."""

    def __init__(self, xv, sc, V, ddev, dFD, p0, idx, alpha="free", switch=False):
        self.xv, self.sc, self.V, self.ddev, self.dFD, self.p0, self.idx = xv, sc, V, ddev, dFD, p0, idx
        self.alpha, self.switch = alpha, switch
        self.basis = K.BASIS(GRID)

    def unpack(self, th):
        n = K.SIZE
        alpha = th[2 * n] if self.alpha == "free" else (35.0 if self.alpha == "35" else 0.0)
        return th[:n], th[n:2 * n], alpha, th[2 * n + 1], np.exp(th[2 * n + 2])

    def simulate(self, th, batches):
        pi, logk, alpha, c_d, ks = self.unpack(th)
        prod, k = self.basis @ pi, np.exp(self.basis @ logk)
        xv, V = self.xv[batches], self.V[batches]
        sw = self.sc[batches] / (ks + self.sc[batches]) if self.switch else np.ones_like(xv)
        add = alpha * self.ddev[batches] + c_d * self.dFD[batches]
        p = self.p0[batches].astype(float).copy()
        out = np.empty((len(batches), len(GRID)))
        out[:, 0] = p / V[:, 0]
        for n in range(len(GRID) - 1):
            c = np.maximum(p / V[:, n], FLOOR)
            p = p + STEP * xv[:, n] * (prod[n] - k[n] * c * sw[:, n]) + add[:, n]
            p = np.maximum(p, FLOOR * V[:, n + 1])
            out[:, n + 1] = p / V[:, n + 1]
        return out

    def residuals(self, th, batches, y):
        sim = self.simulate(th, batches)
        res = [(np.log(sim[j, self.idx[b]]) - np.log(y[b][:, P])) / np.sqrt(len(y[b])) for j, b in enumerate(batches)]
        pi, logk, *_ = self.unpack(th)
        return np.concatenate(res + [3 * np.diff(pi, n=2), 0.3 * np.diff(logk, n=2)])

    def fit(self, batches, y):
        n = K.SIZE
        th0 = np.r_[np.full(n, 0.05), np.full(n, np.log(0.01)), 35.0, 0.0, np.log(0.5)]
        lo = np.r_[np.full(2 * n, -np.inf), 0.0, 0.0, np.log(1e-2)]
        hi = np.r_[np.full(2 * n, np.inf), 300.0, 400.0, np.log(20)]
        self.th = least_squares(self.residuals, th0, args=(batches, y), bounds=(lo, hi), method="trf", x_scale="jac").x
        return self

    def predict(self, batches):
        sim = self.simulate(self.th, batches)
        return {b: sim[j, self.idx[b]] for j, b in enumerate(batches)}


# --------------------------------------------------------------------------- cached members
def c11_byproduct(train_ids, eval_ids):
    """C11's three byproduct members (B1, B2, C) fitted on train_ids; predictions on eval_ids."""
    import run_c11_local as R
    data = STATE["c11_data"]
    R.COLS = STATE["c11_cols"]
    mdl, mm, mf, dh = R.mdl, R.mm, R.mf, R.dh
    B, S, D = mdl.BATCH_FEATS, mdl.SHAPE_FEATS, mdl.DAY_FEATS
    LA, VR, DV = R.COLS["LA"], R.COLS["VR"], R.COLS["DV"]
    F_V = {c: mdl.FEATS_SHAPE[c] + VR + (mf.SUB_NEXT if c == "substrate" else []) for c in dh.TARGET_COLS}
    F_VDV = {c: F_V[c] + DV for c in dh.TARGET_COLS}
    T_DV = {c: B + S + D + LA + VR + DV for c in dh.TARGET_COLS}
    members = {"B1": mm.LogCorrection(mdl.MeanCurve(), F_V, "B1"), "B2": mm.LogCorrection(mdl.MeanCurve(), F_VDV, "B2"),
               "C": mm.TreeLog(T_DV)}
    out = {}
    for name, m in members.items():
        p = m.fit(data, train_ids).predict(data, eval_ids)
        out[name] = p[["batch_id", "elapsed_time", "byproduct"]]
    return out


def pinn_byproduct(train, evaluate):
    import pinn_trials as PT
    T = STATE["pinn_T"]
    y = M.STATE["y"]
    scale = np.vstack([y[i] for i in train]).std(axis=0)
    kin = K.fit_kinetic([M.STATE["batches"][i] for i in train], [y[i] for i in train], scale)
    centre = np.log([T.x0[i, 0] / T.v0[i] for i in train]).mean()
    monod = PT.fit_monod(T, list(train), y, scale, kin.params, centre)
    physics_set = list(train) + [i for i in range(len(T.v0)) if i not in set(train)]
    pinn = PT.fit_pinn(T, list(train), physics_set, y, scale, copy.deepcopy(monod), 1000, 10.0)
    return {i: K.physical(PT.pinn_predict(pinn, T, i))[:, P] for i in evaluate}


# --------------------------------------------------------------------------- harness
STATE = {}


def setup():
    BT.setup()
    import pinn_trials as PT
    import run_c11_local as R
    PT.setup(1000, [10.0])
    STATE["pinn_T"] = PT.STATE["T"]
    data = R.dh.Data(R.DATA / "bioreaction_remaining.csv")
    STATE["c11_cols"] = R.mf.add_mape_features(data)
    STATE["c11_data"] = data


def physics_members(train, evaluate, variants):
    St, S2 = M.STATE, BT.STATE
    groups, times, y, batches = St["groups"], St["times"], St["y"], St["batches"]
    data, all_times, vres = S2["data"], S2["all_times"], S2["vres"]
    both = [*train, *evaluate]
    scale = np.vstack([y[i] for i in train]).std(axis=0)
    kin = K.fit_kinetic([batches[i] for i in train], [y[i] for i in train], scale, relative=True)
    xv_full = np.stack([data.viable(kin.params, b) for b in range(data.n)])
    sc_full = np.zeros_like(xv_full)
    if any(v.get("switch") for v in variants.values()):
        sub = ST.SubstrateModel(data, xv_full, train).fit(list(train), y).calibrate(list(train), y)
        sc_full[both] = BT.substrate_grid(sub, both)
    xv, sc = xv_full[:, ::STRIDE], sc_full[:, ::STRIDE]
    per_sample = BT.sample_volume([vres[i] for i in train], [all_times[i] for i in train])
    base = {i: np.interp(GRID, all_times[i], BT.clean_base(vres[i], all_times[i], per_sample)) for i in both}
    typical = np.median([base[i] for i in train], axis=0)                    # median base trajectory (training)
    ddev = np.zeros_like(xv)
    for i in both:
        ddev[i, 1:] = np.diff(base[i] - typical)
    ddev = ddev[:, 1:]
    dFD = np.diff(data.F[:, ::STRIDE, 3], axis=1)
    V = data.V[:, ::STRIDE]
    idx = [np.minimum(np.floor(t / STEP + 1e-9).astype(int), len(GRID) - 1) for t in times]
    p0 = np.array([m.iloc[0].byproduct * m.iloc[0].volume for _, m in groups])
    out, info = {}, {}
    for name, kw in variants.items():
        model = DeviationModel(xv, sc, V, ddev, dFD, p0, idx, **kw).fit(list(train), y)
        pred = model.predict(list(evaluate))
        out[name] = pred
        _, _, alpha, c_d, ks = model.unpack(model.th)
        info[name] = {"alpha": alpha, "c_D": c_d, "Ks": ks}
    return out, info


def cached(split, fold, member, compute):
    path = CACHE / f"{split}_{fold}_{member}.csv"
    if path.exists():
        return pd.read_csv(path)
    frame = compute()
    frame.to_csv(path, index=False)
    return frame


def to_frame(pred, evaluate):
    ids, times = STATE["ids"], M.STATE["times"]
    return pd.concat([pd.DataFrame({"batch_id": ids[i], "elapsed_time": times[i], "byproduct": pred[i]}) for i in evaluate])


def job(args):
    split, fold, train, evaluate, variants = args
    torch.set_num_threads(1)
    ids = STATE["ids"]
    frames = {}
    c11 = None
    for member in ("B1", "B2", "C"):
        path = CACHE / f"{split}_{fold}_{member}.csv"
        if not path.exists():
            c11 = c11 or c11_byproduct([ids[i] for i in train], [ids[i] for i in evaluate])
            c11[member].to_csv(path, index=False)
        frames[member] = pd.read_csv(path)
    frames["PINN"] = cached(split, fold, "PINN", lambda: to_frame(pinn_byproduct(train, evaluate), evaluate))
    preds, info = physics_members(np.array(train), np.array(evaluate), variants) if variants else ({}, {})
    for name, pred in preds.items():
        frames[name] = to_frame(pred, evaluate)
    return split, fold, evaluate, frames, info


def score(frames, evaluate, combos):
    ids, y, times = STATE["ids"], M.STATE["y"], M.STATE["times"]
    truth = pd.concat([pd.DataFrame({"batch_id": ids[i], "elapsed_time": times[i], "y": y[i][:, P]}) for i in evaluate])
    truth["elapsed_time"] = truth.elapsed_time.round(9)
    logs = {}
    for name, f in frames.items():
        f = f.copy(); f["elapsed_time"] = f.elapsed_time.round(9)
        logs[name] = np.log(np.maximum(truth.merge(f, on=["batch_id", "elapsed_time"], validate="one_to_one").byproduct.to_numpy(), FLOOR))
    late = truth.elapsed_time.to_numpy().round() >= 8
    rows = []
    for combo, members in combos.items():
        pred = np.exp(np.mean([logs[m] for m in members], axis=0))
        ape = 100 * np.abs(pred - truth.y.to_numpy()) / truth.y.to_numpy()
        rows.append({"combo": combo, "mape": ape.mean(), "late": ape[late].mean()})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", default="1")
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    setup()
    groups = M.STATE["groups"]
    STATE["ids"] = ids = [r.batch_id.iloc[0] for r, _ in groups]
    split = pd.read_csv(HERE / "team_benchmark_split.csv").set_index("batch_id").split
    fit = [i for i, b in enumerate(ids) if split[b] == "fit"]
    bench = [i for i, b in enumerate(ids) if split[b] == "benchmark"]
    everything = np.arange(len(groups))
    variants = {"1": {"phys: deviation sensor": dict(alpha="free"), "phys: alpha=35": dict(alpha="35"),
                      "phys: no sensor": dict(alpha="0")},
                "2": {"phys: deviation sensor": dict(alpha="free"),
                      "phys: sensor + substrate switch": dict(alpha="free", switch=True)}}[args.step]
    jobs = [("bench", 0, fit, bench, variants)]
    jobs += [(f"s{s}", f, list(everything[a]), list(everything[b]), variants) for s in range(2)
             for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=s).split(everything), 1)]
    c11 = ["B1", "B2", "C"]
    combos = {"C11": c11, "PINN": ["PINN"], "C11 + PINN (submitted)": c11 + ["PINN"]}
    for v in variants:
        combos[f"{v} alone"] = [v]
        combos[f"C11 + PINN + {v}"] = c11 + ["PINN", v]
    rows, infos = [], []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for split_name, fold, evaluate, frames, info in pool.imap_unordered(job, jobs):
            for r in score(frames, evaluate, combos):
                rows.append({"split": split_name, "fold": fold, **r})
            for v, d in info.items():
                infos.append({"split": split_name, "fold": fold, "variant": v, **d})
            print(f"done {split_name} {fold}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(HERE / f"byproduct_physics_step{args.step}_folds.csv", index=False)
    cv = d[d.split != "bench"]
    piv = cv.pivot_table(index=["split", "fold"], columns="combo", values="mape")
    summary = cv.groupby("combo")[["mape", "late"]].mean().rename(columns={"mape": "cv_mape", "late": "cv_late"})
    ref = piv["C11 + PINN (submitted)"]
    summary["vs_submitted"] = [(piv[c] - ref).mean() for c in summary.index]
    summary["folds_better"] = [int((piv[c] < ref).sum()) for c in summary.index]
    b = d[d.split == "bench"].set_index("combo")
    summary["bench_mape"] = b.mape; summary["bench_late"] = b.late
    summary.sort_values("cv_mape").to_csv(HERE / f"byproduct_physics_step{args.step}_summary.csv")
    print(pd.DataFrame(infos).groupby("variant")[["alpha", "c_D", "Ks"]].agg(["mean", "std"]).round(2).to_string())
    print(summary.sort_values("cv_mape").round(3).to_string())


if __name__ == "__main__":
    main()
