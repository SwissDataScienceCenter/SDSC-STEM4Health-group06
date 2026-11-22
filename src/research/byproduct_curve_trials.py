"""Trial S: whole-curve regression for byproduct (specialist round 2, idea 1; DRSM-style).

Each training batch's log-byproduct trajectory is fitted with a smooth cubic B-spline in time (knots at
days 2.5, 5, 7.5, 10; ridge-shrunk toward the population curve). The spline coefficients are then
predicted from batch inputs known at test time (day-0 state, feed design, base-volume profile) by ridge
regression, so the whole curve shape (peak height and time, decay) shares information across days.
Evaluated alone and as an extra member of C11 / C11 + PINN on the team CV folds (byproduct_cache/).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import byproduct_local_selection as L

HERE = Path(__file__).resolve().parent
KNOTS = np.r_[[0] * 4, 2.5, 5, 7.5, 10, [13.5] * 4]
NB = len(KNOTS) - 4
BASIS = BSpline(KNOTS, np.eye(NB), 3)


def batch_inputs():
    D = L.descriptors()
    raw = L.RAW
    extra = {}
    for b, g in raw.groupby("batch_id"):          # base-volume profile per interval (timing of base addition)
        g = g.sort_values("elapsed_time"); m = g.dropna(subset=["volume"])
        fed = np.array([g.loc[g.elapsed_time < t, ["feed_substrate", "feed_b", "feed_c", "feed_d"]].fillna(0).to_numpy().sum() for t in m.elapsed_time])
        base = m.volume.to_numpy() - m.volume.iloc[0] - fed + 0.05 * np.arange(len(m))
        t = m.elapsed_time.to_numpy()
        extra[b] = {f"base_d{d}": np.interp(d, t, base) for d in (1, 2, 3, 5, 6, 8)}
    E = pd.DataFrame(extra).T
    E = (E - E.mean()) / E.std()
    return D.join(E)


def fit_curves(truth, ids, lam=0.3):
    """Per-batch spline coefficients of log byproduct (including day 0), shrunk toward the pooled curve."""
    t_all = truth[truth.batch_id.isin(ids)]
    Bp = BASIS(t_all.elapsed_time.to_numpy()); pooled = np.linalg.lstsq(Bp, np.log(t_all.y.to_numpy()), rcond=None)[0]
    C = {}
    for b in ids:
        g = truth[truth.batch_id == b]; B = BASIS(g.elapsed_time.to_numpy())
        A = np.vstack([B, np.sqrt(lam) * np.eye(NB)]); r = np.r_[np.log(g.y.to_numpy()), np.sqrt(lam) * pooled]
        C[b] = np.linalg.lstsq(A, r, rcond=None)[0]
    return pd.DataFrame(C).T


def main():
    X = batch_inputs()
    truth = L.RAW.dropna(subset=["byproduct", "volume"])[["batch_id", "elapsed_time", "byproduct"]].rename(columns={"byproduct": "y"})
    labeled = sorted(set(truth.batch_id) - set(L.RAW.loc[L.RAW.to_predict, "batch_id"]), key=lambda s: int(s[5:]))
    ids = np.array([r for r in labeled])
    # fold assignment must match the cache: KFold over load_batches order (batch_id string sort)
    order = np.array(sorted(labeled))
    rows = []
    for alpha in (3.0, 10.0, 30.0):
        for s in (0, 1):
            for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=s).split(order), 1):
                tr, ev = order[a], order[b]
                C = fit_curves(truth, tr)
                model = Ridge(alpha=alpha).fit(X.loc[tr].to_numpy(), C.loc[tr].to_numpy())
                coef = pd.DataFrame(model.predict(X.loc[ev].to_numpy()), index=ev)
                fr = {m: pd.read_csv(HERE / f"byproduct_cache/s{s}_{f}_{m}.csv").assign(et=lambda x: x.elapsed_time.round(9)) for m in ("B1", "B2", "C", "PINN")}
                t = fr["B1"][["batch_id", "et", "elapsed_time"]].copy()
                for m, x in fr.items():
                    t = t.merge(x[["batch_id", "et", "byproduct"]].rename(columns={"byproduct": m}), on=["batch_id", "et"])
                t["curve"] = [np.exp(BASIS(tt) @ coef.loc[bb].to_numpy()) for bb, tt in zip(t.batch_id, t.elapsed_time)]
                yy = truth.assign(et=truth.elapsed_time.round(9))
                t = t.merge(yy[["batch_id", "et", "y"]], on=["batch_id", "et"])
                combos = {"C11": ["B1", "B2", "C"], "curve alone": ["curve"], "C11 + curve": ["B1", "B2", "C", "curve"],
                          "C11 + PINN": ["B1", "B2", "C", "PINN"], "C11 + PINN + curve": ["B1", "B2", "C", "PINN", "curve"]}
                for c, ms in combos.items():
                    p = np.exp(np.log(t[ms].clip(lower=1e-3)).mean(axis=1)); ape = 100 * (p - t.y).abs() / t.y; day = t.elapsed_time.round()
                    rows.append({"alpha": alpha, "split": s, "fold": f, "model": c, "mape": ape.mean(),
                                 "d5_7": ape[(day >= 5) & (day <= 7)].mean(), "d13": ape[day == 13].mean()})
    r = pd.DataFrame(rows); r.to_csv(HERE / "byproduct_curve_folds.csv", index=False)
    for alpha, g in r.groupby("alpha"):
        piv = g.pivot_table(index=["split", "fold"], columns="model", values="mape")
        out = g.groupby("model")[["mape", "d5_7", "d13"]].mean()
        out["vs_C11"] = [(piv[m] - piv["C11"]).mean() for m in out.index]
        out["folds_better_than_C11"] = [int((piv[m] < piv["C11"]).sum()) for m in out.index]
        print(f"== ridge alpha {alpha}"); print(out.sort_values("mape").round(3).to_string())


if __name__ == "__main__":
    main()
