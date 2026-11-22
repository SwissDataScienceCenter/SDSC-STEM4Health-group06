"""Whole-curve member for any target: per-batch B-spline fit of log(target) (shrunk toward the pooled
curve), coefficients ridge-regressed on batch inputs (descriptors, base-volume profile, controller bits)."""
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from sklearn.linear_model import Ridge
import byproduct_curve_trials as CT

RAW = CT.L.RAW


def inputs():
    bits = {}
    for b, g in RAW.groupby("batch_id"):
        m = g.dropna(subset=["volume"]); ev = g[g.feed_substrate > 0].sort_values("elapsed_time").tail(2)
        v = np.interp(ev.elapsed_time, m.elapsed_time, m.volume)
        hi = (ev.feed_substrate.to_numpy() / v > 0.005).astype(float); bits[b] = {"hi11": hi[0], "hi12": hi[1]}
    B = pd.DataFrame(bits).T
    return CT.batch_inputs().join((B - B.mean()) / B.std())


def truth(target):
    return RAW.dropna(subset=[target, "volume"])[["batch_id", "elapsed_time", target]].rename(columns={target: "y"})


def predict(target, knots, train, query, times, X, lam=0.3, alpha=10.0):
    """times: dict batch -> array of query times. Returns dict batch -> predictions."""
    tr = truth(target)
    K = np.r_[[0] * 4, knots, [13.5] * 4]; bas = BSpline(K, np.eye(len(K) - 4), 3)
    ta = tr[tr.batch_id.isin(train)]; Bp = bas(ta.elapsed_time.to_numpy())
    pooled = np.linalg.lstsq(Bp, np.log(ta.y.to_numpy()), rcond=None)[0]; nb = Bp.shape[1]; C = {}
    for b in train:
        g = tr[tr.batch_id == b]; Bm = bas(g.elapsed_time.to_numpy())
        C[b] = np.linalg.lstsq(np.vstack([Bm, np.sqrt(lam) * np.eye(nb)]), np.r_[np.log(g.y.to_numpy()), np.sqrt(lam) * pooled], rcond=None)[0]
    C = pd.DataFrame(C).T
    coef = pd.DataFrame(Ridge(alpha=alpha).fit(X.loc[train].to_numpy(), C.loc[train].to_numpy()).predict(X.loc[query].to_numpy()), index=query)
    return {b: np.exp(bas(np.asarray(times[b])) @ coef.loc[b].to_numpy()) for b in query}
