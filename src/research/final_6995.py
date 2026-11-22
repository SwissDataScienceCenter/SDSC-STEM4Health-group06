"""Rebuild the Team 6 submission that scored 6.995 (attempt_v4third_cells.csv; previous best 7.008 used v4 weight 1/4).

VCD, TCD   = (C21^2 * v4)^(1/3): C21.csv (teammate ensemble) plus the 10-rotation kinetic v4.
Substrate  = C21.csv unchanged.
Byproduct  = geometric mean of six members: C11's B1, B2, C; concentration curves K4, K6; amount curve A6
             (log of byproduct x volume, divided by the measured volume). Then:
             1. day-13 values x exp(-0.0582)   (median day-13 over-prediction in team CV)
             2. every value x exp(-0.5 * offset_b), where offset_b is a batch-level over-prediction predicted
                from the pH-controller base volume (C11 volume-profile increments dv3, dv5).
The two corrections are estimated from out-of-fold predictions in byproduct_kit/oof_with_amount_curves.csv.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "byproduct_kit"))
import final_7026 as F7026  # noqa: E402
import kit  # noqa: E402

TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
OOF = HERE / "byproduct_kit/oof_with_amount_curves.csv"
W_C21 = 2  # C21 weight against v4 (weight 1) for VCD/TCD; 3 gave 7.008, 2 gives 6.995


def amount_curve_member(key):
    """A6: B-spline (knots every 2 days) of log(byproduct x volume) per labeled batch, coefficients
    ridge-regressed (alpha 10) on batch inputs; prediction divided by the measured test volume."""
    raw, X = kit.RAW, kit.batch_inputs()
    amt = raw.dropna(subset=["byproduct", "volume"])[["batch_id", "elapsed_time", "byproduct", "volume"]]
    amt["y"] = amt.byproduct * amt.volume
    test = sorted(raw.loc[raw.to_predict, "batch_id"].unique())
    labeled = sorted(set(amt.batch_id) - set(test))
    K = np.r_[[0] * 4, [2, 4, 6, 8, 10, 12], [13.5] * 4]; bas = BSpline(K, np.eye(len(K) - 4), 3)
    ta = amt[amt.batch_id.isin(labeled)]
    pooled = np.linalg.lstsq(bas(ta.elapsed_time.to_numpy()), np.log(ta.y.to_numpy()), rcond=None)[0]; nb = len(pooled); C = {}
    for b in labeled:
        g = amt[amt.batch_id == b]; Bm = bas(g.elapsed_time.to_numpy())
        C[b] = np.linalg.lstsq(np.vstack([Bm, np.sqrt(0.3) * np.eye(nb)]), np.r_[np.log(g.y.to_numpy()), np.sqrt(0.3) * pooled], rcond=None)[0]
    C = pd.DataFrame(C).T
    coef = pd.DataFrame(Ridge(alpha=10).fit(X.loc[labeled].to_numpy(), C.loc[labeled].to_numpy()).predict(X.loc[test].to_numpy()), index=test)
    tv = raw[raw.to_predict][["batch_id", "elapsed_time", "volume"]].sort_values(["batch_id", "elapsed_time"]).reset_index(drop=True)
    assert (tv.batch_id.values == key.batch_id.values).all() and np.allclose(tv.elapsed_time.values, key.elapsed_time.values)
    return np.array([np.exp(bas(t) @ coef.loc[b].to_numpy()) / v for b, t, v in zip(tv.batch_id, tv.elapsed_time, tv.volume)])


def corrections():
    """Day-13 shift and the base-volume batch-offset model, from out-of-fold predictions."""
    import run_c11_local as R
    T = pd.read_csv(OOF); T["day"] = T.elapsed_time.round()
    e = np.log(kit.geo(T, kit.BEST + ["A6"]) / T.y)
    day13_shift = float(np.median(e[T.day == 13]))
    off = e.groupby([T.split, T.batch_id]).mean().groupby(level=1).mean()
    data = R.dh.Data(R.DATA / "bioreaction_remaining.csv"); R.mf.add_mape_features(data)
    B = data.feat[["dv3", "dv5"]]; B = (B - B.mean()) / B.std()
    model = Ridge(alpha=10).fit(B.loc[off.index].to_numpy(), off.to_numpy())
    test = sorted(kit.RAW.loc[kit.RAW.to_predict, "batch_id"].unique())
    return day13_shift, pd.Series(model.predict(B.loc[test].to_numpy()), index=test)


def build(out_name="final_6995.csv"):
    c21 = pd.read_csv(HERE / "C21.csv", dtype={"batch_id": str, "elapsed_time": str})
    v4 = F7026.v4_rotated()
    out = c21.copy()
    for c in ("vcd", "tcd"):
        out[c] = np.exp((W_C21 * np.log(c21[c].astype(float)) + np.log(v4[c].astype(float))) / (W_C21 + 1))
    out["tcd"] = np.maximum(out.tcd, out.vcd)
    key = F7026.byproduct_members()
    key["A6"] = amount_curve_member(key)
    day13_shift, offset = corrections()
    key["byproduct"] = np.exp(np.log(key[["B1", "B2", "C", "K4", "K6", "A6"]].clip(lower=1e-3)).mean(axis=1))
    key.loc[key.elapsed_time.round() == 13, "byproduct"] *= np.exp(-day13_shift)
    key["byproduct"] *= np.exp(-0.5 * key.batch_id.map(offset))
    key["k"] = key.groupby("batch_id").cumcount(); out["k"] = out.groupby("batch_id").cumcount()
    m = out.merge(key[["batch_id", "k", "elapsed_time", "byproduct"]], on=["batch_id", "k"], suffixes=("_old", ""),
                  validate="one_to_one", sort=False)
    assert len(m) == 156 and (m.elapsed_time_old.astype(float) - m.elapsed_time).abs().max() < 1e-6
    final = m[["batch_id", "elapsed_time_old", *TARGETS]].rename(columns={"elapsed_time_old": "elapsed_time"})
    v = final[TARGETS].astype(float).to_numpy()
    assert np.isfinite(v).all() and (v > 0).all() and (v[:, 1] >= v[:, 0]).all()
    final.to_csv(HERE / out_name, index=False)
    print(f"day-13 shift {day13_shift:.4f}; base-volume offsets (%):", (100 * (np.exp(-0.5 * offset) - 1)).round(1).to_dict())
    return final


def compare(path_a="final_6995.csv", path_b="attempt_v4third_cells.csv"):
    a, b = pd.read_csv(HERE / path_a), pd.read_csv(HERE / path_b)
    return (100 * (a[TARGETS] - b[TARGETS]).abs() / b[TARGETS]).max()


if __name__ == "__main__":
    build()
    print("max % difference vs submitted file:", compare().round(6).to_dict())
