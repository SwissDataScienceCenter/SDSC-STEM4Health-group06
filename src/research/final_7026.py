"""Rebuild the Team 6 submission that scored 7.026 (attempt_best_day13_shift.csv).

VCD, TCD  = (C21^3 * v4)^(1/4): C21.csv (teammate ensemble) plus the 10-rotation v4 as a fourth member.
Substrate = C21.csv unchanged.
Byproduct = geometric mean of C11's members B1, B2, C and two whole-curve members (knots 2.5 and
            2 days apart, inputs include the substrate-controller bits), then day-13 values x exp(-0.0594).
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DAY13_SHIFT = 0.0594   # median day-13 log over-prediction of the byproduct ensemble in team CV (Trial W)
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]


def v4_rotated(force=False):
    """Average of 10 v4 models, each trained on 9/10 of the 49 labeled batches (~10 min)."""
    path = HERE / "official_candidate_v4_rotated.csv"
    if force or not path.exists():
        import export_v4_rotated
        export_v4_rotated.main()
    return pd.read_csv(path, dtype={"batch_id": str, "elapsed_time": str})


def byproduct_members():
    """C11 byproduct members B1, B2, C and curve members K4, K6, fitted on all 49 labeled batches."""
    import curve_member as CM
    import byproduct_late_trials as LT
    import run_c11_local as R
    data = R.dh.Data(R.DATA / "bioreaction_remaining.csv")
    cols = R.mf.add_mape_features(data)
    mem, _ = LT.members(data, cols, data.labeled, data.to_predict, bolus=False, drop_outliers=False)
    key = mem["B1"].sort_values(["batch_id", "elapsed_time"]).reset_index(drop=True)[["batch_id", "elapsed_time"]]
    for m, x in mem.items():
        key[m] = x.sort_values(["batch_id", "elapsed_time"]).byproduct.to_numpy()
    X = CM.inputs()
    times = {b: g.elapsed_time.to_numpy() for b, g in key.groupby("batch_id")}
    for name, knots in (("K4", [2.5, 5, 7.5, 10]), ("K6", [2, 4, 6, 8, 10, 12])):
        p = CM.predict("byproduct", knots, sorted(data.labeled), data.to_predict, times, X)
        key[name] = [p[b][list(times[b]).index(t)] for b, t in zip(key.batch_id, key.elapsed_time)]
    return key


def build(out_name="final_7026.csv"):
    c21 = pd.read_csv(HERE / "C21.csv", dtype={"batch_id": str, "elapsed_time": str})
    v4 = v4_rotated()
    assert (c21[["batch_id", "elapsed_time"]].values == v4[["batch_id", "elapsed_time"]].values).all()
    out = c21.copy()
    for c in ("vcd", "tcd"):
        out[c] = np.exp((3 * np.log(c21[c].astype(float)) + np.log(v4[c].astype(float))) / 4)
    out["tcd"] = np.maximum(out.tcd, out.vcd)
    key = byproduct_members()
    key["byproduct"] = np.exp(np.log(key[["B1", "B2", "C", "K4", "K6"]].clip(lower=1e-3)).mean(axis=1))
    key["k"] = key.groupby("batch_id").cumcount()
    out["k"] = out.groupby("batch_id").cumcount()
    m = out.merge(key[["batch_id", "k", "elapsed_time", "byproduct"]], on=["batch_id", "k"], suffixes=("_old", ""),
                  validate="one_to_one", sort=False)
    assert len(m) == 156 and (m.elapsed_time_old.astype(float) - m.elapsed_time).abs().max() < 1e-6
    day13 = m.elapsed_time.round() == 13
    m.loc[day13, "byproduct"] *= np.exp(-DAY13_SHIFT)
    final = m[["batch_id", "elapsed_time_old", *TARGETS]].rename(columns={"elapsed_time_old": "elapsed_time"})
    v = final[TARGETS].astype(float).to_numpy()
    assert np.isfinite(v).all() and (v > 0).all() and (v[:, 1] >= v[:, 0]).all() and day13.sum() == 12
    final.to_csv(HERE / out_name, index=False)
    return final


def compare(path_a, path_b="attempt_best_day13_shift.csv"):
    a, b = pd.read_csv(HERE / path_a), pd.read_csv(HERE / path_b)
    return (100 * (a[TARGETS] - b[TARGETS]).abs() / b[TARGETS]).max()


if __name__ == "__main__":
    build()
    print("max % difference vs submitted file:", compare("final_7026.csv").round(6).to_dict())
