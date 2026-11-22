"""Shared byproduct kit. Usage (from src/research):  import sys; sys.path.insert(0, 'byproduct_kit'); import kit
OOF table: one row per (split, fold, batch, sample) for the 49 labeled batches, team CV folds
(KFold(5, shuffle=True, random_state=split) over sorted batch ids, split in {0,1}). Columns:
  y = measured byproduct; members: B1, B2, C (C11 members), K4, K6 (whole-curve members), PINN, V4.
CURRENT BEST byproduct = geometric mean of B1, B2, C, K4, K6  (CV 8.658; leaderboard byproduct 8.164).
test_members.csv: the same members predicted for the 12 test batches (batch_id, elapsed_time in sample order).
best_submission(): the current best leaderboard file (7.043); a new attempt must only change its byproduct.
"""
from pathlib import Path
import numpy as np, pandas as pd
K = Path(__file__).resolve().parent
RAW = pd.read_csv(K.parents[2] / "input-data/Project 1 - DTB/bioreaction_remaining.csv")
MEMBERS = ["B1", "B2", "C", "K4", "K6", "PINN", "V4"]
BEST = ["B1", "B2", "C", "K4", "K6"]


def oof():
    return pd.read_csv(K / "oof_byproduct.csv")


def test_members():
    return pd.read_csv(K / "test_members.csv")


def batch_inputs():
    return pd.read_csv(K / "batch_inputs.csv", index_col=0)


def geo(df, members, weights=None):
    w = np.ones(len(members)) if weights is None else np.asarray(weights, float)
    return np.exp(sum(wi * np.log(df[m].clip(lower=1e-3)) for wi, m in zip(w, members)) / w.sum())


def score(T, pred, label="candidate"):
    """Paired per-batch comparison against the current best. T: oof() rows; pred: array aligned with T."""
    ape = lambda p: 100 * (np.asarray(p) - T.y.to_numpy()).__abs__() / T.y.to_numpy()
    a = pd.Series(ape(pred)).groupby([T.split.values, T.batch_id.values]).mean()
    b = pd.Series(ape(geo(T, BEST))).groupby([T.split.values, T.batch_id.values]).mean()
    d = a - b
    res = {"label": label, "cv_mape": a.mean(), "best_cv": b.mean(), "diff": d.mean(), "se": d.std() / np.sqrt(len(d)),
           "batches_better_pct": 100 * (d < 0).mean()}
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
    return res


def best_submission():
    return pd.read_csv(K.parent / "attempt_v4weighted_C_cells_only.csv", dtype={"batch_id": str, "elapsed_time": str})


def write_attempt(byproduct_values, name):
    """byproduct_values: array in the row order of test_members() (sorted by batch_id, elapsed_time)."""
    tm = test_members(); tm["new"] = np.asarray(byproduct_values); tm["k"] = tm.groupby("batch_id").cumcount()
    sub = best_submission(); sub["k"] = sub.groupby("batch_id").cumcount()
    m = sub.merge(tm[["batch_id", "k", "elapsed_time", "new"]], on=["batch_id", "k"], suffixes=("", "_t"), validate="one_to_one", sort=False)
    assert len(m) == 156 and (m.elapsed_time.astype(float) - m.elapsed_time_t).abs().max() < 1e-6
    m["byproduct"] = m.new
    out = m[["batch_id", "elapsed_time", "vcd", "tcd", "substrate", "byproduct"]]
    v = out[["vcd", "tcd", "substrate", "byproduct"]].astype(float).to_numpy()
    assert np.isfinite(v).all() and (v > 0).all() and (v[:, 1] >= v[:, 0]).all()
    path = K.parent / f"attempt_kit_{name}.csv"; out.to_csv(path, index=False); return path
