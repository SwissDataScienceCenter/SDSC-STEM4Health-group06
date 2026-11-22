"""Team cross-validation: out-of-fold (OOF) predictions and the nested ablation of the byproduct steps.

Protocol: the 49 training batches are split into 5 folds (KFold, shuffled, seeds 0 and 1 -> two splits).
Each member was fitted on 4 folds and predicted the 5th, so every training sample has an out-of-fold
prediction per split. The table is cached in byproduct_kit/oof_with_amount_curves.csv (refitting all
members takes hours). Every correction below is also learned *inside* the folds (nested), so the
held-out fold never informs its own correction.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import KIT
from .metrics import per_batch

C11 = ["B1", "B2", "C"]
CURVES = ["K4", "K6"]
AMOUNT = ["A6"]


def load_oof():
    T = pd.read_csv(KIT / "oof_with_amount_curves.csv")
    T["day"] = T.elapsed_time.round().astype(int)
    return T


def geo(T, members):
    """Geometric mean of members = arithmetic mean in log space."""
    return np.exp(np.log(T[members].clip(lower=1e-3)).mean(axis=1)).to_numpy()


def day13_shift_nested(T, pred):
    """Remove the median day-13 log over-prediction, learned on the other folds of the same split."""
    out = pred.copy()
    err = np.log(pred / T.y.to_numpy())
    d13 = (T.day == 13).to_numpy()
    for (s, f), idx in T.groupby(["split", "fold"]).groups.items():
        train = d13 & (T.split == s).to_numpy() & (T.fold != f).to_numpy()
        test = d13 & (T.split == s).to_numpy() & (T.fold == f).to_numpy()
        out[test] *= np.exp(-np.median(err[train]))
    return out


def batch_offsets(T, pred):
    """Each batch's mean log over-prediction, per split."""
    e = pd.Series(np.log(pred / T.y.to_numpy()))
    return e.groupby([T.split.values, T.batch_id.values]).mean()


def base_offset_nested(T, pred, base, features=("interval 3", "interval 5"), alpha=10.0, strength=0.5, raw_pred=None):
    """Batch-level correction: ridge of the batch offset on base-volume features, fitted on the other folds,
    applied as pred * exp(-strength * predicted offset). `raw_pred` (default: pred) defines the offsets."""
    out = pred.copy()
    off = batch_offsets(T, pred if raw_pred is None else raw_pred)
    B = base[list(features)]
    B = (B - B.mean()) / B.std()
    for (s, f), _ in T.groupby(["split", "fold"]).groups.items():
        rows = T[(T.split == s)]
        tr_b = rows.loc[rows.fold != f, "batch_id"].unique()
        te_b = rows.loc[rows.fold == f, "batch_id"].unique()
        y = off.loc[s].loc[tr_b]
        model = Ridge(alpha=alpha).fit(B.loc[tr_b].to_numpy(), y.to_numpy())
        p = pd.Series(model.predict(B.loc[te_b].to_numpy()), index=te_b)
        m = ((T.split == s) & T.batch_id.isin(te_b)).to_numpy()
        out[m] *= np.exp(-strength * T.loc[m, "batch_id"].map(p).to_numpy())
    return out


def ablation(T, base):
    """CV MAPE after each step of the byproduct model, with a paired comparison to the previous step."""
    steps = {}
    steps["C11 members (B1, B2, C)"] = geo(T, C11)
    steps["+ 2 whole-curve members (K4, K6)"] = geo(T, C11 + CURVES)
    steps["+ day-13 correction"] = day13_shift_nested(T, geo(T, C11 + CURVES))
    raw6 = geo(T, C11 + CURVES + AMOUNT)
    steps["+ amount curve (A6)"] = day13_shift_nested(T, raw6)
    steps["+ base-volume batch offset (final)"] = base_offset_nested(T, steps["+ amount curve (A6)"], base, raw_pred=raw6)
    rows, prev = [], None
    for name, p in steps.items():
        a = per_batch(p, T)
        r = {"step": name, "CV MAPE": a.mean()}
        if prev is not None:
            d = a - prev
            r.update({"change": d.mean(), "± SE": d.std() / np.sqrt(len(d)), "% batches better": 100 * (d < 0).mean()})
        rows.append(r); prev = a
    return pd.DataFrame(rows).set_index("step"), steps


def variance_decomposition(T, pred):
    """Share of the log-error variance that is a constant offset per batch, and the MAPE an oracle would get
    if it knew each batch's offset."""
    e = pd.Series(np.log(pred / T.y.to_numpy()))
    key = [T.split.values, T.batch_id.values]
    off = e.groupby(key).transform("mean")
    share = 1 - (e - off).var() / e.var()
    oracle = pred * np.exp(-off.to_numpy())
    return share, per_batch(oracle, T).mean()


def leaderboard_noise(T, pred_a, pred_b, n_batches=12, n_draws=5000, seed=0):
    """How much a MAPE difference between two models varies when only `n_batches` batches are scored, as
    on the leaderboard: standard deviation of the difference over random sets of training batches."""
    d = (per_batch(pred_a, T) - per_batch(pred_b, T)).groupby(level=1).mean().to_numpy()
    rng = np.random.default_rng(seed)
    draws = [rng.choice(d, n_batches, replace=False).mean() for _ in range(n_draws)]
    return float(np.std(draws))
