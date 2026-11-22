"""Trial P: end-of-run byproduct inside C11's byproduct ensemble.

Findings behind it (C11 out-of-fold errors, 49 batches):
- Day 13 has 21% APE vs ~10% on other days. Two batches (Batch29, Batch34) triple their byproduct on
  the last day; no input predicts it. Because C11's members average per day, those two points push
  the day-13 forecast of every other batch up by about 7%.
- For the other batches the late level follows the substrate-bolus cycle: byproduct is consumed as
  substrate runs out after each bolus, and the last sample is taken 8-13 h after the last bolus.

Variants of C11's three byproduct members (B1, B2: mean curve + log ridge correction; C: log trees):
- "+bolus": extra per-sample inputs placing each sample on the bolus cycle: hours since the last
  substrate bolus, its dose level (controller bit: high dose = substrate was low), number of boluses
  so far, and the predicted substrate at the sample time from the controller-conditioned substrate
  model (Trial M), fitted on the training batches of the fold.
- "-outliers": training points where byproduct rises sharply late in the run (day >= 11, log rise
  > 0.5 vs the previous sample) are removed from the training measurements of the fold.
- Robust day-13 correction on top: the median day-13 log error of the same ensemble on the other
  folds is removed (leave-fold-out), so the rare risers cannot shift it.
Same folds as byproduct_physics.py (team protocol, 5 folds x 2 splits over 49 batches, plus benchmark);
C11 and PINN predictions come from its cache.
"""
import argparse
import copy
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import byproduct_physics as BP
import kinetic_trials as K
import mape_trials as M
import substrate_trials as ST

HERE = Path(__file__).resolve().parent
CACHE = BP.CACHE
BOLUS = ["h_bolus", "bolus_hi", "n_bolus", "S_pred", "logS_pred"]
RISE = 0.5


def bolus_table(data, ids):
    """Inputs-only bolus-cycle features for every sample after day 0."""
    rows = []
    for b in ids:
        f = data.feeds[(data.feeds.batch_id == b) & (data.feeds.feed_substrate > 0)]
        pk = data.pk[b]
        for t in pk["t_pred"]:
            before = f[f.elapsed_time < t]
            v = np.interp(before.elapsed_time.iloc[-1], pk["volume"].elapsed_time, pk["volume"].volume) if len(before) else 1
            rows.append({"batch_id": b, "elapsed_time": t,
                         "h_bolus": 24 * (t - before.elapsed_time.iloc[-1]) if len(before) else 99.0,
                         "bolus_hi": float(before.feed_substrate.iloc[-1] / v > ST.HIGH_DOSE_PER_V) if len(before) else 0.5,
                         "n_bolus": float(len(before))})
    return pd.DataFrame(rows)


def substrate_prediction(train, evaluate):
    """Controller-conditioned substrate at each sample, from the Trial M model fitted on `train`."""
    St = M.STATE
    y, batches, data = St["y"], St["batches"], BP.BT.STATE["data"]
    scale = np.vstack([y[i] for i in train]).std(axis=0)
    kin = K.fit_kinetic([batches[i] for i in train], [y[i] for i in train], scale, relative=True)
    xv = np.stack([data.viable(kin.params, b) for b in range(data.n)])
    model = ST.SubstrateModel(data, xv, train).fit(list(train), y).calibrate(list(train), y)
    pred = {**model.conditioned(list(train))[0], **model.conditioned(list(evaluate))[0]}
    ids, times = BP.STATE["ids"], M.STATE["times"]
    return pd.concat([pd.DataFrame({"batch_id": ids[i], "elapsed_time": times[i], "S_pred": p}) for i, p in pred.items()])


def risers(data, train_ids):
    """Late sudden byproduct rises in the training batches: (batch_id, elapsed_time) rows to drop."""
    s = data.samples(train_ids).sort_values(["batch_id", "elapsed_time"])
    s = s.assign(rise=np.log(s.byproduct).groupby(s.batch_id).diff())
    hit = s[(s.elapsed_time.round() >= 11) & (s.rise > RISE)]
    return set(zip(hit.batch_id, hit.elapsed_time))


def members(data, cols, train_ids, eval_ids, bolus, drop_outliers):
    import run_c11_local as R
    mdl, mm, mf, dh = R.mdl, R.mm, R.mf, R.dh
    B, S, D = mdl.BATCH_FEATS, mdl.SHAPE_FEATS, mdl.DAY_FEATS
    LA, VR, DV = cols["LA"], cols["VR"], cols["DV"]
    extra = BOLUS if bolus else []
    F_V = {c: mdl.FEATS_SHAPE[c] + VR + (mf.SUB_NEXT if c == "substrate" else []) + (extra if c == "byproduct" else [])
           for c in dh.TARGET_COLS}
    F_VDV = {c: F_V[c] + DV for c in dh.TARGET_COLS}
    T_DV = {c: B + S + D + LA + VR + DV + (extra if c == "byproduct" else []) for c in dh.TARGET_COLS}
    fit_data = data
    dropped = 0
    if drop_outliers:
        bad = risers(data, train_ids)
        fit_data = copy.copy(data)
        keep = [(b, t) not in bad for b, t in zip(data.meas.batch_id, data.meas.elapsed_time)]
        fit_data.meas = data.meas[keep]
        dropped = len(bad)
    out = {}
    for name, m in {"B1": mm.LogCorrection(mdl.MeanCurve(), F_V, "B1"),
                    "B2": mm.LogCorrection(mdl.MeanCurve(), F_VDV, "B2"),
                    "C": mm.TreeLog(T_DV)}.items():
        m.fit(fit_data, train_ids)
        out[name] = m.predict(data, eval_ids)[["batch_id", "elapsed_time", "byproduct"]]
    return out, dropped


def job(args):
    split, fold, train, evaluate = args
    ids = BP.STATE["ids"]
    data = copy.copy(BP.STATE["c11_data"])
    cols = BP.STATE["c11_cols"]
    tr_ids, ev_ids = [ids[i] for i in train], [ids[i] for i in evaluate]
    feat = bolus_table(data, tr_ids + ev_ids).merge(substrate_prediction(np.array(train), np.array(evaluate)),
                                                   on=["batch_id", "elapsed_time"], how="left")
    feat["logS_pred"] = np.log(np.maximum(feat.S_pred, 1e-3))
    data.feat_day = data.feat_day.merge(feat, on=["batch_id", "elapsed_time"], how="left")
    frames = {m: pd.read_csv(CACHE / f"{split}_{fold}_{m}.csv") for m in ("B1", "B2", "C", "PINN")}
    info = {}
    for tag, bolus, drop in [("+bolus", True, False), ("-outliers", False, True), ("+bolus-outliers", True, True)]:
        preds, dropped = members(data, cols, tr_ids, ev_ids, bolus, drop)
        for name, f in preds.items():
            frames[name + tag] = f
        info[tag] = dropped
    return split, fold, evaluate, frames, info


def day13_robust(table, combo):
    """Leave-fold-out median day-13 log error removed, per split."""
    out = []
    for split, g in table.groupby("split"):
        g = g.copy()
        if split != "bench":
            for f in g.fold.unique():
                shift = np.median(g.loc[(g.fold != f) & (g.day == 13), combo + "_err"])
                m = (g.fold == f) & (g.day == 13)
                g.loc[m, combo] = g.loc[m, combo] * np.exp(-shift)
        out.append(g)
    return pd.concat(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    BP.setup()
    groups = M.STATE["groups"]
    BP.STATE["ids"] = ids = [r.batch_id.iloc[0] for r, _ in groups]
    split = pd.read_csv(HERE / "team_benchmark_split.csv").set_index("batch_id").split
    fit = [i for i, b in enumerate(ids) if split[b] == "fit"]
    bench = [i for i, b in enumerate(ids) if split[b] == "benchmark"]
    everything = np.arange(len(groups))
    jobs = [("bench", 0, fit, bench)]
    jobs += [(f"s{s}", f, list(everything[a]), list(everything[b])) for s in range(2)
             for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=s).split(everything), 1)]
    combos = {"C11": ["B1", "B2", "C"], "C11 +bolus": ["B1+bolus", "B2+bolus", "C+bolus"],
              "C11 -outliers": ["B1-outliers", "B2-outliers", "C-outliers"],
              "C11 +bolus -outliers": ["B1+bolus-outliers", "B2+bolus-outliers", "C+bolus-outliers"]}
    y, times = M.STATE["y"], M.STATE["times"]
    tables, dropped = [], []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for split_name, fold, evaluate, frames, info in pool.imap_unordered(job, jobs):
            truth = pd.concat([pd.DataFrame({"batch_id": ids[i], "elapsed_time": times[i], "y": y[i][:, BP.P]}) for i in evaluate])
            truth["et"] = truth.elapsed_time.round(9)
            t = truth.copy()
            for name, f in frames.items():
                f = f.assign(et=f.elapsed_time.round(9))[["batch_id", "et", "byproduct"]].rename(columns={"byproduct": name})
                t = t.merge(f, on=["batch_id", "et"], validate="one_to_one")
            for c, ms in combos.items():
                t[c] = np.exp(np.log(t[ms].clip(lower=1e-3)).mean(axis=1))
            t = t.assign(split=split_name, fold=fold, day=t.elapsed_time.round())
            tables.append(t)
            dropped.append({"split": split_name, "fold": fold, **info})
            print(f"done {split_name} {fold}", flush=True)
    table = pd.concat(tables, ignore_index=True)
    for c in combos:
        table[c + "_err"] = np.log(table[c] / table.y)
    rows = []
    for c in combos:
        for robust in (False, True):
            tt = day13_robust(table, c) if robust else table
            name = c + (" + robust day 13" if robust else "")
            ape = 100 * (tt[c] - tt.y).abs() / tt.y
            for (s, f), a in ape.groupby([tt.split, tt.fold]):
                d = tt.loc[a.index, "day"]
                rows.append({"split": s, "fold": f, "model": name, "mape": a.mean(), "day13": a[d == 13].mean(),
                             "late": a[d >= 8].mean()})
    res = pd.DataFrame(rows)
    res.to_csv(HERE / "byproduct_late_folds.csv", index=False)
    table.to_csv(HERE / "byproduct_late_predictions.csv", index=False)
    cv = res[res.split != "bench"]
    piv = cv.pivot_table(index=["split", "fold"], columns="model", values="mape")
    summary = cv.groupby("model")[["mape", "late", "day13"]].mean()
    summary["vs_C11"] = [(piv[m] - piv["C11"]).mean() for m in summary.index]
    summary["folds_better"] = [int((piv[m] < piv["C11"]).sum()) for m in summary.index]
    for sp in ("s0", "s1"):
        summary[f"vs_C11_{sp}"] = [(piv[m] - piv["C11"]).loc[sp].mean() for m in summary.index]
    b = res[res.split == "bench"].set_index("model")
    summary["bench"] = b.mape
    summary.sort_values("mape").to_csv(HERE / "byproduct_late_summary.csv")
    print("training points removed as late risers per fold:", pd.DataFrame(dropped)["-outliers"].describe()[["min", "max"]].to_dict())
    print(summary.sort_values("mape").round(3).to_string())


if __name__ == "__main__":
    main()
