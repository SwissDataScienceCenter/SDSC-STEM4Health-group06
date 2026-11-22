"""Trial Q: choose the byproduct ensemble per batch from its most similar labeled batches.

Similarity (inputs only, standardized): day-0 VCD, substrate, byproduct, volume; number of substrate
boluses and time of the earliest one; Feed C dose level and event count; Feed B plateau level (days 6-9,
per volume); cumulative base volume (volume minus feeds plus 0.05 per sample) at days 4 and 13.
Candidates are geometric means of cached out-of-fold members (byproduct_cache/, team CV 5 folds x 2
splits). For each labeled batch, the candidate with the lowest mean byproduct MAPE on its k nearest
other labeled batches is applied to it (its own data is never used), and compared with fixed choices.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = pd.read_csv(HERE.parents[1] / "input-data/Project 1 - DTB/bioreaction_remaining.csv")
CANDIDATES = {"C11": {"B1": 1, "B2": 1, "C": 1}, "C11 + PINN": {"B1": 1, "B2": 1, "C": 1, "PINN": 1},
              "C11 + 2 PINN": {"B1": 1, "B2": 1, "C": 1, "PINN": 2}, "B members only": {"B1": 1, "B2": 1},
              "trees + PINN": {"C": 1, "PINN": 1}}


def descriptors():
    rows = {}
    for b, g in RAW.groupby("batch_id"):
        g = g.sort_values("elapsed_time"); m = g.dropna(subset=["volume"])
        V = lambda t: np.interp(t, m.elapsed_time, m.volume)
        sub, fc, fb = g[g.feed_substrate > 0], g[g.feed_c > 0], g[g.feed_b > 0]
        fed = np.array([g.loc[g.elapsed_time < t, ["feed_substrate", "feed_b", "feed_c", "feed_d"]].fillna(0).to_numpy().sum() for t in m.elapsed_time])
        base = m.volume.to_numpy() - m.volume.iloc[0] - fed + 0.05 * np.arange(len(m))
        t = m.elapsed_time.to_numpy()
        f0 = m.iloc[0]
        rows[b] = {"vcd0": f0.vcd, "S0": f0.substrate, "P0": f0.byproduct, "V0": f0.volume,
                   "n_bolus": len(sub), "first_bolus": sub.elapsed_time.min(),
                   "fc_level": np.median(fc.feed_c / V(fc.elapsed_time)) * 1e4, "fc_n": len(fc),
                   "fb_plateau": np.median((fb.feed_b / V(fb.elapsed_time))[(fb.elapsed_time > 5.5) & (fb.elapsed_time < 9.5)]),
                   "base_d4": np.interp(4, t, base), "base_d13": base[-1]}
    D = pd.DataFrame(rows).T.astype(float)
    return (D - D.mean()) / D.std()


def oof_table():
    truth = RAW.dropna(subset=["byproduct", "volume"])[["batch_id", "elapsed_time", "byproduct"]].rename(columns={"byproduct": "y"})
    truth["et"] = truth.elapsed_time.round(9); truth = truth[truth.groupby("batch_id").cumcount() > 0]
    out = []
    for s in (0, 1):
        for f in range(1, 6):
            fr = {m: pd.read_csv(HERE / f"byproduct_cache/s{s}_{f}_{m}.csv").assign(et=lambda x: x.elapsed_time.round(9))
                  .set_index(["batch_id", "et"]).byproduct for m in ("B1", "B2", "C", "PINN")}
            out.append(pd.DataFrame(fr).reset_index().merge(truth, on=["batch_id", "et"]).assign(split=s))
    a = pd.concat(out, ignore_index=True)
    for name, w in CANDIDATES.items():
        logp = sum(wt * np.log(a[m].clip(lower=1e-3)) for m, wt in w.items()) / sum(w.values())
        a[name] = 100 * (np.exp(logp) - a.y).abs() / a.y
    return a


def main(k_values=(5, 8, 12)):
    D = descriptors()
    a = oof_table()
    per = a.groupby(["split", "batch_id"])[list(CANDIDATES)].mean()        # batch-level MAPE per split
    labeled = per.index.get_level_values(1).unique()
    Dl = D.loc[labeled]
    print("fixed choices, byproduct CV MAPE (mean of batch MAPEs, both splits):")
    print(per.mean().round(3).to_string())
    for k in k_values:
        res = []
        for s in (0, 1):
            p = per.loc[s]
            for b in labeled:
                dist = np.linalg.norm(Dl.drop(b).to_numpy() - Dl.loc[b].to_numpy(), axis=1)
                nb = Dl.drop(b).index[np.argsort(dist)[:k]]
                choice = p.loc[nb].mean().idxmin()
                res.append({"split": s, "b": b, "choice": choice, "mape": p.loc[b, choice],
                            **{c: p.loc[b, c] for c in CANDIDATES}})
        r = pd.DataFrame(res)
        print(f"\nlocal selection, k={k}: MAPE {r.mape.mean():.3f}  vs C11 {r['C11'].mean():.3f}, C11+PINN {r['C11 + PINN'].mean():.3f}"
              f" | per split {[round(x, 3) for x in r.groupby('split').mape.mean()]}"
              f" | batches better than C11+PINN {(r.mape < r['C11 + PINN'] - 1e-9).sum()}, worse {(r.mape > r['C11 + PINN'] + 1e-9).sum()} (of {len(r)})")
        print("   choices made:", r.choice.value_counts().to_dict())
    return D, per


if __name__ == "__main__":
    main()
