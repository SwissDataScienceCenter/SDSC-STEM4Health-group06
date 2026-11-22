"""Run the team's C11 per-target MAPE ensemble locally, and a variant with the 10-rotation v4.

C11's library (notebooks/bioreactor_C11/src) is used unchanged; only its Renku data path is
redirected to the local input-data folder. The variant replaces C11's substrate member
"D: kinetic v4" (one v4 fit) with the average of 10 v4 fits, each trained on 9/10 of the training
batches (Trial N: better than a single v4 in 13 of 15 CV folds, mostly on substrate).
Usage: python run_c11_local.py [c11|c11rot] [benchmark|submission]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # export folder root
sys.path.insert(0, str(HERE.parent / "c11"))  # team C11 library, imported as package `src`
import src.data_handling as dh  # noqa: E402
DATA = ROOT / "input-data/Project 1 - DTB"
dh.DATA_DIR = DATA
import src.evaluate as ev  # noqa: E402
ev.DATA_DIR = DATA
import src.kinetic_v4 as kv4  # noqa: E402
import src.learned_rates as lr  # noqa: E402
import src.mape_features as mf  # noqa: E402
import src.mape_models as mm  # noqa: E402
import src.models as mdl  # noqa: E402


class V4Rotated(kv4.V4Model):
    """Average of 10 v4 models, each fitted without one rotating tenth of the training batches."""
    name = "Julio v4, 10 rotations"

    def fit(self, data, ids):
        ids = np.array(ids)
        self.ps = [kv4.fit_v4([self.tl.groups[b] for b in ids[keep]])
                   for keep, _ in KFold(10, shuffle=True, random_state=11).split(ids)]
        return self

    def predict(self, data, ids):
        req = pd.concat([pd.DataFrame({"batch_id": b, "elapsed_time": data.pk[b]["t_pred"]}) for b in ids], ignore_index=True)
        preds = [kv4.predict_rows(p, self.tl, req) for p in self.ps]
        out = preds[0].copy()
        out[kv4.TARGETS] = kv4.physical(np.mean([p[kv4.TARGETS].to_numpy() for p in preds], axis=0))
        return out


def make(variant):
    """C11 exactly as in bioreactor_C11_mape.ipynb, optionally with the rotated v4."""
    B, S, D = mdl.BATCH_FEATS, mdl.SHAPE_FEATS, mdl.DAY_FEATS
    SN, FB = mf.SUB_NEXT, mf.B_NEXT
    LA, VR, DV = COLS["LA"], COLS["VR"], COLS["DV"]
    F_SN = {c: mdl.FEATS_SHAPE[c] + (SN if c == "substrate" else []) for c in dh.TARGET_COLS}
    F_V = {c: mdl.FEATS_SHAPE[c] + VR + (SN if c == "substrate" else []) for c in dh.TARGET_COLS}
    F_VDV = {c: F_V[c] + DV for c in dh.TARGET_COLS}
    T_COMPACT = {c: B + S + D + SN + FB for c in dh.TARGET_COLS}
    T_ALL = {c: B + S + D + LA for c in dh.TARGET_COLS}
    T_DV = {c: B + S + D + LA + VR + DV for c in dh.TARGET_COLS}
    makers = {"A: learned rates": lambda: mm.LogCorrection(lr.LearnedRates(batch=True), F_SN, "A"),
              "B: mean + volume": lambda: mm.LogCorrection(mdl.MeanCurve(), F_V, "B1"),
              "B: mean + volume profile": lambda: mm.LogCorrection(mdl.MeanCurve(), F_VDV, "B2"),
              "C: ExtraTrees compact": lambda: mm.TreeLog(T_COMPACT),
              "C: forest all feeds": lambda: mm.TreeLog(T_ALL, "rf", 3, 0.75, 200),
              "C: ExtraTrees + volume profile": lambda: mm.TreeLog(T_DV),
              "D: kinetic v4": (lambda: V4Rotated()) if variant == "c11rot" else (lambda: kv4.V4Model())}
    plan = {"vcd": ["A: learned rates", "C: ExtraTrees compact", "C: forest all feeds"],
            "tcd": ["A: learned rates", "C: ExtraTrees compact", "C: forest all feeds"],
            "substrate": ["A: learned rates", "B: mean + volume profile", "D: kinetic v4"],
            "byproduct": ["B: mean + volume profile", "C: ExtraTrees + volume profile", "B: mean + volume"]}
    return lambda: mm.PerTargetEnsemble(makers, plan, variant)


if __name__ == "__main__":
    variant, task = sys.argv[1], sys.argv[2]
    data = dh.Data(DATA / "bioreaction_remaining.csv")
    COLS = mf.add_mape_features(data)
    if task == "benchmark":
        res = ev.benchmark_mape(make(variant), data)
        print(variant, "benchmark MAPE:", res["mape"].round(4).to_dict())
        res["pred"].to_csv(HERE / f"benchmark_predictions_{variant}_local.csv", index=False)
        train, test = ev.split_benchmark(data)
        members = res["model"].member_predictions(data, test)
        print(pd.DataFrame({n: mm.mape(p, data, test) for n, p in members.items()}).T.round(3).to_string())
    else:
        sub = ev.write_submission(make(variant), data, HERE / f"submission_{variant}_local.csv")
        print(f"Wrote submission_{variant}_local.csv: {len(sub)} rows, negatives {(sub[dh.TARGET_COLS] < 0).sum().sum()}, "
              f"TCD<VCD {(sub.tcd < sub.vcd).sum()}")
