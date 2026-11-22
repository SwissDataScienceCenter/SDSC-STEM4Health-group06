"""The final forecast (leaderboard 6.995), step by step.

Thin wrappers around the research code in src/research (final_6995.py, final_7026.py) and the C11
library (src/c11), so the notebook can show each step on its own. `assemble` reproduces the submitted file.
"""
import numpy as np
import pandas as pd

from . import ANALYSIS, TARGETS

SUBMITTED = ANALYSIS / "attempt_v4third_cells.csv"
W_C21 = 2                       # weight of C21 against v4 (weight 1) for VCD and TCD
BYPRODUCT_MEMBERS = ["B1", "B2", "C", "K4", "K6", "A6"]
OFFSET_STRENGTH = 0.5

# Team 6 leaderboard history (mean MAPE over the four targets, 12 test batches)
LEADERBOARD = pd.DataFrame([
    ("v1", "random forest + kinetic ODE", 8.251),
    ("C9", "per-target ensembles + controller features", 7.321),
    ("C21 + curves", "teammate C21 + whole-curve byproduct members", 7.105),
    ("+ v4 cells", "kinetic v4 as a VCD/TCD member (weight 1/4)", 7.043),
    ("+ day 13", "remove day-13 byproduct over-prediction", 7.026),
    ("+ amount curve", "byproduct curve fitted on amount, not concentration", 7.019),
    ("+ base offset", "batch-level byproduct correction from base volume", 7.008),
    ("final", "v4 weight 1/3 for VCD/TCD", 6.995),
], columns=["version", "idea", "leaderboard MAPE"])


def _str_frame(path):
    return pd.read_csv(path, dtype={"batch_id": str, "elapsed_time": str})


def load_c21():
    """Teammate C21 ensemble: forecasts of all four targets for the 156 test samples (an input file)."""
    return _str_frame(ANALYSIS / "C21.csv")


def load_v4():
    """Kinetic model v4, averaged over 10 fits on rotating 90% subsets (cached; refit ~10 min)."""
    import final_7026 as F
    return F.v4_rotated()


def blend_cells(c21, v4, w_c21=W_C21):
    """VCD/TCD: weighted geometric mean (C21^w * v4)^(1/(w+1)); TCD is kept >= VCD."""
    out = c21.copy()
    for c in ("vcd", "tcd"):
        out[c] = np.exp((w_c21 * np.log(c21[c].astype(float)) + np.log(v4[c].astype(float))) / (w_c21 + 1))
    out["tcd"] = np.maximum(out.tcd, out.vcd)
    return out


def byproduct_members():
    """The six byproduct members fitted on all 49 training batches, at the 156 test times."""
    import final_7026 as F
    import final_6995 as F6
    key = F.byproduct_members()
    key["A6"] = F6.amount_curve_member(key)
    return key


def corrections():
    """Day-13 log shift and the predicted base-volume offset of each test batch (from out-of-fold errors)."""
    import final_6995 as F6
    return F6.corrections()


def byproduct_forecast(key, day13_shift, offset):
    """Geometric mean of the members, then the two corrections. Returns the steps as columns."""
    k = key.copy()
    k["ensemble"] = np.exp(np.log(k[BYPRODUCT_MEMBERS].clip(lower=1e-3)).mean(axis=1))
    k["after day-13"] = k.ensemble * np.where(k.elapsed_time.round() == 13, np.exp(-day13_shift), 1.0)
    k["byproduct"] = k["after day-13"] * np.exp(-OFFSET_STRENGTH * k.batch_id.map(offset))
    return k


def assemble(cells, byp):
    """Merge the byproduct forecast into the cell/substrate table in submission order and check it."""
    out = cells.copy()
    k = byp.copy()
    k["k"] = k.groupby("batch_id").cumcount(); out["k"] = out.groupby("batch_id").cumcount()
    m = out.merge(k[["batch_id", "k", "elapsed_time", "byproduct"]], on=["batch_id", "k"], suffixes=("_old", ""),
                  validate="one_to_one", sort=False)
    assert len(m) == 156 and (m.elapsed_time_old.astype(float) - m.elapsed_time).abs().max() < 1e-6
    final = m[["batch_id", "elapsed_time_old", *TARGETS]].rename(columns={"elapsed_time_old": "elapsed_time"})
    v = final[TARGETS].astype(float).to_numpy()
    assert np.isfinite(v).all() and (v > 0).all() and (v[:, 1] >= v[:, 0]).all()
    return final


def compare_to_submitted(final, path=SUBMITTED):
    """Largest relative difference (%) per target between `final` and the submitted file."""
    sub = pd.read_csv(path)
    return (100 * (final[TARGETS].astype(float).to_numpy() - sub[TARGETS].to_numpy()).__abs__()
            / sub[TARGETS].to_numpy()).max(axis=0)


def submit(path, team="Team 6"):
    """Send a file to the leaderboard. Every call is a real, counted attempt."""
    import requests
    base = "https://leaderboard-project1-runai-innovation-thibaut.inference.compute.datascience.ch/api/v1"
    df = pd.read_csv(path, dtype={"batch_id": str})
    r = requests.post(f"{base}/submit", json={"predictions": df.to_dict(orient="records")}, auth=(team, ""), timeout=60)
    return r.status_code, r.json().get("metrics")
