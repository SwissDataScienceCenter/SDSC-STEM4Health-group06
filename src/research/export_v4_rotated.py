"""Export v4-rotated: the average of 10 v4 models, each trained on 9/10 of the 49 labeled batches.

Trial N (rotation_trials.py) found this beats a single v4 fit in 13 of 15 cross-validation folds
(mean MAPE 8.247% vs 8.290%), mostly through steadier substrate forecasts.
"""
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import export_v4_substrate as E
from pooled_ode_fit import DATA, TARGETS
from validate_feed_residuals import load_batches

HERE = Path(__file__).resolve().parent
ROTATIONS = 10
GROUPS = load_batches()[0]
SOURCE = next(root / DATA for root in HERE.parents if (root / DATA).is_file())
RAW = pd.read_csv(SOURCE)
TEMPLATE = pd.read_csv(SOURCE.parent / "bioreaction_template.csv", dtype=str)
BATCHES = list(TEMPLATE.batch_id.unique())
TEST = []
for batch in BATCHES:
    rows = RAW.loc[RAW.batch_id == batch].sort_values("elapsed_time")
    TEST.append(((rows, rows.dropna(subset=[*TARGETS, "volume"])), rows.loc[rows.to_predict, "elapsed_time"].to_numpy()))


def member(keep):
    return E.v4([GROUPS[i] for i in keep], [g for g, _ in TEST], [t for _, t in TEST])


def main():
    folds = [keep for keep, _ in KFold(ROTATIONS, shuffle=True, random_state=11).split(np.arange(len(GROUPS)))]
    with mp.get_context("fork").Pool(5) as pool:
        members = pool.map(member, folds)
    outputs = {b: np.mean([m[j] for m in members], axis=0) for j, b in enumerate(BATCHES)}
    queries = {b: t for b, (_, t) in zip(BATCHES, TEST)}
    values, used = [], set()
    for row in TEMPLATE.itertuples(index=False):
        matches = np.flatnonzero(abs(queries[row.batch_id] - float(row.elapsed_time)) <= 5e-9)
        if len(matches) != 1 or (row.batch_id, matches[0]) in used:
            raise ValueError(f"Unexpected template key: {row.batch_id} {row.elapsed_time}")
        used.add((row.batch_id, matches[0]))
        values.append(outputs[row.batch_id][matches[0]])
    values = np.stack(values)
    assert len(used) == len(TEMPLATE) == 156 and np.isfinite(values).all()
    assert (values >= 0).all() and (values[:, 1] >= values[:, 0]).all()
    out = TEMPLATE.copy(); out[TARGETS] = values
    out.to_csv(HERE / "official_candidate_v4_rotated.csv", index=False)
    single = pd.read_csv(HERE / "official_candidate_v4_substrate.csv")
    change = (100 * (out[TARGETS].astype(float) - single[TARGETS]).abs() / single[TARGETS]).mean()
    print("Wrote official_candidate_v4_rotated.csv (156 rows, checks passed)")
    print("Mean % difference from single v4:", change.round(2).to_dict())


if __name__ == "__main__":
    main()
