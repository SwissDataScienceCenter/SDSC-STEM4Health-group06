"""Refit the v3 candidate on all 49 labeled batches and export the 156 official forecasts.

v3 = v1 trained on the leaderboard loss. Mean of (a) a random forest and (b) the kinetic ODE
fitted on relative errors plus ExtraTrees residuals. Both tree models use the absolute-error
criterion with sample weight 1/y, one model per target, so they minimize MAPE exactly.
All trees use standard + look-ahead feed features.
Before exporting, the same pipeline is fitted on the 39 fitting batches and scored on the team
benchmark to confirm it reproduces mape_trials.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

import kinetic_trials as K
from direct_ml_benchmark import features
from pooled_ode_fit import DATA, TARGETS
from score_mape import actuals, mape
from validate_feed_residuals import load_batches

HERE = Path(__file__).resolve().parent
FLOOR = 1e-3


def design(group, times):
    return np.column_stack([features(*group, times), K.lookahead(group[0], times)])


def fit_v3(groups):
    times = [m.elapsed_time.to_numpy()[1:] for _, m in groups]
    y = [m[TARGETS].to_numpy()[1:] for _, m in groups]
    scale = np.vstack(y).std(axis=0)
    kinetic = K.fit_kinetic([K.Batch(*g, t) for g, t in zip(groups, times)], y, scale, relative=True)
    x = np.vstack([design(g, t) for g, t in zip(groups, times)])
    yt = np.vstack(y)
    base = np.vstack([np.maximum(kinetic(K.Batch(*g, t)), FLOOR) for g, t in zip(groups, times)])
    forests, trees = [], []
    for k in range(4):
        w = 1 / yt[:, k]
        forests.append(RandomForestRegressor(n_estimators=150, min_samples_leaf=3, max_features=.75, random_state=21,
                                             n_jobs=4, criterion="absolute_error").fit(x, yt[:, k], sample_weight=w))
        trees.append(ExtraTreesRegressor(n_estimators=150, min_samples_leaf=2, max_features=.5, random_state=21,
                                         n_jobs=4, criterion="absolute_error").fit(x, yt[:, k] - base[:, k], sample_weight=w))

    def predict(group, query_times):
        xq = design(group, query_times)
        kinetic_base = np.maximum(kinetic(K.Batch(*group, query_times)), FLOOR)
        forest_part = np.column_stack([f.predict(xq) for f in forests])
        kinetic_part = kinetic_base + np.column_stack([t.predict(xq) for t in trees])
        return K.physical((forest_part + kinetic_part) / 2)

    return predict


def main():
    groups, _ = load_batches()
    split = pd.read_csv(HERE / "team_benchmark_split.csv").set_index("batch_id").split
    ids = [r.batch_id.iloc[0] for r, _ in groups]
    # 1. Reproduce the benchmark score.
    predict = fit_v3([g for g, b in zip(groups, ids) if split[b] == "fit"])
    rows = []
    for g, b in zip(groups, ids):
        if split[b] == "benchmark":
            t = g[1].elapsed_time.to_numpy()[1:]
            rows.append(pd.DataFrame(predict(g, t), columns=TARGETS).assign(batch_id=b, elapsed_time=t))
    check = mape(pd.concat(rows), actuals())
    print("Benchmark reproduction (mean MAPE %):", {k: round(v, 3) for k, v in check.items()})
    # 2. Refit on all 49 and export in template order.
    predict = fit_v3(groups)
    source = next(root / DATA for root in HERE.parents if (root / DATA).is_file())
    raw = pd.read_csv(source)
    template = pd.read_csv(source.parent / "bioreaction_template.csv", dtype=str)
    outputs, queries = {}, {}
    for batch in template.batch_id.unique():
        rows = raw.loc[raw.batch_id == batch].sort_values("elapsed_time")
        start = rows.dropna(subset=[*TARGETS, "volume"])
        times = rows.loc[rows.to_predict, "elapsed_time"].to_numpy()
        if len(start) != 1 or len(times) == 0:
            raise ValueError(f"Unexpected test inputs for {batch}")
        queries[batch] = times
        outputs[batch] = predict((rows, start), times)
    values, used = [], set()
    for row in template.itertuples(index=False):
        matches = np.flatnonzero(abs(queries[row.batch_id] - float(row.elapsed_time)) <= 5e-9)
        if len(matches) != 1 or (row.batch_id, matches[0]) in used:
            raise ValueError(f"Unexpected template key: {row.batch_id} {row.elapsed_time}")
        used.add((row.batch_id, matches[0]))
        values.append(outputs[row.batch_id][matches[0]])
    values = np.stack(values)
    assert len(used) == len(template) == 156 and np.isfinite(values).all()
    assert (values >= 0).all() and (values[:, 1] >= values[:, 0]).all()
    output = template.copy()
    output[TARGETS] = values
    path = HERE / "official_candidate_v3_mape.csv"
    output.to_csv(path, index=False)
    print(f"Wrote {path.name}: {len(output)} rows, finite, nonnegative, TCD >= VCD")


if __name__ == "__main__":
    main()
