"""Export v4 = v3 with a physics-based substrate model added to the substrate blend.

VCD, TCD and byproduct are v3 (export_v3_mape.py). Substrate is the mean of v3's two substrate
members (exact-MAPE forest, kinetic ODE + exact-MAPE ExtraTrees) and a third member: the substrate
mass-balance ODE conditioned on the substrate-feed controller bits plus exact-MAPE ExtraTrees
(substrate_trials.py, Trial M). Since v3's substrate is the mean of its two members,
v4 substrate = (2 * v3 substrate + physics member) / 3.
The script first refits on the 39 fitting batches and scores the team benchmark, then refits on all
49 labeled batches and writes the 156 official forecasts in template order.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

import kinetic_trials as K
import substrate_trials as ST
from export_v3_mape import design, fit_v3
from pooled_ode_fit import DATA, TARGETS
from score_mape import actuals, mape
from validate_feed_residuals import load_batches

HERE = Path(__file__).resolve().parent


def physics_member(train_groups, query_groups, query_times):
    """Fit the conditioned substrate model + ExtraTrees on train_groups; predict query batches."""
    groups = train_groups + query_groups
    times = [m.elapsed_time.to_numpy()[1:] for _, m in train_groups] + list(query_times)
    y = [m[TARGETS].to_numpy()[1:] for _, m in train_groups]
    train = list(range(len(train_groups)))
    query = list(range(len(train_groups), len(groups)))
    data = ST.SubData(groups, times)
    data.vcd0 = np.array([m.iloc[0].vcd for _, m in groups])
    scale = np.vstack(y).std(axis=0)
    kin = K.fit_kinetic([K.Batch(*g, times[i]) for i, g in enumerate(train_groups)], y, scale, relative=True)
    xv = np.stack([data.viable(kin.params, b) for b in range(data.n)])
    model = ST.SubstrateModel(data, xv, train).fit(train, y).calibrate(train, y)
    cond_train, _ = model.conditioned(train)
    cond_query, _ = model.conditioned(query)
    x = np.vstack([design(groups[i], times[i]) for i in train])
    ys = np.concatenate([y[i][:, ST.S] for i in train])
    tree = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=2, max_features=.5, random_state=21, n_jobs=4,
                               criterion="absolute_error")
    tree.fit(x, ys - np.concatenate([cond_train[i] for i in train]), sample_weight=1 / ys)
    return [np.maximum(cond_query[i] + tree.predict(design(groups[i], times[i])), ST.FLOOR) for i in query]


def v4(train_groups, query_groups, query_times):
    predict = fit_v3(train_groups)
    phys = physics_member(train_groups, query_groups, query_times)
    out = []
    for g, t, p in zip(query_groups, query_times, phys):
        v = predict(g, t)
        v[:, ST.S] = (2 * v[:, ST.S] + p) / 3
        out.append(K.physical(v))
    return out


def main():
    groups, _ = load_batches()
    split = pd.read_csv(HERE / "team_benchmark_split.csv").set_index("batch_id").split
    ids = [r.batch_id.iloc[0] for r, _ in groups]
    fit = [g for g, b in zip(groups, ids) if split[b] == "fit"]
    bench = [(g, b) for g, b in zip(groups, ids) if split[b] == "benchmark"]
    times = [g[1].elapsed_time.to_numpy()[1:] for g, _ in bench]
    preds = v4(fit, [g for g, _ in bench], times)
    frame = pd.concat([pd.DataFrame(p, columns=TARGETS).assign(batch_id=b, elapsed_time=t)
                       for p, (_, b), t in zip(preds, bench, times)])
    print("Benchmark (mean MAPE %):", {k: round(float(v), 3) for k, v in mape(frame, actuals()).items()})

    source = next(root / DATA for root in HERE.parents if (root / DATA).is_file())
    raw = pd.read_csv(source)
    template = pd.read_csv(source.parent / "bioreaction_template.csv", dtype=str)
    batches = list(template.batch_id.unique())
    test_groups, test_times = [], []
    for batch in batches:
        rows = raw.loc[raw.batch_id == batch].sort_values("elapsed_time")
        start = rows.dropna(subset=[*TARGETS, "volume"])
        t = rows.loc[rows.to_predict, "elapsed_time"].to_numpy()
        if len(start) != 1 or len(t) == 0:
            raise ValueError(f"Unexpected test inputs for {batch}")
        test_groups.append((rows, start)); test_times.append(t)
    outputs = dict(zip(batches, v4(groups, test_groups, test_times)))
    queries = dict(zip(batches, test_times))
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
    bits = {b: len(d) for b, d in zip(batches, ST.SubData(test_groups, test_times).bits)}
    print("Controller bits per test batch:", bits)
    output = template.copy()
    output[TARGETS] = values
    path = HERE / "official_candidate_v4_substrate.csv"
    output.to_csv(path, index=False)
    print(f"Wrote {path.name}: {len(output)} rows, finite, nonnegative, TCD >= VCD")


if __name__ == "__main__":
    main()
