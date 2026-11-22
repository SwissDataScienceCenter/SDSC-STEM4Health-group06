"""Trial K: optimize for the leaderboard metric (mean MAPE over the four targets).

Compares, on the same repeated whole-batch folds as kinetic_trials.py:
- absolute-scale models (as submitted in v1) vs log-scale models, whose squared error approximates
  squared percentage error: forest on log(y), ExtraTrees on log(y) - log(base)
- the kinetic ODE fitted on normalized vs relative (percentage) errors
- arithmetic vs geometric blending
All tree models use the standard + look-ahead feed features.
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

import kinetic_trials as K
from direct_ml_benchmark import features
from pooled_ode_fit import TARGETS, fit_shared_ode
from validate_feed_residuals import load_batches

HERE = Path(__file__).resolve().parent
FLOOR = 1e-3
STATE = {}


def mape_score(prediction, indices, y):
    ape = 100 * np.abs(np.vstack([prediction[i] for i in indices]) - np.vstack([y[i] for i in indices])) \
        / np.vstack([y[i] for i in indices])
    per_target = ape.mean(axis=0)
    return per_target.mean(), per_target


def setup():
    groups, _ = load_batches()
    times = [m.elapsed_time.to_numpy()[1:] for _, m in groups]
    y = [m[TARGETS].to_numpy()[1:] for _, m in groups]
    batches = [K.Batch(*g, times[i]) for i, g in enumerate(groups)]
    standard = [features(*g, times[i]) for i, g in enumerate(groups)]
    ahead = [np.column_stack([standard[i], K.lookahead(groups[i][0], times[i])]) for i in range(len(groups))]
    STATE.update(groups=groups, times=times, y=y, batches=batches, ahead=ahead)


def run(train, evaluate):
    S = STATE
    groups, y, times, ahead, batches = S["groups"], S["y"], S["times"], S["ahead"], S["batches"]
    scale = np.vstack([y[i] for i in train]).std(axis=0)
    both = [*train, *evaluate]
    ode = fit_shared_ode([groups[i] for i in train])
    kinetic = K.fit_kinetic([batches[i] for i in train], [y[i] for i in train], scale)
    kinetic_rel = K.fit_kinetic([batches[i] for i in train], [y[i] for i in train], scale, relative=True)
    bases = {"ODE": {i: ode(*groups[i], times[i]) for i in both},
             "Kinetic": {i: kinetic(batches[i]) for i in both},
             "Kinetic-rel": {i: kinetic_rel(batches[i]) for i in both}}
    x_train = np.vstack([ahead[i] for i in train])
    y_train = np.vstack([y[i] for i in train])
    P = {f"{n} alone": {i: b[i] for i in evaluate} for n, b in bases.items()}

    forest = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, max_features=.75, random_state=21, n_jobs=1)
    forest.fit(x_train, y_train / scale)
    P["Forest"] = {i: forest.predict(ahead[i]) * scale for i in evaluate}
    forest_log = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, max_features=.75, random_state=21, n_jobs=1)
    forest_log.fit(x_train, np.log(y_train))
    P["Forest log"] = {i: np.exp(forest_log.predict(ahead[i])) for i in evaluate}

    def tree():
        return ExtraTreesRegressor(n_estimators=150, min_samples_leaf=2, max_features=.5, random_state=21, n_jobs=1)

    for name, base in bases.items():
        absolute = tree().fit(x_train, np.vstack([y[i] - base[i] for i in train]) / scale)
        P[f"{name} + ET"] = {i: base[i] + absolute.predict(ahead[i]) * scale for i in evaluate}
        logres = tree().fit(x_train, np.vstack([np.log(y[i]) - np.log(np.maximum(base[i], FLOOR)) for i in train]))
        P[f"{name} + ET log"] = {i: np.maximum(base[i], FLOOR) * np.exp(logres.predict(ahead[i])) for i in evaluate}

    def mean(names, geometric=False):
        if geometric:
            return {i: np.exp(np.mean([np.log(np.maximum(P[n][i], FLOOR)) for n in names], axis=0)) for i in evaluate}
        return {i: np.mean([P[n][i] for n in names], axis=0) for i in evaluate}

    P["v1: Forest + Kinetic-ET (submitted)"] = mean(["Forest", "Kinetic + ET"])
    P["3-way: Forest + ODE-ET + Kinetic-ET"] = mean(["Forest", "ODE + ET", "Kinetic + ET"])
    P["v1 log: Forest log + Kinetic-ET log"] = mean(["Forest log", "Kinetic + ET log"])
    P["v1 log-rel: Forest log + Kinetic-rel-ET log"] = mean(["Forest log", "Kinetic-rel + ET log"])
    P["v1 log-rel geometric"] = mean(["Forest log", "Kinetic-rel + ET log"], geometric=True)
    P["3-way log-rel: Forest log + ODE-ET log + Kinetic-rel-ET log"] = mean(["Forest log", "ODE + ET log", "Kinetic-rel + ET log"])
    P["3-way log-rel geometric"] = mean(["Forest log", "ODE + ET log", "Kinetic-rel + ET log"], geometric=True)
    P["4-way log: Forest log + ODE + Kinetic + Kinetic-rel (ET log)"] = mean(
        ["Forest log", "ODE + ET log", "Kinetic + ET log", "Kinetic-rel + ET log"])
    return scale, {n: {i: K.physical(v[i].copy()) for i in evaluate} for n, v in P.items()}


def job(args):
    repeat, fold, train, evaluate = args
    scale, predicted = run(np.array(train), np.array(evaluate))
    rows = []
    for name, values in predicted.items():
        score, per_target = mape_score(values, evaluate, STATE["y"])
        nrmse, _ = K.metrics(values, evaluate, STATE["y"], scale)
        rows.append({"repeat": repeat, "fold": fold, "model": name, "mean_mape": score,
                     **{f"{t}_mape": v for t, v in zip(TARGETS, per_target)}, "normalized_rmse": nrmse})
    return rows, predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    setup()
    split = pd.read_csv(HERE / "team_benchmark_split.csv").set_index("batch_id").split
    ids = [rows.batch_id.iloc[0] for rows, _ in STATE["groups"]]
    fit = np.array([i for i, b in enumerate(ids) if split[b] == "fit"])
    benchmark = np.array([i for i, b in enumerate(ids) if split[b] == "benchmark"])
    jobs = [(r + 1, f, list(fit[a]), list(fit[b])) for r in range(args.repeats)
            for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=72 + r).split(fit), 1)]
    jobs.append(("benchmark", 0, list(fit), list(benchmark)))
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(job, jobs, chunksize=1)
    cv = pd.DataFrame([row for r, _ in results[:-1] for row in r])
    cv.to_csv(HERE / "mape_cv_folds.csv", index=False)
    summary = cv.groupby("model").agg(cv_mape=("mean_mape", "mean"), cv_mape_sd=("mean_mape", "std"),
                                      cv_nrmse=("normalized_rmse", "mean"))
    bench_rows, bench_pred = results[-1]
    for row in bench_rows:
        summary.loc[row["model"], "benchmark_mape"] = row["mean_mape"]
        for t in TARGETS:
            summary.loc[row["model"], f"bench_{t}"] = row[f"{t}_mape"]
    summary.sort_values("cv_mape").to_csv(HERE / "mape_summary.csv")
    pd.DataFrame([{"model": m, "batch_id": ids[i], "elapsed_time": t, **dict(zip(TARGETS, p))}
                  for m, v in bench_pred.items() for i, est in v.items()
                  for t, p in zip(STATE["times"][i], est)]).to_csv(HERE / "mape_benchmark_predictions.csv", index=False)
    print(summary.sort_values("cv_mape").round(3).to_string())


if __name__ == "__main__":
    main()
