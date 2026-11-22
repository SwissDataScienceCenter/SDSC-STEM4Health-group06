"""Test a Feed B/C residual correction on unseen batch trajectories."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from pooled_ode_fit import DATA, TARGETS, fit_shared_ode

HERE = Path(__file__).resolve().parent


def load_batches():
    source = next(root / DATA for root in HERE.parents if (root / DATA).is_file())
    data = pd.read_csv(source)
    test_ids = set(data.loc[data.to_predict, "batch_id"])
    groups, feeds = [], []
    for batch, rows in data.loc[~data.batch_id.isin(test_ids)].groupby("batch_id"):
        rows = rows.sort_values("elapsed_time")
        measured = rows.dropna(subset=[*TARGETS, "volume"])
        groups.append((rows, measured))
        feeds.append(rows[["feed_b", "feed_c"]].sum().to_numpy())
    feeds = np.array(feeds)

    return groups, feeds


def correction_design(t, feed):
    day = t / 13
    return (np.column_stack([day, day**2, day**3])[:, :, None] * feed).reshape(len(t), -1)


def fit_density_models(groups, feeds, train):
    predict = fit_shared_ode([groups[i] for i in train])
    scaler = StandardScaler().fit(feeds[train])
    standardized = scaler.transform(feeds)
    x, residuals = [], []
    for i in train:
        measured = groups[i][1].iloc[1:]
        t = measured.elapsed_time.to_numpy()
        actual = measured[TARGETS[:2]].to_numpy()
        x.append(correction_design(t, standardized[i]))
        residuals.append(actual - predict(*groups[i], t)[:, :2])
    correction = Ridge(alpha=10, fit_intercept=False).fit(np.vstack(x), np.vstack(residuals))

    def forecast(i, t):
        baseline = predict(*groups[i], t)[:, :2]
        adjusted = baseline + correction.predict(correction_design(t, standardized[i]))
        adjusted[:, 0] = np.maximum(adjusted[:, 0], 0)
        adjusted[:, 1] = np.maximum(adjusted[:, 1], adjusted[:, 0])
        return baseline, adjusted

    return forecast


def main():
    groups, feeds = load_batches()

    results = []
    for fold, (train, held) in enumerate(KFold(5, shuffle=True, random_state=6).split(groups), 1):
        forecast = fit_density_models(groups, feeds, train)
        for i in held:
            measured = groups[i][1].iloc[1:]
            t = measured.elapsed_time.to_numpy()
            actual = measured[TARGETS[:2]].to_numpy()
            baseline, adjusted = forecast(i, t)
            median = np.median([
                np.column_stack([np.interp(t, groups[j][1].elapsed_time, groups[j][1][target])
                                 for target in TARGETS[:2]])
                for j in train
            ], axis=0)
            for name, prediction in [("median", median), ("shared_ode", baseline), ("feed_bc", adjusted)]:
                for k, target in enumerate(TARGETS[:2]):
                    error = prediction[:, k] - actual[:, k]
                    results.append({"fold": fold, "batch_id": groups[i][0].batch_id.iloc[0],
                                    "model": name, "target": target, "n": len(error),
                                    "sse": np.sum(error**2), "sae": np.sum(np.abs(error))})
    result = pd.DataFrame(results)
    result.to_csv(HERE / "feed_residual_cv.csv", index=False)
    summary = result.groupby(["target", "model"])[["n", "sse", "sae"]].sum()
    summary["rmse"] = np.sqrt(summary.sse / summary.n)
    summary["mae"] = summary.sae / summary.n
    summary[["rmse", "mae"]].to_csv(HERE / "feed_residual_cv_summary.csv")
    print(summary[["rmse", "mae"]].round(3))
    rng = np.random.default_rng(6)
    for target in TARGETS[:2]:
        subset = result.loc[result.target == target]
        pairs = subset.pivot(index="batch_id", columns="model", values="sse")
        n = subset.loc[subset.model == "shared_ode"].set_index("batch_id").loc[pairs.index, "n"].to_numpy()
        indices = rng.integers(0, len(n), size=(10000, len(n)))
        rmse = np.sqrt(pairs.to_numpy()[indices].sum(axis=1) / n[indices].sum(axis=1)[:, None])
        improvement = rmse[:, pairs.columns.get_loc("shared_ode")] - rmse[:, pairs.columns.get_loc("feed_bc")]
        print(target, "batch bootstrap interval for RMSE improvement:", np.quantile(improvement, [.025, .975]).round(3))


if __name__ == "__main__":
    main()
