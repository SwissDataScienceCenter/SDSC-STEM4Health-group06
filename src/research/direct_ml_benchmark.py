"""Compare data-only tree ensembles on the fixed Project 1 batch holdout."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold

from pooled_ode_fit import FEEDS, TARGETS
from validate_feed_residuals import load_batches

HERE = Path(__file__).resolve().parent


def features(rows, measured, times):
    initial = measured.iloc[0][[*TARGETS, "volume"]].to_numpy(dtype=float)
    columns = [times, *np.broadcast_to(initial[:, None], (len(initial), len(times)))]
    for feed in FEEDS:
        events = rows.loc[rows[feed] > 0, ["elapsed_time", feed]]
        ts, dose = events.to_numpy().T
        cumulative = np.r_[0, dose.cumsum()]
        latest = np.searchsorted(ts, times, side="left")
        columns.extend([latest, cumulative[latest]])
        for window in (.25, 1, 2):
            previous = np.searchsorted(ts, times - window, side="left")
            columns.append(cumulative[latest] - cumulative[previous])
        had_feed = latest > 0
        last_dose = np.zeros(len(times))
        age = np.full(len(times), 14.0)
        last_dose[had_feed] = dose[latest[had_feed]-1]
        age[had_feed] = times[had_feed] - ts[latest[had_feed]-1]
        columns.extend([last_dose, age])
    return np.column_stack(columns)


def main():
    groups, _ = load_batches()
    split = pd.read_csv(HERE / "density_holdout_split.csv").set_index("batch_id").split
    xx, yy, ids, times = [], [], [], []
    for rows, measured in groups:
        later = measured.iloc[1:]
        t = later.elapsed_time.to_numpy()
        xx.append(features(rows, measured, t))
        yy.append(later[TARGETS].to_numpy())
        ids.extend([rows.batch_id.iloc[0]] * len(t))
        times.extend(t)
    x, y, ids, times = np.vstack(xx), np.vstack(yy), np.array(ids), np.array(times)
    train = np.array([split[batch] == "fit" for batch in ids])
    x_train, y_train = x[train], y[train]
    candidates = {
        f"{name}, leaf={leaf}, features={fraction:g}": model(
            n_estimators=200, min_samples_leaf=leaf, max_features=fraction,
            random_state=21, n_jobs=4)
        for name, model in [("ExtraTrees", ExtraTreesRegressor), ("Random forest", RandomForestRegressor)]
        for leaf in (2, 5)
        for fraction in (.5, 1.)
    }
    scores = {}
    for name, estimator in candidates.items():
        squared = []
        for a, b in GroupKFold(5).split(x_train, groups=ids[train]):
            scale = y_train[a].std(axis=0)
            model = clone(estimator).fit(x_train[a], y_train[a] / scale)
            squared.append((model.predict(x_train[b]) - y_train[b]/scale)**2)
        scores[name] = np.sqrt(np.vstack(squared).mean())
    choice = min(scores, key=scores.get)
    scale = y_train.std(axis=0)
    fitted = candidates[choice].fit(x_train, y_train/scale)
    predicted = fitted.predict(x[~train]) * scale
    predictions = pd.DataFrame(
        [{"batch_id": batch, "day": t, "target": target, "actual": actual[j], "direct_ml": pred[j]}
         for batch, t, actual, pred in zip(ids[~train], times[~train], y[~train], predicted)
         for j, target in enumerate(TARGETS)]
    )
    ode = pd.read_csv(HERE / "holdout_all_output_predictions.csv")
    # The exporters use the same source times. Round to protect the CSV round trip.
    for frame in (predictions, ode):
        frame["day"] = frame.day.round(10)
    merged = predictions.merge(ode.drop(columns="actual"), on=["batch_id", "day", "target"],
                               validate="one_to_one")
    if len(merged) != len(predictions):
        raise ValueError("Could not align all held-out predictions with ODE results")
    recent = pd.read_csv(HERE / "last_feed_holdout_predictions.csv").rename(columns={"elapsed_time": "day"})
    recent["day"] = recent.day.round(10)
    merged = merged.merge(recent[["batch_id", "day", "target", "last_feed"]],
                          on=["batch_id", "day", "target"], how="left", validate="one_to_one")
    merged.to_csv(HERE / "direct_ml_holdout_predictions.csv", index=False)
    models = {"ode": "Shared ODE", "last_feed": "Last-feed ODE", "direct_ml": "Data-only ML"}
    metrics = []
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout="constrained")
    fig.suptitle("Data-only ML benchmark · 10 held-out batches", fontweight="bold")
    palette = {"ode": "#64798a", "last_feed": "#a35235", "direct_ml": "#156a6e"}
    for j, target in enumerate(TARGETS):
        p = merged.loc[merged.target == target]
        top, bottom = axes[:, j]
        end = p[["actual", *models]].max().max() * 1.04
        top.plot([0, end], [0, end], ":", color=".45")
        bottom.axhline(0, linestyle=":", color=".45")
        for model, label in models.items():
            available = p.dropna(subset=[model])
            if available.empty:
                continue
            error = available[model] - available.actual
            rmse = np.sqrt(np.mean(error**2))
            metrics.append([target, label, rmse, np.mean(np.abs(error))])
            top.scatter(available.actual, available[model], s=13, alpha=.42, color=palette[model],
                        label=f"{label} {rmse:.2f}")
            bottom.scatter(available.day, error, s=13, alpha=.42, color=palette[model])
        top.set(title=target.upper(), xlabel="Measured", ylabel="Predicted", xlim=(0, end), ylim=(0, end))
        top.legend(frameon=False, fontsize=8)
        bottom.set(xlabel="Elapsed day", ylabel="Predicted − measured")
        for ax in (top, bottom):
            ax.grid(alpha=.16)
            ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / "direct_ml_holdout.png", dpi=175)
    summary = pd.DataFrame(metrics, columns=["target", "model", "rmse", "mae"])
    summary.to_csv(HERE / "direct_ml_holdout_metrics.csv", index=False)
    print("Selected:", choice)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
