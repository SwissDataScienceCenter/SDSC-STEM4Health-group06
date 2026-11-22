"""Re-score saved team-benchmark predictions with the leaderboard metric (mean MAPE, %)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pooled_ode_fit import TARGETS
from validate_feed_residuals import load_batches

HERE = Path(__file__).resolve().parent


def actuals():
    split = pd.read_csv(HERE / "team_benchmark_split.csv").set_index("batch_id").split
    groups, _ = load_batches()
    rows = [m.iloc[1:][["batch_id", "elapsed_time", *TARGETS]] for r, m in groups if split[r.batch_id.iloc[0]] == "benchmark"]
    a = pd.concat(rows); a["elapsed_time"] = a.elapsed_time.round(9)
    return a


def mape(pred, actual):
    p = pred.copy(); p["elapsed_time"] = p.elapsed_time.round(9)
    j = actual.merge(p, on=["batch_id", "elapsed_time"], suffixes=("", "_p"), validate="one_to_one")
    if len(j) != len(actual):
        return None
    ape = 100 * np.abs(j[[t + "_p" for t in TARGETS]].to_numpy() - j[TARGETS].to_numpy()) / j[TARGETS].to_numpy()
    return dict(zip(TARGETS, ape.mean(0))) | {"mean_mape": ape.mean()}


def main():
    actual = actuals()
    files = sys.argv[1:] or sorted(str(f) for f in HERE.glob("*benchmark*predictions*.csv"))
    rows = []
    for f in files:
        d = pd.read_csv(f)
        groups = d.groupby("model") if "model" in d else [(Path(f).stem, d)]
        for model, g in groups:
            r = mape(g, actual)
            if r:
                rows.append({"file": Path(f).name, "model": model, **r})
    out = pd.DataFrame(rows).sort_values("mean_mape")
    out.to_csv(HERE / "benchmark_mape_scoreboard.csv", index=False)
    print(out.head(25).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
