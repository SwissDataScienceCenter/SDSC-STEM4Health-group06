"""Error metrics and paired comparisons."""
import numpy as np
import pandas as pd


def ape(pred, y):
    """Absolute percentage error of each point, in %."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    return 100 * np.abs(pred - y) / y


def mape(pred, y):
    return float(ape(pred, y).mean())


def best_constants(y):
    """Best single-number forecast of a sample y under three losses: squared error -> mean,
    absolute error -> median, absolute percentage error -> median of the 1/y-weighted distribution."""
    y = np.sort(np.asarray(y, float))
    w = np.cumsum(1 / y) / np.sum(1 / y)
    grid = np.linspace(y.min(), y.max(), 2001)
    return pd.Series({"mean (best for squared error)": y.mean(),
                      "median (best for absolute error)": np.median(y),
                      "1/y-weighted median (best for MAPE)": y[np.searchsorted(w, 0.5)],
                      "check: grid minimum of MAPE": grid[np.argmin([ape(g, y).mean() for g in grid])]})


def per_batch(pred, table):
    """Mean APE per (split, batch): the unit of comparison, because points within a batch are correlated."""
    return pd.Series(ape(pred, table.y)).groupby([table.split.values, table.batch_id.values]).mean()


def paired(pred_new, pred_ref, table):
    """Paired comparison of two out-of-fold forecasts, batch by batch."""
    a, b = per_batch(pred_new, table), per_batch(pred_ref, table)
    d = a - b
    return pd.Series({"CV MAPE": a.mean(), "reference": b.mean(), "difference": d.mean(),
                      "standard error": d.std() / np.sqrt(len(d)), "% batches better": 100 * (d < 0).mean()})
