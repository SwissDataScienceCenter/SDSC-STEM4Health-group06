"""Loading the challenge data and deriving the controller signals used in the notebook."""
import numpy as np
import pandas as pd

from . import DATA_DIR, TARGETS

FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
SUBSTRATE_THRESHOLD = 0.966          # substrate level at which the substrate-feed controller switches dose


def load_raw():
    """Event table: one row per sample, feed or volume reading. `to_predict` marks the test sample times."""
    return pd.read_csv(DATA_DIR / "bioreaction_remaining.csv")


def test_batches(raw):
    return sorted(raw.loc[raw.to_predict, "batch_id"].unique())


def training_batches(raw):
    return sorted(set(raw.batch_id) - set(test_batches(raw)))


def samples(raw):
    """Measured samples (training batches) with their day number."""
    s = raw.dropna(subset=TARGETS).copy()
    s["day"] = s.elapsed_time.round().astype(int)
    return s


def overview(raw):
    """Counts that describe the task."""
    test, train = test_batches(raw), training_batches(raw)
    return pd.Series({
        "training batches": len(train),
        "test batches": len(test),
        "measured samples (training)": int(raw.dropna(subset=TARGETS).shape[0]),
        "samples to forecast": int(raw.to_predict.sum()),
        "feed events": int(raw[FEEDS].notna().any(axis=1).sum()),
        "run length (days)": round(raw.elapsed_time.max(), 1),
    })


def feed_events(raw, batch):
    g = raw[raw.batch_id == batch]
    return {f: g.loc[g[f].notna(), ["elapsed_time", f]].to_numpy() for f in FEEDS}


def substrate_controller_events(raw):
    """Every substrate-feed bolus, with the dose per culture volume and the substrate measured at the
    preceding sample (NaN for test batches, where that sample is what we must forecast)."""
    rows = []
    for b, g in raw.groupby("batch_id"):
        g = g.sort_values("elapsed_time")
        sample_rows = g[g.vcd.notna() | g.to_predict]
        vol = g.dropna(subset=["volume"])
        for _, f in g[g.feed_substrate.notna()].iterrows():
            s = sample_rows[sample_rows.elapsed_time <= f.elapsed_time].iloc[-1]
            v = np.interp(f.elapsed_time, vol.elapsed_time, vol.volume)
            rows.append(dict(batch_id=b, bolus_time=f.elapsed_time, sample_time=s.elapsed_time,
                             substrate_at_sample=s.substrate, test=bool(s.to_predict),
                             dose_per_volume=f.feed_substrate / v))
    ev = pd.DataFrame(rows)
    ev["high_dose"] = ev.dose_per_volume > 0.005      # the two dose levels are 0.004 and 0.006 per volume
    return ev


def base_volume_profile(raw):
    """Volume added in each sampling interval that the four listed feeds do not explain.

    Volume changes come from feeds, sampling and base added by the pH controller. The feeds are listed;
    sampling is nearly constant; so the per-interval residual, relative to the typical interval, is a
    proxy for the base added - which neutralises acid, i.e. the byproduct the culture produced."""
    prof = {}
    for b, g in raw.groupby("batch_id"):
        g = g.sort_values("elapsed_time")
        v = g.dropna(subset=["volume"])
        t, V = v.elapsed_time.to_numpy(), v.volume.to_numpy()
        fed = np.array([g.loc[g.elapsed_time < ti, FEEDS].sum().sum() for ti in t])
        prof[b] = np.diff(V - V[0] - fed)[:13]
    P = pd.DataFrame(prof, index=[f"interval {k}" for k in range(1, 14)]).T
    return P - P.loc[training_batches(raw)].median()
