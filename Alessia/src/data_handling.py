"""Data handling: load the bioreactor dataset, build per-batch packages and the batch information (features) used by the models."""

from pathlib import Path # file paths

import numpy as np # numerics
import pandas as pd # tables

DATA_DIR = Path("/home/renku/work/input-data/Project 1 - DTB") # challenge data folder
FEED_COLS = ["feed_substrate", "feed_b", "feed_c", "feed_d"] # feed addition columns
SAMPLE_COLS = ["vcd", "tcd", "substrate", "byproduct", "volume"] # daily sample columns
TARGET_COLS = ["vcd", "tcd", "substrate", "byproduct"] # variables to predict
KEYS = ["batch_id", "batch_num", "elapsed_time"] # row identifiers


# read csv, split into daily samples and feed additions, flag test batches
def load_data(path=DATA_DIR / "bioreaction_remaining.csv"):
    raw = pd.read_csv(path) # raw long-format table
    raw["batch_num"] = raw["batch_id"].str.replace("Batch", "").astype(int) # numeric batch id for sorting
    test_ids = raw.loc[raw["to_predict"], "batch_id"].unique() # batches with rows to predict
    raw["is_test"] = raw["batch_id"].isin(test_ids) # test-batch flag
    is_sample = raw["volume"].notna() # sample rows always have a volume
    meas = raw.loc[is_sample, KEYS + SAMPLE_COLS + ["to_predict", "is_test"]] # daily samples
    feeds = raw.loc[~is_sample, KEYS + FEED_COLS + ["is_test"]] # feed additions
    feeds = feeds.fillna({c: 0.0 for c in FEED_COLS}) # empty feed cell = nothing added
    meas = meas.sort_values(["batch_num", "elapsed_time"]).reset_index(drop=True) # numeric order
    feeds = feeds.sort_values(["batch_num", "elapsed_time"]).reset_index(drop=True) # numeric order
    return meas, feeds


# list labeled (test=False) or to-predict (test=True) batch ids in numeric order
def batch_ids(meas, test=False):
    ids = meas.loc[meas["is_test"] == test, ["batch_id", "batch_num"]].drop_duplicates() # one row per batch
    return ids.sort_values("batch_num")["batch_id"].tolist() # sorted ids


# start state, prediction times, measurements, volume and feed events of one batch
def get_batch(meas, feeds, batch_id):
    m = meas[meas["batch_id"] == batch_id].reset_index(drop=True) # samples of this batch
    f = feeds[feeds["batch_id"] == batch_id] # feeds of this batch
    day0 = m.iloc[0] # first sample
    x0 = day0[SAMPLE_COLS].astype(float).to_dict() # day-0 values
    x0["dead"] = x0["tcd"] - x0["vcd"] # dead cell density at day 0
    events = f.melt(id_vars="elapsed_time", value_vars=FEED_COLS, var_name="feed", value_name="amount") # one row per feed and time
    events = events[events["amount"] > 0].sort_values("elapsed_time").reset_index(drop=True) # real additions only
    return {
        "batch_id": batch_id, # batch name
        "is_test": bool(day0["is_test"]), # to-predict flag
        "x0": x0, # starting point
        "t_pred": m["elapsed_time"].to_numpy()[1:], # sample times after day 0
        "y_meas": m.loc[1:, TARGET_COLS].reset_index(drop=True), # measured targets after day 0 (NaN for to-predict batches)
        "volume": m[["elapsed_time", "volume"]], # volume at every sample
        "feed_events": events, # feeding plan
    }


# batch information known before predicting: day-0 values and feed totals (one row per batch)
def batch_features(meas, feeds):
    day0 = meas[meas["elapsed_time"] == 0].set_index("batch_id") # day-0 sample of every batch
    feat = day0[["vcd", "byproduct", "substrate", "volume"]].add_suffix("0") # start values
    g = feeds.groupby("batch_id") # feed additions per batch
    feat["fb_tot"] = g["feed_b"].sum() # total feed B
    feat["fc_n"] = g["feed_c"].apply(lambda s: (s > 0).sum()) # number of feed C additions
    feat["fs_tot"] = g["feed_substrate"].sum() # total substrate feed
    feat["fs_t1"] = feeds[feeds["feed_substrate"] > 0].groupby("batch_id")["elapsed_time"].min() # time of first substrate feed
    return feat.fillna(feat.mean()) # safety: missing -> mean


# feed-shape information: profile of feed B (dosed on measured cells) and feed C timing (one row per batch)
def feed_shape_features(meas, feeds):
    rows = {} # one dict per batch
    for bid, m in meas.groupby("batch_id"): # loop over batches
        f = feeds[feeds["batch_id"] == bid] # feeds of this batch
        t, vol = m["elapsed_time"].to_numpy(), m["volume"].to_numpy() # sample times and volumes
        fbv = np.array([f.loc[(f["feed_b"] > 0) & (f["elapsed_time"] > t[k]) & (f["elapsed_time"] < t[k] + 0.5), "feed_b"].sum() / vol[k] for k in range(len(t))]) # feed B right after each sample, per volume
        day = np.round(t) # day index

        def fb_mean(d1, d2): # mean feed B/V between two days (dosed days only)
            return fbv[(day >= d1) & (day <= d2) & (fbv > 0)].mean()

        fc_t = f.loc[f["feed_c"] > 0, "elapsed_time"] # feed C addition times
        rows[bid] = {"fb_ramp": np.log(fb_mean(4, 4) / fb_mean(1, 1)) / 3, # early growth of feed B
                     "fb_plateau": fb_mean(6, 9), # plateau level
                     "fb_late": fb_mean(10, 12), # late level
                     "fc_early": float((fc_t < 5).sum()), # feed C additions before day 5
                     "fc_t1": fc_t.min(), # first feed C addition
                     "viab0": m["vcd"].iloc[0] / m["tcd"].iloc[0]} # viability at day 0
    return pd.DataFrame(rows).T.astype(float)


# per-sample feed information: what was fed shortly before each sample (one row per batch and sample)
def day_features(meas, feeds):
    rows = [] # one row per sample after day 0
    for bid, m in meas.groupby("batch_id"): # loop over batches
        f = feeds[feeds["batch_id"] == bid] # feeds of this batch
        for t, v in zip(m["elapsed_time"].iloc[1:], m["volume"].iloc[1:]): # samples after day 0
            last = f[(f["elapsed_time"] < t) & (f["elapsed_time"] > t - 1)] # additions in the 24 h before the sample
            rows.append({"batch_id": bid, "elapsed_time": t, # identifiers
                         "fs_24h": last["feed_substrate"].sum() / v, # recent substrate feed per volume
                         "fb_24h": last["feed_b"].sum() / v, # recent feed B per volume
                         "fs_cum": f.loc[f["elapsed_time"] < t, "feed_substrate"].sum() / v}) # cumulative substrate feed per volume
    return pd.DataFrame(rows)


# everything the models need, built once
class Data:
    # load the data and precompute packages and features
    def __init__(self, path=DATA_DIR / "bioreaction_remaining.csv"):
        self.meas, self.feeds = load_data(path) # samples and feeds
        self.labeled = batch_ids(self.meas) # 49 batches with measurements
        self.to_predict = batch_ids(self.meas, test=True) # 12 batches to submit
        self.pk = {b: get_batch(self.meas, self.feeds, b) for b in self.labeled + self.to_predict} # all 61 packages
        self.feat = batch_features(self.meas, self.feeds) # batch features
        self.feat_shape = feed_shape_features(self.meas, self.feeds) # feed-shape features
        self.feat_day = day_features(self.meas, self.feeds) # per-sample feed features

    # packages of a list of batches
    def packages(self, ids):
        return {b: self.pk[b] for b in ids} # subset of packages

    # measured samples of a list of batches
    def samples(self, ids):
        return self.meas[self.meas["batch_id"].isin(ids)] # subset of samples
